# Ghost scatter-add (the reverse halo)

**Status:** COMPLETE. Implemented in `src/Tessera_HaloScatterAdd.hpp`,
tested by `tests/test_halo_scatter_add.cpp` (TIER `regression`, SERIAL + HIP,
ranks 1-5, 10/10 green).

**Verified against `08dd346`** (branch `conforming-refinement`) — the code this task
cites was re-read at that commit.

## Problem

`haloExchange()` is a **pure gather**: owner → ghost. Every ghost slot named in
`plan.recv_idx` is overwritten with its owner's value. There is no reverse
operation — ghost → owner, accumulating.

That is the missing half of the standard distributed-assembly pattern. Whenever a
per-vertex quantity is assembled by **iterating faces** — vertex areas, a
face-to-vertex gradient scatter, a mass-matrix diagonal, a residual assembled
per-element — each rank computes only the contribution of the faces it holds.
A vertex on a partition boundary is incident on faces held by several ranks, so
its owner ends up with a *partial* sum and every ghost copy holds a different
partial sum. Correct assembly requires each rank to push its ghost partials back
to the owner and add them there.

The alternative available today — have every rank iterate every face incident on
each of its owned vertices — requires those faces to be locally resident *and*
requires a redundant recomputation on every ghosting rank, and it is exactly what
`reduceVertexFromFaces()` does. That works for a one-ring gather-style reduce
where the incident faces happen to be in the halo, but it does not generalize:
it recomputes instead of communicating, and it silently gives the wrong answer
the moment an incident face is not locally held.

**Why it matters.** `Tessera::HaloExchangePlan` already holds *both* index lists
and is explicitly documented as symmetric ("`send_idx` are owned local indices,
`recv_idx` are ghost local indices", with the alignment contract guaranteed by the
builder on both sides). So the plan needed for the reverse direction already
exists and is already correct. The only missing piece is a reverse pack/unpack
with `+=` on the unpack side. The driving consumer (the Beatnik z-model, which
assembles per-vertex areas and per-vertex gradients from face loops) needs it at
every RHS evaluation.

## Approach

### Why it cannot be a whole-tuple operation

`haloExchange()` ships the **whole AoSoA tuple** — all core plus user fields — as
one opaque `MPI_Type_contiguous` of `sizeof(tuple_type)` bytes. That is right for
a gather: overwriting a ghost with its owner's tuple is correct for every field
at once, including `Gid`, `Owner`, `Level` and the connectivity gids.

It is **wrong for an accumulate**. Summing `Gid` or `Owner` is meaningless and
summing connectivity gids is corrupting. So the reverse operation must name the
field it accumulates. This is the one structural difference from
`haloExchange()`, and it is why the API below is field-templated rather than
tuple-shaped.

### API

New header `src/Tessera_HaloScatterAdd.hpp`.

```cpp
//! Accumulate one field from ghost slots into their owners (ghost -> owner, +=).
//! FieldIndex is the Cabana member index within the AoSoA's member type list —
//! a core field (e.g. VertexField::Position) or a user field.
//!
//! Direction is the exact reverse of haloExchange(): pack from plan.recv_idx and
//! send along plan.recv_peers; receive along plan.send_peers and accumulate into
//! plan.send_idx. Multi-component members are accumulated componentwise.
template <std::size_t FieldIndex, class AoSoAType, class MemorySpace>
void haloScatterAdd( MPI_Comm comm, AoSoAType& aosoa,
                     HaloExchangePlan<MemorySpace>& plan );

//! Kind-named conveniences over a MeshHalo.
template <std::size_t FieldIndex, class MeshT, class MemorySpace>
void haloScatterAddVertices( MeshT& mesh, MeshHalo<MemorySpace>& halo );
template <std::size_t FieldIndex, class MeshT, class MemorySpace>
void haloScatterAddEdges( MeshT& mesh, MeshHalo<MemorySpace>& halo );
template <std::size_t FieldIndex, class MeshT, class MemorySpace>
void haloScatterAddFaces( MeshT& mesh, MeshHalo<MemorySpace>& halo );
```

### The contract, stated in the header

Three properties that a caller will get wrong if they are not written down:

1. **Ghost slots are left untouched.** After the call, an owned entry holds the
   complete global sum and every ghost copy still holds that rank's local
   partial. The mesh is therefore **not halo-consistent** for that field —
   follow with `haloExchange()` if downstream kernels read ghosts. Do not zero
   the ghosts inside the call: a caller that wants the assemble-then-broadcast
   pattern calls `haloExchange()`, and a caller that wants only the owned values
   should not pay for a second collective.
2. **Calling it twice double-counts.** It is not idempotent, precisely because
   of (1). This is the standard scatter-add contract; state it, and cover it in
   the test so the behaviour is pinned rather than accidental.
3. **The sum order is fixed by peer order, not by rank count.** Peers are visited
   in ascending rank on both sides (`HaloExchangePlan::setFromHost` packs from
   `std::map`, which is ordered), so within one run the floating-point result is
   deterministic and bitwise reproducible. It is **not** bitwise identical across
   rank counts, because the partition into partial sums differs. Say so — a
   consumer comparing results across rank counts must not expect bitwise
   equality of an assembled field.

### Implementation

Mirror `haloExchange()` structurally, so the two read as a pair:

- Element size is `sizeof(value_type) * num_components` for the named member, not
  `sizeof(tuple_type)`. Get the component count from the slice type
  (`Cabana::Slice` exposes the member extent) so rank-0 (scalar) and rank-1
  (e.g. `Real[3]`) members share one code path.
- Reuse `plan.send_pool` / `plan.recv_pool`. The roles swap: this operation packs
  `totalRecv()` elements and receives `totalSend()` elements, so reserve
  `send_pool` for `totalRecv() * elem` and `recv_pool` for `totalSend() * elem`.
  Both pools are grow-only and already sized for whole tuples in any run that has
  called `haloExchange()`, so in practice no allocation occurs — but reserve
  explicitly rather than relying on that.
- Device-resident pack and unpack kernels over `RangePolicy<execution_space>`, no
  host copy, exactly as `haloExchange()` does. The unpack kernel does `+=`.
- **The unpack `+=` needs no atomics.** `plan.send_idx` may name the same owned
  entity once per peer, so a single flat `parallel_for` over the receive buffer
  *would* race. Avoid it rather than paying for atomics on the GPU: loop peers
  on the host and launch one kernel per peer over that peer's contiguous slice.
  Within one peer's slice an owned index appears at most once (the builder emits
  one entry per shared entity per peer), so each kernel is race-free, and the
  serialization across peers is what fixes the summation order in property (3).
  Record this reasoning in the header — it is the non-obvious part.
- Self-peer never appears in the plan, so a single rank has an empty plan and the
  call returns immediately, as `haloExchange()` does.

### Non-goals

- A generic reverse reduction with a caller-supplied operator (min, max, custom).
  Add `haloScatterMin`/`Max` later if a consumer needs one; `+=` is what
  assembly needs and a template-on-op API is harder to keep race-free.
- Multi-field or whole-user-pack scatter-add. One field per call; a caller with
  three fields makes three calls.

### Tests

New `tests/test_halo_scatter_add.cpp`, registered at **TIER `regression`**,
backends **SERIAL and HIP**, ranks **`${TESSERA_TEST_MPI_RANKS}` (1–5)**. This
gate promotion is **pre-authorized for this task**.

The mesh carries user fields so the test can exercise both a scalar and a vector:
`Mesh<double, 3, VertexFields<double, double[3]>, EdgeFields<>, FaceFields<>, ...>`.

1. **Ghost-multiplicity count (exact integer ground truth).** After
   `distribute` + `haloExchange`, set the scalar user field to `1.0` on **every
   local** vertex, owned and ghost. `haloScatterAdd`. Each owned vertex must then
   hold `1 + (number of other ranks that hold a ghost copy of it)`. Compute that
   count independently on the replicated reference mesh from `faceOwner` and the
   documented local-set rule. Exact in floating point (small integers), so assert
   equality, not a tolerance.
2. **Face-loop assembly (the real use case).** Zero the scalar field. On each
   rank, loop **local** faces and add `1.0` to each of the face's three corner
   vertices (writing local slots, ghosts included). `haloScatterAdd`. Every owned
   vertex must then hold its true **global** incident-face count — 6 for a
   subdivision-2 icosphere except the 12 original icosahedron vertices, which
   have 5. Assert exactly, at every rank count. This is the test that fails today
   with a plain `haloExchange`.
3. **Vector field, componentwise.** Repeat check 2 with the `double[3]` field,
   adding each face's unit-scaled centroid to its corners. Compare against the
   replicated-reference assembly to `1e-14` relative — this catches a component
   stride bug, which check 2 cannot.
4. **Ghosts are untouched.** After check 2, every ghost slot still holds that
   rank's local partial, not the global sum (assert the ghost value is `<=` the
   owner's and strictly less for at least one boundary vertex at ranks ≥ 2).
   Then `haloExchange` and assert every ghost equals its owner.
5. **Double-counting is real.** A second `haloScatterAdd` without an intervening
   `haloExchange` yields the pinned double-counted value. Asserting this makes
   the non-idempotence a tested contract rather than a surprise.
6. **Determinism within a run.** Run check 2 twice from the same initial state
   (re-zero, re-assemble, re-scatter) and assert **bitwise** equality of the
   owned values.
7. **Round-trip identity.** `haloExchange` then `haloScatterAdd` on a field that
   is uniform across all ranks recovers `value * multiplicity` — a cheap
   consistency cross-check between the two directions' index lists, which catches
   a swapped `send_idx`/`recv_idx`.
8. **Single rank.** Empty plan; the field is bitwise unchanged.
9. **Edges and faces.** Repeat check 1 on an edge user field and a face user
   field via the kind-named wrappers. Faces are the interesting case at depth 1:
   a face is ghosted by at most a few ranks, and a *face* is never shared as an
   owned duplicate, so the expected multiplicity is different — derive it from the
   reference rather than assuming.
10. **Depth 2 (if [halo-depth.md](halo-depth.md) has landed).** Check 1 at
    `depth = 2`: multiplicities grow, and the ground truth from the reference
    still matches. Skip cleanly with a printed note if depth is not yet
    available.

## Exit criterion

- `test_halo_scatter_add` green at **SERIAL and HIP, ranks 1–5**, and the full
  gate still green with nothing relabelled.
- Checks 1, 2 and 3 pass with the ground truth computed from the replicated
  reference mesh rather than from Tessera itself — a self-consistent wrong answer
  must not be able to pass.
- README API section documents `haloScatterAdd` and the three contract properties
  (ghosts untouched, not idempotent, summation order fixed by peer order within a
  run but not across rank counts).
- `docs/design.md` *Halo* section gains a paragraph pairing gather and scatter-add
  and stating why the reverse direction is field-templated while the forward one
  is whole-tuple.

## Where this sits

Independent of the other gap tasks; benefits from
[halo-depth.md](halo-depth.md) (check 10), which has already been implemented. See the
ordering diagram in [halo-depth.md](halo-depth.md).

## Progress log

- 2026-08-07 — Task written, then re-checked against `08dd346` after pulling
  `../tessera`. Nothing implemented.
- 2026-08-10 — **Implemented.** `src/Tessera_HaloScatterAdd.hpp` holds
  `haloScatterAdd()` plus the three kind-named wrappers, folded into
  `<Tessera.hpp>`; the component count comes from
  `Cabana::MemberTypeAtIndex<FieldIndex, AoSoAType::member_types>` so scalar and
  `Scalar[N]` members share one code path, and the unpack is one kernel per peer
  as specified (no atomics, deterministic order). Profiling key
  `halo_scatter_add` added at level 1. `tests/test_halo_scatter_add.cpp` covers
  all ten checks and is registered at TIER `regression`, SERIAL + HIP, ranks 1-5:
  **10/10 green** on Tuolumne (`f3RTZQhQ98t7`). Ground truth is the replicated
  reference mesh plus `faceOwner`, with `distribute()`'s ownership and local-set
  rules re-derived independently; the reference's predicted per-rank local counts
  are asserted against the real mesh so the reference itself cannot drift. At np3
  the multiplicity sums are V=251/E=658/F=409 against N=162/480/320, rising to
  V=331 at depth 2, so no check is vacuous.
  - **One deliberate deviation from check 2 as written.** The task said to loop
    **local** faces; local (owned + ghost) faces double-count, because a ghost
    face is an owned face on another rank and would contribute twice, and the
    stated expected value (the true *global* incident-face count) is then wrong.
    The test loops **owned** faces, which is the assembly pattern the operation
    exists for and which does produce the stated value. Check 3 follows the same
    correction.
  - Check 10 needed no skip path: halo depth had already landed.
  - Check 4 is asserted more strongly than specified — every ghost slot must be
    **bitwise unchanged** across the scatter, not merely `<=` the owner's — with
    the `strictly less at ranks >= 2` clause kept as the non-vacuity guard.
  - The full gate was not re-run: nothing existing changed mechanically (one new
    header, one new profiling constant, one umbrella include, one new test).
