# Configurable halo depth

**Status:** DONE (2026-08-09). See *Progress log*. Foundational — several sibling
tasks below depend on it.

**Verified against `08dd346`** (branch `conforming-refinement`), where
`Tessera_HaloRebuild.hpp` already exists. See *What already landed* before
starting.

## Problem

The ghost layer is **1-deep and not configurable**. That covers a one-ring
stencil exactly and nothing wider.

This is already a recorded defect. README *Known Issues* carries it verbatim:

> **`buildVertexStencil(mesh, 2)` (k=2) is incomplete within one hop of a
> partition boundary.** […] the missing outer-ring neighbours are silently absent
> from its CSR row rather than reported as an error. […] Workaround: either widen
> the halo to depth ≥ k before building the stencil, or restrict k=2 stencils to
> single-rank runs.

The first half of that workaround is not currently expressible — there is no way
to build a mesh with a 2-deep halo. So the entry is really a statement that the
capability is missing, and **this task is what deletes it.**

The failure mode is why it matters more than the wording suggests: a short CSR
row looks exactly like a correct one. An operator built on a `k=2` stencil then
produces a plausible field with a small error localized on partition boundaries —
an error that moves when the rank count changes and that no existing invariant
check detects.

**Two successive `haloExchange()` calls do not substitute.** The second call
refreshes the *same* 1-deep ghost set from the same owners; it does not widen the
set. Depth is a property of the local entity closure built by `rebuildHalo()` and
`distribute()`, not of how many times the plan is replayed.

**Why it matters.** A higher-order surface solver's right-hand side is routinely
a two-ring stencil — one surface gradient produces a tangential vector field, a
second gradient is taken of a scalar derived from it. The driving consumer (the
Beatnik z-model, an evolving vortex sheet on an unstructured triangle surface)
has exactly this shape, so today it cannot evaluate its RHS correctly at more
than one rank. Any consumer needing a wider-than-one-ring operator hits the same
wall.

## Approach

### What already landed — read this first

An earlier revision of this task also proposed extracting the halo rebuild out of
`migrate()`. **That is done**, on the `conforming-refinement` branch:

| Commit | What |
| --- | --- |
| `25980f2` | Factor the halo rebuild out of `migrate()` into `Tessera_HaloRebuild.hpp` |
| `a55a8de` | `refine()` rebuilds the halo itself; the identity-`migrate()` workaround is gone |
| `6282257` | Delete `refineImpl()` step 3j — `rebuildHalo()` rebuilds both CSRs completely |

So `rebuildHalo( mesh, halo )` exists at `src/Tessera_HaloRebuild.hpp:733`,
`refine()` calls it at `src/Tessera_RefineParallel.hpp:1077`, `migrate()` calls it
after its own move rounds, and
[halo-rebuild-split-edge-design.md](halo-rebuild-split-edge-design.md) is **closed
out — both its items are done** (item 2, the geometric blue tie-break, landed as
Decision 15 in `4cee602`). Do not re-do any of it, and do not treat that file as
open work.

What remains is **only** the depth parameter. Three steps.

### The shape of the existing rebuild

`Tessera_HaloRebuild.hpp` runs four lettered rounds, and the depth loop belongs
around exactly two of them:

| Round | What it does | Depth-dependent? |
| --- | --- | --- |
| **G** | Gather referenced-but-non-held vertex/edge tuples via a gid coordinator, so every reference of an owned face is held locally with its full field pack. | **No** — a pre-pass about *references*, not about rings. Runs once. |
| **B** | Ownership by lowest-advertising-rank via the vertex/edge coordinators; the vertex coordinator hands each owner the incident faces owned by other ranks. | **No** — see below. |
| **C** | Ghost fetch: each vertex owner requests those faces and ingests them with their vertices and edges. | **Yes** — this is the ring. |
| **D** | Assemble owned-first AoSoAs, rebuild the CSRs, the key side tables, and the three halo plans. | **No** — runs once, at the end. |

Rounds B–D live in `detail::finishHaloAndAssemble()`
(`Tessera_HaloRebuild.hpp:231`), shared verbatim by `rebuildHalo()` and
`migrate()`. **Put the depth loop there**, so both callers get it from one
implementation — that is the payoff of the extraction having already happened.

**Ownership is depth-independent, and this is the key simplification.** A vertex's
owner is the min over the owners of its incident faces, and every incident face is
advertised by its own owner regardless of who holds a copy. So round B's ownership
resolution runs **once**, before the loop, and its result is final at any depth.
Assert it in the test (check 4) rather than trusting the argument.

### Step 1 — The depth loop in `finishHaloAndAssemble()`

- Signature gains a `int depth` parameter; `rebuildHalo( mesh, halo, int depth = 1 )`
  and `migrate( mesh, halo, dest )` (reading `halo.depth`) pass it through.
- Iteration `d` seeds from the **full currently-held vertex set** — owned plus every
  ghost vertex acquired in iterations `< d` — requests the incident faces not yet
  held, and ingests them with their vertices and edges. Iteration 0 keeps the seed
  restricted to **owned** vertices, so `depth == 1` reproduces today's ghost set
  **exactly**; that restriction is what makes check 2 a bitwise regression guard.
- Newly-acquired ghost faces name vertices and edges absent from `vOwner`/`eOwner`.
  They arrive stamped in the fetched tuples — round C already does this — so no
  extra ownership round is needed at any depth.
- Termination: after `depth` iterations, or early when an iteration acquires
  nothing on any rank (`MPI_Allreduce(MPI_SUM)` on the local acquisition count
  == 0). The early exit matters at small rank counts, where the whole mesh becomes
  locally resident before `depth` is reached.
- Round D is unchanged: `detail::buildKindPlan` is a function of the final ghost
  set and does not care how deep it is.
- Round G is unchanged and stays a single pre-pass.

Record `depth` where both callers can see it:

```cpp
template <class MemorySpace> struct MeshHalo { ...; int depth = 0; };  // Tessera_Distribute.hpp
// Mesh: int haloDepth() const;  void setHaloDepth( int );
```

`refine()` and `migrate()` must **preserve** the depth they were handed — read
`halo.depth`, pass it on — so a caller sets depth once at setup and never thinks
about it again. `halo.depth == 0` (never built) is treated as 1, as today.

### Step 2 — `distribute()` depth

`distribute( mesh, halo, faceOwner, int depth = 1 )`. The mesh is replicated
here, so the closure is a local loop with no communication: repeat `depth` times
{ mark every face incident on a marked vertex; mark those faces' vertices and
edges }, seeded from owned vertices. Set `halo.depth`. The existing single-pass
code in `Tessera_Distribute.hpp` is already written as exactly this body executed
once.

### Step 3 — Close the silent-incompleteness hole

`buildVertexStencil( mesh, k )` throws `std::invalid_argument` when
`k > mesh.haloDepth()`, naming both numbers. Update its header comment: the
caller's responsibility ("deeper stencils on a partition boundary need a
correspondingly deeper halo — the caller's responsibility") is now *dischargeable*
and *checked*. This is the whole point of the task — a `k=2` stencil must become
either correct or loud, never quietly short.

### Non-goals

- Widening the *edge* or *face* halo independently of the vertex halo. Depth is
  one number, defined on the vertex closure, with edges and faces following.
- A ghost scatter-add (see [halo-scatter-add.md](halo-scatter-add.md)).
- Anything in [halo-rebuild-split-edge-design.md](halo-rebuild-split-edge-design.md)
  — that file is closed.

### Tests

New `tests/test_halo_depth.cpp`, registered at **TIER `regression`**, backends
**SERIAL and HIP**, ranks **`${TESSERA_TEST_MPI_RANKS}` (1–5)**. This promotion
into the ship gate is **pre-authorized for this task** — do not stop to ask.

```cmake
tessera_add_test(NAME halo_depth BACKEND SERIAL TIER regression
                 RANKS ${TESSERA_TEST_MPI_RANKS} SOURCES test_halo_depth.cpp)
tessera_add_test(NAME halo_depth BACKEND HIP    TIER regression
                 RANKS ${TESSERA_TEST_MPI_RANKS} SOURCES test_halo_depth.cpp)
```

The ground truth is available and exact: the coarse icosphere is generated
identically on every rank, so the test can build a **replicated reference mesh**
alongside the distributed one and compute any k-ring on it directly.

1. **Depth-2 ring completeness (the definitive check).** Subdivision-2 icosphere,
   `distribute(..., depth=2)`, `haloExchange`. For every **owned** vertex, the
   2-ring gid set from `buildVertexStencil(mesh, 2)` equals the 2-ring gid set
   computed on the replicated reference. Zero missing, zero extra, at every rank
   count. This is the check the README entry says cannot pass today.
2. **Depth 1 is bitwise unchanged.** `distribute(..., depth=1)` reproduces the
   pre-task local/owned counts and `topologyChecksum` exactly. Guards the change to
   the shared `finishHaloAndAssemble()`, which `migrate()` and `refine()` also use —
   this check is what protects the 150-test gate from the edit.
3. **Depth is monotone and effective.** At ranks ≥ 2, the ghost vertex count at
   depth 2 is strictly greater than at depth 1, and the 1-ring rows are still
   exactly correct at depth 2 (a wider closure must not perturb the inner ring).
4. **Ownership is depth-invariant.** `checkOwnershipPartition`,
   `globalOwnedVertices/Edges/Faces` and `ownedEulerGlobal` identical at depth 1
   and 2. Pins the argument that round B runs once.
5. **Ghost values are the owners' values.** After `haloExchange` at depth 2, every
   ghost vertex position equals the reference position for its gid, bitwise.
6. **`refine()` preserves depth.** Build at depth 2, then **four** `refine()`
   rounds back-to-back with nothing in between — the sequence
   `tests/test_refine_rehalo.cpp` already exercises at depth 1, which is why this
   check adds depth rather than re-proving the re-halo. After each round:
   `halo.depth == 2`, check 1 still passes, and `checkConforming`,
   `check21Balance`, `checkMidpointAgreement`, `checkOwnedEuler == 2` all pass.
   Include the non-vacuity guard `test_refine_rehalo.cpp` uses — plans non-empty
   at ranks ≥ 2, and a deliberately corrupted ghost resynced by `haloExchange` —
   since the structural checks pass on an empty plan.
7. **`migrate()` preserves depth.** Identity migrate and a real `loadBalance()` at
   depth 2; `halo.depth == 2` and check 1 still passes.
8. **Idempotence.** `rebuildHalo()` twice yields identical local counts and
   identical plans (`totalSend`/`totalRecv` and the flattened index arrays).
9. **Stencil guard.** `buildVertexStencil(mesh, 2)` throws on a depth-1 mesh and
   does not on a depth-2 mesh.
10. **Single rank.** Depth 1 and 2 both give an empty plan and identical meshes;
    `haloExchange` is a no-op; the early exit in step 1 fires.

## Exit criterion

- `test_halo_depth` green at **SERIAL and HIP, ranks 1–5**, and the full gate
  (`ctest -L regression -R "SERIAL|HIP"`) still **150/150** with nothing
  relabelled or weakened. Check 2 is what makes that credible, since the edit is
  in code `refine()` and `migrate()` share.
- `buildVertexStencil( mesh, 2 )` gives complete rows for every owned vertex on a
  depth-2 mesh at every rank count, and throws on a depth-1 mesh.
- The README *Known Issue* **"`buildVertexStencil(mesh, 2)` (k=2) is incomplete
  within one hop of a partition boundary"** is **deleted** — that entry is this
  task's target, and deleting it is the definition of done.
- README API section documents the `depth` arguments on `rebuildHalo()` and
  `distribute()`, `mesh.haloDepth()`, and that `refine()`/`migrate()` preserve
  depth.
- `docs/design.md` gains a *Halo depth* subsection beside the existing
  `rebuildHalo()` material (§ around line 291 and line 530): depth is a property
  of the local closure, the loop widens the closure and not the exchange, and
  ownership is depth-invariant.

## Where this sits

Recommended order across the eleven gap tasks (dependencies only, not priority).
Because `rebuildHalo()` already exists, several tasks that an earlier revision
listed as blocked on this one are **no longer blocked** — they need a halo rebuild,
which they now have, and only want depth if they want a wider ring.

```
global-reductions ─┐
latlon-sphere ─────┤   independent
edge-split ────────┤   (needs rebuildHalo — already in tree)
mesh-compaction ───┤   (same)
distributed-coarse-build ─┤ (same)
face-adjacency ────┤
distributed-loadbalance-solve ─┘

halo-depth ─→ halo-scatter-add (optional, for a depth-2 check)
          └─→ edge-collapse    (HARD: link condition needs depth >= 2)

face-adjacency ─→ edge-flip
mesh-compaction ─→ edge-collapse
edge-split ─────→ edge-flip, edge-collapse   (Editing families decision)
```

Only [edge-collapse.md](edge-collapse.md) has a hard dependency on this task.

## Progress log

- 2026-08-07 — Task written, then revised against `08dd346` after pulling. The
  `rebuildHalo()` extraction and the geometric blue tie-break had both landed in
  the interim, so the task narrowed from "extract the rebuild **and** add depth"
  to "add depth to the existing rebuild". Nothing implemented.
- 2026-08-09 — **DONE.** All three steps implemented, `tests/test_halo_depth.cpp`
  registered at TIER `regression` on SERIAL and HIP at ranks 1–5. Full gate
  **160/160** (150 pre-existing, unchanged and unrelabelled, plus the 10 new),
  diagnostic tier 62/62. The README *Known Issue* is deleted.

  Two notes on how it landed, both about the ring loop rather than the plumbing.

  * The ring loop went where the task said — `detail::finishHaloAndAssemble()` —
    but round B's ghost PUSH (the vertex coordinator handing each owner the
    incident faces owned by others) was replaced by a PULL against the same
    coordinator table `byV`. The push only ever reaches a vertex's owner, which
    is the ring-0 seed and nothing else; ring *d* needs the incidences of
    vertices the rank merely holds. One mechanism serving every ring is simpler
    than two, and it is exactly equivalent at depth 1: the coordinator returns
    all incidences of a seed vertex and the ones this rank already owns are
    filtered out, which is precisely the set the push sent. Ownership
    resolution still runs once, before the loop, as the task argued.
  * The early exit is the one place this was genuinely subtle, and it was a real
    bug caught by the tests. Growth must be measured on the newly-held
    **vertices**, not only on the newly-held faces: a rank all of whose owned
    vertices are interior to its own faces acquires no new face in ring 0 — yet
    its own faces' boundary vertices are what ring 1 expands from. Testing faces
    alone made such a rank break after one ring and hand back a silently 1-deep
    halo, which is the exact failure mode this task exists to remove. It showed
    up first in `distribute()`'s local loop (rank 1 of a 2-rank latitude-band
    split had V=93 instead of 115) and the same counting fix was applied to the
    distributed loop. Check 1b in the test — cross-checking `distribute()`'s
    local closure against `rebuildHalo()`'s coordinator rings at depth 2 — is
    what makes that class of bug loud, because a short closure otherwise only
    surfaces as a short k-ring on the few owned vertices next to the gap.
