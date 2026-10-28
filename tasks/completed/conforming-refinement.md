# Conforming Refinement

> **Handoff contract.** This file is the source of truth for the conforming-refinement
> work: the design, the task breakdown, and what is done. **At the start of any
> session touching this task, read this file first.**
>
> **How to work this file.** It is *not* a log. When a task completes, **rewrite** the
> affected sections so the file always describes the *current* state of the code —
> update the status table, delete or rewrite any design text the implementation
> superseded, and fold what was learned into the design sections. Only the
> short *Decision record* and *Progress log* at the bottom are append-only.
> **Commit and push at the completion of each task.**
>
> **Do not run tests in Tasks 1–7.** End each of those tasks by committing and
> pushing — nothing else. Do not run `ctest`, do not submit
> `run_regression_minset.flux`, do not run a one-off test binary. **Task 8** is the
> single verification task that runs the full suite and fixes whatever it finds.
> Consequently the per-task **Acceptance** sections in Tasks 1–7 are a
> *specification of what must hold* — write the tests, do not execute them; Task 8
> verifies them. Building (`make -j`) and `--target format-check` are not tests and
> are still expected, so each task's code at least compiles clean when it lands.

## Status

| # | Task | Status |
|---|------|--------|
| 1 | Refinement-mode plumbing (`RefinementMode`, conditional face fields, dispatch) | **Done** (compiles clean as of Task 2) |
| 2 | Serial closure kernel: `closeFaces` / `unclose` + patterns | **Done** |
| 3 | Distributed split-edge discovery (extend Phase 2 to kept faces) | **Done** |
| 4 | Wire closure into distributed `refine()` | **Done** |
| 5 | `migrate()` / `loadBalance()` / halo rebuild on a closed mesh | **Done** |
| 6 | I/O round-trip, `markByQuality`, example + docs | **Done** |
| 7 | Dedicated conforming test suite; flip the default to `Conforming` | **Done** |
| 8 | **Run the full suite and fix everything it finds** (the only task that runs tests) | **Done** (2026-08-05) — sub-tasks D0–D8 in [conforming-refinement-debug.md](conforming-refinement-debug.md), all done. Gate **140/140**, unit **62/62** at HEAD |

All eight tasks have landed, so **conforming refinement is feature-complete,
verified, and the default**: the closure kernel and its inverse are pure local
functions in `src/Tessera_RefineClosure.hpp`, both `refineLocal()` and the
distributed `refine()` run un-close → mask translation → red split → close,
`migrate()` / `loadBalance()` keep closure siblings co-resident and weight a red
parent's children as one unit, the closure bookkeeping round-trips through HDF5,
`markByQuality` drives the whole thing through the mask translation, every
mode-sensitive test is registered in both modes, and `Mesh`'s `Mode` parameter
defaults to `Conforming`. There is no abort stub left anywhere.

**Verification is complete (Task 8, sub-tasks D0–D8).** At HEAD the ship gate
(`regression` × {SERIAL, HIP} × ranks 1–5) is **140/140** and the diagnostic tier
(`ctest -L unit`) is **62/62** — 202 instances over 28 registrations, zero
failures, zero tests relabelled or excluded. The `Conforming` default **survived**
(Decision 13). Task 8 found and fixed **nine defects**: three in the library (all
in the red engine's handling of a persistent hanging node — Decision 8/9 — plus
locally-minted edge gids, Decision 10) and six test-side, of which **six separate
occurrences were the same defect**: a missing re-halo between two back-to-back
`refine()` calls. Two design limits are recorded rather than fixed: the visible
layer is rank-count dependent up to blue closure diagonals (Decision 11), and
defects 2–3 of Decision 8 remain in `HangingNode2to1` mode by construction. The
per-sub-task verdicts, root causes and evidence live in
[conforming-refinement-debug.md](conforming-refinement-debug.md); the measurements
are tabulated under *Task 8 → Report back* below.

**Two findings from Task 7's analysis, both since resolved by Task 8** — both were
found by reading the code while writing tests against it, not by running anything,
and both are recorded in full under *Task 8 → risk points* below:

1. **The split-edge map the closure consumes is this-round-only, but the level
   jumps it must close are persistent.** This looks like a defect in the Task-4
   wiring that every multi-round conforming test will trip. Risk point 9.
   **Confirmed and fixed — Task 8 D2** (Decisions 8 and 9); the fix also closed a
   latent midpoint-duplication and 2:1-propagation gap around hanging nodes.
2. **Rank-count independence of anything gid-keyed is not achievable**, because
   new vertex and face gids come from an `MPI_Exscan` over ranks. Only
   position-canonical comparisons are rank-count independent, which is how
   `conforming_determinism` is written. Risk point 10. **Confirmed and accepted as
   a design limit — Task 8 D6** (Decision 11): it fires at np5 only, on the blue
   closure diagonal, and the geometric tie-break that would fix it cannot be
   evaluated locally because `closeFaces()` runs on the un-closed red layer whose
   corners a rank may not hold (risk point 4). `conforming_determinism` now asserts
   the sharper statement that the visible layer differs *only* through blue
   diagonals.

---

## Problem

Tessera's adaptive refinement is **non-conforming**: a partial refine mask leaves
T-junctions (hanging nodes), bounded to a 2:1 level jump across any edge
(`README.md` → Known Issues; `src/Tessera_RefineParallel.hpp`). A hanging node means
a half-edge `(a, m)` has only **one** incident face, so:

- owned-only Euler `V − E + F = 2` holds only for a *uniform* refine;
- `CurvatureCriterion`'s edge coordinator silently skips such edges
  (`inc.size() != 2 → continue` in the propagation fixpoint and the analogous
  guard in `Tessera_MarkQuality.hpp`);
- a downstream solver that assembles a surface operator (`applyStencil`,
  `reduceVertexFromFaces`) sees an inconsistent 1-ring at the T-junction vertex.

The consuming codebase requires a conforming triangulation. This task adds one.

---

## Decision 1 — option, not replacement

**Conforming refinement is added as an additional mesh option; the 2:1 hanging-node
path is retained.** Rationale:

1. **It is strictly additive, not a fork.** The design below *builds on* the existing
   2:1-balanced red refinement: the 2:1 balance stays, and conforming refinement is a
   **closure post-pass** over exactly the mesh the current `refine()` already
   produces. Deleting the hanging-node mode would not simplify the conforming path —
   it is a prerequisite of it.
2. **The 2:1 mode is cheaper** — no closure faces, no closure bookkeeping fields, no
   un-close pass — and is sufficient for consumers doing cell-centred work.
3. **The existing regression tests pin the 2:1 mode.** Keeping the mode keeps that
   coverage as a live regression instead of deleting it.

### How the option is spelled

`RefinementMode` is a **compile-time `Mesh` template parameter**, appended last so
every existing `Mesh<...>` spelling stays source-compatible:

```cpp
enum class RefinementMode
{
    HangingNode2to1,   //!< 2:1-bounded hanging nodes (today's behavior)
    Conforming         //!< red-green-blue transient closure; no hanging nodes
};

template <class Scalar, int Dim = 3, class VertexUserFields = VertexFields<>,
          class EdgeUserFields = EdgeFields<>, class FaceUserFields = FaceFields<>,
          class MemorySpace = ..., class ExecutionSpace = ...,
          RefinementMode Mode = RefinementMode::Conforming>
class Mesh;
```

Compile-time was chosen over a runtime flag so the closure bookkeeping fields exist
in the face AoSoA **only** in `Conforming` mode — a hanging-node mesh pays zero
memory. `refine()` dispatches on `MeshT::refinement_mode` with `if constexpr`.

**The default is `Conforming`** (Decision 3), as of Task 7. The rationale for
which way round the default goes: a hanging node breaks a surface operator
*silently* — the vertex's incident-face set, edge 1-ring and `vertexFaces()` row
are all self-consistent, they simply describe a half-disc — so the mode that
gives a plausible wrong answer is the one a consumer should have to ask for.
Because the pre-existing tests and examples assert hanging-node behavior, the
flip was sequenced last among the implementation tasks: Tasks 1–6 shipped with
the default still `HangingNode2to1`, and Task 7 pinned every test that calls
`refine()` to `HangingNode2to1` explicitly *before* flipping. Since no task
before Task 8 runs tests, the flip landed *before* conforming had ever been
executed — Task 8 verifies both modes together, and reverting the flip is Task
8's documented escape hatch if conforming cannot be made green (Decision 5).

---

## Design — red–green–blue transient closure

### Why closure, and why transient

Two alternatives were rejected:

- **Red-only conforming refinement** (propagate the 1→4 red split until no edge has a
  level jump) degenerates to *uniform* refinement: face A refined forces its three
  edge-neighbours to refine, which cascades over the whole connected component. Not
  usable for AMR.
- **Newest-vertex / longest-edge bisection** is conforming by construction with
  provable shape bounds, but replaces the red engine wholesale: new `Level`
  semantics, a new cross-rank compatible-edge propagation loop, and it breaks the
  icosphere-subdivision correspondence the uniform-refine Euler test relies on.
  Recorded as a possible Milestone-2 alternative; not this task.

So: keep red refinement, add a **green/blue closure** layer. The closure must be
**transient** — recomputed from scratch each refine call — because repeatedly
bisecting an already-bisected closure triangle degrades its shape without bound.
Transient closure gives a bounded number of triangle similarity classes: every
visible triangle is a red-refined triangle or one of three fixed retriangulations of
one.

### Two layers

| Layer | Persistence | Contents |
|---|---|---|
| **Red layer** | persistent, authoritative | Faces produced by red 1→4 splits only, 2:1 balanced. Exactly what `refine()` produces today. Each face carries its red `Level`. |
| **Closure layer** | transient, derived | Green/blue/red retriangulation of red faces that have hanging nodes, so the visible mesh is conforming. |

The **visible mesh** — what `mesh.faces()`, the slices, I/O, and the geometry/stencil
API see — is the closure layer. The red layer is recovered by the un-close pass.

### The key simplification: closure creates no new vertices

A hanging node on a kept face's edge is a midpoint vertex that the *neighbour's* red
split already created. Closure only reconnects existing vertices into new triangles.
Therefore:

- **no new vertex gids**, no `MPI_Exscan` for vertices, no position/field
  interpolation, and **no `RefinePolicy` involvement** in the closure;
- un-closing removes only faces and edges — **never a vertex** — so no vertex state
  is ever discarded by the transient closure;
- closure is **purely local** to one red face, hence to one rank (a face is owned by
  exactly one rank). No communication in the closure step itself.

### Closure patterns

For a kept red face with corners `(a,b,c)`, edges `ab, bc, ca`, let `S` be the subset
of its edges that are bisected in the red layer, with midpoints `m_ab, m_bc, m_ca`.
The 2:1 balance bounds the jump to one level, so each split edge contributes exactly
one midpoint and `|S| ∈ {0,1,2,3}`:

| `\|S\|` | Pattern | Children | Child count |
|---|---|---|---|
| 0 | none | face emitted unchanged | 1 |
| 1 | **green** (say `ab` split) | `(a, m_ab, c)`, `(m_ab, b, c)` | 2 |
| 2 | **blue** (say `ab`, `bc` split) | `(a, m_ab, c)`, `(m_ab, b, m_bc)`, `(m_ab, m_bc, c)` | 3 |
| 3 | **red-closure** | `(a,m_ab,m_ca)`, `(b,m_bc,m_ab)`, `(c,m_ca,m_bc)`, `(m_ab,m_bc,m_ca)` | 4 |

Rules:

- **Every closure child inherits the parent's `Level` and the parent's face user
  fields.** The closure never changes the red layer — in particular the `|S| = 3`
  red-closure does **not** increment `Level` and does **not** promote the face into
  the red layer. This deliberately avoids a promotion cascade.
- **Winding is preserved**: children are emitted in the parent's CCW order so
  `faceNormalRaw` / `CurvatureCriterion` keep the consistent
  outward-seen-from-outside orientation.
- The **blue** pattern has two valid diagonals. Choose deterministically — connect
  the midpoint with the **lower gid** to the opposite corner — so the output is
  reproducible and rank-count independent (needed for the topology checksum used by
  the migrate/IO tests).
- Face `e[3]` follows the existing convention `e[k] = edge(v[k], v[(k+1)%3])` and is
  re-derived with the rest of the edge table.

**As implemented (Task 2).** The `|S| = 1` and `|S| = 2` patterns are written once
for a *rotated* corner triple `(A,B,C)`, where the rotation is chosen by which edge
is split (green: rotate the split edge to position 0) or unsplit (blue: rotate the
unsplit edge to position 2, `rot = (u+1) % 3`). Two consequences:

- a cyclic rotation of a CCW triple is CCW, so winding preservation is structural
  rather than checked per pattern;
- the working triple `(A,B,C)` is a function of *the triangle and which edge is
  (un)split*, **not** of which corner happens to be stored at `v[0]`. The emitted
  child-triangle set is therefore invariant under a cyclic relabelling of the
  parent's corners, which is the concrete form of partition independence. Pinned by
  the `refine_closure` rotation-invariance check.

**Blue tie-break, precisely.** With `(A,B,C)` as above, `q0 = mid(A,B)` and
`q1 = mid(B,C)`; the corner triangle `(q0, B, q1)` is always cut off and the
remaining quad `(A, q0, q1, C)` is split along the diagonal from the lower-gid
midpoint to *its opposite corner* — `q0 ↔ C` when `q0 < q1`, else `q1 ↔ A`:

| condition | children |
|---|---|
| `q0 < q1` | `(A,q0,C)`, `(q0,B,q1)`, `(q0,q1,C)` |
| `q1 < q0` | `(A,q0,q1)`, `(q0,B,q1)`, `(A,q1,C)` |

This is partition-independent because *both* inputs are global: the two midpoint
gids are bit-identical on every rank sharing the bisected edge (`refine()`'s Phase-2
guarantee, verified by `checkMidpointAgreement`), and `(A,B,C)` is rotation-derived
as above. Nothing partition-local — local index, owner rank, map iteration order,
`v[0]` — enters the comparison. `ClosureStats` counts both branches so a test can
assert the tie-break actually fires in both directions rather than being vacuous.

**What "partition independent" does and does not cover (Task 7 finding).** The
tie-break makes the closure a function of *(red layer, split-edge map)* alone —
no local index, no owner rank, no iteration order — so the same red mesh with the
same midpoint gids closes identically no matter which rank owns the face. That is
the property the design needed and it is real. It is **not** the same as
rank-count independence, because the midpoint gids themselves are not: Phase 2c
sorts each rank's owned midpoint keys and then `MPI_Exscan`s, so at np1 the
midpoints are numbered in global `EdgeKey` order while at np > 1 they are grouped
by owner rank first. Two midpoints of the same kept face can therefore compare in
one order at np1 and the other at np3, flipping the diagonal. The consequence
propagates: corner gids of later rounds are earlier rounds' midpoint gids, so
**nothing gid-keyed is comparable across rank counts** after the first refine.
What *is* rank-count independent, and what `conforming_determinism` therefore
compares, is everything positional: the split-edge set, the red layer, the `|S|`
histogram, the closure-vertex set, and V/E/F. Whether the visible layer is too —
i.e. whether the tie-break happens to be stable across rank counts on these
meshes — is asserted *and* measured directly (a count of blue parents whose
diagonal differs from a serial reference), so Task 8 gets the answer rather than
just a checksum mismatch. See risk point 10.

### Data model — closure bookkeeping

Un-closing must reconstruct the red parent from its children. Each closure child
stores the parent outright, so **any single child determines its parent** with no
sibling lookup:

| Field | Type | Meaning |
|---|---|---|
| `ClosureParent` | `GlobalId` | Parent red face's gid; `invalid_gid` for a red face (no closure). |
| `ClosureParentVerts` | `GlobalId[3]` | Parent's three corner vertex gids, in the parent's winding order. |

**As implemented (Task 1).** The two members live in `ClosureFaceMembers` in
`Tessera_Fields.hpp` and are appended to the face member list **after** the user
pack — *not* added to `CoreFaceMembers` — only when `Mode == Conforming`:

```cpp
using ClosureFaceMembers = Cabana::MemberTypes<GlobalId, GlobalId[3]>;

template <class UserFaceFields, RefinementMode Mode>          // primary
struct FaceMemberTypesImpl { using type = MemberTypesCat_t<CoreFaceMembers, UserFaceFields>; };
template <class UserFaceFields>                                // partial spec on the value
struct FaceMemberTypesImpl<UserFaceFields, RefinementMode::Conforming>
{ using type = MemberTypesCat_t<MemberTypesCat_t<CoreFaceMembers, UserFaceFields>,
                                ClosureFaceMembers>; };
```

so the layout is `[core | user | (closure)]`. Rationale: `FaceField::UserBegin` is
the hard-coded constant `5` and `userFaceField<M>()` is `UserBegin + M`; inserting
closure fields among the core members would shift every user field index and
silently break existing consumers. With the append, `UserBegin` stays `5` and the
closure indices are computed from the user pack size —
`closureParentField<FaceUserFields>()` / `closureParentVertsField<...>()`, also
re-exported as `MeshT::closure_parent_field` / `closure_parent_verts_field`.

> **Consequence — the user pack is no longer the tuple's suffix on faces.** Any code
> sizing a face user-field loop as `face_member_types::size - FaceField::UserBegin`
> over-counts by two in `Conforming` mode and would treat the closure members as
> user data. Task 1 added `numFaceUserFields<UserFaceFields>()` (== the pack size)
> plus `detail::copyUserFieldsN<UserBegin, N>` and switched both face-field copy
> sites in `Tessera_Refine.hpp` / `Tessera_RefineParallel.hpp` to it. The
> tuple-suffix-derived `detail::copyUserFields<UserBegin>` remains, correct for
> vertices and edges. Task 6 discharged the same hazard in the I/O path: the HDF5
> writer and reader bound their face user-field loops (and the `n_user_f_fields`
> attribute) with `detail::forEachUserFieldN<UserBegin, numFaceUserFields<>()>`
> and write the closure members as their own datasets.

> **Consequence — the closure members must be explicitly initialized.** A Cabana
> `AoSoA`'s backing View is *zero*-initialized, so an untouched `ClosureParent`
> reads as face gid `0` — a perfectly valid gid — and `unclose()` would restore a
> bogus parent for every face. Task 2 added
> `initClosureFaceMembers<MeshT>( hostFaceAoSoA, begin, end )` (a no-op in
> `HangingNode2to1` mode) and calls it from `buildTriangleMesh()` in
> `Tessera_MeshBuilder.hpp`. `distribute()` / `migrate()` / `haloExchange()` are
> generic over the whole face tuple and carry the members through unchanged, so they
> need nothing. Task 4 found that `distribute()` *does* (it builds a fresh face
> AoSoA), and Task 6 the same for the HDF5 reader; both now call it.
> `unclose()` guards the failure mode anyway: it aborts if two restored red
> faces share a gid, which is exactly what an uninitialized field produces.

**Memory.** Only the *visible* face array carries the two fields, in `Conforming`
mode only: 4 extra `GlobalId` per face on a ~78 B face core. A compressed encoding
(parent gid + a pattern/child-index byte pair, reconstructing parent corners from
siblings) is a possible follow-up — record it in README *Future Optimizations*, not
in this task.

### Distributed algorithm — what changes in `refine()`

**As implemented (Task 4).** There is no separate conforming driver. The former
`detail::refineHangingNode()` is now `detail::refineImpl()`, shared by both modes,
and `refine()` is a one-line forward to it — the mode branch lives *inside*, as
four `if constexpr ( kConforming )` blocks. That was the right shape because the
conforming path differs from the hanging-node path in exactly three inserted local
steps and one widened count; everything phases 1–3 do is byte-for-byte the same
code operating on the same arrays. Concretely, the shared body was re-expressed
over a **red-layer snapshot** (`fV` / `fG` / `fL` corner gids, gid, level, plus
`fSrc[r]` = the pre-call visible row supplying red face *r*'s face user fields) and
a red-indexed `mark`. In `HangingNode2to1` mode that snapshot is the identity read
of the visible faces and `fSrc[f] == f`; in `Conforming` mode it is the un-close,
and `mark` is `translateMask()`'d. Phase 1 and Phase 2 then iterate `nRedF` instead
of `nOwnedF` and are otherwise untouched.

```
Conforming refine( mesh, halo, mask ):

  0. UN-CLOSE (new, local, no comm)
       For each distinct ClosureParent gid held locally, emit one red parent face:
         gid   = ClosureParent
         v[3]  = ClosureParentVerts
         level = the child's Level (closure children carry the parent's level)
         user fields = from the lowest-gid child (all children inherited an
                       identical copy from the parent)
       Red faces (ClosureParent == invalid_gid) pass through unchanged.
       Result: the persistent red layer, exactly the shape phases 1-3 expect.

  0b. MASK TRANSLATION (new, local)
       A red parent is marked iff ANY of its closure children was marked in the
       caller's visible-face mask. The caller's mask is indexed by VISIBLE owned
       faces; phases 1-3 want it indexed by RED owned faces.

  0c. RECOVER THE PERSISTENT SPLIT-EDGE MAP (new, local, no comm — Task 8 D2)
       Also from the closure bookkeeping: for each closed red parent, which of
       its edges is bisected in the red layer and at which midpoint. Being
       bisected outlives the round that caused it; phase 2 only ever learns
       THIS round's bisections. See the Phase 2 extension below.

  1. 2:1 mark-propagation fixpoint          [EXTENDED: keyed on half-edges]
  2. midpoint-gid assignment                [EXTENDED, see below]
  3. red face list: kept + 4 children each  [unchanged; a split of an already-
                                             bisected edge REUSES its midpoint]

  3b. CLOSE (new, local, no comm)
       For each KEPT red face, look up its 3 edges in the split-edge map — this
       round's from phase 2 UNIONED with 0c's; apply the |S| pattern above; emit
       the closure children with ClosureParent / ClosureParentVerts set. A red
       child of a REFINED face has |S| = 0 unless one of its two inherited
       boundary half-edges is bisected this round, which is possible exactly
       when its parent's edge carried a reused midpoint — assert that narrower
       form (closeFaces()'s `firstNewVertexGid`).

  3c. face gid allocation                   [EXTENDED: must cover closure children]
  3d..3i. edges / keys / CSR / ownership    [unchanged, driven by the final face list]
```

**Phase 2 extension — kept faces must learn their split edges. (Landed, Task 3.)**
Before Task 3, Phase 2 advertised only *refining* faces' edges to the edge
coordinator, and only refining participants received the midpoint gid. A kept face
on rank R needs to know which of its edges were bisected by a refining neighbour on
rank R′. The existing rounds were extended — **no new communication rounds**:

- **2a:** `detail::OwnerMsg` gained an `unsigned char refining` field and is now
  emitted for the edges of *all* owned faces, refining and kept.
- **2b:** the coordinator aggregates per edge into `{participants, refOwner,
  anyRefining}`. An edge is **split iff `anyRefining`** — one with no refining
  incidence is dropped right there, so the extra advertisements create no
  downstream state. For a split edge the midpoint owner is the minimum *refining*
  participant (rule unchanged), and `(key, midOwner)` is replied to **all**
  participants, with `(key, cosharer)` sent to the owner for each other
  participant.
- **2c:** unchanged code — the midpoint owner therefore ships the gid to **all**
  co-sharers, refining or kept.

Every rank then holds `EdgeKey → midpoint gid` for every split edge it touches, which
is exactly the split-edge map step 3b needs. It is published as
**`RefineResult::midpoints`**, whose contract was widened from "edges of my refining
faces" to "edges of *any* of my owned faces that were bisected" — a superset, so
`checkMidpointAgreement` (its other consumer) only gets stronger. `RefineResult`
also gained `phase2Adverts` / `phase2AdvertsRefining` counters so a test can report
the added message volume without instrumenting the library at the call site.

**Why the midpoint assignment is bit-identical to before.** Ownership reads only
refining participants, and the set of split edges is unchanged (an edge with no
refining incidence is dropped), so `myMid` — the owned-midpoint list each rank
exscans over — is the same list in the same order as before. Vertex gids, counts,
and positions are therefore untouched; only the *distribution* of already-decided
gids got wider. *(True of Task 3. The Task-8 D2 extension below does change the
`Conforming` assignment — deliberately, since minting a second midpoint for an
already-bisected edge is the defect it fixes. `HangingNode2to1` is still
bit-identical: its persistent map is always empty.)*

**Persistent split edges — the map is a property of the red layer, not of a round.
(Landed, Task 8 D2; this is the fix for risk point 9.)** Phase 2 above answers
"which edges did *this round* bisect". Step 3b needs "which edges are bisected",
full stop: a kept face closed in round 1 is still the coarse side of that hanging
node in round 2, and the coordinator drops its edge (no refining incidence), so
the closure re-emits it unclosed and the mesh stops being conforming from round 2
on. Three coupled changes, all driven by step 0c's recovered map and none of them
adding a message round or a stored field:

- **Recovery (`recoverSplitEdges()`, `UncloseResult::splitEdges`).** For a closed
  parent `(a,b,c)`, a parent edge is split **iff it is not an edge of any child**,
  and the midpoint of a split `(x,y)` is the unique child corner `m ∉ {a,b,c}` for
  which `(x,m)` and `(m,y)` are each an edge of **exactly one** child. The
  one-child qualifier is load-bearing: an edge shared by two children is a
  *diagonal* of the parent's fan, and without it the blue pattern is ambiguous —
  in the `q0 < q1` blue both `(q0,B)` and `(q0,C)` exist, so `q0` would answer for
  `(B,C)` as well as for `(A,B)`. Fan-boundary edges appear exactly once. Purely
  local (a face is owned by one rank), so no communication and no new field. It
  aborts if the recovery is not unique, which is precisely the signature of a
  closure family split across ranks.
- **Union into phase 2's map.** Consulted by step 3b for kept faces and by step 3
  for refining ones — a coarse face refining across a hanging node must **reuse**
  the existing midpoint, not mint a coincident second vertex, which would crack
  the mesh. The recovered entries need no agreement step: a bisected edge has
  exactly one incident coarse face, hence exactly one rank that can consult it.
- **Half-edge keying at the coordinator (`forEachSubEdge()`).** A hanging node
  means the coarse face still spans `(x,y)` while the fine faces opposite carry
  `(x,m)` and `(m,y)`, so keying on `(x,y)` leaves *both* sides with a single
  incidence and every coordinator rule that needs two — phase 1's 2:1
  propagation, phase 2's split decision — silently skips the pair. Advertising the
  two halves instead makes the coarse face meet its true neighbours. **This is
  what makes "at most one midpoint per red edge", the precondition of the closure
  patterns, actually hold across a hanging node**; before D2 the level jump there
  was unbounded (and still is in `HangingNode2to1` mode, which keeps no record to
  recover from — see *Known limits*). One level of expansion suffices exactly
  because the bound then holds inductively.
  A half is advertised in phase 2 with `refining = 0` **regardless of the face's
  own mark**: a refining coarse face bisects the whole edge, at the midpoint it
  already has, and bisects neither half. Advertising a half as refining makes the
  coordinator mint a midpoint for it — a spurious refinement that cascades into a
  red edge carrying two midpoints and no applicable pattern, which is exactly the
  `closeFaces()` "bisected more than once" abort seen in round 3 while D2 was
  being brought up.

**A closure child may reference a non-local vertex.** The midpoint gid arrives, the
midpoint *position* does not. This is not new: `refine()` already leaves an
owned-only mesh whose faces reference vertex gids the rank does not hold (see the
post-refine gotcha in `docs/design.md`). The halo rebuild inside `migrate()` brings
them in. Do **not** add a position gather to the closure.

**Face gid allocation. (Landed, Task 4.)** Closure children are new faces and need
new gids, folded into the existing single `MPI_Exscan`:
`4 * nRefining + countClosureChildren(newRed, midpointOf)` per rank. Two details
the design under-specified, both now settled in the code:

- **Ordering.** The red 1→4 split is built *topology first* — corner gids do not
  depend on face gids — so `countClosureChildren()` can run on the post-split red
  layer before a single gid is handed out. After the exscan, the fresh red children
  take the low part of this rank's block and `closeFaces()` is handed the first
  gid of the remainder as its `firstChildGid`.
- **Which "global max".** The block sits above the global max of the **pre-refine
  VISIBLE** face gids, not the max red gid. The visible layer contains every red
  gid *plus* the closure children allocated above them, so this bound is monotone
  across rounds and no allocation can collide with anything still referenced —
  including a retired parent gid, which un-close reuses on the next round. Basing
  it on the red max would be *nearly* safe (a discarded closure child's gid is
  referenced by nothing) but gives up monotonicity for no gain. Live face gids stay
  sparse, which is already true today and already handled.

### Interaction with the rest of the library

| Area | Impact |
|---|---|
| `Level` semantics | Face `Level` remains the **red** level. A closure child carries its parent's level, so in `Conforming` mode `Level` no longer maps 1:1 to triangle size. Edge level stays `min` of incident face levels. Document in `docs/design.md`. |
| `migrate()` / `loadBalance()` | **Done, Task 5.** The visible (closed) mesh migrates. Un-close is local *per child*, so siblings need not stay co-resident for correctness — but two ranks each holding a child of the same parent would each restore the parent, duplicating it. `migrate()` round S constrains `dest` so all closure siblings follow the lowest-gid sibling's destination (purely local — siblings are co-resident on entry); `ownedFaceWeights()` weights a child `1/nsiblings`. |
| Halo rebuild | Unchanged in mechanism. On a conforming mesh every edge has exactly 2 incident faces, so the 1-ring closure invariant is cleaner, not harder. |
| I/O | **Done, Task 6.** Format version 2: the two closure members are their own `/faces/closure_parent` + `/faces/closure_parent_verts` datasets (persistent gids, no dense translation), the face user-field loops are bounded by `numFaceUserFields<>()` instead of the tuple suffix, and a `refinement_mode` root attribute distinguishes the two file shapes. The reader also runs the collective `repairClosureCohesion()` — its fresh block partition splits sibling groups, which `migrate()`'s local round S cannot see. |
| `markByQuality` | **Done, Task 6 — no library change needed.** Returns a mask over **visible** owned faces; step 0b translates it. `CurvatureCriterion`'s "exactly two incident faces" coordinator assumption becomes *true* in conforming mode rather than silently skipped — a correctness improvement, measured directly by `checkConforming()` (which counts exactly the edges that guard would skip). |
| `MeshGeometry` / `VertexStencil` | No change; they read the visible mesh and are already generation-guarded. Conforming topology makes the k=1 stencil consistent at former T-junctions. |
| `RefinePolicy` | Untouched. The closure interpolates nothing. |
| Profiling | Level-2 `refine_unclose` / `refine_close` and level-3 `refine_closure_patterns` — added, Task 4. |

### Invariants and acceptance

New shared checks in `tests/MeshInvariants.hpp` (**all landed, Task 4**):

- `checkConforming(mesh)` — every edge in the global mesh has **exactly two**
  incident faces, verified through the edge coordinator (`edgeCoordRank` +
  `allToAllV`), *not* the halo, so the verdict is rank-count independent and needs
  no ghost layer (`refine()` leaves an owned-only mesh).
- `checkOwnedEuler(mesh) == 2` — owned-only `V − E + F = 2` for an **arbitrary**
  (adaptive, not just uniform) mask. The headline acceptance criterion; it is
  precisely what the hanging-node mode fails. Value-identical to the existing
  `ownedEulerGlobal()`, named separately because it is *the criterion*.
- `checkNoInteriorVertex(mesh)` — **geometric**: no vertex is collinear with and
  strictly between the endpoints of any edge. The topological reading is true of
  every ordinary triangle, so geometry is unavoidable — and that forces the one
  design departure here: this check *cannot* go through an edge coordinator,
  because it needs vertex positions and a post-`refine()` face may name a vertex
  only its owner holds (always so across a partition boundary, and in `Conforming`
  mode also for a closure child's midpoint corner). It therefore replicates the
  global owned vertices and owned faces on rank 0 and runs the exact test there —
  affordable at these sizes, and rank-count independent by construction, in the
  same partition-free-reference style the rest of the suite uses. It also flags a
  face naming a vertex **no** rank owns, which is a free extra check.
- `checkClosureInverse(mesh, midpoints)` — takes `RefineResult::midpoints` (the
  very map the closure consumed) and verifies three things, all local: **fidelity**
  — re-closing the un-closed red layer reproduces the visible layer as a multiset
  of (sorted corner triple, level, parent gid, parent corners); **inverse** —
  un-closing *that* returns the red layer bit-for-bit on gid/corners/level; and
  **gid sanity** — visible gids are locally distinct and a passed-through red face
  carries no closure bookkeeping (a zero-filled `ClosureParentVerts` would read as
  vertex gid 0). Child gids are excluded from the multiset comparison only because
  their base is an exscan result.
- `check21BalanceRed(mesh)` — the existing check applied to the **red layer**
  (after un-close). Identical to `check21Balance()` in `HangingNode2to1` mode. The
  coordinator logic was factored into `check21BalanceOn(comm, size, verts, levels)`
  so both wrappers share it.

The existing `checkMidpointAgreement` and `checkOwnershipPartition` must keep
passing unchanged.

**Gate.** New conforming tests join the ship gate at label `regression` × backends
{SERIAL, HIP} × ranks {1,2,3,4,5} — the gate definition itself does not change, so
no edits to `CLAUDE.md`, `run_regression_minset.flux`, or CI are needed. Existing
hanging-node tests stay in the gate.

### Known limits to document, not solve

- Face **user** fields on closure faces are the parent's, copied. If a solver writes
  per-closure-face state, un-close takes the lowest-gid child's values and the rest
  are discarded. Document as the contract (it parallels the existing "edge user
  fields are reset by `refine()`" rule).
- Edge user fields continue to be reset by `refine()`.
- Coarsening / edge collapse remains out of scope.
- **`HangingNode2to1` mode does not track hanging nodes across rounds** (found while
  fixing risk point 9; see Decisions 8 and 9). Two consequences, both pre-existing
  and both invisible to that mode's own tests, which assert non-conformity anyway:
  the 2:1 mark propagation cannot see across a hanging node (the coordinator's rule
  needs two incident faces and a hanging node leaves one on each side), so a
  sequence of adaptive rounds can drive an unbounded level jump there; and a coarse
  face refining across a hanging node mints a second midpoint coincident with the
  existing one instead of reusing it. Fixing either needs the persistent split-edge
  map, which only the closure bookkeeping can supply locally — a `HangingNode2to1`
  rank has no record of it and would need a new field or a new message round. The
  `Conforming` mode does not inherit either limit.

---

## Tasks

Each task is one Claude Code session. Each session must **rewrite** the affected
sections of this file (design text and status table) to match what landed — not
append to it.

**Tasks 1–7 end at a commit + push, with no test run** (see the handoff contract
above). Their **Acceptance** sections specify the tests to *write* and the properties
those tests must pin; **Task 8** is where the suite is executed and failures are
fixed. Do compile and keep `format-check` clean as you go — a task that lands
non-compiling code makes Task 8 unable to distinguish a build break from a real
defect.

Work on the `conforming-refinement` branch (it exists and currently points at
`redesign`).

---

### Task 1 — Refinement-mode plumbing — **DONE**

**Goal.** The option exists and is inert. No behavior change.

**What landed.**

- `src/Tessera_RefinementMode.hpp` (new, in `<Tessera.hpp>`) —
  `enum class RefinementMode { HangingNode2to1, Conforming }` with the contract of
  each mode documented on the enumerators.
- `Mesh` gained an 8th and last template parameter
  `RefinementMode Mode = RefinementMode::HangingNode2to1`, exposed as
  `static constexpr RefinementMode refinement_mode`. **No call site changed** —
  every existing spelling in `src/`, `tests/`, and `examples/` passes exactly seven
  arguments. *(Default flips to `Conforming` in Task 7.)*
  `Mesh` also now re-exports `vertex_user_fields` / `edge_user_fields` /
  `face_user_fields` (needed to size a user-field loop, see below) and the
  `closure_parent_field` / `closure_parent_verts_field` slice indices.
- `Tessera_Fields.hpp` — `ClosureFaceMembers`, the `FaceMemberTypesImpl` partial
  specialization on the mode value, `numFaceUserFields<>()`,
  `closureParentField<>()`, `closureParentVertsField<>()`. See *Data model —
  closure bookkeeping* above for the spelling and the user-pack-is-no-longer-the-
  suffix consequence.
- `refineLocal()` and `refine()` are now thin dispatchers on
  `MeshT::refinement_mode`; the existing bodies moved verbatim into
  `detail::refineLocalHangingNode()` / `detail::refineHangingNode()`. The
  `Conforming` branch is a runtime `Kokkos::abort("... not implemented yet ...")`
  naming the task that will implement it — deliberately not a `static_assert`, so
  a `Conforming` mesh type stays instantiable (Task 2+ tests need to construct one
  before the kernels exist).
- `docs/design.md` gained an *Adaptive refinement → Refinement modes* subsection;
  `README.md`'s API sketch documents the 8th parameter and its *Known Issues*
  non-conforming entry now records conforming as in progress with the abort stub.
- `tests/test_refinement_mode.cpp` + registration (`unit`, SERIAL + HIP, np1).

**Notes for later tasks.**

- **Cabana friction: none.** `Cabana::MemberTypes` concatenation via the existing
  `MemberTypesCat_t` composed cleanly, and a partial specialization on the
  `RefinementMode` *value* works without a tag-type detour. `AoSoA`,
  `MemberTypeAtIndex`, and `Cabana::slice<I>` are all indifferent to the two extra
  members.
- The `Conforming` face tuple is 9 members for an empty user pack (7 + 2), and the
  closure members are always the **last two**, at `tupleSize-2` / `tupleSize-1`.
- The `Kokkos::abort` in a host-side `if constexpr` branch compiles but is not a
  `[[noreturn]]` the compiler can see through, so `refine()`'s `Conforming` branch
  also returns a default `RefineResult{}` to keep the return path well-formed.

**Acceptance.**
`unit` test `refinement_mode` (SERIAL + HIP, np1) — written, not run (see the
handoff contract). It pins: a `Conforming`-typed mesh instantiates, constructs,
resizes, and round-trips written closure-member values host↔device;
`FaceField::UserBegin == 5` and `userFaceField<0..1>()` indices *and types* are
identical in both modes; the `Conforming` face tuple is exactly two members larger
and the closure indices are its last two, with types `GlobalId` / `GlobalId[3]`;
vertex and edge member lists are mode-independent;
`numFaceUserFields<>()` counts the pack while the tuple suffix over-counts by two.
Most of these are `static_assert`s (a `UserBegin` shift must be a build failure,
not a silent wrong read) mirrored by run-time checks so a failure names itself.
The test deliberately does **not** assert what the 7-argument default mode *is*, so
it survives Task 7's flip.

**Report back.** *(delivered — see Notes above; the short version: no Cabana
friction, no call site broken, and the one real hazard the layout choice creates is
the face user pack no longer being the tuple suffix, which the I/O path still
assumes — flagged for Task 6.)*

---

### Task 2 — Serial closure kernel — **DONE**

**Goal.** The closure and its inverse exist as pure, testable local functions.

**What landed.**

`src/Tessera_RefineClosure.hpp` (new, in `<Tessera.hpp>`, depends only on
`Tessera_Fields` / `Tessera_RefinementMode` / `Tessera_Types` so both
`Tessera_MeshBuilder.hpp` and `Tessera_Refine.hpp` can include it without a cycle):

| Symbol | Role |
|---|---|
| `RedFace { v[3], gid, level }` | one face of the persistent red layer |
| `VisibleFace { v[3], gid, level, parent, parentVerts[3] }` | one face of the visible layer; `parent == invalid_gid` ⇒ a passed-through red face |
| `ClosureStats` | `\|S\|` histogram, visible/closure-child counts, per-diagonal blue counts |
| `closureChildCount(nSplit)` | `nSplit + 1` |
| `faceSplitEdges(v, midpointOf, mid)` | fills `mid[k]` per edge, returns `\|S\|` |
| `countClosureChildren(red, midpointOf)` | new-gid count, for Task 4's `MPI_Exscan` *before* closing |
| `closeFaces(red, midpointOf, firstChildGid, freshChild = {})` → `CloseResult` | the patterns; children take consecutive gids from `firstChildGid` |
| `unclose(visible)` → `UncloseResult` | the inverse |
| `translateMask(visibleMask, uncloseResult)` | visible-face mask → red-face mask |
| `readVisibleFaces<MeshT>(hostFaceAoSoA, n)` | AoSoA → `VisibleFace` vector |
| `writeClosureFace<MeshT>(hf, f, vf)` / `initClosureFaceMembers<MeshT>(hf, b, e)` | the write side; both no-ops in `HangingNode2to1` mode |

`CloseResult` carries `sourceRed[i]` (which red face visible face *i* takes its face
user fields from) and `UncloseResult` carries `sourceVisible[r]` (the lowest-gid
child) plus `redOfVisible[i]` — the latter is what makes `translateMask()` a plain
scatter with no gid map, and Task 4 will reuse it.

`detail::refineLocalConforming()` in `Tessera_Refine.hpp` runs
un-close → mask translation → red 1→4 split → close, then re-derives edges, keys, and
the vertex 1-ring CSR from the **visible** face list. `refineLocal()`'s dispatcher
calls it; the abort stub is gone. Two contract changes, both documented at the
function:

- **Face gids are no longer local indices** in `Conforming` mode. A closed red face's
  gid is retired into its children's `ClosureParent`, so it must not be reissued:
  red gids persist across the call and closure children are allocated above the
  current max. (Vertex and edge gids still equal their index.) Live face gids are
  therefore sparse — already true of the distributed `refine()`. As a knock-on, the
  edge AoSoA's incident-face field now stores real face **gids** rather than indices
  that happened to equal them.
- **`refineLocal()` enforces no 2:1 balance** (neither mode does — that is
  `refine()`'s Phase 1). *One* call from a balanced mesh bisects each edge of a kept
  face at most once, so the patterns apply; a *sequence* of adaptive `refineLocal()`
  calls can drive a >2:1 jump, at which point an edge carries more than one midpoint
  and no fixed pattern applies. `closeFaces()` detects exactly that — it checks
  whether either half-edge `(v[k], m)` / `(m, v[k+1])` of a split edge is itself in
  the split-edge map — and aborts, rather than silently emitting a mesh that still
  has hanging nodes.

**Where the "red child of a refined face has `|S| = 0`" assert lives.** Inside
`closeFaces()`, driven by the optional `freshChild` flag array that
`refineLocalConforming()` passes (Task 4's `refine()` will pass the same). It sits
where `|S|` is already computed, so it costs nothing, and it is a property of the
*pair* (red list, split-edge map) — checking it in the caller would duplicate the
`|S|` computation. **Why it holds by construction:** the 1→4 split replaces parent
`(a,b,c)` with children whose nine edges are the six half-edges `(a,m_ab)`,
`(m_ab,b)`, … and the three interior edges `(m_ab,m_bc)`, … Every one of those edges
has a *midpoint vertex of this round* as an endpoint, and the split-edge map is keyed
by edges of the **pre-split** red layer, whose endpoints are all pre-existing
vertices. So no child edge can be a key in the map. It can only fire if the map is
polluted with an edge that did not exist before the split — which is the real bug it
is there to catch. Whether it ever fires is a Task 8 observation.

**Extra hazard found and fixed: closure-member initialization.** See the consequence
box under *Data model — closure bookkeeping* above. This was not in the design and is
the one thing Task 2 had to add outside the closure header.

**Acceptance.** New `unit` test `refine_closure` (SERIAL + HIP, np1) — written and
compiled, **not run** (see the handoff contract). It pins:

- each of the four `|S|` cases against hand-written child triangles, in emission
  order, for a hand-built parent; CCW winding via a planar embedding of the parent
  (signed area > 0); consecutive child gids from `firstChildGid`; every child
  carrying the parent's level and `ClosureParent`/`ClosureParentVerts`; the `|S| = 3`
  red-closure *not* incrementing the level; `unclose()` restoring the parent exactly
  from any single child;
- the green pattern for each of the three edges in turn (the rotation);
- the blue tie-break in **both** directions, by relabelling the two midpoint gids so
  the diagonal must flip, with `ClosureStats`' per-diagonal counters asserted;
- **rotation invariance**: cyclically relabelling the parent's corners leaves the
  emitted triangle *set* (sorted corner-gid triples) unchanged, for all four `|S|`;
- `translateMask()`: any single marked child marks the parent, no marked child marks
  nothing, and a random visible mask maps to exactly the owning red set;
- `unclose ∘ close == identity` on a hand-built red layer from a random (seeded LCG)
  ~35 % mask over `buildIcosphere(3)`, compared gid-keyed on gid/corners/level;
  plus `countClosureChildren()` agreeing with the number actually emitted, and
  `sourceVisible[r]` being the lowest-gid child;
- `refineLocal()` with a **partial** mask on `buildIcosphere(2)` in `Conforming`
  mode: owned Euler `V−E+F == 2`, every edge with exactly two incident faces, no
  vertex in the interior of an edge, all `faceNormalRaw` outward, and the **same
  vertex count** as `HangingNode2to1` mode; face user fields inherited (all children
  of one parent agree, every value is one of the seeded base-face values).

The "no interior vertex" check is **geometric, not topological**: the topological
reading ("some `m` has edges `(u,m)` and `(m,w)`") is true of every ordinary
triangle. It intersects the neighbour sets of `u` and `w` and tests each candidate for
collinearity with *and* strict betweenness on the segment.

**Non-vacuity guard.** The three conformity checks are also run on the
`HangingNode2to1` result *with the same mask*, and the test **fails** if they pass
there — otherwise a mask too weak to create a hanging node would make the conforming
case prove nothing.

**Report back.** *(delivered — see the blue-tie-break and assert subsections above.
Short version: the tie-break compares the two globally-agreed midpoint gids over a
rotation-derived corner triple, so it reads no partition-local quantity; the
`|S| = 0` assert lives in `closeFaces()` behind the `freshChild` flag and holds
because every edge of a fresh child has a this-round midpoint as an endpoint while
the split-edge map is keyed by pre-split edges. The one design gap was closure-member
initialization.)*

---

### Task 3 — Distributed split-edge discovery — **DONE**

**Goal.** Every rank learns the midpoint gid of every split edge it touches,
including on kept faces. No closure yet.

**What landed.** All of it in `src/Tessera_RefineParallel.hpp` — see *Phase 2
extension* above for the spelling. `detail::OwnerMsg` gained `refining`; 2a
advertises every owned face; 2b aggregates `{participants, refOwner, anyRefining}`
and drops non-split edges; 2c is unchanged and now reaches kept co-sharers.
`RefineResult::midpoints` is documented as the split-edge map, and
`phase2Adverts` / `phase2AdvertsRefining` were added for the Task-8 volume
measurement. No call site changed; `refine_parallel` and both `markquality` tests
were not edited.

**The `inc.size() != 2` guard is unaffected.** That guard lives in **Phase 1**, the
2:1 mark-propagation fixpoint, which *already* advertised every owned face's edges
(one `PropMsg` per (owned face, edge)). Task 3 did not touch Phase 1, so the
coordinator still receives exactly two `PropMsg`s per edge of a closed surface — one
from each incident face, since a face is owned by exactly one rank and every face
advertises. **Phase 2 has no such guard**: it aggregates by *participant rank*, not
by incidence, into a `std::set<Rank>` whose size is 1 (both incident faces on one
rank) or 2. A count-of-two assumption would have been wrong there both before and
after this change, and none is made.

**Estimated message-volume increase (Task 8 measures the actual).** Phase 2a goes
from `3·nRefining` to `3·nOwnedF` messages per rank — a factor of `1/ρ` where
`ρ = nRefining/nOwnedF` is the post-fixpoint refine fraction. Downstream rounds grow
far less: 2b's replies and 2c's gid sends are per *(split edge, participant)* and a
split edge has at most 2 participants either way, so their volume rises only by the
edges whose kept side sits on a different rank — an O(boundary) term, not O(F). The
test's fixture (`gid % 7`, subdiv-2 icosphere, no fixpoint propagation at level 0)
has `ρ ≈ 1/7`, so 2a should be roughly **7×** on round 1 and closer to `1/ρ` for the
later rounds' larger `ρ`. Since 2a is one `allToAllV` of a 24-byte struct over
`3·nOwnedF` entries, this is a constant-factor bump on the cheapest of Phase 2's
rounds, not a complexity change. The test prints total-vs-refining-only per round.

**Acceptance.** New `regression` test `refine_splitedges` (SERIAL + HIP, ranks 1–5)
— written and compiled, **not run** (handoff contract). It pins:

- **round-1 exactness against a partition-free reference.** On the level-0 subdiv-2
  icosphere the 2:1 fixpoint provably cannot fire (all levels equal ⇒ no
  final-level gap reaches 2), so the refining set *is* the caller's mask and the
  split set is exactly the edges of `{f : gid % 7 == 0}` — computed from a
  replicated, un-partitioned mesh with no MPI. Each rank's key set must equal that
  reference restricted to the edges of its own pre-refine owned faces.
- **the midpoint gid block**: the globally distinct gids are exactly
  `[V0, V0 + |S|)`, checked by count *and* sum *and* XOR against the closed forms,
  so a duplicate and a gap both fail. This is the check that would catch the
  extension accidentally creating or renumbering a midpoint.
- **non-vacuity**: at ranks ≥ 2 the global count of split edges a rank holds that
  lie on *none* of its refining owned faces must be **positive** — those are
  exactly the kept-side discoveries the pre-Task-3 code missed. Zero ⇒ the test
  proved nothing ⇒ it fails. (It is legitimately 0 at np1, so the check is
  rank-gated.)
- **multi-round soundness + completeness** over three further adaptive rounds,
  where levels do differ and the fixpoint really propagates so no cheap serial
  reference exists. `checkSplitEdgeCoverage()` derives the ground truth from the
  edge coordinator instead: an edge is split iff *some* rank reports it; then every
  rank touching it must report it (completeness) and no rank may report an edge
  none of its faces touches (soundness). Plus `checkMidpointAgreement` and
  `check21Balance` each round.

**Report back.** *(delivered — see the `inc.size() != 2` and message-volume
subsections above. Short version: the two-incidence guard is Phase 1's and Phase 1
was already advertising every owned face, so nothing there changed; Phase 2 never
assumed two incidences. Estimated 2a growth is `1/ρ` ≈ 7× on the test fixture's
round 1, with the reply/gid rounds growing only by an O(partition-boundary) term.)*

---

### Task 4 — Wire closure into distributed `refine()` — **DONE**

**Goal.** `Conforming` distributed refinement works. This is the milestone.

**What landed.**

- `src/Tessera_RefineParallel.hpp`: `detail::refineHangingNode()` →
  `detail::refineImpl()`, shared by both modes with the mode branch inside; the
  red-layer snapshot (`fV`/`fG`/`fL`/`fSrc`, `nRedF`) and the red-indexed `mark`;
  steps 0 / 0b / 3b / the widened 3c gid exscan; `writeClosureFace()` at the face
  materialization site; step 3e/3g/3h/3i now read the `newVis` visible list.
  `refine()` is a one-line forward. `RefineResult` gained a `ClosureStats closure`
  member so a test can report the `|S|` histogram, the closure-face fraction, and
  both blue-diagonal tallies without instrumenting the library at the call site.
- `src/Tessera_Distribute.hpp`: **`distribute()` needed
  `initClosureFaceMembers<MeshT>()`.** It builds a fresh face AoSoA rather than
  copying tuples, so the closure members were left zero-filled — and a zero
  `ClosureParent` reads as face gid 0, which `unclose()` would take for a real
  retired parent. This is the same hazard Task 2 found in the builder and Task 6
  must still fix in the HDF5 reader. `migrate()` copies whole tuples and needs
  nothing.
- `src/Tessera_Profiling.hpp`: level-2 `refine_unclose` / `refine_close`, level-3
  `refine_closure_patterns`.
- `tests/MeshInvariants.hpp`: `checkConforming`, `checkOwnedEuler`,
  `checkNoInteriorVertex`, `checkClosureInverse`, `check21BalanceRed`,
  `check21BalanceOn`, `ownedVisibleFaces`, `ownedVisibleFaceLevels` — see
  *Invariants and acceptance* above for what each pins and the one place the
  design had to bend (`checkNoInteriorVertex` cannot use a coordinator).
- `tests/test_refine_conforming.cpp` + registration (`regression`, SERIAL + HIP,
  ranks 1–5).

**Nothing leaked into a caller-visible API.** The red/visible distinction is
entirely internal: `refine()`'s signature, the mask's indexing (visible faces, as
before), and every mesh accessor are unchanged, and no existing call site was
edited. The only additions are the new `RefineResult::closure` member and the
already-existing `ClosureParent`/`ClosureParentVerts` face slices.

**Two things the design section got wrong, now rewritten above.** (1) The face-gid
block must sit above the max **visible** gid, not the max red gid, to stay monotone
across rounds — see *Face gid allocation*. (2) The design assumed all four new
invariant checks could be coordinator-routed; `checkNoInteriorVertex` cannot,
because positions are not available for the vertices a closure child names.

**Acceptance.** `regression` test `refine_conforming` (SERIAL + HIP, ranks 1–5) —
written and compiled, **not run** (handoff contract). It pins:

- **three successive adaptive rounds** (`gid % 7`, then `gid % 5` twice) with
  `checkConforming`, owned Euler `== 2`, `checkNoInteriorVertex`,
  `checkOwnershipPartition`, `checkMidpointAgreement`, `check21BalanceRed`,
  `checkClosureInverse`, fixpoint termination, and `RefineResult::closure.nVisible`
  agreeing with the mesh's global owned-face count;
- **non-vacuity, twice over**: closure children must actually be emitted
  (`totalClosure > 0`), *and* the identical mask sequence is run on a
  `HangingNode2to1` mesh as a control whose conformity checks must **fail** —
  otherwise a mask too weak to create a hanging node would prove nothing. Both are
  hard failures, not warnings;
- **empty mask**: V/E/F counts *and* all three gid checksums unchanged, no closure
  children, no midpoints, Euler 2 — i.e. the closure is the identity on an
  already-closed mesh;
- **full (uniform) mask**: no kept faces ⇒ `|S| = 0` everywhere ⇒ no closure
  children, and the global V/E/F must equal a `HangingNode2to1` mesh's under the
  same mask. A mismatch means the closure fired when it should have been inert.

It prints per round: the `|S|` histogram, closure-face count and fraction, both
blue-diagonal tallies, and the control mesh's Euler / bad-incidence / T-junction
figures.

**Report back.** *(delivered — see above. Short version: nothing leaked into a
caller-visible API; the two design errors were the gid-allocation bound and the
assumption that every new invariant could be coordinator-routed; and the one
hazard found outside the refine path was `distribute()` leaving the closure
members zero-filled.)*

---

### Task 5 — migrate / loadBalance / halo on a closed mesh — **DONE**

**Goal.** A conforming mesh survives redistribution and re-haloing.

**What landed.**

- `src/Tessera_MeshMigrate.hpp` — **round S**, a `Conforming`-only local pass at
  the head of `migrate()` that makes every closure child follow the **lowest-gid**
  sibling's destination. `MigrateStats { siblingFixups, siblingGroups }` is now
  `migrate()`'s return type (was `void`); every pre-existing call site is
  unaffected because they all discard it. `ownedFaceWeights()` gained the
  per-parent weighting (below). Header rounds list and the block comment at the
  fixup carry the rationale.
- `src/Tessera_Zoltan2Balancer.hpp` — `loadBalance()` forwards the `MigrateStats`.
  `computeLoadBalance()` is **unchanged**: it already fed `ownedFaceWeights()` to
  Zoltan2, so the weighting needed no signature or call-order change.
- `src/Tessera_Profiling.hpp` — level-2 `migrate_sibling_cohesion`.
- `tests/MeshInvariants.hpp` — `checkSiblingCoresidency()` (coordinator-routed, so
  rank-count independent and no ghost layer needed) and `closureSiblingGroups()`.
- `tests/test_conforming_migrate.cpp` + registration (`regression`, SERIAL + HIP,
  ranks 1–5).

**Repair, not reject.** A `dest` that splits a sibling group is the *normal* case,
not a caller error: `computeLoadBalance()` partitions by face **centroid** and
closure siblings have distinct centroids, so Zoltan2 scatters them on essentially
every call. Rejecting would make `loadBalance()` unusable on a conforming mesh,
and pushing the repair onto callers would duplicate the same loop at each one. So
the fixup repairs and *reports*: `MigrateStats::siblingFixups` is the loud
channel, and it is a returned value rather than a log line so a test can assert on
it — `conforming_migrate` fails if it is zero at ranks ≥ 2, which is what keeps
the case from being vacuous. The repair reads only globally-agreed face gids
(lowest-gid sibling wins), so it does not reintroduce a partition dependence of
the kind the blue-diagonal tie-break was careful to avoid.

**Co-residency is a real invariant, not a nicety.** Two ranks each holding a child
of one parent would each `unclose()` it into a red face — a duplicated face, a
broken ownership partition, and a global face count that grows every round. It
holds by construction after `refine()` (`closeFaces()` runs on one rank's red
layer), and `migrate()` is the only operation that can break it, hence the fixup
sits there and nowhere else. Detecting a *pre-existing* violation would need
communication, so that check lives in the test (`checkSiblingCoresidency`) rather
than in `migrate()`'s hot path.

**Weighting: contract unchanged, values changed.** `ownedFaceWeights()` still
returns one `double` per owned face in local index order. In `Conforming` mode a
closure child now gets `1/nsiblings` and a passed-through red face `1.0`, so the
total weight is exactly the **red**-face count and a sibling group weighs 1.0
however it is split — which is precisely what makes the post-fixup partition's
load equal the load Zoltan2 optimized. Without it the closure (an
O(level-jump-boundary) set, rebuilt on every refine) would read as real load and
pull parts toward refinement fronts.

**Halo rebuild needed nothing.** Round D is generic over the face tuple, and on a
conforming mesh every edge has exactly two incident faces, so the 1-ring it closes
is cleaner rather than harder. The one place the wider face tuple could bite is
the ghost pack/unpack, which the test exercises directly.

**Acceptance.** New `regression` test `conforming_migrate` (SERIAL + HIP, ranks
1–5) — written and compiled, **not run** (handoff contract). Two cases, both after
`buildIcosphere(2)` → `distribute` → two adaptive conforming rounds:

- **adversarial `dest`** — `dest[f] = faceGid % size`. Siblings hold consecutive
  gids, so at size > 1 essentially every group is scattered and the fixup must
  fire. Then `checkSiblingCoresidency`, `checkOwnershipPartition`,
  `owned1RingLocal`, `checkConforming`, owned Euler `== 2`,
  `checkNoInteriorVertex`, `check21BalanceRed`, and an unchanged topology
  checksum; plus a ghost **corrupt-and-resync** over all three halo plans, which
  is what exercises the two extra closure members of the wider conforming face
  tuple (Task 8 risk point 6). Non-vacuity: the global fixup count must be
  positive at ranks ≥ 2, and the closure-group count must be positive at every
  rank count.
- **`loadBalance`** — dump every owned face on rank 0, then `loadBalance()`.
  Asserts the max owned-face count improves and that the **weighted** max load
  (the quantity `ownedFaceWeights()` defines and Zoltan2 optimizes) lands within
  2× ideal, then re-runs the same invariant sweep.

Both print their figures: sibling groups, `dest` fixups, and pre/post max face
count and max weighted load against ideal. The topology checksum is printed
per rank count by the gate's ranks-1–5 sweep, which is how the cross-rank-count
comparison is obtained; the dedicated rank-count-independence assertion is Task
7's `conforming_determinism`.

**Report back.** *(delivered — see above. Short version: a violating `dest` is
**repaired**, because Zoltan2 produces one on every call and rejecting would make
`loadBalance()` unusable on a conforming mesh, with `MigrateStats::siblingFixups`
as the reported channel; `computeLoadBalance()`'s signature and
`ownedFaceWeights()`'s contract are both unchanged — only the weight *values*
differ, and only in `Conforming` mode. `migrate()`/`loadBalance()` now return
`MigrateStats` instead of `void`, which no existing call site notices.)*

---

### Task 6 — I/O, marking, example, docs — **DONE**

**Goal.** The feature is usable and documented end to end.

**What landed.**

- `src/Tessera_IoCommon.hpp` — `detail::forEachUserFieldN<UserBegin, N, AoSoA>`,
  an explicit-count user-field loop; `forEachUserField` now delegates to it with
  `N = tuple size − UserBegin`. Needed because the face user pack is not the face
  tuple's suffix in `Conforming` mode.
- `src/Tessera_HDF5Writer.hpp` / `src/Tessera_HDF5Reader.hpp` — **format version
  2.** Both face user-field loops (and `n_user_f_fields`) are now bounded by
  `numFaceUserFields<MeshT::face_user_fields>()`; the two closure members are
  written as their own `/faces/closure_parent` (u64) and
  `/faces/closure_parent_verts` (u64 × 3) datasets; a new root attribute
  `refinement_mode` (0/1) is written in both modes and validated on read; the
  reader calls `initClosureFaceMembers<MeshT>()` on its freshly-materialized face
  block and then fills the two members from the datasets.
- `src/Tessera_MeshMigrate.hpp` — `repairClosureCohesion( mesh, dest )`, the
  **collective** counterpart of `migrate()`'s local round S (below).
- `examples/02_mesh_pipeline` — `--refine-mode {hanging,conforming}` (default
  `conforming`), dispatching to two instantiations of the same `run()`; the mode
  tag is part of the frame stem so the two can be compared frame by frame.
- `tests/test_io.cpp` — `runConforming()` case; `tests/test_markquality_conforming.cpp`
  (new, `regression`, SERIAL + HIP, ranks 1–5); README + `docs/design.md` updated.

**The one thing the design missed: the reader breaks sibling co-residency.**
Task 5 established that a split sibling group means two ranks each un-close the
same parent, and put the repair in `migrate()` — deliberately **local**, on the
stated grounds that "siblings are co-resident on entry". `readMesh()` is the one
caller for which that premise is false: it hands every rank a *fresh contiguous
block of dense face indices*, deliberately unrelated to the writer's partition,
so a group straddling a block boundary arrives already split and round S — which
only ever sees one rank's holdings — cannot detect it. Hence
`repairClosureCohesion()`: one `allToAllV` round trip keyed on the parent gid
(`parent % size`), the coordinator picking the destination of the lowest-gid
sibling, i.e. the *same* globally-agreed rule round S applies, so running one
after the other is idempotent and neither introduces a partition dependence.
It lives next to round S rather than inside the reader because it is a property
of conforming meshes, not of HDF5. Round S stays local and unchanged: paying a
collective on every `loadBalance()` to handle a case only the reader creates
would be the wrong trade.

**Closure fields are not user fields, and the distinction is load-bearing.**
Writing them as `u<n>`/`u<n+1>` would have round-tripped *by accident* — they are
plain `GlobalId`/`GlobalId[3]` members and the writer and reader derive the field
set with the same (over-counting) formula, so the two would have agreed. It is
still wrong: it inflates `n_user_f_fields`, puts closure bookkeeping in the XDMF
attribute list as if it were solver data, and makes the on-disk schema of a
conforming file differ from a hanging-node file in a way nothing declares. The
count fix plus explicit datasets plus the `refinement_mode` attribute makes the
difference explicit and checkable.

**No dense translation.** Unlike `/edges/verts` and `/faces/verts`, the closure
datasets hold **persistent** gids — a retired red face gid (which names no live
entity in the file at all) and three vertex gids — so they need neither the
dense-index map nor the ghost fetch. A parent's corners are always corners of the
parent's own children, hence of faces the writing rank holds.

**`markByQuality` needed no library change.** A criterion returns a mask over
*visible* owned faces and `refine()`'s step 0b translates it; neither criterion
reads anything mode-dependent. `CurvatureCriterion`'s `inc.size() != 2` skip is
the interesting part: on a conforming mesh there are no such edges, so the
coordinator's assumption becomes true rather than usually-true. The test measures
that directly rather than by proxy — `checkConforming()` counts exactly the edges
that guard would skip, through the same `edgeCoordRank` routing.

**Acceptance.** All written and compiled, **not run** (handoff contract).

- `io` (`regression`, SERIAL + HIP, ranks 1–5) gained `runConforming()`: two
  adaptive conforming rounds, then write → read → **red-layer identity**. The
  red layer is compared as a rank-count-independent checksum (count + BXOR + SUM
  over a per-red-face hash of gid, the three corner gids *in order*, and level) —
  order kept rather than sorted so a winding change fails too. Plus
  `checkSiblingCoresidency` (what pins `repairClosureCohesion`),
  `checkConforming`, owned Euler `== 2`, `check21BalanceRed`,
  `checkOwnershipPartition`, `owned1RingLocal`, unchanged topology and
  user-field checksums (the latter doubling as the `UserBegin`-shift canary),
  and finally **one more conforming `refine()` on the read-back mesh** — the
  check that the recovered red layer is not just equal but usable. Non-vacuity:
  the round-trip fails if no closure children were emitted before the write.
- `markquality_conforming` (new, `regression`, SERIAL + HIP, ranks 1–5): one
  fixture (subdiv-2 icosphere, vertex gid 0 pushed out 3×, so the spike ring is
  both long-edged and steeply folded) drives both criteria for two rounds of
  mark → refine → migrate → halo, asserting the full conforming sweep each
  round. `EdgeLengthCriterion`'s threshold (1.0) sits in a wide empty gap;
  `CurvatureCriterion`'s is derived from the replicated reference's own dihedral
  distribution (midpoint of the largest gap), the rule `markquality_curv` uses.
  Non-vacuity, all hard failures: the mask must be a proper non-empty subset
  every round, closure children must be emitted, and the same criterion on a
  `HangingNode2to1` control must leave T-junctions behind.

**Report back.** *(delivered — see above. Short version: yes, the reader needed a
format-version bump, 1 → 2, because the new `refinement_mode` root attribute is
written unconditionally and a v1 file has none; that attribute is also how the
two file shapes are told apart, as a hard `abortOnMismatch` against the mesh
type's own mode rather than a fallback. The computed on-disk delta is
`+4 × sizeof(GlobalId)` = **32 B per owned face** (`closure_parent` 8 B +
`closure_parent_verts` 24 B); the `io` test prints that figure against the actual
`.h5` size for Task 8. The one design gap was the reader breaking the
sibling-co-residency premise round S is built on.)*

---

### Task 7 — Dedicated conforming test suite; flip the default — **DONE**

**Goal.** Conforming refinement has first-class test coverage of its own — not just
the per-feature tests Tasks 2–6 added along the way — and `Conforming` becomes the
default.

Landed as three commits, one per checkpoint.

#### Checkpoint A — mode-parity registration (landed)

Pinning the legacy tests to `HangingNode2to1` (needed for the flip) would otherwise
*remove* conforming coverage from those paths. So each was pinned **and** its
conforming counterpart confirmed present — all of which Tasks 2–6 had already
written, so Checkpoint A added no new test, only explicit mode spellings:

| Existing test | Pinned | Conforming counterpart |
|---|---|---|
| `refine` (unit, np1) — 3 mesh types | yes | `refine_closure` (Task 2) — covers `refineLocal` conforming |
| `refine_parallel` (regression) | yes | `refine_conforming` (Task 4) |
| `refine_splitedges` (regression) | yes | `refine_conforming` (see below) |
| `migrate_mesh`, `loadbalance` (regression) | yes | `conforming_migrate` (Task 5) |
| `io` (regression) | yes | `io`'s `runConforming()` case (Task 6) |
| `markquality_edge`, `markquality_curv` (regression) | yes | `markquality_conforming` (Task 6) |
| `staleslice_guard` (unit, np1) | yes | its own new conforming case (Checkpoint B) |
| `distribute`, `connectivity`, `keys`, `data_model`, `halo`, `migrate`, `geometry`, `global_reduce`, `stencil_topology`, `apply_stencil`, `reduce_faces` | n/a — no `refine()` call | — |

**`refine_splitedges` was pinned although the table in the original plan did not
list it**, and the reason is worth recording: it isolates `refine()`'s **Phase 2**,
which both modes share, and its whole reference construction rests on "the edges of
my owned faces" naming a single set. That is true only when the visible layer *is*
the red layer. In `Conforming` mode `RefineResult::midpoints` is keyed by the red
layer while `mesh.faces()` shows the closure, so both the round-1 partition-free
reference and the multi-round coordinator ground truth would have to un-close
first — a different test. The composition is what `refine_conforming` covers.

**Nothing resisted pinning**, and no test needed a behavioral edit: every site was a
one-line change from the seven-argument `Mesh<...>` spelling to an eight-argument one
naming `RefinementMode::HangingNode2to1`, plus a comment saying which conforming test
covers the same ground. The mode-insensitive tests were left on the default
deliberately, so they now run in `Conforming` mode and act as a free check that the
wider face tuple does not disturb construction, distribution, halo, geometry, or the
stencil/reduction operators.

#### Checkpoint B — new conforming-specific tests (landed)

1. **`conforming_operators`** — `regression`, SERIAL + HIP, ranks 1–5.
   `tests/test_conforming_operators.cpp`. The payoff test.
   `buildIcosphere(2)` → `distribute` → two adaptive conforming rounds →
   `migrate` → `haloExchange`, then, at the **closure vertices** specifically:
   the 1-ring is a closed fan; `buildVertexStencil(mesh, 1)`'s `k=1` row equals an
   independently *face-derived* 1-ring and the `vertexFaces()` row equals the set of
   local faces containing the vertex; `applyStencil` with `f(p) = p_x` and uniform
   weights matches a position-derived reference; `reduceVertexFromFaces` with the
   one-third-area op reproduces the global area identity *and*, per closure vertex,
   the true one-third sum over its full incident-face set.

   **How the closure-vertex set is identified**, and why it needs no new library
   support: a closure child stores its retired parent's three corners outright, so a
   corner of a child that is **not** one of `ClosureParentVerts` is by construction a
   midpoint of a parent edge — i.e. exactly the hanging node the closure absorbed.
   The scan runs over **all locally held faces, owned and ghost**, because an owned
   vertex's incident faces may be owned by a neighbour; the closure members travel
   with the ghosts through the face halo, which `conforming_migrate` already
   exercises. **How it fails loudly if the set is empty:** the global count of owned
   closure vertices is reduced and a non-positive value is a hard `++fails`, not a
   warning — a mask too weak to create a hanging node would otherwise make every
   check below it vacuously true.

   The two `applyStencil` max errors (closure vs interior) and the two area errors
   are asserted **separately at the same tolerance**, so a closure-specific
   regression cannot hide behind the far more numerous interior vertices, and both
   are printed.

   The **closed-fan** check is the local form of conformity and is what a
   hanging-node mesh fails: every edge incident to an owned vertex must be shared by
   exactly two of that vertex's incident faces. At a hanging node the fan is an open
   half-disc and its two boundary edges have one incident face each.

2. **`conforming_determinism`** — `regression`, SERIAL + HIP, ranks 1–5.
   `tests/test_conforming_determinism.cpp`. Three cases:
   - **Rank-count independence** against a reference *the same run* recomputes on
     `MPI_COMM_SELF` — every rank redundantly refines the whole mesh alone — under a
     **geometric** mask (centroid above a z threshold), which selects the same faces
     at any partition where a gid mask would not. The comparison is by **quantised
     position**, for the reason given under *Blue tie-break* above; the breakdown
     (V/E/F, `|S|` histogram, red layer, visible layer, closure-vertex set) is
     printed component by component, plus a direct count of blue parents whose
     chosen diagonal differs from the reference. The diagonal is read off the mesh
     without assuming which branch fired: among a three-child group the internal edge
     with exactly one endpoint in the parent's corner set *is* the diagonal.
   - **Closure idempotence** — an empty-mask `refine()` on an already-closed mesh is
     un-close → no red split → re-close, and must reproduce the visible layer. The
     comparison excludes each face's own gid (the re-closure allocates a fresh block
     above the global max, so child gids legitimately move) and covers the sorted
     corner gids, level, parent gid and parent corners, plus the red layer by gid and
     the V/E/F counts. **This is the sharpest probe of risk point 9** and, on the code
     as it stands, is expected to expose it.
   - **Cross-mode equivalence under a uniform mask** — strengthened past
     `refine_conforming`'s count comparison to full topology: identical face gids,
     corner gids and levels, and identical vertex/edge gid checksums, over two
     rounds. Valid across *modes* (they share the red engine and its exscans) even
     though it is not valid across rank counts.

3. **`conforming_quality`** — registered **`unit`** by Task 7; **promoted to
   `regression`** (SERIAL + HIP, ranks 1–5) by Task 8 sub-task D7 once its bounds
   were measured, and extended from 8 rounds to **16**. See Decision 12 for the
   measured bounds that replaced the provisional ones below.
   `tests/test_conforming_quality.cpp`. Eight adaptive rounds over a shrinking
   geodesic cap (half-angle `0.35 × 0.6^k`, chosen so the marked set neither dies out
   nor engulfs the sphere as the faces halve), tracking per round the global minimum
   triangle angle, the global maximum radius ratio `Q = abc(a+b+c)/(16A²)` (1 for
   equilateral), and the closure-face fraction — each asserted against a **fixed**
   bound every round, since a bound that had to grow with the round count would mean
   the closure is not transient after all. Conformity, red-layer 2:1 balance,
   sibling co-residency and Euler are re-checked each round too, so the quality
   figures are not measured on a mesh that has quietly stopped being conforming.

   **Provisional bounds and their derivation** (marked provisional in the source, in
   the CMake registration, and in README *Known Issues*): on an exactly equilateral
   parent the patterns give min angle 60° (`|S|` = 0), 30° (green — a median cut
   gives 30/60/90), 30° (blue — the 30-30-120 sliver off the midline is the worst of
   the three children), 60° (red-closure); the corresponding worst `Q` are 1.37 for
   the 30-60-90 green child and 2.16 for the 30-30-120 blue child. Icosphere red
   faces are near- but not exactly equilateral, so the realised worst case sits
   below those. The registered bounds — **min angle ≥ 20°, `Q` ≤ 4.0, closure
   fraction ≤ 0.50** — allow roughly a further third of shape loss for that
   distortion. They are *derived, not measured*: this is why the test is `unit` and
   not in the gate. **Superseded by Decision 12** — D7 measured them; the
   derivation above proved sound in direction (blue is the worst family, the green
   worst is below its ideal-parent value) but loose by ~1.6× on `Q`.

4. **`staleslice_guard` extension** (landed in the existing SERIAL-only np1 test).
   The conforming refine path un-closes and re-closes, so it reallocates the face
   AoSoA and changes the local face count **even when nothing refines**. A face
   slice, a `MeshGeometry`, and a `VertexStencil` held across it must each abort when
   copied; all three are checked, since each reaches the guard by a different route
   (a bare slice; the position `GenerationHandle` inside the geometry accessor; the
   CSR `GenerationHandle` inside the stencil).

#### Checkpoint C — flip the default and sync docs (landed)

- `Mesh`'s `Mode` parameter now defaults to `RefinementMode::Conforming`.
  **The flip broke nothing at compile time** — the whole suite, both backends, and
  the example built clean with no further edits, which is the payoff of Checkpoint A
  going first and of `FaceField::UserBegin` having been held fixed in Task 1.
- `src/Tessera_RefinementMode.hpp` — the enumerator comments now say which mode is
  the default and why (a hanging node breaks an operator *silently*).
- `docs/design.md` — *Refinement modes* table reordered with `Conforming` first, the
  code sketch inverted, a new *Why `Conforming` is the default* paragraph, and the
  stale "the default is still `HangingNode2to1`" sentence corrected.
- `README.md` — the API sketch's mode comment inverted, the `refine()` line
  clarified, *Known Issues* rewritten to say the default is now `Conforming` and
  still unproven (with reverting named as the escape hatch), and a new entry for
  `conforming_quality`'s provisional bounds.
- `CLAUDE.md` — task-log row updated to Tasks 1–7 done, Task 8 next.

**Acceptance.** Both modes are registered and covered across the suite; the three
new tests are written and registered at their intended tiers and ranks; the `Mesh`
default is `Conforming`; docs are in sync; the whole suite (SERIAL + HIP) and the
example compile clean; `clang-format` is clean on every touched file under **both**
the v21 (`/usr/bin`) and v19 (`/opt/rocm-6.4.2/llvm/bin`) binaries on this machine.
No test was run (handoff contract).

**Report back.** *(delivered — see above, plus risk points 9 and 10 which this task
contributed to Task 8. Short version: no legacy test resisted pinning and none
needed a behavioral edit; `refine_splitedges` needed pinning although the plan's
table omitted it; the closure-vertex set is identified from
`ClosureParentVerts` — a child corner that is not a parent corner — over owned and
ghost faces, with an empty set a hard failure; the quality bounds are 20° / 4.0 /
0.50, derived from the patterns' ideal-parent geometry plus about a third of margin
for icosphere distortion; and the default flip broke nothing at compile time. The
two findings from reading the code while writing tests against it are risk points 9
and 10 below — the first is a probable defect, the second a scoping correction the
design needed.)*

---

### Task 8 — Run the full suite and fix everything it finds

**Goal.** The whole test suite is green. **This is the only task in this plan that
runs tests** — Tasks 1–7 wrote code and tests without executing them, so this is the
first execution of any of it.

Budget a long session. If the failure list is large, treat each fix as its own
commit + push and keep the *Open failures* list below current, so the task can be
resumed by a later session.

**Procedure.**

```bash
hostname                                          # expect tuolumne* → systems/tuolumne/claude.md
spack env activate ~/spack_envs/tuolumne_trilinos/
cd build-tuolumne && bash ../run_cmake_toulumne.sh && make -j $(nproc)
cmake --build . --target format-check

flux batch ../scripts/tuolumne/run_regression_minset.flux   # gate: regression x {SERIAL,HIP} x ranks 1-5
ctest -L unit --output-on-failure                            # unit tier
```

Fix in **dependency order**, not failure-count order — a Task 1 field-index bug will
manifest as a dozen unrelated failures downstream:
build errors → `refinement_mode`/`refine_closure` (units) → `refine_splitedges` →
`refine_conforming` → **`conforming_determinism`'s idempotence case (risk point 9 —
the minimal reproducer; fix this before chasing anything downstream of it)** →
`conforming_migrate` → `io` → `markquality` → `conforming_operators` →
`conforming_determinism`'s other two cases → `conforming_quality` →
`staleslice_guard` → the legacy hanging-node tests.

**Triage first at these known risk points** (each is a place the design admits a
plausible defect):

1. **`FaceField::UserBegin` shift.** If `userFaceField<0>()` differs between modes,
   every user-field read on a conforming mesh is silently wrong. Symptom: garbage
   face user data, not a crash. Check this before anything else.
2. **Blue-diagonal tie-break.** A partition-dependent tie-break passes every test
   except `conforming_determinism`'s rank-count checksum. If that checksum differs
   across ranks 1–5, the tie-break is reading a partition-local quantity.
3. **Face gid collisions across rounds.** Retired parent gids are reused by un-close
   while children are allocated above the global max. Surfaces as duplicate face gids
   in `checkOwnershipPartition` *only after* a `migrate()` brings two ranks together —
   so a multi-round test, not a single-round one.
4. **Non-local vertices referenced by closure children.** By design a closure child
   may name a vertex gid the rank does not hold until the halo is rebuilt. Anything
   that assumes post-refine locality (see the post-refine gotcha in
   `docs/design.md`) will abort or read garbage.
5. **Sibling co-residency after `migrate()`/Zoltan2.** If the `dest` fixup missed a
   case, two ranks each restore the same parent → duplicate face, caught by
   `checkOwnershipPartition`.
6. **HIP-specific.** The closure derivation is host-side, but the face tuple grew by
   two members: check the `deep_copy` of the wider face AoSoA and any place a face
   tuple size or member index was assumed. A SERIAL pass with a HIP failure points
   here.
7. **I/O field-set mismatch.** The writer emitting the two closure fields and the
   reader not expecting them (or vice versa) fails as a round-trip checksum
   mismatch, not a read error.
8. **Generation-guard over-firing.** Un-close bumps `generation()` an extra time; a
   handle the tests expect to survive may now abort. Distinguish a real stale-handle
   bug from an over-eager bump.

9. **THE SPLIT-EDGE MAP IS THIS-ROUND-ONLY, BUT THE LEVEL JUMPS IT MUST CLOSE ARE
   PERSISTENT.** — **RESOLVED, Task 8 D2 (2026-08-04).** Confirmed at np1 exactly as
   predicted below, then fixed. The recovery is in `recoverSplitEdges()` /
   `UncloseResult::splitEdges`; see *Persistent split edges* under the distributed
   algorithm for what landed and Decisions 8 and 9 for what was chosen and why.
   Three notes for anyone reading the prediction below against the code:
   the uniqueness rule needed a **one-child** qualifier on the two half-edges (the
   rule as written below is ambiguous for the blue pattern); the union turned out to
   be needed for *refining* faces too, not "kept faces only", because a coarse face
   refining across a hanging node must reuse the existing midpoint; and the map's
   real reach is wider than the closure — keying phases 1 and 2 on half-edges is what
   makes the 2:1 bound hold across a hanging node at all. *(Original Task-7
   analysis preserved below.)*

   Step 3b′ calls `closeFaces( newRed, midGid, … )`, and `midGid` is built entirely
   inside Phase 2 from **this** round's refinements: the coordinator drops every
   edge with `!anyRefining`. But a kept face needs closing whenever its edge is
   bisected **in the red layer**, which is a persistent condition. Concretely:

   - Round 1 refines A; its neighbour B is kept, so B's edge `(a,b)` is bisected at
     `m` and B is closed into two green children. Correct.
   - Round 2 refines something else. Un-close restores B as a red face with corners
     `(a,b,c)` at level 0, while A's four children sit at level 1. The red layer
     still has a hanging node at `m` on B's edge — the 2:1 fixpoint permits it, so
     B is not forced to refine. But `(a,b)` is advertised in Phase 2a only by B,
     with `refining = 0`; A's children advertise `(a,m)` and `(m,b)`, which are
     different keys. So `(a,b)` has no refining incidence, is dropped, and `midGid`
     does not contain it. `closeFaces()` computes `|S| = 0` for B and passes it
     through **unclosed** — the mesh reverts to having a T-junction.

   So conformity should hold after the *first* conforming round and be lost on the
   second and later ones, for every face that was closed earlier and neither it nor
   its refined neighbour refines again. Expected symptom: `refine_conforming` round 1
   green, rounds 2–3 failing `checkConforming` and `checkOwnedEuler`; the same in
   `markquality_conforming`, `conforming_migrate`, `conforming_operators` and
   `conforming_quality`; and the **minimal reproducer** is
   `conforming_determinism`'s idempotence case, where an *empty* mask makes `midGid`
   empty outright and the visible layer collapses to the bare red layer in one step.

   **Proposed fix, for Task 8 to weigh** (not applied in Task 7 — it is a design
   change and the contract reserves fixes for the verification pass). The persistent
   split-edge map is recoverable **locally, with no communication and no new field**,
   from the closure bookkeeping that is already stored: for a parent `P` with corners
   `(a,b,c)` and its children, a parent edge is split **iff it does not appear as an
   edge of any child** — check the four patterns: green replaces `(a,b)` by
   `(a,m),(m,b)` while `(b,c)` and `(c,a)` survive as child edges; blue removes two;
   red-closure removes all three; `|S| = 0` removes none. The midpoint of a split
   parent edge `(x,y)` is then the unique child corner `m` outside `{a,b,c}` with
   both `(x,m)` and `(m,y)` present as child edges. So `unclose()` can return an
   `EdgeKey → GlobalId` map alongside the red layer, and step 3b′ closes against
   `midGid` **unioned with** it. Two things to get right: the recovered map must be
   consulted for kept faces only (a face refined this round has its edges replaced
   anyway), and `countClosureChildren()` in step 3c must be given the same union, or
   the gid exscan will under-allocate. Whether the union can also be reached through
   Phase 2 — e.g. by having the coordinator treat a red edge with a *single*
   incidence as split — is worth considering, but it needs the midpoint gid, which
   only the finer side knows and which the local reconstruction above already has.

10. **RESOLVED (Task 8 D6) — Decision 11: the gid tie-break stays, and the visible
    layer is documented as rank-count dependent up to blue diagonals.** The
    prediction below was exactly right, including the signature to look for: np1–4
    agree and np5 differs by 4 of 20 blue parents with every other component in
    exact agreement. The geometric alternative it proposes was implemented and
    measured, then reverted — not because of the floating-point fragility the
    paragraph anticipates (the exact-tie case is real but handled by a positional
    fallback), but because **the closure cannot see the positions**: it runs on the
    un-closed red layer, whose corners come from `ClosureParentVerts` and may name
    vertices the rank does not hold (risk point 4), which a 1-deep halo does not
    fix. No purely local rule can be rank-count stable on the data the closure has.
    `conforming_determinism` case A now asserts the sharper "differs only through
    blue diagonals" instead of dropping the check. Original analysis preserved:

    **Rank-count independence is narrower than the design assumed.** New vertex and
    face gids come from an `MPI_Exscan` over ranks, so which gid a midpoint gets
    depends on the partition, and after one round nothing gid-keyed is comparable
    across rank counts — see *What "partition independent" does and does not cover*
    above. `conforming_determinism` therefore compares by quantised **position** and
    reports its breakdown per component. If it fails only on the visible layer while
    the red layer, the `|S|` histogram, the closure-vertex set and V/E/F all agree,
    and `blueDiagMismatch > 0`, the cause is the blue tie-break comparing gids that
    the exscan ordered differently — **not** a partition-local read, and not a bug in
    the sense risk point 2 means. That is a genuine design choice to make: leave it
    (documenting that the visible layer is rank-count dependent up to blue
    diagonals), or replace the tie-break with a rank-count-stable rule. Note that no
    *gid*-based rule can be rank-count stable, for the recursive reason above; only a
    geometric one (e.g. the shorter diagonal) could, at the cost of floating-point
    fragility on symmetric quads. Record the decision here either way.

**Rules while fixing.**

- **A failing test is never silently excluded from the gate.** If something cannot be
  made green in this task, relabel it `unit`, record it in README *Known Issues* with
  how it reproduces and whether it predates this work, list it under *Open failures*
  below, and report it. Never delete a test or weaken an assertion to get green.
- **`conforming_quality` bound calibration.** Task 7 registered it as `unit` with
  provisional bounds. Replace them with values justified by the measured per-round
  data, then promote it to `regression` at ranks 1–5 *only if* it is stable across
  repeated runs. If the measured quality is genuinely unbounded with round count, that
  is a **design** finding, not a tolerance to loosen — stop, record it, and report it.
  **DONE (D7) — Decision 12.** The bound is fixed: `maxQ` saturates at 2.5254 by
  round 11 and is flat through round 16. Bounds recalibrated, the round count raised
  to 16, saturation and the red layer's own shape now asserted, and the test promoted
  to `regression` × {SERIAL, HIP} × ranks 1–5.
- **Escape hatch on the default flip.** Task 7 flipped the `Mesh` default to
  `Conforming` before anything had been executed. If conforming cannot be made green
  in this task, revert the default to `HangingNode2to1` (keeping conforming opt-in and
  fully built), record why in README *Known Issues* and under *Open failures*, and
  report it. Reverting the default is preferable to leaving a broken default.
  **Not exercised (D8) — Decision 13.** Conforming *is* green: 140/140 gate and
  62/62 unit at HEAD, zero open failures, so the escape hatch's precondition never
  arose and `Conforming` stays the default.
- If a fix changes a **design** decision, rewrite the affected design section of this
  file — do not leave the design describing something the code no longer does.
- Recompute and record the **new expected test totals** (they were 70/70 regression
  and 28/28 unit before this work) in the *Status* section here and in the CLAUDE.md
  task-log row.

**Acceptance.** `flux batch scripts/tuolumne/run_regression_minset.flux` fully green
at the gate definition (label `regression` × {SERIAL, HIP} × ranks 1–5), `ctest -L
unit` fully green, `format-check` clean, and the *Open failures* list empty.

**Open failures.** Task 8 is being worked in
[tasks/completed/conforming-refinement-debug.md](conforming-refinement-debug.md), which
breaks it into ordered sub-tasks D1–D8 and carries the live failure list, the
build/run recipe, and the harness traps. **Read that file, not this section, to
resume Task 8.**

First execution (2026-08-04, D0): **78 test instances pass**, including all of
Tasks 1 and 2 (`refinement_mode`, `refine_closure`) and the whole pre-existing
suite at np1–4 — so risk points 1, 6 and 8 are clear. Three failures:
`refine_splitedges` hangs at np≥2 (**fixed, D1** — a rank-0-guarded
`MPI_Allreduce` hidden in a `printf` argument, plus a missing between-rounds
re-halo; `refine_splitedges` is now green SERIAL + HIP at np1–5);
`refine_conforming` fails adaptive rounds 2
and 3 at np1 (**risk point 9 confirmed exactly as predicted** — round 2's
`euler=-136` equals round 1's hanging-node control, i.e. the mesh reverts to
hanging-node behaviour from the second round on); and `refine_conforming` aborts
at np2 with `std::out_of_range: unordered_map::at`. Nine registrations
(`conforming_migrate`, `loadbalance`, `io`, `markquality_edge`,
`markquality_curv`, `conforming_operators`, `conforming_determinism`,
`conforming_quality`, `markquality_conforming`) had not yet been executed at that
point; **all nine have since run and pass** at ranks 1–5 on both backends (D3–D6),
so the suite has 28 of 28 registrations passing.

**Report back.** *(delivered — D8, 2026-08-05. Per-sub-task root causes and evidence
are in [conforming-refinement-debug.md](conforming-refinement-debug.md); this is the
summary the task asked for.)*

**Totals at HEAD.** Gate (`regression` × {SERIAL, HIP} × ranks 1–5) **140/140**,
job `f3QTwKZkEkSK`, 14.3 min. Diagnostic tier (`ctest -L unit`) **62/62**, job
`f3QTwKgRBWzK`, 4.9 min. **202 instances over 28 registrations, zero failures.**
Before this work (at `f50a91b^`) the suite was 70 `regression` + 28 `unit` = **98
instances over 19 registrations** (7 `regression` + 12 `unit`), so conforming
refinement **more than doubled the instance count**: **+9 registrations, +104
instances**, and the gate itself went 7 → 14 registrations and 70 → 140 instances. `format-check` is clean on all 34 source files this work touched,
under both clang-format v21 (`/usr/bin`) and v19 (`/opt/rocm-6.4.2/llvm/bin`); the
only tree violation is `src/Tessera_Geometry.hpp`, untouched since `5dcb202` and so
predating Task 1.

**No test was relabelled to `unit`, and none was deleted or weakened.** The traffic
went the other way: `conforming_quality` was **promoted** `unit` → `regression`
(D7, Decision 12), moving 10 instances, and three assertions got *stronger* —
`empty-mask`'s gid check became a gathered sorted set comparison instead of an XOR
(D3), `conforming_determinism`'s idempotence case now demands `midpoints` equal the
recovered split-edge map exactly rather than merely be non-empty (D6), and its
rank-count case now asserts the visible layer differs *only* through blue diagonals
(D6).

**The `Conforming` default survived** — Decision 13.

**Failure list as first observed, and root cause.** D0 saw 3 failing registrations
and 9 that had never executed; the final tally is **nine defects**.

| # | First seen | Where | Root cause | Fixed in |
|---|---|---|---|---|
| 1 | `refine_splitedges` hangs np≥2 | test | `globalOwnedFaces()` — an `MPI_Allreduce` — was an *argument to a `printf` guarded by `if (rank==0)`* | D1 |
| 2 | `refine_splitedges` aborts np≥2 | test | **missing re-halo** between two `refine()`s | D1 |
| 3 | `refine_conforming` rounds 2–3 fail np1 | **library** | the closure's split-edge map was this-round-only (risk point 9) | D2 |
| 4 | *(unmasked by #3)* | **library** | a refining coarse face minted a second, coincident midpoint across a hanging node — *pre-existing since Step 6b* | D2 |
| 5 | *(unmasked by #3)* | **library** | 2:1 propagation was blind across a hanging node (both coordinator rules need two incidences) — *pre-existing* | D2 |
| 6 | `refine_conforming` aborts np≥2 | test | **missing re-halo** | D3 |
| 7 | *(unmasked by #6)* | **library** | `refine()` minted edge gids from an exscan over each rank's *local* edge count — gaps in the owned gid space, and the two sides of a boundary edge carried different gids — *pre-existing* | D3 |
| 8 | `conforming_migrate` / `conforming_operators` / `conforming_determinism` abort np≥2 | test | **missing re-halo**, in four more places (one fixture each, plus determinism case C's round loop) | D5, D6 |
| 9 | `conforming_determinism` idempotence fails np1 | test | a **stale assertion** (`!midpoints.empty()`) contradicting the contract D2 established and its own neighbour two lines below | D6 |

Three of the nine are in the library and **all three are in the red engine's handling
of a persistent hanging node**; #4, #5 and #7 were *pre-existing* and invisible before
conforming refinement existed, because `HangingNode2to1`'s own tests assert
non-conformity. Six of the nine are test-side, and **six separate occurrences (#2, #6,
#8×4) are the same defect** — the missing identity-`migrate()` + `haloExchange()`
between two back-to-back `refine()` calls. That is the single most repeated finding of
Task 8 and the first thing to check on any future `unordered_map::at` abort at np≥2.

**Which risk points fired.** Of the ten: **9 fired** and was the central defect (D2,
three coupled library defects where one was predicted). **10 fired** at np5 and became
a recorded design limit rather than a fix (D6, Decision 11). **4 fired, but only
indirectly and only once** — it is benign for the closure itself, and surfaced solely
as the reason a *geometric* blue tie-break cannot be evaluated locally. **1, 2, 3, 5,
6, 7 and 8 never fired**: `refinement_mode` and `refine_closure` passed in D0's first
sweep, and risk points 3, 4 and 5 all passed on the first run the moment
`conforming_migrate`'s fixture stopped throwing, so Task 5's sibling fixup and
`repairClosureCohesion()` were correct as written.

**Design sections rewritten.** *Persistent split edges* under the distributed
algorithm plus steps 0c/1/3/3b of the pseudocode (D2); the refine section's edge-gid
assignment, here and in `docs/design.md` (D3); the closure section's determinism
claim, here, in `docs/design.md` and in `Tessera_RefineClosure.hpp`'s header note,
which had conflated partition-independence with rank-count independence (D6); and the
closure section now carries D7's measured shape table so the "similarity classes are
bounded" claim cites numbers. Decisions **8–13** were added.

**Plus the measurements Tasks 2–7 deferred to here** — collected from the run output.
Results are in the right-hand column; the fuller tables are in the debug file's
per-sub-task *What landed* sections.

| Deferred from | Measurement | Result |
|---|---|---|
| Task 2 | Whether the "red child of a refined face has `\|S\| = 0`" assert in `closeFaces()` ever fired; and the `refine_closure` printout (per-`\|S\|` histogram, closure-face count, both blue-diagonal counts, conforming vs hanging-node Euler / bad-incidence / T-junction figures) | **The assert fired — and was the right kind of alarm.** D2's defect 4 made it legitimately possible for a fresh red child to have `\|S\| > 0` on an inherited half-edge, so it was *narrowed*, not removed: it now reads "no split edge may touch a vertex created this round" (new `firstNewVertexGid` parameter). `refine_closure` at HEAD: inverse `red=2639 marked=453 visible=3506 closure=1478`, `\|S\|` hist `[2028,382,202,27]`, blue diagonals **120 / 82** (both populated); `refineLocal` conforming `V=424 E=1266 F=844 Euler=2` with 361 closure faces, against hanging-node `Euler=-201 bad=609 T-junctions=203`. Identical in `[Serial]` and `[Default]`. |
| Task 3 | Actual message-volume increase on the extended Phase-2 rounds vs the estimated `1/ρ`; and the `refine_splitedges` printout (split-edge count, kept-side-discovered count per rank count, phase-2a total vs refining-only per round) | Phase-2a total vs refining-only: **×6.96 / ×5.03 / ×10.01 / ×16.38** over rounds 1–4, **rank-count independent**. Round 1 `refining=46 splitEdges=138`. Kept-side-discovered `keptOnlyDiscovered` = **0 / 6 / 13 / 24 / 27** at np1–5, i.e. non-vacuity *strengthens* with rank count — the cross-boundary discovery Task 3 exists to provide is genuinely exercised. Global face counts identical at every rank count (458 / 731 / 950 / 1124). The ratio is well above the `1/ρ` estimate and grows with round because ρ (refining fraction) falls as the mesh grows; the cost is bounded in absolute terms, not as a ratio. D1. |
| Task 4 | Closure-face fraction vs mask fraction, and the `\|S\|`-pattern histogram, per rank count | Measured at np1–5 (job `f3QUMwrCeAU7`). **Closure fraction is flat in the round and near-flat in the rank count**: round 1 **0.438** at every np; round 2 0.456 / 0.455 / 0.460 / 0.464 / 0.455; round 3 0.461 / 0.469 / 0.450 / 0.452 / 0.461. Round 1's mask is 46 of 320 faces = **0.144**, so closure fraction ≈ **3× mask fraction** — the O(perimeter) scaling, and the reason D7's 16-round run sees it *decline* to 0.0623 once the marked region stops growing as fast as the mesh. `\|S\|` histogram round 1 `[335,108,15,0]` at every np; round 3 e.g. `[1278,388,82,18]` at np1, `[1263,370,97,12]` at np5. Every round `euler=2` at every rank count; the control fails every round with growing defects (414 → 1031 → ~1470). |
| Task 5 | How many `dest` entries the sibling-cohesion fixup moved (both the adversarial `dest` and Zoltan2's); pre/post `loadBalance` max face count and max weighted load vs ideal | Adversarial `dest = gid % size` scatters **essentially every** sibling group: **248 fixups against 242 groups** at np2, then 306 / 318 / 318 at np3–5. Zoltan2 needed no help — it splits siblings **unaided**, 4 / 20 / 14 / 25–26 fixups at np2–5, confirming `Tessera_Zoltan2Balancer.hpp:180`. `loadBalance` cuts maxFaces 1232 → 251 at np5, and lands the **weighted** load within **0.6 %** of ideal at every rank count (447.0 vs 446.5 at np2; 184.0 vs 182.8 at np5) against a 2× bound — so the per-parent weighting is what makes the sibling fixup load-neutral: it moves faces without moving weight. D5. |
| Task 6 | Actual on-disk size delta for a conforming vs hanging-node file | **The +32 B/face prediction is exact.** Closure payload 39104 B / 1222 faces, 38208 / 1194, 39040 / 1220 — **32.000 B per visible face at three different rank counts**, and **19.4 %** of the file in all three. D5. |
| Task 7 | `conforming_operators`: closure-vertex count per rank count (non-vacuity), closed-fan and 1-ring mismatch counts split closure/interior, closure-vertex vs interior-vertex `applyStencil` max error, the same split for the accumulated-area error, and the vertex-area vs face-area totals. `conforming_determinism`: the per-component agreement breakdown (counts / `\|S\|` histogram / red layer / visible layer / closure vertices) against the `MPI_COMM_SELF` reference at each rank count, plus `blueDiagMismatch` and `parentMissing` — the numbers risk point 10 turns on. `conforming_quality`: per-round marked count, F, min angle, max radius ratio, closure count and fraction over 16 rounds, split by closure pattern `\|S\|` and against the red layer that feeds them, plus the worst-of-all-rounds figures against the bounds | `conforming_operators`: `closureVerts` **311 / 301 / 322 / 318 / 318** at np1–5 (an empty set is a hard failure by design), `fanBad=0` and `ringBad=0` split closure/interior, `stencilMaxErr` and `areaMaxErr` at closure vertices **identical to interior** (0 exactly on Serial, ~7e-18 on HIP), `vArea == fArea` to 9 digits. D6. `conforming_determinism` case A: counts / `\|S\|` hist / red layer / closureVerts / `parentMissing=0` **all agree at every rank count** — `V=676 E=2022 F=1348`, `\|S\|` hist `[1820,60,20,0]`, `closureVerts=74`, byte-identical np1–5; the **visible layer** agrees at np1–4 with `blueDiagMismatch=0` and at np5 differs with `blueDiagMismatch=4` (4 of 20 blue parents flip) — the number risk point 10 turns on, now Decision 11. D6. `conforming_quality`: **collected by D7, tabulated in Decision 12** — the headline is that the red layer feeding the closure is *dead flat* at `Q`=1.0278 / 54.397° across all 16 rounds, green is flat at 1.5672 from round 1, `\|S\|`=3 is never realised on this workload, and blue is the sole mover, saturating at 2.5254 by round 11 and flat through 16. |

---

## Decision record

*(append-only)*

- **2026-07-31 — Decision 1: option, not replacement.** Conforming refinement is an
  additional mesh option; the 2:1 hanging-node path is retained. It is additive —
  conforming refinement is a closure pass over the 2:1-balanced red mesh, so the
  hanging-node path is a prerequisite, not a competitor.
- **2026-07-31 — Decision 2: compile-time template parameter.** `RefinementMode` is a
  `Mesh` template parameter (appended last, source-compatible), not a runtime flag,
  so the closure bookkeeping fields exist only in `Conforming` mode.
- **2026-07-31 — Decision 3: `Conforming` becomes the default**, hanging-node
  opt-in. Sequenced last (Task 7) so the flip lands only after conforming is
  gate-green at ranks 1–5 over multiple adaptive rounds.
- **2026-07-31 — Decision 5: supersedes the sequencing clause of Decision 3.** Since
  Tasks 1–7 run no tests, Task 7's default flip lands *before* conforming has ever
  been executed — not after it is gate-green, as Decision 3 stated. Task 8 verifies
  both modes together, and reverting the default is Task 8's escape hatch. The choice
  of `Conforming` as the default (Decision 3 proper) is unchanged.
- **2026-08-04 — Decision 6: the closure's determinism claim is scoped to
  *partition* independence, not *rank-count* independence.** New gids come from an
  `MPI_Exscan` over ranks, so the midpoint-gid assignment — and therefore, after one
  round, every corner gid — is a function of the partition. The blue tie-break reads
  only globally-agreed values, which is what makes the closure independent of *which
  rank owns a face*; it does not make it independent of *how many ranks there are*.
  `conforming_determinism` compares by quantised position accordingly and measures
  the blue-diagonal agreement separately. Whether to keep the gid tie-break or
  replace it with a rank-count-stable rule is left to Task 8 (risk point 10).
- **2026-08-04 — Decision 7: Task 7 flagged the persistent-split-edge defect rather
  than fixing it.** The analysis in risk point 9 was produced while writing tests
  against the code, but the fix changes the un-close contract and cannot be validated
  without running the suite, which the handoff contract reserves for Task 8. The
  finding, its expected symptoms, and a proposed local fix are recorded in full so
  Task 8 does not have to rediscover them.
- **2026-08-04 — Decision 8: the persistent split-edge map is recovered locally
  from the closure children, not published by the Phase-2 coordinator.** Risk point
  9's two candidate fixes were (a) reconstruct the map from the closure bookkeeping
  already stored, (b) have the coordinator treat a red edge with a *single*
  incidence as split. (a) was chosen and (b) is strictly worse, as Task 7 suspected:
  the midpoint gid is known only to the coarse side, which is exactly the side that
  reconstructs it in (a), so (b) would need (a)'s machinery *plus* a message round.
  (a) also needs no new stored field, and it is the same one-rank-per-edge locality
  that makes the closure itself communication-free.
  The map's *third* consumer decided the shape: once it exists, phase 1 and phase 2
  can key the coordinator on the two half-edges of a bisected edge, which is what
  finally bounds the level jump across a hanging node. That was not part of either
  candidate — Task 7's analysis treated risk point 9 as purely a closure-input bug.
- **2026-08-04 — Decision 9: `refine()` reuses an existing midpoint when a coarse
  face refines across a hanging node.** Previously the 1→4 split minted a fresh
  midpoint for the whole edge `(a,b)` while the fine side already had one at the
  same position, cracking the mesh. This was latent in `HangingNode2to1` too and
  invisible there (that mode's meshes fail `checkConforming` by construction, so
  the crack changed a failing number into a differently-failing number). The fix is
  `Conforming`-only because it needs the persistent map, which only the closure
  bookkeeping can supply. Recorded under *Known limits* for the other mode.
- **2026-08-04 — Decision 10: refined edge gids are assigned by the edge
  coordinator, not by a local exscan.** `refine()` re-derives its edge list from the
  new visible faces, and used to gid it from an `MPI_Exscan` over each rank's
  **local** edge count — which includes the boundary edges a rank holds but does not
  own. Two consequences, both wrong: the owned gid space had gaps wherever a
  non-last rank held such a duplicate (so an *empty-mask* refine changed the global
  owned edge-gid set at np ≥ 3 — np2 escaped only because with two ranks every
  duplicate lands on the last one), and the two sides of a boundary edge carried
  **different gids for the same edge**, contradicting what `distribute()`, the mesh
  builder, `migrate()`'s gid-keyed edge maps and the HDF5 writer/reader all assume.
  Each `EdgeKey` reaches exactly one coordinator, so the coordinator can number its
  own keys densely from an exscan over its key count and return the gid in the reply
  of the round trip that already establishes ownership and level — **no extra
  message round**, and edges now match vertices and faces in having globally
  consistent gids. The numbering is hash-scattered rather than rank-contiguous;
  nothing depends on edge-gid locality. Found by Task 8 D3.
- **2026-08-04 — Decision 11: the blue tie-break stays gid-valued, and the visible
  layer is documented as rank-count dependent up to blue diagonals.**
  **RESOLVED 2026-08-06 by Decision 15** — the tie-break is now geometric and
  `blueDiagMismatch` is 0 at np1–5 on both backends. The original analysis is
  preserved below unchanged, because it is what made the second attempt work: it is
  the source of both the algebraic identity the new rule uses and the diagnosis
  ("the length must be attached to the edge, not computed from corner positions")
  that the reverted attempt could not reach. This settles
  risk point 10 and the open question Decision 6 deferred to Task 8. Measured: the
  visible layer agrees with the `MPI_COMM_SELF` reference at np1–4 and differs at
  np5 by exactly **4 of 20 blue parents**, with the red layer, the `|S|` histogram,
  the closure-vertex set, V/E/F and `parentMissing` all in exact agreement — i.e.
  precisely the signature risk point 10 predicted, and not a partition-local read.

  The geometric alternative risk point 10 proposed was **implemented in full and
  measured, then reverted**, because it cannot be done locally. Two things worth
  keeping:

  * The shorter-diagonal rule is simpler than it looks. With `q0 = (A+B)/2` and
    `q1 = (B+C)/2`, `dQ0C - dAQ1 = (3/4)(|C-B|^2 - |B-A|^2)`, so "take the shorter
    diagonal" is exactly "connect the midpoint of the **longer split edge** to its
    opposite corner" — the standard rule, needing only the two split edges' lengths.
  * It still cannot be evaluated locally. The closure runs on the **un-closed** red
    layer, whose corners come from closure children's `ClosureParentVerts`, and a
    child may name a vertex gid its rank does not hold — the already-documented
    risk point 4. A 1-deep halo does not help, because those are *parent* corners,
    not neighbours. Instrumented at np2, a single `closeFaces()` call asked for 12
    positions the rank did not have, including original icosphere vertices, so the
    geometric rule aborts (or silently picks a diagonal from the origin) at np ≥ 2.

  Since every gid-valued quantity is exscan-derived and positions are unreachable,
  **no purely local rule can be rank-count stable on the data the closure currently
  has.** Making a geometric rule work needs communication — the split edge's
  squared length carried in Phase 2's existing coordinator reply, contributed by a
  rank holding both endpoints (the refining side always does; the persistent split
  edges of Decision 8 have only a coarse incidence and would need a separate
  answer). That is deferred, not rejected: it is one extra field on an existing
  round trip, the same shape as Decision 10's fix.

  What `conforming_determinism` case A asserts was therefore **narrowed, but not to
  nothing**: the visible layer must differ *only* through blue diagonals. A
  visible-layer difference with no diagonal mismatch to explain it fails, and so
  does a diagonal mismatch that leaves the visible layer identical. Every quantity
  that is provably a function of the global mesh still has to agree exactly. Found
  and settled by Task 8 D6.
- **2026-08-05 — Decision 12: the shape bound is fixed in the round count, measured
  at 16 rounds, and `conforming_quality` is promoted into the gate.** This settles
  the last open question of Task 8. Task 7's bounds (min angle ≥ 20°, `Q` ≤ 4.0,
  closure fraction ≤ 0.50) were derived from the ideal-parent patterns; D4's 8-round
  run held them all but left `maxQ` *stepping upward at the last round measured*
  (1.5672 → 1.7759 → 2.2344), which 8 rounds cannot distinguish from unbounded
  growth. The round count is therefore now **16**, and the answer is that the bound
  is fixed:

  | quantity | measured over 16 rounds |
  |---|---|
  | `\|S\|=0` (red — the closure's **input**) | `Q` 1.0278, min angle 54.397° — identical in **every** round |
  | `\|S\|=1` (green) | `Q` 1.5672 from round 1, never moves |
  | `\|S\|=2` (blue) | 1.7759 (r6) → 2.2344 (r8) → 2.5254 (r11), then **flat through r16** |
  | `\|S\|=3` (red-closure) | never realised on this workload |
  | min angle (all faces) | 25.987° from round 2 on, flat |
  | closure fraction | peaks 0.1864 at r6, declines to 0.0623 at r16 |
  | amplification `Q(child)/Q(parent)` | 2.4906, flat from r11 |

  Blue is the only family that moves, and it moves in **discrete steps separated by
  flat rounds, then saturates** — rounds 11–16 are identical while F grows 4348 →
  24608 and the marked set 384 → 2388. That is a maximum over a finite set being
  progressively discovered (red similarity class × which edges are split × which
  diagonal), not a drift with depth. The derivation in Task 7 was right in
  *direction* — blue is the worst family, green sits below its ideal-parent value —
  and loose by ~1.6× on `Q`.

  Calibrated bounds, each the measured worst plus ~10%: **min angle ≥ 24.0°,
  `Q` ≤ 2.8, closure fraction ≤ 0.25, amplification ≤ 2.8**, plus two new
  assertions that make the *claim* checkable rather than merely printed:

  * **The red layer is bounded separately** (`Q` ≤ 1.10, min angle ≥ 50°). The red
    layer is the closure's input and is round-independent by construction; bounding
    it apart from the aggregate is what distinguishes "the closure degrades shape"
    from "the red engine does", which a single aggregate number cannot.
  * **Saturation.** The worst `Q` over the final 4 rounds may not exceed the worst
    over the preceding rounds. This is the assertion 8 rounds could not support and
    is the reason the round count is 16 rather than 12 — the last step is at round
    11.

  **Promoted to `regression` × {SERIAL, HIP} × ranks 1–5** with explicit user
  confirmation (`CLAUDE.md` requires it). The gate *definition* is unchanged, so no
  gate-definition edit was needed. Stability evidence: 320 printed round lines over
  np1–5 × {SERIAL, HIP} × {`[Serial]`, `[Default]`} reduce to exactly **16 distinct
  lines** — every measured quantity, including all the new per-pattern ones, is
  byte-identical at every rank count on both backends — at 11–16 s per instance.
- **2026-08-05 — Decision 14: `refine()` rebuilds the halo, and `rebuildHalo()`
  canonicalises the local layout by gid.** Follow-up 1 of
  [halo-rebuild-split-edge-design.md](halo-rebuild-split-edge-design.md). The general
  non-replicated 1-deep halo rebuild was welded inside `migrate()`; the five round
  letters hid a sixth (round G, gather referenced-but-non-held tuples), and only
  rounds S and A are migration proper. Rounds G and B/C/D moved to
  `Tessera_HaloRebuild.hpp` as two `detail::` helpers plus a public
  `rebuildHalo( mesh, halo )`, which `refine()` now calls in place of clearing the
  three plans. This retires the wart in Decision 13's second bullet — the one that
  caused **six of Task 8's nine defects** and threw from *inside the next*
  `refine()` rather than at the call the caller got wrong. Ten workaround sites
  across eight tests were deleted; `refine_rehalo` (regression, SERIAL+HIP, ranks
  1–5, confirmed with the user) is the acceptance test, and contains no `migrate()`
  and no `rebuildHalo()` call by design.

  **Two sub-decisions worth pinning:**

  * **Round D's gid-sort is kept, not generalised** (option A of the design's open
    question). `rebuildHalo()` reuses round D verbatim, so **local index order is a
    pure function of the owned gid set**: a mesh's layout is reproducible regardless
    of how it was reached, and the old "with vs without re-halo differ" ambiguity —
    which is why `refine_conforming` round 3 at np1 reported `F=2372` with the
    identity migrate and `F=2386` without — no longer exists, because there is one
    behaviour. Since `rebuildHalo()` is exactly the identity `migrate()` minus round
    A's all-to-all and round S's fixup, and round A with an identity `dest` is
    content-preserving, every test that already used the workaround saw **byte-
    identical** output. Measured: of the eight gid-masked/conforming tests probed at
    np1 and np5, seven were byte-identical and one number moved —
    `conforming_migrate`'s `loadBalance destFixups` 26 → 25, a Zoltan2 cut shifted by
    the now-canonical owned-face ordering, with `F`, `maxFaces`, `maxWeight`, `ideal`
    and `inv=0` all unchanged.
  * **No `bool rebuild` escape hatch.** A single-shot `refine()` on a mesh nobody
    will halo now pays for a halo it may not need, but every multi-round caller
    already paid strictly more via the identity `migrate()` (which adds round A's
    all-to-all on top), so real drivers get *faster*. Gate wall time: 814 s\*proc
    before the change, measured after at the same node count. The parameter is not
    added speculatively.

  `MigrateStats` needed no threading — both its fields are populated by round S,
  which stays in `migrate()`. The two shared rounds' profiling keys were renamed
  `migrate_round_{g,b,c,d}_*` → `halo_round_*` since either caller can drive them.
  Also deleted: `refineImpl()` step 3j, the deliberately incomplete best-effort
  owned-vertex 1-ring CSR, which round D now rebuilds completely.
- **2026-08-06 — Decision 15: the blue diagonal is chosen geometrically, and the
  length is carried on the edge rather than computed from corner positions.**
  Follow-up 2 of [halo-rebuild-split-edge-design.md](halo-rebuild-split-edge-design.md);
  this **resolves Decision 11** and the first bullet of Decision 13. The blue quad
  is split along its **shorter diagonal**, which by Decision 11's own identity
  `|q0−C|² − |A−q1|² = ¾(|C−B|² − |B−A|²)` is exactly *connect the midpoint of the
  longer split edge to its opposite corner* — the two split edges' lengths suffice
  and the diagonals' are never needed.

  **The one insight that separates this from the attempt D6 reverted** is not the
  rule; D6 had the rule. It is *where the length lives*. `closeFaces()` runs on the
  un-closed red layer, whose corners come from closure children's
  `ClosureParentVerts` and may name vertices the rank does not hold — D6 measured
  12 missing positions in a single `closeFaces()` call at np2, including original
  icosphere vertices — and a 1-deep halo cannot help because those are *parent*
  corners. So nothing is computed at the point of use: the length is **attached to
  the edge** and delivered with the midpoint gid.

  * `detail::KeyGid` gains a `double len2`, populated by the midpoint owner — which
    is by construction the owner of an incident *refining* face and therefore holds
    both endpoints of the edge it is bisecting — and delivered to every co-sharer on
    Phase 2's **existing** `allToAllV`. **No extra message round**, the same shape as
    Decision 10's edge-gid fix and as Decision 11 predicted.
  * A **persistently** split edge never appears in that round trip at all, because
    `forEachSubEdge()` advertises it only as its two halves. It gets its length
    **locally** instead (option 1 of the design's three), from the closure children's
    own corners, next to where `recoverSplitEdges()` reconstructs its midpoint. This
    is sound only because of Decision 14: after `rebuildHalo()` every vertex an owned
    face references is held *with its position*. Compose that with sibling
    co-residency and "a closed parent's corners are the union of its children's" and
    it needs no communication. **This is the load-bearing reason follow-up 1 came
    first**, and it held exactly as designed.
  * Every producer goes through one `edgeLen2Canonical()`, which orders the endpoints
    by gid before subtracting and computes in `double` regardless of the mesh's
    `scalar_type`. Two ranks that disagreed on a length by one bit would cut the same
    quad along different diagonals and crack the mesh. The four producers are the two
    `refine()` paths, `refineLocalConforming()`, and `checkClosureInverse()`.

  **Exact ties are counted, not resolved by geometry, and they are not zero.** On an
  exact `len2` equality the rule falls back to the old lower-midpoint-gid comparison
  and increments a new `ClosureStats::nBlueDiagTie`. The design anticipated ties on a
  symmetric icosphere and it was right: **4** blue parents tie on
  `conforming_determinism`'s workload, and 11 of 202 on `refine_closure`'s inverse
  case. The proposed escalation — deliver the midpoint *position* and compare
  quantised coordinates — was deliberately **not** implemented, because the measurement
  it was contingent on came back saying it would buy nothing: `blueDiagMismatch` is
  **0** even with the 4 ties present, i.e. the gid fallback happens to agree at every
  rank count tested. The residual rank-count dependence is therefore real in
  principle and unobserved in practice, and the counter keeps it an ongoing
  measurement rather than an assumption. `conforming_determinism` case A asserts
  `blueTie` itself matches the reference and tolerates at most `blueTie` mismatches,
  so a tie can never silently absorb a real divergence.

  **Measured (the acceptance).** `vis=ok`, `blueDiagMismatch=0`, `parentMissing=0`,
  `blueTie=4` at np1–5 SERIAL and np1–4 HIP, in both the `[Serial]` and `[Default]`
  exec spaces — where Decision 11 measured 4 of 20 blue parents flipping at np5.
  Both diagonals stay populated (a rule that collapsed to one would be inverted):
  `refine_closure`'s inverse case 101/101/11, `refine_conforming` 6/9/3 at round 1
  rising to 41/41/11 at round 3, and `q0C + q1A` equals the `|S|=2` count exactly at
  every round. SERIAL and HIP are byte-identical at equal rank count, which is the
  cross-backend float-determinism check. `F`, `euler`, the `|S|` histogram and the
  closure fraction are **unchanged** — the red layer is not a function of the
  diagonal, and that is the signature that says only the diagonal moved. Full sweep
  **193/193** (135 regression + 58 unit instances, both backends, SERIAL np1–5 and
  HIP np1–4).

  **The shape bound improved, and Decision 12's numbers move.** Re-measured at 16
  rounds: blue's worst radius ratio **2.5254 → 2.2344** and worst amplification
  **2.4906 → 2.2310**, with saturation now at round **8** instead of 11 and flat
  through 16. This is the predicted direction and is the second independent check
  that the comparison is not inverted — the rule takes the *shorter* diagonal, so a
  rule that got it backwards would have made `maxQ` *worse*. Red, green, min angle,
  the closure fraction and every `F` are identical to D7's. Decision 12's calibrated
  **bounds are deliberately left as they are** (`Q ≤ 2.8`, amplification `≤ 2.8`):
  they still hold with more margin than before, and re-tightening a gate bound onto a
  freshly measured number buys nothing but a future false failure. Only the
  *measured* numbers recorded alongside them are updated.

  **`checkClosureInverse` remains the only check that catches a wrong diagonal** —
  D6's regression reported `euler=2`, the correct `F`, the correct `|S|` histogram and
  both diagonals populated, and failed that check alone. The `refine_closure` blue
  unit cases were accordingly rewritten to drive **both** diagonals from geometry (the
  reference triangle has `|ab|²=1, |bc|²=2, |ca|²=1`, so which pair of edges is split
  selects the diagonal with no gid relabelling), plus two new sub-cases: `blue/relabel`
  swaps the midpoint gids and asserts the diagonal does **not** move — that is the
  direct regression test for the property this change buys — and `blue/tie` drives a
  synthetic exact tie and asserts both the gid fallback and `nBlueDiagTie == 1`.
  A missing `len2` for an edge that *is* split prints the edge and parent gid and
  `Kokkos::abort`s rather than defaulting, which is D6's other lesson: a lookup that
  returns a default on a miss converts a hard failure into a plausible mesh.
  `nBlueDiagLowFirst`/`LowSecond` were renamed `nBlueDiagQ0C`/`nBlueDiagQ1A`, since
  the old names describe a gid comparison that no longer happens.
- **2026-08-05 — Decision 13: `Conforming` stays the `Mesh` default.** Decision 3
  chose it, Decision 5 accepted that Task 7 would flip it *before* anything had ever
  been executed, and Task 8 held the escape hatch: revert to `HangingNode2to1` if
  conforming could not be made green. **The hatch is not used.** At HEAD the ship
  gate is 140/140 and the diagnostic tier 62/62 — 202 instances over 28
  registrations, zero failures, zero tests relabelled or excluded, on both backends
  at ranks 1–5. Conforming is green at *more* rank counts and with *stronger*
  assertions than the hanging-node path had before this work, so the precondition for
  reverting never arose and the safer default is also the verified one.

  Two limits are accepted with the default rather than blocking it, both recorded in
  README *Known Issues* because they are user-visible:

  * **The visible layer is rank-count dependent up to blue closure diagonals**
    (Decision 11). Practical consequence: do not compare a conforming mesh's visible
    face set bitwise across rank counts — compare the red layer, or compare by
    position. Everything provably a function of the global mesh is invariant and
    asserted as such.
    **RESOLVED 2026-08-06 by Decision 15** — the blue tie-break is geometric, the
    visible layer agrees at np1–5 on both backends, and the README entry is gone.
    **Neither accepted limit of the `Conforming` default survives**; the default is
    now carried by a mesh with no recorded user-visible limit at all.
  * **A distributed mesh must be re-haloed between two `refine()` calls.** This is
    pre-existing and mode-independent, but Task 8 made it sharper and much more
    prominent: the consequence is a **throw from inside the next `refine()`**, not
    merely an inert `haloExchange()`. It accounted for **six** of Task 8's nine
    defects. It is a library-contract wart, not a conforming-mode defect, and is the
    strongest candidate for the next round of work — see the Milestone-2 note.
    **RESOLVED 2026-08-05 by Decision 14** — `refine()` rebuilds the halo itself.
    The README entry is gone; only the blue-diagonal limit above remains.
- **2026-07-31 — Decision 4: transient red–green–blue closure**, not
  newest-vertex bisection and not red-only propagation. Red-only propagation
  degenerates to uniform refinement; bisection replaces the red engine wholesale
  (recorded as a Milestone-2 alternative).

## Progress log

*(append-only)*

- 2026-07-31 — Design scoped; Decisions 1–4 recorded. No code yet. Next: Task 1.
- 2026-07-31 — Task 7 expanded: it now owns a dedicated conforming test suite
  (`conforming_operators`, `conforming_determinism`, `conforming_quality`, plus a
  `staleslice_guard` case) alongside the mode-parity registration and the default
  flip, split across three commit checkpoints.
- 2026-07-31 — Added Task 8 (run the full suite, fix everything). Tasks 1–7 are now
  explicitly **no-test** tasks that end at a commit + push; their Acceptance sections
  specify tests to write, not to run. Task 8 is the single verification point.
- 2026-07-31 — **Task 1 landed.** `RefinementMode` enum + `Mesh`'s 8th template
  parameter, conditional `ClosureParent`/`ClosureParentVerts` face members appended
  after the user pack, index/count helpers, `if constexpr` dispatch with an
  aborting `Conforming` stub, `test_refinement_mode`, docs. **Not compiled:** the
  build was skipped at the user's request this session, so Task 2 should expect to
  absorb any Task-1 build break — build before adding to it.
  `clang-format` (v21, `/usr/bin/clang-format` on Tuolumne) is clean on every file
  Task 1 touched, but note that several *untouched* files (e.g.
  `src/Tessera_Geometry.hpp`) already violate v21, so a wholesale `format-check`
  under that version is not clean and predates this work.
  Also, running a bare `cmake .` in `build-tuolumne` without the spack env active
  deleted its `CMakeCache.txt`; that directory needs
  `bash ../run_cmake_toulumne.sh` (with `spack env activate
  ~/spack_envs/tuolumne_trilinos/`) re-run before the next `make`.
- 2026-07-31 — **Task 2 landed.** `src/Tessera_RefineClosure.hpp` (`RedFace`,
  `VisibleFace`, `ClosureStats`, `closeFaces`, `unclose`, `translateMask`,
  `countClosureChildren`, `faceSplitEdges`, `readVisibleFaces`, `writeClosureFace`,
  `initClosureFaceMembers`), `detail::refineLocalConforming()` replacing
  `refineLocal()`'s `Conforming` abort stub, `initClosureFaceMembers()` wired into
  `buildTriangleMesh()`, and `tests/test_refine_closure.cpp` registered `unit`
  SERIAL + HIP np1. **Task 1's build break risk is discharged:** the Task-1 code
  compiled clean with no changes needed. Note for later sessions — the Bash tool
  runs a *non-login* shell, where no `PrgEnv-*` module is loaded and the Cray `CC`
  wrapper fails at configure with "A PrgEnv-* modulefile must be loaded"; run the
  build under `bash -lc` (and export
  `SPACK_USER_CONFIG_PATH=$HOME/.spack/tuolumne`). `clang-format` on this checkout
  resolves to `/opt/rocm-6.4.2/llvm/bin/clang-format` (v19), under which every file
  Task 2 touched is clean; pre-existing violations in untouched files (e.g.
  `src/Tessera_Geometry.hpp`) remain and predate this work.
- 2026-07-31 — **Task 3 landed.** `detail::OwnerMsg` gained a `refining` flag and
  Phase 2a now advertises every owned face's edges; the coordinator aggregates
  `{participants, refOwner, anyRefining}`, drops non-split edges, and replies to all
  participants, so 2c's gid delivery reaches kept co-sharers.
  `RefineResult::midpoints` is now the complete split-edge map (contract widened and
  documented), plus new `phase2Adverts` / `phase2AdvertsRefining` counters.
  `tests/test_refine_splitedges.cpp` registered `regression` SERIAL + HIP ranks 1–5.
  `docs/design.md`'s Phase-2 paragraph rewritten, and its stale "the `Conforming`
  branch is a stub that aborts" line corrected — that is now true of the
  *distributed* `refine()` only, since Task 2 implemented `refineLocal()`.
  Midpoint **assignment** is provably unchanged (ownership still reads refining
  participants only, and the split-edge set is unchanged), so no existing test's
  expected gids move. Build note: `spack env activate` needs
  `. /usr/WS2/stewartj/spack/share/spack/setup-env.sh` sourced first even under
  `bash -lc`. `clang-format` is clean on both touched files under **both** the v21
  (`/usr/bin`) and v19 (`/opt/rocm-6.4.2/llvm/bin`) binaries on this machine.
- 2026-07-31 — **Task 4 landed — the milestone.** Distributed conforming
  refinement works. `detail::refineHangingNode()` became `detail::refineImpl()`,
  shared by both modes: a red-layer snapshot (`fV`/`fG`/`fL`/`fSrc`, `nRedF`) plus
  a red-indexed `mark` replaced the owned-face snapshot, and four
  `if constexpr ( kConforming )` blocks add un-close, mask translation, close, and
  the widened face-gid exscan. `RefineResult::closure` publishes the `ClosureStats`.
  Two design corrections, both folded into the sections above: the gid block must
  sit above the max **visible** face gid (monotone across rounds), and
  `checkNoInteriorVertex` cannot be coordinator-routed because a closure child may
  name a vertex whose position no rank but its owner holds — it replicates to rank
  0 instead. One hazard found outside the refine path: **`distribute()` builds a
  fresh face AoSoA and so left the closure members zero-filled**; it now calls
  `initClosureFaceMembers()`. `tests/test_refine_conforming.cpp` registered
  `regression` SERIAL + HIP ranks 1–5, with a `HangingNode2to1` control mesh run on
  the same mask as a hard non-vacuity guard. Whole suite (including every HIP
  target) compiles clean; `clang-format` clean on all five touched files under both
  the v21 (`/usr/bin`) and v19 (`/opt/rocm-6.4.2/llvm/bin`) binaries.
- 2026-07-31 — **Task 5 landed.** `migrate()` gained round S, a `Conforming`-only
  local sibling-cohesion fixup on `dest` (every closure child follows the
  lowest-gid sibling), and now returns `MigrateStats { siblingFixups,
  siblingGroups }` instead of `void` — no call site noticed, since all of them
  discard it. A violating `dest` is **repaired, not rejected**: Zoltan2 partitions
  by centroid and so produces one on every call. `ownedFaceWeights()` weights a
  closure child `1/nsiblings` so a red parent is one unit of work; the
  per-owned-face contract and `computeLoadBalance()`'s signature are unchanged.
  New level-2 timer `migrate_sibling_cohesion`;
  `TesseraTest::checkSiblingCoresidency()` / `closureSiblingGroups()`;
  `tests/test_conforming_migrate.cpp` registered `regression` SERIAL + HIP ranks
  1–5, with an adversarial `dest = faceGid % size` case (fixup count asserted
  positive at ranks ≥ 2 as the non-vacuity guard) and a `loadBalance` case that
  checks weighted load, plus a ghost corrupt-and-resync over all three halo plans
  to exercise the wider conforming face tuple. The halo rebuild itself needed no
  change. Whole suite compiles clean (SERIAL + HIP); `clang-format` clean on every
  touched file under both the v21 (`/usr/bin`) and v19
  (`/opt/rocm-6.4.2/llvm/bin`) binaries. `README.md` Known Issues and
  `docs/design.md`'s *Load balancing* section updated (new *Redistributing a
  conforming mesh* subsection).
- 2026-07-31 — **Task 6 landed — the feature is complete.** HDF5 **format version
  2**: `/faces/closure_parent` + `/faces/closure_parent_verts` datasets (persistent
  gids, no dense translation), a `refinement_mode` root attribute written in both
  modes and hard-validated on read, and both face user-field loops re-bounded by
  `numFaceUserFields<>()` via the new `detail::forEachUserFieldN<UserBegin, N>` —
  the tuple-suffix formula over-counts by two in `Conforming` mode. The reader also
  calls `initClosureFaceMembers()` on its fresh face block. **The one design gap:
  `readMesh()` breaks the co-residency premise `migrate()`'s round S is built on** —
  its fresh dense-index block partition splits sibling groups, which a local pass
  cannot detect — so `repairClosureCohesion( mesh, dest )` was added next to round S
  as its collective counterpart (one `allToAllV`, keyed `parent % size`, lowest-gid
  sibling wins, idempotent with round S). `markByQuality` needed **no** library
  change: a criterion's mask is over visible faces and step 0b translates it.
  `examples/02_mesh_pipeline` gained `--refine-mode {hanging,conforming}` (two
  instantiations, mode tag in the frame stem); `tests/test_io.cpp` gained a
  conforming red-layer round-trip case and `tests/test_markquality_conforming.cpp`
  is new (`regression`, SERIAL + HIP, ranks 1–5). README (Known Issues rewritten,
  compressed-closure-encoding added to Future Optimizations) and `docs/design.md`
  (both modes, conforming-marking subsection, on-disk format 2) updated. Whole
  suite compiles clean (SERIAL + HIP); `clang-format` clean on every touched file.
- 2026-07-31 — Audited every task's Acceptance / Report-back for consistency with the
  no-test rule: dropped the "gate green" claims from Tasks 2–7, moved all
  runtime-measurement report-back items into a deferred-measurements table in Task 8,
  required the tests to *print* those measurements, made `conforming_quality` land as
  `unit` with provisional bounds for Task 8 to calibrate, and added the default-flip
  escape hatch (Decision 5).
- 2026-08-04 — **Task 7 landed — the suite and the default flip.** Three commits.
  *(A)* Every test that calls `refine()` now names `RefinementMode::HangingNode2to1`
  explicitly, with a note pointing at its conforming counterpart; `refine_splitedges`
  needed pinning although the plan's parity table omitted it, because it isolates the
  shared Phase 2 and its reference construction assumes visible == red. Nothing
  resisted, and no test needed a behavioral edit. *(B)* Three new tests:
  `conforming_operators` (`regression`, SERIAL + HIP, ranks 1–5 — the payoff test:
  closed 1-ring fan, stencil topology against a face-derived reference, `applyStencil`
  and `reduceVertexFromFaces` errors asserted separately at closure vs interior
  vertices, with an empty closure-vertex set a hard failure), `conforming_determinism`
  (`regression`, ranks 1–5 — rank-count independence against an `MPI_COMM_SELF`
  reference compared by quantised position, closure idempotence, cross-mode
  equivalence), and `conforming_quality` (**`unit`**, ranks 1–5 — min angle, radius
  ratio and closure fraction over 8 rounds against provisional 20° / 4.0 / 0.50
  bounds), plus a conforming case in `staleslice_guard` covering a face slice, a
  `MeshGeometry` and a `VertexStencil` held across a conforming `refine()`. *(C)*
  `Mesh`'s `Mode` defaults to `Conforming`; **the flip broke nothing at compile
  time**; `README.md`, `docs/design.md`, `src/Tessera_RefinementMode.hpp` and
  `CLAUDE.md` synced. Whole suite compiles clean (SERIAL + HIP); `clang-format` clean
  on every touched file under both the v21 (`/usr/bin`) and v19
  (`/opt/rocm-6.4.2/llvm/bin`) binaries. **Two findings recorded for Task 8, both
  from reading the code while writing tests against it, neither fixed here:** risk
  point 9 — the split-edge map the closure consumes is this-round-only while the
  level jumps it must close are persistent, so conformity should be lost from the
  second round on (with `conforming_determinism`'s idempotence case as the minimal
  reproducer, and a local no-communication fix proposed); and risk point 10 — nothing
  gid-keyed is rank-count independent, because gids come from an `MPI_Exscan`, which
  narrows the closure's determinism claim (Decisions 6 and 7).
- 2026-08-05 — **Task 8 complete; conforming refinement is done.** Sub-tasks D0–D8
  (see [conforming-refinement-debug.md](conforming-refinement-debug.md)) took the
  feature from *never executed* to green: at HEAD the ship gate is **140/140** and
  the diagnostic tier **62/62** — 202 instances over 28 registrations, both backends,
  ranks 1–5, zero failures. `format-check` clean on all 34 touched files under
  clang-format v21 and v19. **Nine defects**, of which three were in the library and
  all three in the red engine's handling of a persistent hanging node: the
  this-round-only split-edge map that risk point 9 predicted (Decision 8/9), plus two
  *pre-existing* ones it unmasked — a coincident duplicate midpoint across a hanging
  node, and 2:1 propagation being blind across one — and, separately, `refine()`
  minting edge gids from a local exscan so the two sides of a boundary edge disagreed
  (Decision 10). The other six were test-side and **six occurrences were one defect**:
  a missing re-halo between back-to-back `refine()` calls. Of the ten risk points, 9
  and 10 fired, 4 fired only indirectly, and the other seven never did — Task 5's
  sibling fixup and `repairClosureCohesion()` passed on the first run that reached
  them. Two limits are accepted rather than fixed: the visible layer is rank-count
  dependent up to blue diagonals, because no purely local rule can be rank-count
  stable on the data the closure has (Decision 11, geometric tie-break deferred with
  a design for it), and defects 2–3 above remain in `HangingNode2to1` by construction.
  No test was relabelled or weakened; `conforming_quality` was **promoted** into the
  gate after its bound was measured to saturate (Decision 12), and three assertions
  were strengthened. `Conforming` **stays the default** (Decision 13). Decisions 8–13
  and the deferred-measurements table were added; the design's persistent-split-edge,
  edge-gid and determinism sections were rewritten here and in `docs/design.md`.
