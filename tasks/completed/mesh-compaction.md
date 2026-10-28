# Mesh compaction (tombstone removal and renumbering)

**Status:** DONE (2026-08-15). Green on SERIAL **and HIP** at ranks 1–5, and the
full gate is 240/240. See the *Met.* paragraph at the end of the *Progress log*.
Prerequisite of [edge-collapse.md](edge-collapse.md).
**Read the "Editing families" section of [edge-split.md](edge-split.md) first.**

**Verified against `579b543`** (branch `conforming-refinement`) — the code this
task cites was re-read at that commit. (It was originally written against
`08dd346`; the citations below were corrected at `579b543`.)

## Problem

Tessera has no way to remove entities. Nothing orphans a vertex today —
`refine()` is split-only, `Level` never falls, and every entity that exists stays
— so the gap has been invisible. The moment a coarsening operation exists
([edge-collapse.md](edge-collapse.md)), it produces dead entities and there is
nowhere for them to go: the AoSoAs have no delete, the owned-first ordering has no
way to close a hole, the vertex→face and vertex→edge CSRs would index removed
slots, the `edgeKeys`/`faceKeys` side tables would carry stale keys, and the halo
plans would name ghost slots that no longer exist.

Consequential rather than independent — but it must exist **before** collapse, not
alongside it, or collapse ends up growing its own private half-compaction.

**A second, less obvious hazard.** Several code paths index by **global id** into
a dense host array, and every one of them is sized by a *max gid*, not by the
entity count:

- `detail::buildKindPlan` takes `const std::vector<LocalIndex>& gid2local` and
  indexes it by raw gid (`src/Tessera_Distribute.hpp:83`);
- `rebuildHalo()`'s round D **builds** those vectors, sized to the local max gid
  (`detail::make_g2l`, `src/Tessera_HaloRebuild.hpp:519–533`) — so **`compact()`
  is itself on the leak path**, not merely a fixer of it;
- `distribute()`'s `build_order` builds `v2l`/`e2l`/`f2l` the same way
  (`src/Tessera_Distribute.hpp:294–316`).

Gids are assigned by `MPI_Exscan` onto a monotonically rising global count, so a
long run of split-and-collapse rounds grows the gid space **without bound** even
while the mesh stays the same size. The dense arrays then grow without bound too.
`migrate()` already avoids this (it uses `std::map<GlobalId, ...>`), so the hazard
is confined, but it is real and this task is where it gets addressed. Both of the
newly-named paths belong in the `docs/design.md` *Compaction* subsection the exit
criterion requires.

## Approach

New header `src/Tessera_Compact.hpp`.

### Tombstones

Uniform rule for all three entity kinds: **`Gid == invalid_gid` marks a dead
entity.** Uniform beats per-kind cleverness here — `VertexField::Flags` exists and
edges/faces have no equivalent, so using `Flags` for vertices only would mean two
mechanisms.

```cpp
//! Mark a local entity dead. Tombstoning does NOT repair connectivity: a caller
//! that kills a face must also kill the entities that become orphaned, and must
//! not leave a live entity referencing a dead gid. compact() verifies this and
//! throws on a dangling reference rather than producing a corrupt mesh.
template <class MeshT> void tombstoneVertex( MeshT& mesh, LocalIndex v );
template <class MeshT> void tombstoneEdge  ( MeshT& mesh, LocalIndex e );
template <class MeshT> void tombstoneFace  ( MeshT& mesh, LocalIndex f );
```

### `compact()`

```cpp
struct CompactStats
{
    long long verticesRemoved = 0, edgesRemoved = 0, facesRemoved = 0;
    long long gidSpaceBefore = 0, gidSpaceAfter = 0; //!< ALWAYS filled, by both
                                                     //!< calls (check 11 reads
                                                     //!< compact()'s)
};

//! Remove every tombstoned entity, restore owned-first ordering, rebuild the two
//! CSRs, the key side tables, and the three halo plans. GIDS ARE PRESERVED —
//! a surviving entity keeps the gid it had, so every cross-rank reference held by
//! another rank stays valid and no communication is needed to agree on renaming.
//! Collective (the halo rebuild is). Bumps generation.
template <class MeshT>
CompactStats compact( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo );

//! compact(), then renumber gids contiguously from 0 per kind via MPI_Exscan over
//! owned counts, so the gid space stops growing across many edit rounds. Strictly
//! more expensive (every rank must learn the new gid of every entity it ghosts,
//! and every connectivity field must be rewritten), so it is a SEPARATE call the
//! caller invokes periodically rather than something compact() does every time.
template <class MeshT>
CompactStats compactAndRenumberGids( MeshT& mesh,
                                     MeshHalo<typename MeshT::memory_space>& halo );
```

**Decision — gid preservation is the default, renumbering is opt-in and periodic.**
Preserving gids makes `compact()` a purely local reordering plus one halo rebuild;
renumbering makes it a global relabelling that invalidates every gid a peer holds.
A remesher calling compact every timestep wants the cheap one; the gid-space
growth is a slow leak that a call every few hundred steps fixes. Both are needed;
conflating them would make the common case pay for the rare one.

### Implementation

**`compact()` reuses `rebuildHalo()`; it does not build a private permutation.**
Steps 2–6 of the original plan *are* `detail::finishHaloAndAssemble()`'s round D
(`src/Tessera_HaloRebuild.hpp:474–690`): it orders owned-first and gid-ascending,
whole-tuple copies every AoSoA, rebuilds both CSRs and both key tables, and
replaces the three plans. Critically it derives the held vertex and edge set
**only from owned faces** (`src/Tessera_HaloRebuild.hpp:783–797`), so vertices and
edges that no surviving face references are dropped with no explicit compaction of
those two AoSoAs. So `compact()` has three steps of its own:

1. **Verify before mutating.** Throw on a dangling reference — a live face
   naming an entity its owner marked dead — naming the offending face and the
   dead gid. A tombstone set that does not close is the single most likely caller
   bug and it must not produce a corrupt mesh silently. `tombstoneVertex`/
   `tombstoneEdge` feed **this check**, not a removal pass; an orphan the caller
   forgot to tombstone disappears anyway, and check 6 must still throw.
2. **Drop the tombstoned faces** from the face AoSoA, owned live faces only (the
   survivors are already gid-ascending, so filtering in place preserves the
   ordering the rebuild expects on entry), and `setOwnedCounts` with the live
   owned face count. Ghost faces go wholesale: the ghost set genuinely changes —
   a ghost whose owner deleted it must disappear — so this cannot be a plan patch.
3. **`rebuildHalo( mesh, halo, effectiveHaloDepth( halo ) )`**
   (`effectiveHaloDepth` is `src/Tessera_Distribute.hpp:56`), exactly as
   `src/Tessera_EdgeSplit.hpp:866` does. [halo-depth.md](halo-depth.md) is DONE,
   so this is unconditional. Gid preservation, the canonical ordering, and the
   generation bump all fall out of round D unchanged; connectivity fields hold
   gids and gids are preserved, so **no connectivity rewrite is needed** — that is
   the payoff of the preservation decision.

Then:

4. **`compactAndRenumberGids`** adds, after `compact()`: a new gid per owned
   entity; write them onto the owned entities; `haloExchange()` all three AoSoAs
   so every ghost receives its owner's new gid **inside the tuple** (there is no
   generic per-entity value exchange — `haloExchange()` is whole-AoSoA,
   whole-tuple, `src/Tessera_HaloExchange.hpp:134`); build the local old→new map
   from the gids saved beforehand; rewrite every connectivity field and both key
   side tables; rebuild the halo **again** because the plans were keyed on old
   gids. Two halo rebuilds is the honest cost; do not try to fuse them.
5. **All-dead-on-one-rank** must work: that rank ends with zero owned entities, its
   peers drop it from their plans, and the collectives still complete. This is a
   real load-balance state, not a pathological one.

**Both calls claim the Remesh editing family** via
`requireEditFamily( mesh, EditFamily::Remesh, "compact" )`, matching what
`src/Tessera_EditFamily.hpp:38,67,76` already documents. Consequence for the
README: compacting a `refine()`d mesh throws, and compacting a freshly built mesh
tags it Remesh so a later `refine()` is refused.

### Non-goals

- Deciding *what* to tombstone. That is collapse's job, or the caller's.
- Shrinking AoSoA capacity. Cabana resize semantics are Cabana's; compaction
  changes the logical size and lets capacity be whatever Cabana keeps.
- Repairing a non-closed tombstone set. It throws (step 1).

### Tests

New `tests/test_compact.cpp`, registered at **TIER `regression`**, backends
**SERIAL and HIP**, ranks **`${TESSERA_TEST_MPI_RANKS}` (1–5)**. Gate promotion is
**pre-authorized for this task**.

The test needs a way to produce a valid tombstone set without collapse existing
yet. Use a **closed sub-mesh removal**: pick a set of faces by a rank-count-
invariant rule, then tombstone exactly those faces plus the edges and vertices
that no surviving face references. That set closes by construction, and the
resulting mesh is a surface **with boundary** — which is a legitimate exercise of
the machinery even though it is not what collapse will produce.

1. **No-op.** Nothing tombstoned. V/E/F, `topologyChecksum`, and every gid
   unchanged; `owned1RingLocal` still consistent; `CompactStats` all zero.
2. **Idempotence.** `compact()` twice: the second call is a no-op by check 1's
   criteria (`topologyChecksum` bitwise equal).
3. **Closed sub-mesh removal.** Remove the faces of one base-icosahedron patch
   (a fixed gid range at subdivision 2). Assert: local and owned counts drop by
   exactly the tombstoned counts on each rank; `CompactStats` matches;
   **every surviving entity's gid is unchanged** (compare against a pre-edit
   gid→position snapshot); owned-first ordering holds with gids ascending in each
   block; `owned1RingLocal` passes; and the surviving faces' corner-position
   triples are exactly the pre-edit set minus the removed ones.
4. **Halo correctness after removal.** `haloExchange()` leaves every ghost vertex
   position equal to its owner's; `checkOwnershipPartition` passes; the removed
   entities appear in no plan.
5. **Euler on the result.** For a disc-like removal, `V - E + F == 1` per boundary
   component (Euler characteristic of a sphere minus one disc). Compute the
   expected value from the removed patch's boundary rather than hardcoding — this
   is a real check that connectivity survived, and it is why removing a *closed*
   sub-mesh is a better test fixture than removing scattered faces.
6. **Dangling reference throws.** Tombstone one vertex and nothing else. Step 1
   must throw naming a live face and the dead vertex gid.
7. **Everything dead on the last rank.** At ranks ≥ 2, tombstone every entity on
   rank `size-1`. That rank ends with zero owned entities; the call completes on
   every rank; `globalOwnedEuler` reflects the reduced mesh; `haloExchange`
   succeeds; the empty rank appears as no peer's send or recv target.
8. **Generation guard.** A `VertexStencil` (or a slice) taken before `compact()`
   aborts when used after, matching `test_staleslice_guard.cpp`.
9. **`compactAndRenumberGids`.** After check 3's removal: gids are `[0, N)`
   contiguous per kind globally (verify by gathering owned gid sets);
   `gidSpaceAfter < gidSpaceBefore`; every connectivity field resolves; the
   surviving faces' corner-**position** triple multiset is **unchanged** from
   before renumbering (positions are the gid-independent identity of the mesh);
   `haloExchange` correct; `owned1RingLocal` passes.
10. **Renumbering is deterministic.** `compactAndRenumberGids` on the same input at
    ranks 1–5 gives the same *global* gid→position map. Assert as a sorted
    (gid, position) list comparison.
11. **Repeated rounds do not leak gid space.** Ten rounds of
    "tombstone a small patch, `compact()`" and assert `gidSpaceBefore` is
    monotone; then one `compactAndRenumberGids` and assert it returns to the live
    count. Pins the hazard this task exists to close.

## Exit criterion

- `test_compact` green at **SERIAL and HIP, ranks 1–5**, and the full gate still
  green with nothing relabelled.
- Check 3 confirms **gid preservation** and check 9 confirms **renumbering**; both
  are required, since the two calls have opposite contracts.
- Check 7 passes: a rank with zero owned entities is a supported state.
- Check 11 demonstrates the gid-space leak and its fix, with the measured numbers
  recorded in the progress log.
- README API section documents `tombstoneX`, `compact`, `compactAndRenumberGids`,
  the `Gid == invalid_gid` tombstone convention, and the gid-preservation decision
  with its rationale (cheap per-step compaction vs. periodic renumbering).
- `docs/design.md` gains a *Compaction* subsection, including the gid-space growth
  hazard and which code paths index densely by gid.

## Where this sits

**No hard prerequisite** — `rebuildHalo()` already exists in tree
(`Tessera_HaloRebuild.hpp`, `25980f2`), so step 6 is available today, and
[halo-depth.md](halo-depth.md) is only needed if a caller wants depth > 1.
**Required by
[edge-collapse.md](edge-collapse.md)**, which calls `compact()` internally. See the
ordering diagram in [halo-depth.md](halo-depth.md).

## Progress log

- 2026-08-07 — Task written, then re-checked against `08dd346` after pulling
  `../tessera`. Nothing implemented.

- 2026-08-14 — **IN PROGRESS, paused mid-verification.** Implementation and test
  are written; SERIAL is green at ranks 1–5; HIP is built but was never run
  against the fixed test. Nothing is committed.

  **What exists in the tree** (all uncommitted):

  | File | State |
  | --- | --- |
  | `src/Tessera_Compact.hpp` | **New.** `CompactStats`, `tombstoneVertex/Edge/Face`, `compact`, `compactAndRenumberGids`, and `detail::{gidSpace, ownedCounts, verifyTombstoneClosure, dropDeadOwnedFaces, orderPreservingRenumber, compactImpl}`. |
  | `src/Tessera.hpp` | `#include "Tessera_Compact.hpp"` added. |
  | `src/Tessera_Profiling.hpp` | `TIMER_COMPACT`, `TIMER_COMPACT_RENUMBER` added and used. |
  | `tests/test_compact.cpp` | **New**, all eleven checks. |
  | `tests/CMakeLists.txt` | `compact` registered SERIAL + HIP, TIER `regression`, `RANKS ${TESSERA_TEST_MPI_RANKS}`. |
  | `tasks/completed/halo-depth.md` | `rebuildHalo()` citation corrected 642 → 733. |
  | `tessera_compact.flux` | Scratch batch runner, **not for committing**. |

  **Decisions, as they were settled and implemented.**

  * **`compact()` is a halo rebuild, not a private permutation** — the shape in
    the *Implementation* section above. Round D does the reordering, the CSRs,
    the key tables and the plans; `compact()` only verifies, drops dead owned
    faces, and calls `rebuildHalo()`.
  * **The closure check had to become GLOBAL, and this was the one real design
    change.** The document's step 1 describes a local sweep. A local sweep is
    wrong in *both* directions. A rank that tombstones a **ghost** while keeping
    a live face on it is fine — round G re-fetches the entity from its owner —
    so a local sweep would reject valid input; and a rank that tombstones a
    vertex it **owns** whose last local face also died, while a neighbour still
    has a live face on that vertex, is a fatal caller bug that no local sweep on
    either rank can see. Left unchecked that second case does not corrupt the
    mesh, it **crashes**: the vertex is gone from the owner's advertisement, so
    round G's `coord.at()` throws `std::out_of_range` on one rank while the
    others sit in the next collective. So deadness is **owner-scoped** and the
    check is a coordinator round: every rank advertises the live entities it owns
    and the references of its live owned faces to `gid % size`; the coordinator
    reports every reference with no live claim back to the referencing rank. The
    throw is made collective by an `MPI_Allreduce` on the report count — a throw
    on one rank alone would deadlock the rest.
  * **Renumbering is an ORDER STATISTIC, not an `MPI_Exscan` block.** The
    document's step 7 says exscan-over-owned-counts. That **fails check 10**: an
    exscan hands rank *r* a contiguous block, so the old→new map depends on who
    owned what and two runs at different rank counts disagree. Implemented
    instead as *new gid = the number of live gids of that kind strictly below the
    old one*, computed without gathering (bucket by gid **value**, one bucket per
    rank, `MPI_Exscan` over the per-bucket counts, sorted position within the
    bucket). Same contiguous `[0, N)` result, and identical at every rank count.
  * **`CompactStats` is entirely GLOBAL** — the removal counts are owned-count
    differences summed across ranks. A local count is not a statement about the
    mesh: a rank's local count also moves when the ghost set changes.
  * **`gidSpaceBefore`/`gidSpaceAfter` are always filled by both calls**
    (one `MPI_Allreduce(MAX)` over the three kinds, each contributing
    max-live-owned-gid + 1). Equal for `compact()` **unless** the removal
    contained the globally maximal gid of some kind, in which case the space
    shrinks by the vacated tail — the doc now says so, and check 3 asserts the
    exact expected value computed from the reference rather than assuming
    equality.
  * **`EdgeField::Faces` is neither checked nor repaired by `compact()`.** It is
    best-effort by design (`src/Tessera_FaceAdjacency.hpp:215–223`: `migrate()`
    carries it verbatim, so it can already name a face no rank holds), and a
    surviving boundary edge whose second incidence was removed is exactly what a
    legitimate compaction produces. `compactAndRenumberGids()` **does** map it,
    to `invalid_gid` where the face is not held locally: a gid left in the old
    space after a renumbering would silently **alias** a different live face,
    which is worse than absent.
  * **Test fixture** is `buildIcosphere` + `distribute()` with a replicated
    `MPI_COMM_SELF` reference mesh as ground truth, so every expected count, gid
    set and position is computed rather than hardcoded.

  **Signatures as they ended up:**

  ```cpp
  template <class MeshT> void tombstoneVertex( MeshT&, LocalIndex );
  template <class MeshT> void tombstoneEdge  ( MeshT&, LocalIndex );
  template <class MeshT> void tombstoneFace  ( MeshT&, LocalIndex );

  template <class MeshT> CompactStats
  compact( MeshT&, MeshHalo<typename MeshT::memory_space>& );
  template <class MeshT> CompactStats
  compactAndRenumberGids( MeshT&, MeshHalo<typename MeshT::memory_space>& );
  ```

  **Check 11's measured numbers** (SERIAL and HIP-default execution space, np1;
  ten rounds, round *r* removing base patches `[0, r]`, so the dead set is
  cumulative). `gid space` is `CompactStats::gidSpaceAfter`, `live` is the global
  owned V+E+F:

  | round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | gid space | 962 | 962 | 962 | 962 | 962 | 962 | 962 | 962 | 962 | 962 |
  | live | 925 | 881 | 837 | 793 | 741 | 697 | 653 | 609 | 565 | 521 |
  | wasted | 37 | 81 | 125 | 169 | 221 | 265 | 309 | 353 | 397 | 441 |

  The gid space is **pinned at its initial 962** across all ten rounds while the
  mesh loses 46% of its entities — that gap is what the dense-by-gid vectors in
  `buildKindPlan`/`make_g2l`/`build_order` are sized by, and what grows without
  bound under a real split/collapse workload. One `compactAndRenumberGids()`
  returns it to **exactly** 521 == live.

  Case 3's removal (one base patch): V/E/F 162/480/320 → **159/462/304**, i.e.
  3 vertices, 18 edges and 16 faces removed; one boundary component; **χ = 1**,
  the Euler characteristic of a sphere minus one disc.

  **Bugs only running revealed.**

  * `test_compact.cpp` called the **collective** `ownedEulerGlobal()` inside a
    `rank == 0` print guard, which deadlocked every other rank at np ≥ 2 with no
    output. The value is now computed before the guard. The reason it took a run
    to find is worth recording: the test buffers stdout, so the hang looked like
    it was in case 1 rather than case 3. `main()` now sets
    `setvbuf( stdout, nullptr, _IONBF, 0 )` so a hang localizes to the case that
    did not print.
  * Nothing in `Tessera_Compact.hpp` itself has needed a fix so far.

  **What remains** — in order: *(all six items completed 2026-08-15; see the
  entry below.)*

  1. `make tessera_test_compact_HIP` (the HIP binary in the tree predates the
     deadlock fix) and run `ctest -R compact_HIP` at ranks 1–5.
  2. Check 7 (the emptied rank) has only ever run at np1, where it **returns
     immediately**. It is unexercised. Read its output at np ≥ 2 before believing
     it.
  3. README: document `tombstoneX`, `compact`, `compactAndRenumberGids`, the
     `Gid == invalid_gid` convention, the gid-preservation rationale, and the
     Remesh-family consequence; add `compact()` to the *Editing families* table
     (it is currently listed as "to follow").
  4. `docs/design.md`: add the *Compaction* subsection, including the gid-space
     hazard and all three dense-by-gid paths named above. Suggested location:
     after *Edge-addressed splitting*, before *Quality-based refinement marking*.
  5. Full gate (`ctest -L regression -R "SERIAL|HIP"`), which should be 170/170:
     160 pre-existing plus the 10 new.
  6. Delete `tessera_compact.flux`, then mark this document **DONE** with a
     **Met.** paragraph.

  **Affects:** [edge-collapse.md](edge-collapse.md) — it calls `compact()`
  internally, and three findings change what it can assume. (a) It must tombstone
  on the **owner** and its tombstone set must close **globally**, or `compact()`
  throws; a purely local decision to kill a boundary vertex is exactly the bug
  the collective check catches. (b) It need **not** tombstone orphaned vertices
  and edges at all — `compact()` drops every vertex and edge no surviving face
  references — so collapse only has to get the **face** set right. (c) `compact()`
  claims `EditFamily::Remesh`, so a collapse test cannot build its fixture with
  `refine()`.

- 2026-08-15 — **DONE.** The six remaining items were worked in order; nothing in
  `Tessera_Compact.hpp` or `test_compact.cpp` needed a change to close them.

  1. **HIP.** The stale binary was rebuilt (`make tessera_test_compact_HIP`) and
     `ctest -R compact_HIP` is **5/5 Passed at np1–np5** (8.4 / 5.4 / 6.3 / 7.1 /
     7.8 s). Both execution spaces run inside each binary (`[Serial]` and
     `[Default]`), so HIP exercises every check twice. Check 11's numbers on the
     HIP-default space are **identical to the SERIAL table above** at every rank
     count — 962 pinned across ten rounds, live falling 925 → 521, and exactly
     521 == live after one `compactAndRenumberGids()`.
  2. **Check 7 is real at np ≥ 2, and was read rather than assumed.** It is
     non-vacuous at every rank count and the emptied rank's face count scales as
     the partition does: **160 / 106 / 80 / 64** faces removed at np2 / 3 / 4 / 5
     (of 320). What it asserts on the emptied rank is stronger than the exit
     criterion asked: zero **owned** *and* zero **local** entities, all three
     plans empty in both directions; and on every peer, that rank appears in no
     `send_peers` or `recv_peers`. Plus the global Euler number against the
     reference, `checkOwnershipPartition`, `owned1RingLocal`, the ordering check
     and a ghost round-trip. A rank emptied to nothing is a supported state.
  3. **README** — new *Compaction: removing entities* subsection (after *Editing
     families*, before *Example programs*): the three `tombstoneX` calls and the
     `Gid == invalid_gid` convention, what is removed and why marking a vertex or
     edge is optional-for-removal-but-load-bearing-for-the-check, the collective
     closure check and its collective throw, the gid-preservation rationale
     against periodic renumbering, the order-statistic definition of a new gid,
     `CompactStats` being global, the zero-owned-entity state, the Remesh-family
     consequence, and the non-goals including the `EdgeField::Faces` asymmetry.
     The *Editing families* table row now reads
     `splitEdges(), compact(), compactAndRenumberGids()` with only
     `collapseEdges()`/`flipEdges()` left as "to follow".
  4. **`docs/design.md`** — new top-level *Compaction* section in the suggested
     location (after *Edge-addressed splitting*, before *Quality-based refinement
     marking*), with three subsections: compaction as a halo rebuild rather than
     a private permutation; why the closure check is necessarily global (both
     directions of the local-sweep argument, including the `std::out_of_range`
     crash the check prevents); and the gid-space hazard, with **all three
     dense-by-gid paths in a table** (`buildKindPlan`, `make_g2l`,
     `build_order`), the measured 962-vs-521 numbers, and the order-statistic
     rationale.
  5. **Full gate: 240/240, 0 failed**, SERIAL + HIP, ranks 1–5, 1668 s
     (`tessera-regression-gate.f3SRzacqwR7u.out`). **The "170/170" figure in the
     item-5 note above was stale** — it was carried from an Aug-10 gate of 180
     tests and predates the halo-depth and distributed-build work. The gate is
     **230 pre-existing + the 10 new** = 240. Nothing was relabelled and no
     pre-existing test changed status. All of `src/` was rebuilt first, since
     `Tessera.hpp` and `Tessera_Profiling.hpp` are included everywhere.
  6. `tessera_compact.flux` deleted. (`tessera_build.flux`, also scratch and also
     untracked, is left alone — it is not this task's.)

  **Met.** `compact()` and `compactAndRenumberGids()` are in tree and green at
  **SERIAL and HIP, ranks 1–5**, with the full gate still green and nothing
  relabelled. Check 3 confirms **gid preservation** against a pre-edit
  gid→position snapshot and check 9 confirms **renumbering** to a contiguous
  `[0, N)`; the two calls' opposite contracts are both pinned, and check 10 pins
  that the renumbering is identical at every rank count. Check 7 passes with a
  rank emptied to zero owned *and* zero local entities, named in no peer's plan,
  at np2–np5. Check 11 demonstrates the leak — the gid space pinned at 962 while
  the mesh loses 46% of its entities, 441 wasted slots after ten rounds — and its
  fix, one `compactAndRenumberGids()` returning the space to exactly the live
  count of 521; the numbers are recorded above and are identical on both
  execution spaces. The README documents the API, the tombstone convention and
  the gid-preservation decision with its rationale, and `docs/design.md` has the
  *Compaction* section naming every dense-by-gid path. The one design change made
  during implementation — the closure check being **global**, not local — is
  recorded in the 2026-08-14 entry and is now documented in both places.
  Unblocks [edge-collapse.md](edge-collapse.md).
