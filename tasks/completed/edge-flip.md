# Edge flip

**Status:** DONE (2026-08-17). **Read the "Editing families" section of
[edge-split.md](edge-split.md) first** — a flip is a remesh-family operation.

**Met.** `src/Tessera_EdgeFlip.hpp` implements `flipEdges()`, `FlipResult` and
`DefaultFlipPolicy`; the mesh's `EditFamily::Remesh` claim is made through
`requireEditFamily`. `tests/test_flip_edges.cpp` is registered at TIER
`regression`, SERIAL + HIP, ranks 1–5, and is **green in all ten registrations**
in both execution spaces. Verified there: counts invariant with Euler 2,
conformity and no interior vertex after every flip (check 1); the flip is an exact
**involution** on the multiset of face corner-**gid** triples and on the edge key
set (check 2); flipping into an existing edge is rejected across the extra
coordinator round, on a hand-built subdivided tetrahedron whose original corners
keep valence 3 (check 3, `rejectedDuplicateEdge == 1`, `accepted == 0`, mesh
bitwise unchanged); no face is rewritten twice with every owned edge marked
(check 4, 480 requested → 80 accepted, 400 conflict); **rank-count invariance**
against an `MPI_COMM_SELF` reference — identical `accepted`, identical verdict
histogram and an identical multiset of face corner-**position** triples (check 5);
`minQuality = 0.99` rejects all 480 and moves nothing (check 6); the valence
histogram does not degrade over three valence-driven rounds (check 7); the edge
gid set is unchanged while the edge key set moves by **exactly** the accepted
flips, and `edgeKeys()`/`faceKeys()` match the AoSoA entry for entry with no
global duplicates (check 8); the halo is valid on return and a second
`flipEdges()` follows immediately (check 9); the empty mask is a no-op with all
counters zero (check 10); the family guard throws both ways and flip-after-split
is allowed and still involutive (check 11); and five split+flip rounds hold above
the measured floor of 0.025 (check 12). README documents `flipEdges`,
`DefaultFlipPolicy`, the three decisions and the caller-loop idiom, and the
*Editing families* row now names `flipEdges()`; `docs/design.md` gains *Edge
flip*. Full gate **250/250**.

**Verified against `2dd8da1`** (branch `conforming-refinement`) — the code this task
cites was re-read at that commit, which is where implementation started.
`splitEdges()`, `compact()`/`compactAndRenumberGids()`, configurable halo depth and
the distributed initial build have all landed since `08dd346`, which this document
was originally stamped against.

## Problem

Edge flipping does not exist at any level. It is not merely a missing convenience:
a flip needs **both faces incident on the edge**, and the 1-deep *vertex* halo does
not guarantee they are co-resident (an owned face all three of whose corners are
ghosts can have edge-neighbours held on neither this rank nor any single other
rank's local set). So a correct distributed flip needs the same edge-coordinator
machinery `refine()`'s 2:1 balance uses, not a local rewrite.

**Why it matters.** Flipping is the cheapest of the three remeshing operations and
the one that does the most for element quality per unit cost: it changes
connectivity without moving a vertex or changing V, E or F. Valence equalization
(drive interior valences toward 6) and Delaunay-style quality repair are both
flip-only passes. The driving consumer (the Beatnik z-model's mesh-quality
maintenance) uses flips to keep a tightening roll-up from producing slivers, and
without them a long run dies on element quality rather than on physics.

## Approach

New header `src/Tessera_EdgeFlip.hpp`.

### What a flip is, precisely

For an interior manifold edge `(a,b)` with incident faces `(a,b,c)` and `(b,a,d)`:
delete edge `(a,b)`, create edge `(c,d)`, and replace the two faces with `(a,c,d)`
and `(b,d,c)`. **V, E and F are all unchanged.** Only connectivity changes, plus:

- the flipped edge's `EdgeField::Verts` changes, so its **`EdgeKey` changes** — and
  therefore its `detail::edgeCoordRank` routing changes, and the `edgeKeys()` side
  table must be rebuilt. Easy to miss; it is the bookkeeping trap of this task.
- the two faces' `Verts` and `Edges` change, so `faceKeys()` must be rebuilt.
- the vertex→edge and vertex→face CSRs change for all four of `a`, `b`, `c`, `d`.

**Decision 1 — gids are preserved.** The flipped edge keeps its gid (with a new
key) and the two faces keep theirs (with new corners). Nothing is created or
destroyed, so there is nothing to assign, and preserving gids means no peer's
reference is invalidated. Consequence: a gid no longer determines a key, so any
code assuming `gid ↔ key` is a bijection breaks. Audit for that assumption as part
of the task.

### API

```cpp
struct FlipResult
{
    long long requested = 0;             //!< marked owned edges, globally
    long long accepted = 0;
    long long rejectedBoundary = 0;      //!< fewer than two incident faces
    long long rejectedDuplicateEdge = 0; //!< (c,d) already exists
    long long rejectedGeometric = 0;     //!< policy test failed
    long long rejectedConflict = 0;      //!< lost the independent-set round
};

struct DefaultFlipPolicy
{
    //! Reject if either new face's normal points more than this far from the
    //! area-weighted average of the two old normals (radians). Guards against
    //! flipping a nearly-flat pair into a fold.
    double maxNormalDeviation = 0.35;
    //! Reject if either new face's radius ratio (inradius/circumradius, scaled to
    //! 1 for equilateral) falls below this.
    double minQuality = 0.05;
};

//! Flip the marked edges. `edgeMask.size() == mesh.numOwnedEdges()`; the OWNER of
//! an edge decides. At most an independent set is applied per call — two flips
//! sharing a FACE conflict and only one survives, so a caller wanting more
//! progress calls again. Collective. Ends with rebuildHalo().
template <class MeshT, class Policy = DefaultFlipPolicy>
FlipResult flipEdges( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
                      const std::vector<char>& edgeMask,
                      const Policy& policy = Policy{} );
```

### Decision 2 — valence selection is the caller's, not Tessera's

The dominant use of flipping is valence equalization, which needs the **full
valence** of `a`, `b`, `c` and `d`. A vertex's valence is a local quantity at its
owner (the vertex→edge CSR owned rows are complete at halo depth 1), so the caller
can compute it and encode it in the mask. Keeping it out of `flipEdges` means the
operation itself needs **only depth 1** — the topological and geometric tests below
are all evaluable from the two incident faces, which the coordinator supplies.

Pulling valence inside would force `flipEdges` to need depth 2 and would hard-code
one selection criterion into a general operation. Document the intended pattern
(caller computes valences from `mesh.vertexEdges()`, builds the mask, calls
`flipEdges`, repeats) so consumers do not each reinvent it.

### Implementation — owner decides, both face owners apply

1. **Advertise.** Each rank sends, per **owned** face, `(EdgeKey, faceGid,
   ownerRank, oppositeVertexGid, oppositeVertexPosition)` for each of its three
   edges to `detail::edgeCoordRank`. The opposite vertex is the face corner not on
   the edge — `c` for one face, `d` for the other. Carrying the **position** is
   what lets the coordinator run the geometric test without needing a second round,
   and it is cheap (one `double[3]` per advertisement).
   **There is in-tree precedent: `refine()`'s Phase-2 coordinator reply already
   carries the split edge's squared length** — `detail::KeyGid`'s `double len2`
   rider (`Tessera_RefineParallel.hpp:196–201`, whose rationale is the comment at
   `:183–195`), added as Decision 15 in `4cee602` so the blue closure diagonal
   could be chosen geometrically. (Earlier revisions of this document named a
   `SplitLenMsg` at `:182`; no such struct exists.) Follow that message shape, and
   compute every length comparison through `Tessera::edgeLen2Canonical()`
   (`Tessera_RefineClosure.hpp:212`) so two ranks comparing the same edge get
   bit-identical doubles regardless of endpoint order — that helper is what makes
   the priority in step 3 genuinely rank-count invariant.
2. **Decide, at the coordinator.** For each edge with exactly two advertisements
   and the owner's mask bit set, the coordinator has `a`, `b`, `c`, `d` and all
   four positions, so it evaluates:
   - **Boundary**: fewer than two incident faces → reject.
   - **Duplicate edge**: does `(c,d)` already exist? The coordinator for
     `makeEdgeKey(c,d)` is a *different* rank in general, so this needs one extra
     round — the deciding coordinator queries the `(c,d)` coordinator and only
     accepts on a negative answer. Do not skip it: flipping into an existing edge
     produces a non-manifold mesh, and it happens routinely on a coarse mesh (any
     valence-3 vertex).
   - **Geometric**: the policy's normal-deviation and quality tests, from the four
     positions.
3. **Conflict resolution — a deterministic independent set.** Two flips sharing a
   **face** conflict (that face would be rewritten twice); two sharing only a
   vertex do not. Build the conflict graph from
   [face-adjacency.md](face-adjacency.md)'s `nbrGid`/`nbrOwner` — actually the
   relation needed is "two candidate edges belong to the same face", which each
   face owner can evaluate locally over its own three edges' verdicts, so no graph
   traversal is required: **a face with two or more accepted candidate edges keeps
   only the highest-priority one.** Priority is `(squared length descending,
   EdgeKey ascending)` — computable at the coordinator from the positions it
   already has, totally ordered, and independent of rank count. Face owners
   exchange their local decisions back through the coordinator so both sides of
   each edge agree before anything is written.
4. **Apply.** Each face owner rewrites its face's `Verts`/`Edges`; the edge owner
   rewrites the edge's `Verts`/`Faces`. `setOwnedCounts` with unchanged counts to
   bump the generation. **Do not hand-roll the side-table rebuild:** step 5's
   `rebuildHalo()` round D already redoes `edgeKeys()`, `faceKeys()` and both CSRs
   (`Tessera_HaloRebuild.hpp:474`, `:646`, `:661`), so step 5 subsumes most of this
   step. Follow the finalize sequence in `src/Tessera_EdgeSplit.hpp:834–866`
   verbatim.
5. **Halo.** `rebuildHalo( mesh, halo, effectiveHaloDepth( halo ) )` before
   returning (`src/Tessera_Distribute.hpp:56`), as every other editor does, so the
   depth the caller chose at setup is preserved rather than silently narrowed to 1.
6. **Empty mask** and **all-rejected** are no-ops with the counters reporting why.

**Decision 3 — one independent set per call, not a loop to exhaustion.** Iterating
inside `flipEdges` would hide an unbounded number of collectives behind one call
and make the cost unpredictable. The caller loops and watches `accepted`, which is
also what lets it re-derive valences between rounds. State the idiom in the header.

This deliberately does **not** reproduce a serial "rebuild the edge map after each
accepted flip" pass, which is inherently sequential and order-dependent. The
independent-set result differs from the serial one; both are valid. A consumer must
compare **quality statistics**, not the flip set.

### Non-goals

- Boundary edge flips (undefined — a boundary edge has one face).
- Choosing which edges to flip (Decision 2).
- Looping to exhaustion (Decision 3).

### Tests

New `tests/test_flip_edges.cpp`, registered at **TIER `regression`**, backends
**SERIAL and HIP**, ranks **`${TESSERA_TEST_MPI_RANKS}` (1–5)**. Gate promotion is
**pre-authorized for this task**.

1. **Counts are invariant.** For every case below: `globalOwnedVertices`,
   `globalOwnedEdges`, `globalOwnedFaces` unchanged, `globalOwnedEuler == 2`,
   `checkConforming` and `checkNoInteriorVertex` pass. The cheapest possible
   detector of a botched rewrite.
2. **Involution — the sharpest check.** Mark one edge (globally smallest
   `EdgeKey`), flip, then mark the *same edge gid* and flip again. The multiset of
   face corner-**gid** triples must return **exactly** to the original, and so must
   the edge key set. A flip is its own inverse; nothing else in the test pins the
   rewrite this tightly.
3. **Duplicate-edge rejection.** On a subdivision-0 icosahedron every vertex has
   valence 5, so construct the case deliberately: a hand-built soup with a
   valence-3 vertex, where flipping the opposite edge would duplicate an existing
   edge. Assert `rejectedDuplicateEdge == 1`, `accepted == 0`, mesh unchanged, and
   that the edge key set contains no duplicates afterwards.
4. **Independent set.** Mark **every** owned edge on a subdivision-2 icosphere.
   Assert no face was rewritten twice — i.e. for each face gid, at most one of its
   three edges is in the accepted set. Derive the accepted set by gathering it and
   checking against the pre-flip face→edge map.
5. **Rank-count invariance of the accepted set.** Check 4 at ranks 1–5: identical
   `accepted`, and the identical multiset of face corner-**position** triples
   afterwards. This is the payoff of the deterministic priority in step 3; if it
   fails, the priority is not total or not rank-independent.
6. **Geometric rejection.** A policy with `minQuality = 0.99` rejects everything:
   `accepted == 0`, `rejectedGeometric == requested - rejectedBoundary`, mesh
   bitwise unchanged (`topologyChecksum`).
7. **Valence use case.** Compute valences from `mesh.vertexEdges()`, mask the edges
   whose flip would reduce total valence deviation from 6, and run three rounds.
   The subdivision-2 icosphere starts optimal (12 vertices of valence 5, 150 of
   valence 6), so assert the valence histogram does **not degrade** — no vertex
   below 4 or above 8, and the count of valence-6 vertices does not decrease. Also
   report minimum radius ratio before and after. Statistics, not edit sets, per
   Decision 3.
8. **Key and gid bookkeeping.** After any accepted flip: the edge **gid** set is
   unchanged; the edge **key** set has changed by exactly the accepted flips;
   `edgeKeys()` and `faceKeys()` contain no duplicates and match the AoSoA
   connectivity entry for entry.
9. **Halo valid on return.** `haloExchange()` immediately after leaves ghost
   positions equal to owners'; a second `flipEdges` with no intervening `migrate()`
   succeeds; `checkOwnershipPartition` passes.
10. **Empty mask** is a no-op (`topologyChecksum` unchanged), all counters zero.
11. **Family guard.** `flipEdges` on a `refine()`d mesh throws per
    [edge-split.md](edge-split.md) Decision 1; `flipEdges` after `splitEdges` is
    allowed and passes checks 1 and 2.
12. **Composed with split.** `splitEdges` (length-threshold mask) then three
    rounds of valence flips, five times over. Euler `== 2` and `checkConforming`
    after every operation; report the minimum radius ratio per round and assert it
    stays above a floor **measured in the first implementation run and recorded
    here** rather than guessed. **Measured** (inradius/circumradius, 0.5 for an
    equilateral triangle; byte-identical at np1–5 on both backends and in both
    execution spaces — all twenty instances print the same five lines):

    | round | 1 | 2 | 3 | 4 | 5 |
    |---|---|---|---|---|---|
    | min r/R | 0.2452 | 0.0727 | 0.0727 | **0.0309** | 0.0330 |
    | min angle (deg) | 30.382 | 14.744 | 14.744 | 8.666 | 5.968 |
    | F | 800 | 1880 | 4520 | 9126 | 21284 |

    so the floor is **0.025**, just below the measured worst of 0.0309 — which is
    itself just above `DefaultFlipPolicy::minQuality`'s bound expressed in these
    units (0.05 in the convention where an equilateral triangle scores 1 is 0.025
    where it scores 0.5). The floor is a statement about the DRIVE, not about
    either operation; see the `kMinRadiusRatioFloor` comment.

## Exit criterion

- `test_flip_edges` green at **SERIAL and HIP, ranks 1–5**, and the full gate still
  green with nothing relabelled.
- Check 2 (involution) passes exactly, and check 5 (rank-count invariance) passes —
  those two together are what make the operation trustworthy.
- Check 3 passes: flipping into an existing edge is detected, including across the
  extra coordinator round.
- README API section documents `flipEdges`, `DefaultFlipPolicy`, the three
  decisions (gid preservation and the broken `gid ↔ key` bijection; valence
  selection is the caller's; one independent set per call), and the caller-loop
  idiom.
- `docs/design.md` gains an *Edge flip* subsection with the owner-decides protocol,
  the duplicate-edge round, and the independent-set priority rule — plus the
  explicit note that the result differs from a serial shortest-first pass and that
  consumers compare statistics.

## Where this sits

**Requires [edge-split.md](edge-split.md)** (Editing families, Decision 1).
`rebuildHalo()` already exists in tree (`Tessera_HaloRebuild.hpp`, `25980f2`), so
[halo-depth.md](halo-depth.md) is **not** a prerequisite — Decision 2 keeps this
operation depth-1-safe on purpose. halo-depth has nonetheless landed since, so the
call must PRESERVE the depth: `rebuildHalo( mesh, halo, effectiveHaloDepth( halo ) )`. [face-adjacency.md](face-adjacency.md) is useful
context for step 3 but the final design avoids needing the face graph. See the
ordering diagram in [halo-depth.md](halo-depth.md).

## Progress log

- 2026-08-07 — Task written, then re-checked against `08dd346` after pulling
  `../tessera`. Nothing implemented.

- 2026-08-17 — **Implemented.** New `src/Tessera_EdgeFlip.hpp` (`flipEdges()`,
  `FlipResult`, `DefaultFlipPolicy`), added to the `Tessera.hpp` umbrella; new
  `tests/test_flip_edges.cpp` at TIER `regression`, SERIAL + HIP, ranks 1–5.
  `minRadiusRatio()` moved from `tests/test_split_edges.cpp` into
  `tests/MeshInvariants.hpp` and is now shared by both tests, following the
  precedent set when `edgeSetOf()`/`checkSplitEdgeCoverage()` were moved there —
  `test_split_edges` is unchanged in behaviour and in every measured number.
  README gains the API line, an *Edge flip* subsection and a new Known Issue; the
  *Editing families* row now reads `splitEdges(), flipEdges(), compact(),
  compactAndRenumberGids()` with only `collapseEdges()` left as "to follow".
  `docs/design.md` gains *Edge flip*.

  **The three decisions handed down with the task, all taken as stated:**

  1. **The advertisement carries the opposite vertex's position** (and, in the
     end, all three of the advertising face's corner positions). `splitEdges()`
     deliberately did *not* put a length in a message (its log, 2026-08-10,
     departure 2) because its operands were the deciding face's own corners; that
     reasoning does not transfer, because the flip coordinator is a **third rank
     holding neither incident face** and needs all four of `a`, `b`, `c`, `d`. The
     precedent followed is `detail::KeyGid`'s `len2` rider.
  2. **The independent-set priority is `(squared length descending, EdgeKey
     ascending)` and contains no gid.** Every length goes through
     `edgeLen2Canonical()`. Check 5 passes at np1–5, which is the direct evidence
     that the rule is total and rank-count invariant.
  3. **`EdgeField::Faces` is not trusted as input.** Incidence is derived from the
     advertisements, which are built from owned faces and are therefore true.

  **Two things the task text got wrong, corrected in the implementation and in
  the document above:**

  1. **The winding.** The task says the two new faces are `(a,c,d)` and `(b,d,c)`;
     that pair is REVERSED relative to the input. With the incident faces written
     `(u,v,w)` and `(v,u,x)` — which is what a consistently oriented manifold
     gives — the quad's boundary cycle is `u → x → v → w` and the only
     orientation-preserving retriangulation on the diagonal `(w,x)` is
     **`(u,x,w)` and `(v,w,x)`**. Implemented that way; the header states it.
  2. **Which of the two old face gids lands on which new face is not free**, and
     most rules fail check 2. The rule that works is stated in canonical terms:
     with `p < q` the EdgeKey's ids and `r < s` the two opposite corners, the old
     face `{p,q,r}` keeps its gid on the new face `{p,r,s}` and `{p,q,s}` on
     `{q,r,s}`. Applying the same rule to the flipped edge maps them back exactly,
     which is why a second flip restores the original (corners, gid) pairing. The
     obvious alternatives — "the forward face keeps the face containing `u`", or
     "smallest old gid to the canonically-first new face" — both fail the
     involution, and the failure is not visible without check 2.

  **Signatures as they ended up:**

  ```cpp
  struct FlipResult
  {
      long long requested = 0, accepted = 0;
      long long rejectedBoundary = 0, rejectedDuplicateEdge = 0;
      long long rejectedGeometric = 0, rejectedConflict = 0;
      //! ADDED beyond the task's API: (old EdgeKey, new EdgeKey) for every flip
      //! this rank TOUCHES, sorted and unique by the old key. The same role
      //! SplitResult::midpoints plays -- without it, checks 4 and 8 would have to
      //! instrument the library at the call site to learn the accepted set.
      std::vector<std::pair<EdgeKey, EdgeKey>> flipped;
  };

  struct DefaultFlipPolicy
  {
      double maxNormalDeviation = 0.35;   // radians
      double minQuality = 0.05;           // r/R, 1 for an equilateral triangle
  };

  template <class MeshT, class Policy = DefaultFlipPolicy>
  FlipResult flipEdges( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
                        const std::vector<char>& edgeMask,
                        const Policy& policy = Policy{} );
  ```

  as designed. The five verdict counters partition `requested`, which the test
  asserts after every call. Rounds: one advertisement pair, one duplicate-edge
  query/reply pair, one offer/vote pair, one apply — seven `allToAllV`s plus the
  rebuild, against `splitEdges()`' six.

  **Two bugs only running revealed.**

  1. **`rebuildHalo()` keeps a stale ghost edge, and it corrupts the heap.** The
     first np2 run died with `double free or corruption (out)` before printing a
     single case line. Cause: `rebuildHalo()` seeds its gid → tuple map from
     **every locally held edge, owned and ghost**, and round G only fetches the
     ones that are MISSING — so a rank holding a ghost copy of a flipped edge kept
     the OLD endpoints and handed them to round D, which derives the edge key,
     the ownership AND the vertex→edge CSR from them. The CSR build indexes a
     dense gid → local array, so a stale endpoint the rank does not hold writes
     out of bounds. `splitEdges()` never meets this because it hands the rebuild a
     freshly built owned-only mesh; a rewrite-**in-place** operation has to say so
     explicitly. Fix: `resizeEdges(nOwnedE)` / `resizeFaces(nOwnedF)` before the
     rebuild, with a comment saying it is correctness and not tidiness. Ghost
     VERTICES are deliberately left alone — a flip moves no vertex.
  2. **`distribute()` leaves `edgeKeys()`/`faceKeys()` stale**, which is a
     PRE-EXISTING defect this task merely tripped over. `distribute()` rebuilds
     the AoSoAs, both CSRs and the three halo plans but never calls
     `setEdgeKeys()`/`setFaceKeys()`, so a freshly distributed mesh still carries
     the REPLICATED builder's tables, sized to the global entity count. Invisible
     until now because every consumer of those tables runs after an editor or an
     explicit `rebuildHalo()`, all of which rebuild them, and because at np1 the
     replicated and distributed meshes coincide. Recorded in README *Known
     Issues*; NOT fixed here, as it is outside this task. Test-side, check 8's
     side-table assertion is made in the three cases whose mesh has actually been
     through a flip, and case 7 carries a comment saying why it is not made there.
     *Since fixed inside `distribute()` itself* — one round-D-style local rebuild
     of both key Views after `setOwnedCounts()`, with `checkKeyTables()` moved to
     `tests/MeshInvariants.hpp` and now asserted by `test_distribute` and by case
     7; the README *Known Issues* bullet has been removed.

  **First-run measurements** (subdivision-2 icosphere, V=162 E=480 F=320; all
  byte-identical at np1–5, SERIAL and HIP, `Serial` and `Default` execution
  spaces unless noted):

  | check | result |
  |---|---|
  | 1+2. involution | one flip accepted, then the same edge GID flipped back; face corner-gid multiset and edge key set both return exactly; edge gid set never moves |
  | 3. duplicate edge | subdivided tetrahedron V=10 E=24 F=16; `requested=1 rejectedDuplicateEdge=1 accepted=0`, mesh bitwise unchanged |
  | 4. independent set | every owned edge marked: `requested=480 accepted=80 conflict=400`, boundary/dup/geometric all 0; **no face has two accepted edges** |
  | 5. rank-count invariance | `accepted=80` and every verdict counter identical to the `MPI_COMM_SELF` reference; face corner-position multiset bitwise equal |
  | 6. geometric rejection | `minQuality=0.99` → `rejectedGeometric=480`, `accepted=0`, gid checksums and connectivity signatures unchanged |
  | 7. valence | fixture is optimal (12 × valence 5, 150 × valence 6); 1440 candidate edges evaluated over three rounds, **0 accepted**, histogram and min r/R (0.4865) / min angle (54.397°) unmoved |
  | 8. key/gid bookkeeping | post-flip edge key checksum predicted exactly from the pre-flip one plus the flip map; no duplicate edge or face key globally |
  | 9. halo | 52 ghost positions corrupted and restored at np2, plan size 420; a second `flipEdges()` accepts 124 |
  | 10. empty mask | all six counters 0, `flipped` empty, checksums unchanged |
  | 11. family guard | throws in both directions naming both families; flip-after-split accepted and still involutive |
  | 12. split+flip | see the table under check 12 above; min r/R 0.2452 → 0.0330 with worst 0.0309, floor set to 0.025 |

  The 80-of-480 acceptance in check 4 is the expected shape of a one-round
  independent set on a closed triangulation: 320 faces, each able to endorse one
  edge, and an edge needs both of its faces — 80 is a quarter of the faces, i.e.
  the flips are well spread rather than clustered.

  Case 12's per-round FLIP COUNTS do move with the rank count (up to ~1.5%),
  while the per-round SHAPE numbers do not. That is the caller's valence mask, not
  the operation: the mask skips an owned edge whose second incident face is not
  resident, and which edges those are is a property of the partition. It is a
  legitimate depth-1 caller choice and is called out in the test; check 5 asserts
  `flipEdges()`' own rank-count invariance directly, with a mask that has no such
  dependence.

  **Verification:** `test_flip_edges` green in all ten registrations (SERIAL and
  HIP × np1–5), 12–23 s each. Full regression gate **250/250, 0 failed**, 1806 s
  (`tessera-gate.f3Spe2ZDWTFd.out`) — 240/240 before, plus this test's ten
  entries, and nothing relabelled. `test_split_edges` re-run and unchanged, its
  seven case-8 round lines and every measured number identical after the
  `minRadiusRatio()` move.

  **Affects:**
  * **[edge-collapse.md](edge-collapse.md)** — three findings transfer directly.
    (a) A collapse also rewrites entities in place around a neighbourhood, so it
    inherits bug 1 above: it must drop ghost edges/faces before `rebuildHalo()`,
    or hand the rebuild an owned-only mesh as `splitEdges()` does. (b) Its
    independent set has the same shape and should reuse this priority rule
    verbatim — `(squared length ..., EdgeKey ...)`, no gid — rather than inventing
    one; note the conflict relation is WIDER for a collapse (the whole 1-ring of
    both endpoints, not just the two incident faces), so the "each face owner
    decides locally" shortcut does NOT carry over and a real conflict graph or a
    vertex-coordinator round is needed. (c) The link condition a collapse must
    test — "the edge's two endpoints share exactly the two opposite corners" — is
    the same *shape* of query as this task's duplicate-edge round, and can be
    answered the same way, because the advertisement over every owned face's three
    edges gives a coordinator the exact global edge set.
  * **[face-adjacency.md](face-adjacency.md)** — no change, but confirmed in
    passing: `EdgeField::Faces` really is unusable as an input to a topological
    operation, and `flipEdges()` only repairs it on the flipped edge (the six side
    edges of a flipped quad may name the flip's sibling face). If a consumer ever
    needs `Faces` to be exact, that is a separate task and it needs to state which
    operations maintain it.
  * **[halo-depth.md](halo-depth.md)** — no change. Decision 2 held: `flipEdges()`
    is depth-1-safe in practice, and the only depth-dependent thing in the whole
    exercise is the caller's valence mask.
