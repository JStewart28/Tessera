# Edge collapse

**Status:** **DONE** (2026-08-18). `src/Tessera_EdgeCollapse.hpp` and
`tests/test_collapse_edges.cpp` are in tree, green in all ten registrations
(SERIAL + HIP × np1–5), and the remesh family is complete. See the
`## Progress log` entry for the measured numbers, the signatures that ended up
different from the API block below, and the two bugs only running revealed — one
of them a latent defect in `splitEdges()` that only a coarsening operation could
trigger. Largest of the eleven gap tasks, and the **last** of the
four topological edits — [edge-split.md](edge-split.md),
[mesh-compaction.md](mesh-compaction.md) and [edge-flip.md](edge-flip.md) have all
landed, so every prerequisite is in tree. **Read the "Editing families" section of
[edge-split.md](edge-split.md) first**, and the `**Affects:**` bullets in the
progress logs of [edge-flip.md](edge-flip.md) and
[mesh-compaction.md](mesh-compaction.md) — both name findings that change what this
task can assume.

**Verified against `82c02df`** (branch `edge-collapse`) — the code this task cites
was re-read at that commit, where `src/Tessera_EdgeSplit.hpp`,
`src/Tessera_EdgeFlip.hpp`, `src/Tessera_Compact.hpp`,
`src/Tessera_EditFamily.hpp` and configurable halo depth all exist. (It was
originally stamped against `08dd346`, which predates all four.)

## Problem

Edge collapse does not exist at any level, and unlike flip it is not merely
unimplemented — **the data model has no coarsening path at all.**
`FaceField::Level` and `EdgeField::Level` only ever rise, `refine()` is split-only,
`RefinementMode::Conforming` tracks a `ClosureParent` for refinement children with
no inverse, and nothing anywhere removes an entity. So neither the link condition
nor a cross-rank owner-decides protocol has anywhere to attach.

**Why it matters.** Collapse is the only operation that removes degrees of freedom.
Without it, a remeshing scheme is monotone: a metric-driven remesher can refine
where the metric demands resolution but can never coarsen where it does not, so the
mesh grows without bound and a long run dies on memory or on timestep. It is also
half of the standard quality-repair pair — collapse removes short edges and
slivers that no amount of flipping fixes. The driving consumer (the Beatnik
z-model's dynamic remesher) is built out of `collapse_short_edges` alongside split
and flip, and its evolving vortex sheet both stretches and compresses, so
coarsening is not optional for it.

## Approach

New header `src/Tessera_EdgeCollapse.hpp`. Four hard problems, each with a decision
to make and record.

### What a collapse is, precisely

For an interior manifold edge `(a,b)` with incident faces `(a,b,c)` and `(b,a,d)`:
merge `a` and `b` into one vertex at the policy-chosen position, delete the two
incident faces, delete edge `(a,b)`, and merge the two pairs of edges that become
coincident. Net: **V−1, E−3, F−2**, so Euler is preserved. Every other face that
referenced `b` now references `a`.

### Problem 1 — where coarsening attaches to the data model

**Decision 1 — collapse is a remesh-family operation and requires a mesh with no
transient closure and no hanging nodes.** It is defined on a pristine mesh and on a
mesh edited by `splitEdges`/`flipEdges`; it is **not** defined on a mesh produced by
`refine()`, in either refinement mode. Under `Conforming` there is a transient
closure layer whose children reference retired red parents, and collapsing across
it would have to un-close first; under `HangingNode2to1` there are T-junctions,
and the link condition is not even well-posed at one. The existing `EditFamily`
guard from [edge-split.md](edge-split.md) Decision 1 enforces this with a clear
message. `Level` on the merged vertex's incident entities is set to the `min` of
the merged values and is **advisory** thereafter, per that same decision.

This is the decision that makes the task tractable at all. The alternative — a real
inverse to `refine()`, un-refining a red split back to its parent — is a different
and much larger feature (it needs the parent's identity retained, sibling
co-residency, and an inverse to the closure); it is **not** what a metric remesher
wants, because it can only coarsen along the refinement tree. Record it as future
work and do not attempt it here.

### Problem 2 — the link condition needs a two-ring

Collapsing `(a,b)` is topologically valid iff
`link(a) ∩ link(b) == {c, d}` — the two vertices opposite the edge and nothing
else. Violating it welds two distant parts of the surface together and produces a
non-manifold mesh that no later check will untangle.

Evaluating it needs the **full one-rings of both endpoints**, i.e. a two-ring
around the edge. **Requires [halo-depth.md](halo-depth.md) at `depth >= 2`**, and
`collapseEdges` enforces it: **throw iff `0 < mesh.haloDepth() < 2`**, naming the
required and the actual depth. This is precisely the class of silent-seam bug the
depth task exists to close, so do not let it recur here.

The guard is `0 <` and not `>= 2` because **`haloDepth() == 0` means "never
distributed"** — a replicated mesh straight out of the builder, where every entity
is local and no ring is missing (`src/Tessera_Mesh.hpp:171–177`). `>= 2` would
reject exactly that mesh, which is the fixture the hand-built soup checks need.
`buildVertexStencil()` already polices depth this way, treating 0 as unconstrained
and any positive value as the real bound (`src/Tessera_Stencil.hpp:92`); follow it
rather than inventing a second convention.

**Correctness does not rest on local ring residency, only the precondition does.**
Depth `d` guarantees the `d`-ring of every **owned vertex**, and the owner of edge
`(a,b)` need own neither `a` nor `b` — an owned face can have all three corners as
ghosts, which is the case [edge-flip.md](edge-flip.md) had to design around. So the
one-rings are assembled from **vertex-coordinator advertisements** (Problem 4
step 2), which are exact by construction at any depth.

### Problem 3 — conflicts, and why not shortest-first

Two collapses sharing a vertex conflict: applying one changes whether the other is
valid, and applying both can produce garbage. Serial reference codes process
**shortest-first**, re-evaluating validity after each acceptance. That is inherently
sequential and its result depends on the processing order.

**Decision 2 — a deterministic maximal independent set, one round per call.** Each
candidate edge gets a priority `(squared length ascending, EdgeKey ascending)` —
shortest first, exactly matching the serial *preference* — and an edge is accepted
only if it has the strictly best priority among all candidate edges incident on
either of its endpoints' one-rings. That set is pairwise non-conflicting, is
computable in one coordinator round plus one exchange, and is **independent of rank
count** because the priority is a total order on geometric quantities and keys.

The accepted set will **not** equal the serial shortest-first set. It is a subset of
it, reached in fewer passes. A caller wanting more progress calls again and watches
`accepted`, exactly as with [edge-flip.md](edge-flip.md) Decision 3. Consumers must
compare **statistics** — face count, quality distribution, edge-length histogram —
not the edit set. State this in the header, in the README, and in `docs/design.md`;
it is the single most likely source of a "why doesn't this match the serial code"
question.

### Problem 4 — cross-rank owner-decides

The one-rings of `a` and `b` may span several ranks even at depth 2, and the merged
vertex's identity must be agreed before any rank rewrites connectivity.

Protocol, built on the existing edge coordinator (`detail::edgeCoordRank`,
`allToAllV`), the same machinery `refine()`'s 2:1 balance uses:

1. **Advertise**, per owned face: to each of its three **edge** coordinators
   `(EdgeKey, faceGid, ownerRank, oppositeVertexGid, oppositeVertexPosition)`, and
   to each of its three **corner** coordinators the face's three corner gids. The
   edge advertisement is the same *shape* as [edge-flip.md](edge-flip.md) step 1
   but **there is no shared helper to reuse**: `detail::FlipAdvert`
   (`src/Tessera_EdgeFlip.hpp:281`) is private to that header and its payload
   carries one opposite corner, which is not what this task needs. Declare a
   `detail::CollapseAdvert` here. Do **not** refactor `Tessera_EdgeFlip.hpp` to
   extract a common helper — it is green at ten registrations and a shared
   abstraction over two payloads that differ is not worth destabilising it.
2. **The candidate is assembled from coordinator state, not from local residency.**
   The edge coordinator for `(a,b)` holds both incident faces' advertisements, hence
   `a`, `b`, `c`, `d` and their positions, so it evaluates the geometric tests and
   computes the priority. The **link condition** is answered by the *vertex*
   coordinators: a corner coordinator holds every owned face incident on its
   vertex, so `link(a)` and `link(b)` are exact there by construction at any depth.
   The edge coordinator queries both and accepts only on
   `link(a) ∩ link(b) == {c, d}` — the same round shape as
   [edge-flip.md](edge-flip.md)'s duplicate-edge query, and for the same reason:
   the deciding coordinator is in general a third rank holding neither face.
3. **Independent-set round.** Each *vertex* coordinator collects the candidates
   incident on its vertex, and replies to each with whether it holds the best
   priority. An edge is accepted iff both endpoints' coordinators say yes. One
   `allToAllV` round trip. Priority is
   `(squared length ascending, EdgeKey ascending)`, containing **no gid**, and every
   length goes through `Tessera::edgeLen2Canonical()`
   (`src/Tessera_RefineClosure.hpp:212`) so two ranks comparing the same edge get
   bit-identical doubles regardless of endpoint order. That is
   [edge-flip.md](edge-flip.md)'s rule with the length ordering **reversed** —
   shortest-first here, longest-first there — and reusing its shape is what makes
   check 5 reachable.
4. **Apply.** The surviving vertex is `min(gid(a), gid(b))` — deterministic and
   rank-independent — moved to the position at parameter `policy.t` along `(a,b)`,
   through the policy hook. Every rank holding a face or edge that references the
   dying gid rewrites it to the surviving gid; that set is exactly the ranks in the
   dying vertex's one-ring, reachable through the vertex coordinator. Only the **two
   incident faces** are **tombstoned** (per
   [mesh-compaction.md](mesh-compaction.md)).

   **Do not tombstone the collapsed edge or the coincident edge pairs.** `compact()`
   drops every vertex and edge that no surviving face references
   (`src/Tessera_HaloRebuild.hpp:783–797`), so getting the *face* set right is
   sufficient — and tombstoning an edge that a surviving face still names trips
   `compact()`'s **global** closure check and throws. What is genuinely required is
   the connectivity rewrite: every surviving face's `Verts` and `Edges` must name
   the surviving gid of each merged pair, or the tombstone set does not close.

   **Drop stale ghost tuples before the rebuild.** Because this operation rewrites
   entities **in place**, it must `resizeEdges( numOwnedEdges() )` and
   `resizeFaces( numOwnedFaces() )` first. `rebuildHalo()` seeds its gid → tuple map
   from every locally held edge, owned *and ghost*, and only re-fetches the ones
   that are missing — so a rank holding a ghost copy of a rewritten edge keeps the
   **old** endpoints, and the CSR build indexes a dense gid → local array and writes
   out of bounds, corrupting the heap. This is not tidiness; it is the bug
   [edge-flip.md](edge-flip.md) hit as a `double free or corruption (out)` at np2
   before printing a single case line. Ghost *vertices* are left alone.
5. **`compact()`, and nothing after it.** `compact()` already ends with
   `rebuildHalo( mesh, halo, effectiveHaloDepth( halo ) )`
   (`src/Tessera_Compact.hpp:432`), so the incoming depth is preserved without a
   second call — adding one is a redundant collective, not a safety net.
   `collapseEdges` returns a compact, fully-haloed mesh at the depth it was handed;
   a caller must never see tombstones.

### API

```cpp
struct CollapseResult
{
    long long requested = 0, accepted = 0;
    long long rejectedBoundary = 0;
    long long rejectedLinkCondition = 0;
    long long rejectedNormalFlip = 0;
    long long rejectedQuality = 0;
    long long rejectedConflict = 0;
    long long verticesRemoved = 0, edgesRemoved = 0, facesRemoved = 0;
};

struct DefaultCollapsePolicy
{
    //! 0.5 = midpoint. 0.0 keeps a, 1.0 keeps b (a = lower gid).
    double t = 0.5;
    //! Reject if any surviving incident face's normal rotates more than this
    //! (radians) — the standard fold guard.
    double maxNormalRotation = 0.5;
    //! Reject if any surviving incident face's radius ratio falls below this.
    double minQuality = 0.05;

    //! Merged position: out[0..dim) = (1-t)*a[d] + t*b[d]. `a` is the LOWER-gid
    //! endpoint, so t is oriented by gid and not by local index — that is what
    //! makes the result rank-count invariant.
    void interpolatePosition( double* out, const double* a, const double* b,
                              double t, int dim ) const;
    //! Blend one scalar component of the vertex user field at ABSOLUTE member
    //! index M, same M convention as DefaultRefinePolicy. Default (1-t)*a + t*b.
    template <std::size_t M>
    double interpolateVertexField( double a, double b, double t ) const;
};

//! Collapse the marked edges. `edgeMask.size() == mesh.numOwnedEdges()`; the
//! OWNER decides. At most a maximal independent set is applied per call. Requires
//! mesh.haloDepth() >= 2 (link condition) and throws otherwise. Compacts and
//! rebuilds the halo before returning. Collective.
template <class MeshT, class Policy = DefaultCollapsePolicy>
CollapseResult collapseEdges( MeshT& mesh,
                              MeshHalo<typename MeshT::memory_space>& halo,
                              const std::vector<char>& edgeMask,
                              const Policy& policy = Policy{} );
```

**The policy carries its own `t`-aware hooks; `RefinePolicy` cannot be reused
verbatim.** `DefaultRefinePolicy::interpolatePosition( mid, a, b, dim )` and
`interpolateVertexField<M>( a, b )` are hard-coded 0.5 averages with **no `t`
argument** (`src/Tessera_RefinePolicy.hpp:72` onward), so a collapse at
`t != 0.5` is not expressible through them. The two hooks above mirror that
interface with `t` added, keeping the per-field override pattern
`DefaultRefinePolicy` documents (derive, shadow `interpolateVertexField`, dispatch
on `M` with `if constexpr`) so every user field is still handled automatically and
a consumer's existing refine policy transfers by inspection. Drive them over the
vertex user fields with the same member-index walk `splitEdges()` uses.

### Non-goals

- Un-refining a `refine()`d mesh (Decision 1's alternative).
- Vertex removal / valence-3 cleanup, half-edge collapse to a fixed endpoint beyond
  `policy.t`, or quadric error metrics. All are consumer policy.
- Looping to exhaustion (Decision 2).
- Boundary collapses. A boundary edge is rejected; a boundary-preserving collapse is
  a separate feature.

### Tests

New `tests/test_collapse_edges.cpp`, registered at **TIER `regression`**, backends
**SERIAL and HIP**, ranks **`${TESSERA_TEST_MPI_RANKS}` (1–5)**. Gate promotion is
**pre-authorized for this task**.

1. **One edge, exact deltas.** Subdivision-2 icosphere at depth 2, mark the
   globally smallest `EdgeKey`. `V−1`, `E−3`, `F−2`; `globalOwnedEuler == 2`;
   `checkConforming`, `checkNoInteriorVertex`, `owned1RingLocal` all pass. The
   surviving vertex gid is `min(a,b)` and sits at the midpoint.
2. **Link condition is enforced.** Mark all three edges of one face. Collapsing two
   of them is topologically invalid; assert `accepted <= 1`, Euler `== 2`, and
   conformity. Then a deliberate positive: on a hand-built soup containing two
   triangles that share an edge *and* a third vertex (the classic link-condition
   violation), assert `rejectedLinkCondition == 1` and `accepted == 0`.
3. **Depth guard.** `collapseEdges` on a depth-1 mesh **throws**, naming the
   required and actual depth. Pins Problem 2.
4. **Independent set.** Mark every owned edge. Assert no two accepted edges share a
   vertex (gather the accepted set and check against the pre-collapse
   vertex→edge map), and that `verticesRemoved == accepted`,
   `edgesRemoved == 3 * accepted`, `facesRemoved == 2 * accepted`.
5. **Rank-count invariance — the strongest determinism claim in the task.** Check 4
   at ranks 1–5: identical `accepted`, identical resulting V/E/F, and an identical
   multiset of face corner-**position** triples. If this fails, the priority order
   is not total or the surviving-vertex rule leaked a local index.
6. **Geometric rejection.** `minQuality = 0.99` rejects everything: `accepted == 0`,
   `topologyChecksum` bitwise unchanged, and the mesh is not compacted into a
   different ordering (a no-op must be a genuine no-op).
7. **Normal-flip rejection.** Build a soup with a near-degenerate spike whose
   collapse folds a neighbour; assert `rejectedNormalFlip >= 1` and that no
   surviving face has a normal reversed relative to its pre-collapse orientation
   (dot product with the pre-collapse normal positive for every surviving face).
8. **Decimation to a floor.** Loop `collapseEdges` with an all-edges mask until
   `accepted == 0` or 20 rounds. After every round: Euler `== 2`, conformity,
   `owned1RingLocal`, and no duplicate edge keys. Report face count and minimum
   radius ratio per round. Assert the face count is strictly decreasing while
   `accepted > 0`.

   **MEASURED** (identical at np1–5, SERIAL and HIP, both execution spaces):

   | round | 1 | 2 | 3 | 4 | 5 | 10 | 15 | 20 |
   |---|---|---|---|---|---|---|---|---|
   | accepted | 8 | 8 | 4 | 4 | 3 | 4 | 3 | 3 |
   | F | 304 | 288 | 280 | 272 | 266 | 236 | 210 | **174** |
   | min r/R | .3769 | .3812 | .3769 | .3812 | .3001 | .2634 | .1747 | **.1747** |

   **Terminal face count 174 after 20 rounds, quality floor 0.1747** — and the loop
   is STILL ACCEPTING at the cap, because an all-edges mask has no target scale.
   What eventually ends it is the fold and quality guards, not the round count:
   check 9's same-mask measurement ran 68 rounds from 1280 faces down to 144. The
   asserted floor is `kDecimateFloor = 0.02`, just under check 9's lower 0.0676.
9. **Round trip with split.** `splitEdges(all edges)` (→ 1280 faces) then collapse
   loops until no progress. Assert Euler `== 2` throughout and that the final face
   count is within a factor of 2 of the original 320 — a coarse but real check that
   coarsening actually undoes refinement rather than jamming.

   **MEASURED, and the mask had to be length-driven rather than all-edges.** With
   an all-edges mask the loop ran 68 rounds to genuine exhaustion and landed at
   **144 faces** — it does not jam, it overshoots the band, because a mask with no
   target scale keeps going until the quality guards stop it. The rule that
   expresses "coarsening undoes refinement" is a statement about a LENGTH:
   collapse every edge shorter than 4/5 of the pre-split mean edge length (0.2993),
   the same 4/5-of-target threshold check 14 uses. That terminates AT that scale:
   **F 320 → 1280 → 198–204 in 59–62 rounds, quality floor 0.0676–0.1219**, inside
   the factor-of-two band [160, 640] at every rank count.

   The small spread across rank counts is real and is not `collapseEdges()` losing
   determinism — see the progress log's note on `splitEdges()` midpoint gids being
   rank-count dependent, which makes the post-split EdgeKeys (and so the priority
   tie-breaks) differ. Check 5 asserts the operation's own invariance directly on a
   fixture whose gids come from the replicated builder.
10. **User-field blend.** A vertex user field seeded to a linear function of
    position: after a collapse at `t = 0.5`, the merged vertex's value equals the
    linear function evaluated at its new position, to `1e-14` relative. Both
    `double` and `double[3]`.
11. **Halo and compaction on return.** No tombstones remain (`Gid != invalid_gid`
    for every local entity); `haloExchange()` leaves ghost positions equal to
    owners'; `checkOwnershipPartition` passes; `halo.depth == 2` still; a second
    `collapseEdges` with no intervening `migrate()` succeeds.
12. **Empty mask** is a no-op; all counters zero; `topologyChecksum` unchanged.
13. **Family guard.** `collapseEdges` on a `refine()`d mesh throws per
    [edge-split.md](edge-split.md) Decision 1, in both refinement modes.
14. **Composed remesh loop — the integration test.** Twenty rounds of
    { `splitEdges` above a length threshold → `flipEdges` for valence →
    `collapseEdges` below a length floor }. After every operation: Euler `== 2`,
    conformity, no duplicate keys, no tombstones. Report face count, edge-length
    histogram and minimum radius ratio per round, and assert the face count stays
    within a band and the quality floor holds. This is the test that proves the
    four remesh operations compose, which is the actual deliverable of the family.

    **MEASURED, with a FIXED target length — a relative threshold is vacuous
    here.** "Split above 1.3 × the current mean, collapse below 0.7 × it" marks
    NOTHING on the subdivision-2 icosphere, whose edges span 0.276–0.325 around a
    mean of 0.30: the first implementation run printed
    `split=0 flipped=0 collapsed=0` for all twenty rounds. The drive is therefore
    the Botsch–Kobbelt rule against a fixed target of 0.6 × the initial mean
    (**0.1796**): split above 4/3 × target, collapse below 4/5 × target, flip for
    valence in between.

    | round | 1 | 2 | 5 | 10 | 15 | 20 |
    |---|---|---|---|---|---|---|
    | split / flip / collapse | 480/0/20 | 0/0/16 | 2/28/8 | 18/26/12 | 20/18/11 | 12/14/10 |
    | F | 1240 | 1208 | 1196 | 1224 | 1250 | **1240** |
    | edge len | .133–.207 | .133–.252 | .116–.260 | .098–.276 | .069–.269 | .069–.269 |
    | min r/R | .3792 | .3671 | .1130 | .1207 | .1206 | .0488 |

    **Per-round band [1196, 1250] over all twenty rounds and quality floor 0.0255**
    (worst at round 16), identical at np1–5 on both backends and in both execution
    spaces, tail included. Round 1 splits all 480 edges up to 1240 faces and the
    drive then HOLDS the mesh there — it neither runs away under repeated splitting
    nor decimates under repeated collapsing, which is the property the assertion is
    about. Asserted band `[1000, 1500]`, floor `0.02`, ~1340–1370 total edits.

## Exit criterion

- `test_collapse_edges` green at **SERIAL and HIP, ranks 1–5**, and the full gate
  still green with nothing relabelled. The gate stands at **250/250** with
  [edge-flip.md](edge-flip.md) landed, so this task's ten registrations make the
  expected figure **260/260**; a smaller total means something was relabelled or
  dropped.
- Checks 2, 3 and 7 pass — the three ways a collapse silently corrupts a mesh are
  each detected and reported through a named counter.
- Check 5 passes: the accepted set is rank-count invariant.
- Check 14 passes: split, flip and collapse compose over 20 rounds without
  degrading Euler, conformity or the quality floor.
- README API section documents `collapseEdges`, `DefaultCollapsePolicy`, the
  `haloDepth() >= 2` precondition, and **Decision 2 with its consequence** (the
  independent set differs from a serial shortest-first pass; compare statistics,
  not edit sets).
- README *Known Issues* records the accepted limits: collapse is undefined on a
  `refine()`d mesh, boundary edges are always rejected, and one call applies at most
  an independent set.
- `docs/design.md` gains an *Edge collapse* subsection with all four problems and
  their decisions, and states Decision 1's alternative (a true inverse to `refine()`)
  as explicitly out of scope.

## Where this sits

**Every prerequisite is landed.** [halo-depth.md](halo-depth.md) — this is the one
task with a hard dependency on it — supplies `depth >= 2` for the link-condition
precondition (Problem 2); [mesh-compaction.md](mesh-compaction.md) supplies
`compact()`; [edge-split.md](edge-split.md) supplies Editing families and the
coordinator idiom; [edge-flip.md](edge-flip.md) supplies the advertisement *pattern*
and three findings recorded in its `**Affects:**` bullet, but **no shared helper**
(Problem 4 step 1). See the ordering diagram in
[halo-depth.md](halo-depth.md).

## Progress log

- 2026-08-07 — Task written, then re-checked against `08dd346` after pulling
  `../tessera`. Nothing implemented.

- 2026-08-18 — **Reconciled against the four prerequisites' progress logs, which
  had all landed since the `08dd346` stamp. Nothing implemented.** Corrected above:
  the shared advertisement helper the task promised does not exist and this task
  declares its own `detail::CollapseAdvert` (Problem 4 step 1); the depth guard is
  `0 < haloDepth() < 2`, not `>= 2`, which would have rejected the replicated
  builder mesh the soup fixtures use (Problem 2); the link condition comes from
  vertex-coordinator advertisements rather than local depth-2 residency, because
  the edge owner may own neither endpoint (Problem 2, Problem 4 step 2); only the
  two incident **faces** are tombstoned and the coincident edges are left to
  `compact()`, with the connectivity rewrite named as the real requirement
  (Problem 4 step 4); ghost edge/face tuples are dropped before the rebuild, per
  edge-flip's heap-corruption bug (Problem 4 step 4); the trailing `rebuildHalo()`
  is deleted as a double rebuild (Problem 4 step 5); `DefaultCollapsePolicy` gained
  its own `t`-aware hooks because `DefaultRefinePolicy`'s take no `t`, and step 4's
  undeclared `policy.position( a, b )` is gone (API); the expected gate figure is
  260/260 (Exit criterion).

## edge-collapse

- 2026-08-18 — **DONE.** `src/Tessera_EdgeCollapse.hpp` (~1030 lines) and
  `tests/test_collapse_edges.cpp` (~1250 lines, 14 checks) implemented on branch
  `edge-collapse` from `780f102`. Ten registrations, TIER `regression`, SERIAL +
  HIP × `${TESSERA_TEST_MPI_RANKS}` (np1–5), all green.

  **The four decisions, as taken.**

  * **`DefaultCollapsePolicy` carries its own `t`-aware hooks** —
    `interpolatePosition( out, a, b, t, dim )` and
    `interpolateVertexField<M>( a, b, t )`, both defaulting to the lerp, both
    keeping the `M`-indexed `if constexpr` override pattern so a consumer's refine
    policy transfers by inspection. Necessary and not merely tidy:
    `DefaultRefinePolicy`'s hooks are hard-coded 0.5 averages with **no `t`
    argument** (`src/Tessera_RefinePolicy.hpp:72` onward), so `t != 0.5` is not
    expressible through them. Driving them needed a local
    `detail::blendVertexUserFieldsT()` — `Tessera_Refine.hpp`'s
    `blendVertexUserFields()` could not be reused for the same reason.
  * **The link condition is answered from vertex-coordinator advertisements**, not
    from local depth-2 residency, because the owner of `(a,b)` may own neither
    endpoint. Every owned face advertises its three corners to those corners'
    coordinators, so `ring(v)` is exact at the coordinator of `v` **at any depth**.
    `haloDepth() >= 2` remains a documented precondition with its own throw and its
    own check (check 3), and **no correctness argument rests on a ring being locally
    resident.** The geometric tests moved to the vertex coordinators too, for the
    same reason: that is the only place where every surviving face incident on an
    endpoint is known.
  * **The depth guard throws iff `0 < mesh.haloDepth() < 2`.** Depth 0 is
    "never distributed", and it is what lets checks 2 and 7 run on hand-built
    replicated soups.
  * **Collapse declares its own `detail::CollapseAdvert`.** `Tessera_EdgeFlip.hpp`
    was not touched. Its three geometry helpers (`flipTriNormal`,
    `flipAngleBetween`, `flipRadiusRatio`) ARE reused unchanged — reuse of three
    inline functions, not a refactor.

  **One design addition beyond the document.** The vertex link test
  `link(a) ∩ link(b) == {c,d}` is necessary but **not sufficient**: on a mesh that
  is a single tetrahedron (or anywhere the merge fuses two triangles into one) it
  passes and the result carries two faces with the same three corners. The
  deciding coordinator already holds both endpoints' incident faces, so it rewrites
  the dying endpoint's faces and rejects a candidate whose rewritten face set has a
  duplicate `FaceKey`. Same counter (`rejectedLinkCondition`), no extra round.

  **Signatures as they ended up:**

  ```cpp
  struct CollapseResult
  {
      long long requested = 0, accepted = 0;
      long long rejectedBoundary = 0, rejectedLinkCondition = 0;
      long long rejectedNormalFlip = 0, rejectedQuality = 0;
      long long rejectedConflict = 0;
      long long verticesRemoved = 0, edgesRemoved = 0, facesRemoved = 0;
      //! ADDED beyond the task's API block: the EdgeKey of every accepted
      //! collapse this rank TOUCHES, sorted and unique. The same role
      //! SplitResult::midpoints and FlipResult::flipped play -- without it check
      //! 4 could not learn the accepted set without instrumenting the library.
      std::vector<EdgeKey> collapsed;
  };
  ```

  `DefaultCollapsePolicy` and `collapseEdges()` are exactly as in the API block,
  with two notes: the policy is **not** templated on `Scalar` (its hooks are
  `double`, per the block, unlike `DefaultRefinePolicy<Scalar>`), and the hooks'
  `t` parameter is spelled `t_` to avoid shadowing the member. The removal counts
  come straight from `compactImpl()`'s `CompactStats`, so
  `verticesRemoved == accepted`, `edgesRemoved == 3 * accepted`,
  `facesRemoved == 2 * accepted` is an end-to-end statement that the connectivity
  rewrite closed, and the test asserts it after every single call.

  **Rounds:** one advertisement round (five messages: face→edge-coordinator,
  face-corner→vertex-coordinator, owned-vertex→its coordinator,
  owned-edge→both endpoints' coordinators, marked-edge verdict), one
  query/reply pair against the two endpoints' vertex coordinators (the reply
  carries ring, owner, incident faces, incident edges and the geometric verdict),
  one offer/endorse pair for the independent set, one apply round (six messages),
  then `compactImpl()` and **nothing after it** — ten `allToAllV`s plus the one
  rebuild inside the compaction, against `flipEdges()`' seven.

  **Two bugs only running revealed.**

  1. **`splitEdges()` based new midpoint VERTEX gids on the vertex COUNT, and a
     coarsening operation makes that false.** *(A latent defect in `splitEdges()`,
     not in collapse — but only collapse can trigger it, so it is this task's to
     fix, and it is fixed in `src/Tessera_EdgeSplit.hpp`.)* Phase B assigned
     `midGid = globalOwnedVertices(mesh) + exscan + i`, while step 3c correctly
     bases child FACE gids on `globalMaxF + 1` with a comment about retired parents
     leaving the face gid space sparse. Vertex gids were dense-and-count-equal for
     as long as nothing removed a vertex; `collapseEdges()` removes vertices and
     `compact()` **preserves** the survivors' gids, so after one collapse the max
     vertex gid exceeds the count and the next `splitEdges()` hands a midpoint a
     gid a LIVE vertex still holds. Symptom, from check 14 round 3: `V` stops
     growing, the Euler number breaks by **exactly one per split** (`V=600 E=1818
     F=1212` → −6), an edge appears between two nearly antipodal points
     (length 1.955 on a unit sphere), and the quality collapses to 0.0239. Fixed by
     the max-gid rule, with the comment naming the check that found it.
     `refine()`'s midpoint base (`Tessera_RefineParallel.hpp:565`) has the same
     shape but is **not reachable** — it belongs to the hierarchical family and can
     never see a mesh a collapse has coarsened — so it was deliberately left alone.
  2. **A test bug that only np ≥ 4 could show: the reference midpoint must be
     agreed globally.** Check 1 computed the expected merged position locally from
     the two endpoints and compared it on every rank holding the survivor. At np1–3
     every such rank happened to hold both endpoints; at np4–5 one always holds the
     surviving endpoint as a ghost **without** the far one, so its reference was
     zero and the check failed while the operation was correct. Fixed with an
     `MPI_MAX` over a sentinel — ghost positions are whole-tuple copies, so every
     contributing rank agrees bitwise. Only case 1 failed, at np4 and np5, on both
     backends; everything else including the 68-round loop passed at every rank
     count first time.

  **Two findings about the DRIVE, not the operation** (both changed the test, and
  both are recorded in the check bodies with the measurement that forced them):
  a relative length threshold (split above 1.3 × the current mean) marks
  **nothing** on the near-uniform subdiv-2 icosphere and made check 14 twenty
  vacuous rounds; and an all-edges mask in check 9 does not jam but **overshoots**,
  running 68 rounds to 144 faces, because a mask with no target scale coarsens
  until the quality guards stop it. Both now use a fixed target length with the
  Botsch–Kobbelt 4/3 and 4/5 thresholds.

  **First-run measurements** (subdivision-2 icosphere V=162 E=480 F=320 at halo
  depth 2 unless noted; identical at np1–5, SERIAL and HIP, `Serial` and `Default`
  execution spaces, except where the spread is called out):

  | check | result |
  |---|---|
  | 1. one edge | `requested=1 accepted=1`, V/E/F 162/480/320 → 161/477/318, `removed v=1 e=3 f=2`, surviving gid 0 exactly at the midpoint on every copy |
  | 2. link condition | negative: three edges of one face → `accepted=1` (≤1); positive: the 4-triangle violation soup → `rejectedLinkCondition=1 accepted=0`, mesh untouched |
  | 3. depth guard | throws at depth 1 naming "depth >= 2" and "depth 1"; runs at depth 2; the mesh is not mutated by the throw |
  | 4+5. independent set + rank-count invariance | every owned edge marked: `requested=480 accepted=8 conflict=472`, boundary/link/normal/quality all 0 → V=154 E=456 F=304. **No two accepted collapses share a vertex.** Byte-identical to the `MPI_COMM_SELF` reference at np1–5: same `accepted`, same five verdict counters, same V/E/F, same face corner-POSITION multiset |
  | 6. geometric rejection | `minQuality=0.99` → `rejectedQuality=480 accepted=0`; `topologyChecksum` unchanged AND the local gid SEQUENCES unchanged (a rejected-everything call skips the compaction entirely) |
  | 7. normal flip | closed decagonal bipyramid with a z=3 spike, V=12 E=30 F=20: `requested=30 accepted=1 rejectedNormalFlip=20 conflict=9`; 18 surviving faces compared against their own pre-collapse normals by face gid, **0 reversed** |
  | 8. decimation | F 320 → **174** in 20 rounds, still accepting at the cap; quality floor **0.1747**. Full table above |
  | 9. round trip | F 320 → 1280 → **198–204** in 59–62 rounds (terminated on `accepted == 0`), floor **0.0676–0.1219**, inside [160, 640]. The spread is the post-split gid space, not the operation |
  | 10. user field | `double` and `double[3]` seeded to a linear function of position: worst relative error **0.000e+00** at every rank count, checked on all 1–4 copies of the merged vertex |
  | 11+12. halo, compaction | empty mask: all ten counters 0 and the gid sequences fixed. After a real call: no tombstone anywhere, all 165/236/324 ghost positions (np3/4/5) corrupted and restored, `halo.depth == 2` and `mesh.haloDepth() == 2` still, plans 872–3296, second call accepts 8 |
  | 13. family guard | throws in both directions naming both families, in BOTH refinement modes |
  | 14. composed loop | 20 rounds of split/flip/collapse at target 0.1796: F band **[1196, 1250]**, quality floor **0.0255**, ~1340–1370 edits, edge lengths held around the target. Table above |

  The 8-of-480 acceptance in check 4 is the expected shape of Decision 2's
  conflict relation, not a defect: accepting one candidate excludes every candidate
  in the **two-ring** of either endpoint, which on a valence-6 triangulation is
  ~40 edges. It is recorded in README *Known Issues* as an accepted limit, with the
  loop that works around it.

  **Verification:** `test_collapse_edges` green in all ten registrations, 14–23 s
  each, 160 s wall for the ten (`tessera-collapse.f3T3NGU4gg1V.out`). Full
  regression gate **260/260, 0 failed**, 2051 s
  (`tessera-gate.f3T3gfXPwVC3.out`) — 250/250 before, plus this test's ten
  entries, and **nothing relabelled**. All of `src/` was rebuilt first, since
  `Tessera.hpp` and `Tessera_Profiling.hpp` are included everywhere and
  `Tessera_EdgeSplit.hpp` changed. `docs/testing.md` needed no change: it defines
  the gate by label + backends + ranks and enumerates neither names nor totals, and
  this task adds a test at an existing tier.

  **Met.** `collapseEdges()` is in tree and green at **SERIAL and HIP, ranks 1–5**,
  with the full gate at **260/260** and nothing relabelled or dropped. The three
  ways a collapse silently corrupts a mesh are each detected and reported through a
  named counter, measured rather than asserted-by-construction: check 2 gets
  `rejectedLinkCondition == 1` on a soup built so that the edge has exactly two
  incident faces and the boundary test cannot fire, and `accepted <= 1` when all
  three edges of one face are marked; check 3 throws on a depth-1 mesh naming both
  the required and the actual depth, and runs at depth 2; check 7 gets
  `rejectedNormalFlip == 20` of 30 on the spike bipyramid with **0 of 18** surviving
  faces reversed against their own pre-collapse normals. Check 5 holds exactly: at
  np1–5 the accepted set, all five verdict counters, V/E/F and the face
  corner-position multiset are identical to an `MPI_COMM_SELF` reference the same
  run computes. Check 14 composes all four operations over twenty rounds with the
  face count inside **[1196, 1250]**, the quality floor at **0.0255**, and Euler,
  conformity, key uniqueness and tombstone-freedom asserted after **every single
  operation** rather than once per round. The measured floors are check 8's
  **174 faces / 0.1747**, check 9's **198–204 faces / 0.0676** from 1280, and check
  14's band and floor above; `verticesRemoved == accepted`,
  `edgesRemoved == 3·accepted`, `facesRemoved == 2·accepted` after every call in
  every case. README documents `collapseEdges`, `DefaultCollapsePolicy` with both
  `t`-aware hooks, the `haloDepth() >= 2` precondition and Decision 2 **with its
  consequence** (compare statistics, not edit sets, and 8 of 480 in one call is the
  shape of that consequence); README *Known Issues* records the three accepted
  limits; and `docs/design.md` has an *Edge collapse* section with all four problems,
  their decisions, Decision 1's alternative as explicitly out of scope, and the
  `splitEdges()` gid-space consequence. Two things were found by running rather than
  by reading, and one of them was a latent defect in a landed operation. **The
  remesh family is complete.**

  **Affects:**
  * **[face-adjacency.md](face-adjacency.md)** — `EdgeField::Faces` is now
    demonstrably worse than best-effort on output, and the task should say so.
    `collapseEdges()` does not repair it **at all**: a surviving merged edge keeps
    the `Faces` it had, one entry of which names a **tombstoned face**, and
    `compact()`'s closure check cannot catch that because it checks the references
    of live owned FACES only (`Tessera_Compact.hpp`, `verifyTombstoneClosure`).
    `flipEdges()` at least rewrote the flipped edge's `Faces` exactly. So after a
    collapse the field can name a face that exists nowhere — a stronger statement
    than "it can name a face no rank holds", which was the pre-existing
    `migrate()` case. Any consumer that needs `Faces` exact needs that task, and
    that task must now list which operations maintain it and which actively break
    it.
  * **[halo-depth.md](halo-depth.md)** — collapse is the **first** operation with a
    hard `depth >= 2` requirement, and the interesting part is that the requirement
    turned out to be a *precondition only*: the exact one-rings come from vertex
    coordinators, which are correct at any depth, so nothing in `collapseEdges()`
    would break at depth 1 except the contract it advertises. That is worth
    recording as the pattern for future depth-hungry operations — police the depth
    at the entry point, derive the neighbourhood from coordinators — because it
    keeps the operation's correctness independent of the halo. The `0 <` form of
    the guard, and the depth-0-means-replicated convention it shares with
    `buildVertexStencil()`, are now used by two call sites rather than one.
  * **[edge-split.md](edge-split.md)** — its midpoint-gid rule was **changed** by
    this task (bug 1 above): the block now sits above the global max vertex gid,
    not above the vertex count. Anything in that task or its log that describes the
    old rule is stale.
