# Distributed initial mesh construction

**Status:** DONE (2026-08-10). See *Progress log*.

**Verified against `08dd346`** (branch `conforming-refinement`) — the code this task
cites was re-read at that commit.

## Problem

The initial mesh is **built in full on every rank before it is cut**.
`buildFromTriangleSoup()` is host-side, serial, and takes a replicated
`TriangleSoup`; `buildIcosphere()` generates the whole soup on every rank and hands
it over; `distribute()` then computes ownership locally (which it can *because* the
mesh is replicated) and cuts each rank down to its own share.

So peak memory per rank is proportional to the **global** mesh size, not to
`global / ranks`, and it is paid on every rank simultaneously. The edge derivation
inside `buildFromTriangleSoup` compounds it: a `std::map<EdgeKey, int>` over every
edge of the global mesh, plus three `std::vector<std::array<...>>` side tables, all
host-resident, all replicated.

This is scalability, not correctness. At the resolutions Tessera is tested at it is
irrelevant — subdivision 2 is 162 vertices. It becomes a hard ceiling exactly when
someone wants an initial mesh whose resolution is comparable to the refined running
mesh, which is the normal case for a production run that does not want to spend its
first hundred steps refining up from a coarse sphere.

**Why it matters.** The `distribute()` path also constrains the *shape* of any
non-icosphere initial mesh: a caller supplying its own geometry (a lat/lon sphere,
a scanned surface, a mesh from another code) must materialize the whole thing on
every rank first. The driving consumer (the Beatnik z-model) builds its own initial
surfaces and would like to build them in parallel; the replication requirement is
the reason it currently cannot.

## Approach

Two deliverables. The second is the general capability; the first is the concrete
case that proves it.

### Deliverable A — `buildFromTriangleSoupDistributed()`

New header `src/Tessera_DistributedBuilder.hpp`.

```cpp
//! Build a distributed mesh from PER-RANK LOCAL triangle patches, with no rank
//! ever holding the global mesh.
//!
//! The caller supplies its own patch of triangles and, for each of its local
//! vertices, a CANONICAL KEY: a rank-independent 128-bit identifier that is equal
//! on two ranks exactly when they mean the same vertex. Tessera dedups by key,
//! assigns globally unique contiguous gids, derives the unique edge set, resolves
//! ownership, and builds the ghost layer and halo plans.
//!
//! Patches must cover the surface with no gaps; overlap is allowed and is resolved
//! by the key dedup, so a caller may generate a patch plus a boundary ring.
//! Collective.
template <class MeshT, class Scalar>
void buildFromTriangleSoupDistributed(
    MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
    const TriangleSoup<Scalar>& localSoup,
    const std::vector<VertexKey>& localVertexKeys,  //!< one per local vertex
    int haloDepth = 1 );
```

**The canonical key is the whole design.** Deduplicating shared vertices is the one
thing that genuinely needs global agreement, and a position-based dedup would need
a tolerance and would be non-reproducible. Requiring the caller to supply a key
pushes the problem to where the answer is known for free: a generator always knows
*why* two patches share a vertex. Reuse the existing `EdgeKey` machinery's shape for
`VertexKey` (a pair of `GlobalId`, with `makeVertexKey`), which covers the two cases
that matter — a base vertex (`{i, invalid_gid}`) and a midpoint of two vertices
(`{min(a,b), max(a,b)}`, recursively resolvable) — and let a caller with a different
scheme hash into it.

Implementation, all of it a reuse of machinery that already exists:

1. **Dedup and gid assignment.** Route each local vertex key to
   `detail::gidCoordRank`-style coordinator by key hash, via `allToAllV`. The
   coordinator sees every rank claiming that key, picks the owner by the lowest-rank
   rule, counts its owned vertices, `MPI_Exscan`es a contiguous global block, and
   replies `(key → gid, owner)` to every claimant. Exactly `refine()` Phase 2's
   pattern with midpoint keys replaced by vertex keys.
2. **Edge derivation, locally.** Each rank derives the unique edges of **its own**
   triangles with a local `std::map<EdgeKey, int>` — `O(local)` memory, which is the
   point. Edge gids and ownership go through `detail::edgeCoordRank` exactly as in
   step 1, so an edge shared across a patch boundary gets one gid agreed by both
   sides.
3. **Face gids** by `MPI_Exscan` over local triangle counts, after dropping
   duplicate triangles (a face is owned by the lowest rank claiming its `FaceKey`,
   which the existing `faceKeys` machinery already defines).
4. **Assemble owned-first AoSoAs**, set counts, build the CSRs — the same Round D
   assembly `migrate()` performs; factor it into a shared `detail` helper rather
   than a third copy.
5. **Ghost layer and halo** by calling `rebuildHalo( mesh, halo )`
   (`Tessera_HaloRebuild.hpp`) — the general, non-replicated ghost builder, which
   is exactly what this path needs and which `25980f2` made reachable from outside
   `migrate()`. Pass `haloDepth` once [halo-depth.md](halo-depth.md) has landed;
   until then the `haloDepth` parameter accepts only 1.
   Note that `rebuildHalo()`'s **round B resolves ownership by communication**, so
   this builder does not need the replicated-mesh ownership shortcut `distribute()`
   relies on — that is the structural reason the whole task is now feasible.

`distribute()` is then **not** on this path at all, and `buildFromTriangleSoup()`
stays exactly as it is for the replicated/serial case (it is the right tool for a
small mesh and for every existing test).

### Deliverable B — `buildIcosphereDistributed()`

```cpp
//! Generate and build an icosphere of the given subdivision level with no rank
//! materializing the global mesh. Collective.
template <class MeshT>
void buildIcosphereDistributed( MeshT& mesh,
                                MeshHalo<typename MeshT::memory_space>& halo,
                                int subdivisions, int haloDepth = 1 );
```

**Decision — partition by the subdivision tree, not by an axis sort.**
`facePartitionByAxis()` sorts every face centroid globally, which requires every
centroid, which requires the global mesh — the exact thing being avoided. The
icosphere is generated by a deterministic 1→4 recursion, so face index space at
depth `s` is a perfect 20-ary-then-4-ary tree: face `f` at depth `s` descends from
base face `f / 4^s`. Assign rank `r` the contiguous index range
`[r * F/P, (r+1) * F/P)` and generate **only those faces** by descending only the
subtrees that intersect the range. Memory and time are `O(F/P + depth)` per rank.

This gives a hierarchical, locality-preserving partition (children of a base face
stay together), which is a *better* starting partition than a single-axis sort, and
it needs no communication and no global sort. For `size > 20` the range simply cuts
inside a base patch, which is fine.

Vertex canonical keys come free from the recursion: a base vertex is
`{i, invalid_gid}`, a midpoint is `{min(parentKeyGid), max(...)}` resolved to the
already-assigned gids of its two parents — so keys are assigned level by level,
one coordinator round per subdivision level. That is `subdivisions` extra
collectives at setup, which is nothing.

**Reproducibility requirement:** the generated positions must be **bitwise
identical** to `generateIcosphere()`'s for the same subdivision, so the two paths
are interchangeable. That means the same base table, the same
`(v_a + v_b)` then `normalize3` order of operations, and the same `double`
intermediate precision regardless of `Scalar`. Do not restructure the arithmetic.

### Non-goals

- Replacing `buildFromTriangleSoup()` or `distribute()`. Both remain, and every
  existing test keeps using them.
- A distributed **reader** for an arbitrary mesh file. `readMesh` is separate.
- Load balancing the initial partition (see
  [distributed-loadbalance-solve.md](distributed-loadbalance-solve.md)); the
  subdivision-tree partition is a starting point, and `loadBalance()` refines it.

### Tests

New `tests/test_distributed_build.cpp`, registered at **TIER `regression`**,
backends **SERIAL and HIP**, ranks **`${TESSERA_TEST_MPI_RANKS}` (1–5)**. Gate
promotion is **pre-authorized for this task**.

1. **Counts and invariants.** `buildIcosphereDistributed` at subdivision 2:
   `globalOwnedVertices == 162`, `globalOwnedEdges == 480`,
   `globalOwnedFaces == 320`, `globalOwnedEuler == 2`,
   `checkOwnershipPartition`, `owned1RingLocal`, `checkConforming` all pass, at
   every rank count.
2. **Equivalence with the replicated path — the definitive check.** Build the same
   subdivision level twice: once via `buildIcosphere` + `distribute`, once via
   `buildIcosphereDistributed`. Gid numbering differs (different partitions), so
   compare the gid-independent identity: the **vertex position multiset** sorted
   lexicographically must be **bitwise equal**, and so must the multiset of face
   corner-position triples. Run at subdivisions 1, 2 and 3 and at every rank count.
3. **Nobody holds the global mesh.** At ranks ≥ 2 and subdivision 5 (10 242
   vertices, 20 480 faces): assert `mesh.numVertices() < globalOwnedVertices` and
   `mesh.numFaces() < globalOwnedFaces` on **every** rank, and further that
   `numOwnedFaces() <= 2 * globalOwnedFaces / size`. This is the property the task
   exists to establish; without it the test suite cannot tell the new path from the
   old one. Print the per-rank local counts so the numbers are in the log.
4. **Reproducibility across rank counts.** The position multiset from check 2 is
   identical at ranks 1–5, bitwise.
5. **Halo correctness.** `haloExchange()` leaves every ghost vertex position equal
   to its owner's; `halo.depth` is as requested; at `haloDepth = 2` the owned
   2-rings are complete against the replicated reference (the same check as
   [halo-depth.md](halo-depth.md) check 1, here proving the new builder feeds
   `rebuildHalo` correctly).
6. **`refine()` works on the result.** Uniform refine of a distributed-built mesh:
   `V' = V+E`, `E' = 2E+3F`, `F' = 4F`, `checkConforming`,
   `checkMidpointAgreement`, `check21Balance`. The new builder must produce a mesh
   indistinguishable from `distribute()`'s output as far as every downstream
   operation is concerned.
7. **`migrate()` and `loadBalance()` work on the result.** Identity migrate then a
   real `loadBalance`; invariants hold.
8. **I/O round trip.** `writeMesh` then `readMesh`; the position multiset is
   unchanged, per `test_io.cpp`'s idiom.
9. **Deliverable A directly, with a non-icosphere.** Hand-build two overlapping
   patches of a small hand-written surface (e.g. an octahedron split into two
   4-face patches with a shared boundary ring), supply canonical keys, and assert
   the resulting mesh equals the same octahedron built via
   `buildFromTriangleSoup` + `distribute` by the position-multiset criterion.
   Exercises overlap resolution, which the icosphere path (disjoint ranges) does
   not.
10. **Key collision is caught.** Supply two genuinely different vertices with the
    same canonical key; assert it throws naming the key, rather than silently
    welding them. A caller bug that must not be silent.
11. **Degenerate configurations.** `size > numFaces` (e.g. 5 ranks, subdivision 0
    with 20 faces is fine — use a 2-face patch case instead) so some rank owns zero
    faces: the build completes, that rank holds zero owned entities, and the
    collectives do not deadlock.

## Exit criterion

- `test_distributed_build` green at **SERIAL and HIP, ranks 1–5**, and the full
  gate still green with nothing relabelled.
- Check 2 passes **bitwise** at subdivisions 1–3 — the two builders are
  interchangeable.
- Check 3 passes, with the per-rank local counts at subdivision 5 recorded in the
  progress log. That measurement is the deliverable: state the observed peak local
  vertex count per rank versus the global count.
- README API section documents `buildFromTriangleSoupDistributed`,
  `buildIcosphereDistributed`, `VertexKey`, and the **canonical-key contract**
  (rank-independent, equal iff the same vertex, collisions throw).
- README notes that `buildIcosphere` + `distribute` remains supported and is the
  right choice for a small initial mesh.
- `docs/design.md` gains a *Distributed initial construction* subsection with the
  subdivision-tree partition decision and why an axis sort cannot be used here.

## Where this sits

**No hard prerequisite** — `rebuildHalo()` already exists in tree
(`Tessera_HaloRebuild.hpp`, `25980f2`), which is what makes step 5 possible today;
[halo-depth.md](halo-depth.md) is implemented. It is needed for the `haloDepth = 2` half of
check 5. Nothing depends on this task. See the ordering diagram in
[halo-depth.md](halo-depth.md).

## Progress log

- 2026-08-07 — Task written, then re-checked against `08dd346` after pulling
  `../tessera`. Nothing implemented.
- 2026-08-10 — **Implemented and green.** `src/Tessera_DistributedBuilder.hpp`
  (`VertexKey`, `makeVertexKey`, `buildFromTriangleSoupDistributed`,
  `buildIcosphereDistributed`), registered in `Tessera.hpp`; five new profiling
  keys; `tests/test_distributed_build.cpp` at TIER `regression`, SERIAL and HIP,
  ranks 1–5 (pre-authorized promotion). **10/10 green on the first run**, and the
  full gate re-run clean afterwards with nothing relabelled.

  All eleven checks pass at every rank count on both backends and on both
  execution spaces, non-vacuously:

  * **Check 2/4 — the definitive one — passes bitwise at subdivisions 1, 2 and 3.**
    The vertex position multiset checksum is byte-identical to the replicated
    reference's and to itself at np1–np5: `fcbcbf9882b5278b` (subdiv 1),
    `2d164c6173f3535b` (2), `8de0e93be5228d3b` (3); face corner-triple checksums
    `0655797e641a38c7` / `d3a9c719aab373bf` / `b0a9373bb99472ff`, likewise
    identical everywhere. The two builders are interchangeable.
  * **Check 3 — the measurement that is the deliverable.** At subdivision 5,
    global V = 10 242 and F = 20 480. Worst per-rank **local** counts:

    | ranks | peak local V | peak local F | peak owned F |
    |---|---|---|---|
    | 1 | 10 242 | 20 480 | 20 480 |
    | 2 | 5 591 | 10 870 | 10 240 |
    | 3 | 4 097 | 7 616 | 6 827 |
    | 4 | 3 110 | 5 750 | 5 120 |
    | 5 | 2 465 | 4 603 | 4 096 |

    So at five ranks the peak local vertex count is **2 465 against a global
    10 242 — 24%**, the residual over `1/P` being the ghost ring, and no rank ever
    holds the global mesh. np1 holds everything because there is nothing to
    distribute, which is why the assertion is guarded at `size >= 2`.
  * **Check 5** is non-vacuous at ranks ≥ 2 — 30 (depth 1) and 50 (depth 2) ghost
    positions deliberately corrupted at np2 and restored by `haloExchange()` — and
    the `haloDepth = 2` half confirms the builder feeds `rebuildHalo()` correctly
    at depth > 1, with every owned vertex's 2-ring exact against the
    global-adjacency reference.
  * Checks 1, 6, 7, 8: counts/invariants, uniform `refine()` (162→642, 480→1920,
    320→1280), identity `migrate()` + a real `loadBalance()`, HDF5 round trip.
  * Check 9: the octahedron in two **overlapping** patches (V=6, E=12, F=8),
    equal to the replicated build by the position-multiset criterion. Check 10:
    the key collision throws naming the key at every rank count. Check 11: a
    2-face patch supplied identically by every rank, so ranks 1..P−1 own zero
    entities and the collectives still complete.

  **Two deliberate deviations from the plan above**, both noted here rather than
  silently:

  1. **Face gids** come from an `MPI_Exscan` over the deduplicated owned counts as
     specified, but face *ownership* is resolved by a `FaceKey` coordinator round
     first (routing by a `faceKeyCoordRank` hash, lowest claimant wins). The plan
     implied ownership fell out of `faceKeys` locally; it cannot, because two ranks
     holding the same triangle have no way to agree who keeps it without one round.
     That round costs the same two `allToAllV` calls a combined gid+owner reply
     would, and keeping the exscan preserves per-rank contiguous face gids.
  2. **Step 4 was not factored into a shared `detail` helper** with `migrate()`'s
     round D, because it turned out not to be a third copy of it: this builder only
     materializes the **owned** entities and hands everything else — the ghost
     rings, the key Views, the CSRs, the plans, the canonical owned-first ordering
     — to `rebuildHalo()`'s round D verbatim. The assembly that remains is
     ~70 lines of "write the owned tuples", with no overlap worth extracting.

  `haloDepth` is forwarded to `rebuildHalo()` in full (halo-depth.md has landed),
  so the `haloDepth = 2` half of check 5 is live rather than deferred.
  README gains a *Distributed initial construction* subsection documenting both
  entry points, `VertexKey`, the three-property canonical-key contract, and the
  statement that `buildIcosphere` + `distribute` remains supported and is the right
  choice for a small initial mesh; `docs/design.md` gains the matching design
  section with the subdivision-tree partition decision and why an axis sort cannot
  be used here.
