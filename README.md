# Tessera

A distributed, MPI- and GPU-aware **unstructured triangle-mesh library** built on
[Cabana](https://github.com/ECP-copa/Cabana) and
[Kokkos](https://github.com/kokkos/kokkos).

Tessera provides the local, halo-able mesh machinery for evolving closed surfaces:
entity storage, full connectivity, a hand-built MPI halo of configurable depth,
split-based adaptive refinement, optional migration/load-balancing, and parallel
Paraview-readable I/O. It is a dependency of the [Canopy](https://github.com/) FMM
solver and the Beatnik/rocketrig Rayleigh–Taylor problem; the global
Birkhoff–Rott/FMM velocity solve lives downstream in Canopy and is **not** part of
Tessera.

> **Status: early development — design reference.** This README and
> [docs/design.md](docs/design.md) document the *design* of milestone 1 (a single
> rising bubble). The API below is the target the
> implementation is being built against (see `plans/compiled-hopping-torvalds.md`
> and the `tasks/completed/milestone1_mesh.md` handoff contract). Sections marked *(planned)*
> are not yet implemented.

---

## Documentation

Detailed descriptions of the algorithms and design decisions used in this
repository — entity storage and AoSoA layout, global ID scheme, ownership and the
halo (including halo depth), adaptive refinement, load balancing, and parallel I/O
— live in **[docs/design.md](docs/design.md)**. This README covers the public API, build
instructions, and per-example arguments; the design rationale is not duplicated
here.

Per-system build/run instructions (one file per machine) live under
[systems/](systems/).

---

## Usage / API

The full pipeline below is implemented and gate-tested end to end (see
`examples/02_mesh_pipeline/` for a complete, runnable version with CLI args).

```cpp
#include <Tessera.hpp>

using namespace Tessera;
using MeshT = Mesh<double, /*Dim=*/3, VertexFields<>, EdgeFields<>, FaceFields<>,
                   MemSpace, ExecSpace>;
// An optional 8th parameter selects the refinement conformity contract:
//   ..., ExecSpace, RefinementMode::Conforming>        // default: no T-junctions
//   ..., ExecSpace, RefinementMode::HangingNode2to1>   // opt-in: 2:1 hanging nodes
// See docs/design.md → Adaptive refinement → Refinement modes.

MeshT mesh( MPI_COMM_WORLD );
buildIcosphere( mesh, /*subdivisions=*/3 );        // initial coarse closed surface,
                                                     // replicated on every rank. For a
                                                     // LARGE initial mesh, see
                                                     // "Distributed initial construction"
                                                     // below -- buildIcosphereDistributed()
                                                     // replaces these next three calls and
                                                     // never replicates anything

auto faceOwner = facePartitionByAxis( mesh, /*axis=*/2 );  // deterministic geometric
                                                             // partition of the faces
MeshHalo<MemSpace> halo;
distribute( mesh, halo, faceOwner, /*depth=*/1 );   // cut to owned + a `depth`-deep ghost
                                                     // layer, build the halo exchange
                                                     // plans. Pass depth=k once, at setup,
                                                     // for a k-ring operator; refine() and
                                                     // migrate() PRESERVE it thereafter
haloExchange( mesh, halo );                         // fill the ghost layer

auto vort = mesh.vertexSlice<Tessera::VertexField::Vorticity>();   // typed Cabana slice
// ... fill/evolve owned vertices ...
haloExchange( mesh, halo );                         // refresh ghosts (whole field pack)

refine( mesh, halo, face_refine_mask );             // 2:1-balanced red split, plus the
                                                     // conforming closure in the default
                                                     // mode; REBUILDS `halo` on the way
                                                     // out, at its recorded depth, so it
                                                     // may be called again immediately and
                                                     // haloExchange() is meaningful
                                                     // straight afterwards

// mesh.haloDepth() reports the depth in force. rebuildHalo( mesh, halo, depth ) rebuilds
// the ghost layer in place from the owned entities -- refine()/migrate() call it
// themselves, so it is only needed when something else changed the owned set.

splitEdges( mesh, halo, edge_split_mask );          // bisect EXACTLY the marked edges (an
                                                     // EDGE mask, sized numOwnedEdges());
                                                     // every incident face becomes 2, 3 or
                                                     // 4 children, conforming on exit with
                                                     // no closure and no 2:1 pass. Also
                                                     // rebuilds `halo`. REMESH family --
                                                     // see "Editing families" below

FlipResult fr = flipEdges( mesh, halo, edge_flip_mask );
                                                     // swap the diagonal of each marked
                                                     // edge's quad. V, E and F UNCHANGED and
                                                     // every gid preserved. At most an
                                                     // INDEPENDENT SET per call, so a caller
                                                     // loops on fr.accepted. Also rebuilds
                                                     // `halo`. REMESH family -- see "Edge
                                                     // flip" below

CollapseResult cr = collapseEdges( mesh, halo, edge_collapse_mask );
                                                     // merge each marked edge's two
                                                     // endpoints: V-1, E-3, F-2 per accepted
                                                     // collapse, so Euler is preserved. The
                                                     // ONLY operation that removes degrees of
                                                     // freedom. REQUIRES haloDepth() >= 2 and
                                                     // throws otherwise. At most an
                                                     // INDEPENDENT SET per call, so a caller
                                                     // loops on cr.accepted. Compacts and
                                                     // rebuilds `halo`. REMESH family -- see
                                                     // "Edge collapse" below

loadBalance( mesh, halo );           // internal Zoltan2 rebalance + halo rebuild (optional).
                                     // Nothing is gathered to rank 0 -- see "Load
                                     // balancing modes" below for the three modes and
                                     // the trade-off between them
// or, external (e.g. Canopy-driven):
//   auto c = ownedFaceCentroids( mesh );
//   std::vector<Rank> dest = external_partition( c );
//   migrate( mesh, halo, dest );   // also rebuilds the halo
haloExchange( mesh, halo );                         // re-sync the field pack (refine() and
                                                     // migrate() leave ghost values already
                                                     // equal to the owners')

writeMesh( mesh, "bubble_0000" );    // bubble_0000.h5 + bubble_0000.xmf; returns an
                                     // XdmfFrame describing the frame (discardable)
// writeMesh( mesh, "bubble_0001", 1.5 );  // same, plus <Time Value="1.5"/> in the .xmf
```

#### Time series: one Paraview dataset instead of N

A run that calls `writeMesh()` once per frame produces N unrelated `.xmf` files:
Paraview opens each separately, with no time slider over the sequence. `MeshSeries`
adds **one master `.xmf`** — an XDMF temporal collection — which Paraview opens as a
single dataset with one timestep per frame, animatable, with the same attribute set
selectable at every step. Per-frame `.h5` and `.xmf` files are still written and
unchanged, so `readMesh()` still reads any individual frame and a single frame stays
individually inspectable.

```cpp
#include <Tessera.hpp>   // Tessera_XdmfSeries.hpp

Tessera::MeshSeries series( "bubble" );          // -> bubble.xmf, the file to open
for ( int step = 0; step < nsteps; ++step )
{
    // ... refine / migrate / advance ...
    series.write( mesh, "bubble_" + pad( step ), t );   // collective on mesh.comm()
}
```

- `time` is the caller's own quantity (physical time, or the step index if there is
  no physical time) and must be **strictly increasing** across a series.
- The master is **rewritten after every frame** (via a temp file and `rename`), so a
  run killed at frame 40 leaves a valid master with 40 timesteps rather than a
  missing or truncated file.
- The master must sit in the **same directory** as every frame it names, because an
  `.xmf` references its `.h5` by basename. `write()` throws `std::runtime_error` on a
  frame stem in a different directory, and on a non-increasing `time`.
- In Paraview, only the **temporal** XDMF3 reader (`Xdmf3ReaderT`) walks a temporal
  collection. If the master opens showing a single timestep, check the reader choice
  first: `grep -c '<Time Value=' bubble.xmf` tells you what the file actually holds.

### Halo: gather and scatter-add

`haloExchange()` is a **gather** — owner → ghost, overwrite, the whole AoSoA tuple
at once. `haloScatterAdd()` is its **reverse** — ghost → owner, `+=`, one named
field per call — and it is what makes distributed assembly correct. Any per-vertex
quantity assembled by iterating **owned faces** (vertex areas, vertex normals, a
face→vertex gradient scatter, a per-element residual) leaves the owner of a
partition-boundary vertex holding only a *partial* sum, with every ghost copy
holding a different partial; the reverse accumulate pushes those partials home.

```cpp
#include <Tessera.hpp>   // Tessera_HaloScatterAdd.hpp

constexpr std::size_t Area = Tessera::userVertexField<0>();   // e.g. Scalar
// ... loop OWNED faces, adding each face's contribution to its three corners'
//     LOCAL slots (a corner of an owned face may be a ghost) ...
haloScatterAddVertices<Area>( mesh, halo );   // owned slots now hold the global sum
haloExchange( mesh, halo );                   // ONLY if ghosts are read downstream

// Edge and face fields via the sibling wrappers, or the plan-level primitive:
haloScatterAddEdges<Tessera::userEdgeField<0>()>( mesh, halo );
haloScatterAddFaces<Tessera::userFaceField<0>()>( mesh, halo );
haloScatterAdd<Field>( mesh.comm(), mesh.vertices(), halo.vplan );
```

`FieldIndex` is any Cabana member index — a core field
(`VertexField::Position`) or a user field — of scalar or fixed-array
(`Scalar[3]`, accumulated componentwise) shape. Unlike `haloExchange(mesh, halo)`
it is **not** an all-three-kinds call: an accumulate names a field, and a field
belongs to one entity kind. Three contract properties:

1. **Ghost slots are left untouched.** Afterwards an owned entry holds the
   complete global sum while every ghost copy still holds that rank's local
   partial, so the mesh is **not halo-consistent** for that field — follow with
   `haloExchange()` if downstream kernels read ghosts. Ghosts are deliberately
   not zeroed inside the call, so a caller that only reads owned values does not
   pay for a second collective.
2. **Calling it twice double-counts.** It is not idempotent, precisely because of
   (1): the second call re-sends the same ghost partials. This is the standard
   scatter-add contract, and it is pinned by the test.
3. **The summation order is fixed by peer order, not by rank count.** Peers are
   visited in ascending rank on both sides and the accumulate is serialized per
   peer (one kernel per peer, which is also how the unpack avoids atomics), so
   within one run the floating-point result is deterministic and bitwise
   reproducible. It is **not** bitwise identical across rank counts, because the
   partition into partial sums differs — do not write a cross-rank bitwise
   comparison of an assembled field (the same caveat as `globalSum`).

Not provided: a caller-supplied reduction operator (min/max/custom) and
multi-field packing. One field per call; three fields is three calls.

### Load balancing modes

`loadBalance()` and `computeLoadBalance()` take a `LoadBalanceMode` selecting how
the Zoltan2 geometric solve is run. All three use **`Zoltan2` MultiJagged and never
`rcb`** — Zoltan2's deterministic RCB breaks on Tuolumne, and that finding stands
regardless of which communicator the solve runs over.

```cpp
#include <Tessera.hpp>   // Tessera_Zoltan2Balancer.hpp

// dest only -- computes nothing else, moves nothing:
std::vector<Rank> dest = computeLoadBalance( mesh, /*imbalanceTolerance=*/0.05,
                                             LoadBalanceMode::Sampled );
// compute + migrate, with optional instrumentation:
LoadBalanceStats st;
MigrateStats ms = loadBalance( mesh, halo, 0.05, LoadBalanceMode::Distributed, &st );
// st.rootSolveFaces -- how many faces rank 0 received for its solve
// st.cuts           -- Sampled: the broadcast cut structure, identical on every rank
```

| Mode | Rank-0 solve input | Reproducible run to run | Partition quality |
|---|---|---|---|
| `GatherRoot` | the **global** face count | yes | best (one solve over every face) |
| `Distributed` | **zero** — one solve over a `Teuchos::MpiComm` on `mesh.comm()` | **no** (see *Known Issues*) | best |
| `Sampled` *(default)* | `O(comm size)` — a `64 · nparts` gid-order sample | yes, and partition-independent | slightly looser (cuts fitted to a sample) |

`GatherRoot` is the original behaviour, kept as the reference implementation and
the fallback: it `MPI_Gatherv`s every rank's centroids/weights to rank 0, solves
the whole problem there over a `Teuchos::SerialComm`, and `MPI_Scatterv`s the
assignment back. Rank 0's memory and solve time therefore scale with the
**global** face count — a memory ceiling on one rank, not a distributed cost.

`Distributed` removes that entirely: each rank feeds the adapter its **own**
owned-face centroids/weights, keyed by the real face gids (already globally
unique), and `getPartListView()` comes back in this rank's owned-face order with
nothing to scatter. It is the right choice when partition quality matters more
than bitwise repeatability.

`Sampled` is the default. Each rank selects the owned faces whose gid satisfies
`gid % stride == 0` — a rule on **globally agreed gids**, so the sampled *set* is
a property of the mesh and not of its current partition — rank 0 gathers the
`O(nparts)` sample, **sorts it by gid** (the gather arrives in rank order, which
is partition-dependent), solves it, and broadcasts MultiJagged's axis-aligned
per-part boxes. Every rank then classifies its own centroids against those boxes
in exact local arithmetic: inside one box, that part; inside several (exactly on
a cut), the lowest part id; inside none, the nearest box. Consequently the cuts
are bit-identical across two different starting partitions of the same mesh, and
the assignment is reproducible by construction.

The returned `dest` is in exactly the order `ownedFaceCentroids/Gids/Weights`
produce, in every mode, because `migrate()` depends on that.

### Mesh generators

Tessera ships two closed-surface generators. Both produce a **triangle soup** —
a flat `positions` array plus a flat per-face `triangles` index array — which
`buildFromTriangleSoup()` turns into a fully-connected replicated mesh. The
`build*` wrappers do both steps. A caller is free to supply its own soup
instead; the soup interface exists precisely for that.

```cpp
TriangleSoup<double> s1 = generateIcosphere<double>( /*subdivisions=*/3 );
TriangleSoup<double> s2 = generateLatLonSphere<double>( /*nLat=*/33, /*nLon=*/64 );
buildFromTriangleSoup( mesh, s2 );          // or, in one step:
buildLatLonSphere( mesh, /*nLat=*/33, /*nLon=*/64 );
buildIcosphere( mesh, /*subdivisions=*/3 );
```

| Generator | Counts | Character |
|---|---|---|
| `generateIcosphere(n)` / `buildIcosphere` | `V=12, E=30, F=20` at `n=0`; each level `V'=V+E, E'=2E+3F, F'=4F` | Nearly isotropic, nearly equilateral, almost every vertex valence 6 |
| `generateLatLonSphere(nLat,nLon)` / `buildLatLonSphere` | `V = 2 + (nLat−2)·nLon`, `F = 2·nLon·(nLat−2)`, `E = 3·nLon·(nLat−2)` | **Anisotropic** by construction; the two poles have valence `nLon` |

**Why both.** An icosphere is *too good* a test surface. A lat/lon sphere
stretches its triangles toward the poles and puts two valence-`nLon` outliers on
the surface, so it exercises what an icosphere never reaches: quality-based
marking, the cotangent weight at a high-valence vertex, stencil rows of very
different lengths, and the poles as valence outliers. At `(33,64)` the measured
max/min triangle-area ratio is **10.21** and the max/min edge-length ratio
**14.41** (against ~1 for an icosphere).

`generateLatLonSphere` parameters and guarantees:

- **`nLat`** is the number of latitude rings **including both poles**, `nLat >= 3`;
  **`nLon`** is the number of meridians, `nLon >= 3`. Anything smaller throws
  `std::invalid_argument`. `nLat == 3` is the degenerate-but-legal bipyramid:
  two pole fans, no interior quads, `2·nLon` faces.
- **Vertex ordering** is deterministic and documented, so a consumer can address
  a vertex arithmetically: index `0` is the north pole, then ring
  `j = 1 .. nLat−2` each contributing `nLon` vertices in ascending `i` — so
  `ring(j,i) == 1 + (j−1)·nLon + i` — then the south pole last, at `V−1`.
  Positions are
  `(sin θ·cos φ, sin θ·sin φ, cos θ)` with `θ_j = πj/(nLat−1)` and
  `φ_i = 2πi/nLon` for `i = 0 .. nLon−1` — **no seam duplicate**; `i` wraps.
- **The poles are exact**, written literally as `(0,0,+1)` and `(0,0,−1)`.
  `sin(π)` is not zero in floating point, so a south pole evaluated from the
  formula is off the unit sphere in its last bits *and different for different
  `φ`* — which is how a duplicate-pole bug hides.
- **Winding** is CCW seen from outside, matching `generateIcosphere()`.
- **The quad diagonal is fixed:** each interior quad
  `(i,j)-(i+1,j)-(i+1,j+1)-(i,j+1)` is split along the `(i,j+1)-(i+1,j)`
  diagonal, the same way around the whole sphere. Not adaptive — a caller
  wanting a different triangulation flips edges. This convention is what fixes
  the valence histogram: `2` poles at `nLon`, `2·nLon` vertices at valence 5
  (rings `1` and `nLat−2`), `(nLat−4)·nLon` at valence 6, and for `nLat == 3` a
  single ring of `nLon` valence-4 vertices.
- **Reproducibility caveat.** Unlike the icosphere, whose positions come from a
  rational base table plus `sqrt`, these come from `sin`/`cos` at computed
  angles, which may differ in the last bit across libm implementations and
  platforms. Positions are **not** guaranteed bit-reproducible across machines,
  so compare against a gold file generated elsewhere with a tolerance. Within
  one run they are identical on every rank, which is what the replicated coarse
  build needs.

A **distributed** lat/lon generator is a natural follow-on now that
`buildFromTriangleSoupDistributed` exists (the canonical key is just
`makeVertexKey( j·nLon + i )`); it is deliberately not built here. Other
generators (torus, plane, cylinder, cube-sphere) on demand.

### Distributed initial construction

Everything above builds the initial mesh **in full on every rank** and then cuts
it: `buildFromTriangleSoup()` is serial and takes a *replicated* soup, and
`distribute()` can compute ownership locally only *because* the mesh is
replicated. Peak memory per rank is therefore proportional to the **global** mesh
size, paid on every rank at once. That is fine for a small coarse sphere and is a
hard ceiling when the initial mesh's resolution should be comparable to the
running refined mesh. Two entry points remove the replication requirement
(`Tessera_DistributedBuilder.hpp`); **no rank ever holds the global mesh**, and
`distribute()` is not on this path at all.

```cpp
#include <Tessera.hpp>   // Tessera_DistributedBuilder.hpp

MeshT mesh( MPI_COMM_WORLD );
MeshHalo<MemSpace> halo;

// (B) the icosphere, generated and built in parallel -- a drop-in replacement for
//     buildIcosphere() + facePartitionByAxis() + distribute():
buildIcosphereDistributed( mesh, halo, /*subdivisions=*/5, /*haloDepth=*/1 );
haloExchange( mesh, halo );          // halo plans are valid on return

// (A) the general capability: THIS RANK'S OWN patch, plus a canonical key per
//     local vertex. Patches must COVER the surface; OVERLAP is fine.
TriangleSoup<double> myPatch = my_generator( mesh.rank(), mesh.commSize() );
std::vector<VertexKey> keys = my_keys( myPatch );          // one per local vertex
buildFromTriangleSoupDistributed( mesh, halo, myPatch, keys, /*haloDepth=*/1 );
```

**The canonical-key contract is the whole interface, and it is three properties.**
`VertexKey` is the same shape as `EdgeKey` — a sorted pair of `GlobalId`, 128 bits,
structured rather than hashed:

```cpp
VertexKey base = makeVertexKey( i );        // == { i, invalid_gid }
VertexKey mid  = makeVertexKey( a, b );     // == { min(a,b), max(a,b) }
```

1. **Rank-independent.** Two ranks meaning the same vertex compute the same key
   with no communication. Deduplicating a shared vertex is the one thing that
   genuinely needs global agreement, and a position-based dedup would need a
   tolerance and would not be reproducible — so the *caller* supplies the key,
   because a generator always knows *why* two patches share a vertex.
2. **Equal iff the same vertex.** Patches must **cover** the surface with no gaps;
   **overlap is allowed** and is resolved by the key dedup, so a caller may
   generate a patch plus a boundary ring. A duplicated *triangle* is likewise kept
   by the lowest rank claiming its `FaceKey` and dropped by the others, so seams
   need no coordination.
3. **Collisions throw.** Two vertices given the same key at *different* positions
   are a caller bug, not a weld: the key coordinator compares every claimant's
   position **bitwise** and every rank throws a `std::runtime_error` naming the
   key. Welding them would produce a mesh that passes every structural invariant
   and is geometrically wrong.

Postconditions are exactly `buildIcosphere` + `distribute`'s, so every downstream
operation is indifferent to which builder ran: globally unique agreed gids,
ownership partitioning each entity kind, the canonical owned-first-then-ghost
layout, and valid halo plans (`haloDepth` is forwarded to `rebuildHalo()`, and
`refine()`/`migrate()` preserve it). Gid **numbering** differs — a different
partition numbers differently — while the gid-independent identity, the vertex
position multiset and the face corner-position multiset, is **bitwise identical**
to the replicated path at every rank count.

`buildIcosphereDistributed` partitions **by the subdivision tree, not by an axis
sort**: `facePartitionByAxis()` needs every centroid, hence the global mesh, which
is the thing being avoided. See `docs/design.md` → *Distributed initial
construction*.

**`buildIcosphere` + `distribute` remains fully supported, and is the right
choice for a small initial mesh.** It is simpler, needs no keys, and is what every
pre-existing test uses; the replication only becomes a problem when the coarse
mesh is large. Not provided: a distributed reader for an arbitrary mesh file
(`readMesh` is separate), and load balancing of the initial partition — the
subdivision-tree partition is a locality-preserving starting point and
`loadBalance()` refines it.

### Editing families

Tessera has **two disjoint families of topological edit, and a mesh belongs to
exactly one of them**:

| Family | Operations | Invariant maintained | `Level` semantics |
|---|---|---|---|
| **Hierarchical** | `refine()`, `refineLocal()` | 2:1 level balance; conforming closure | `Level` is **authoritative** |
| **Remesh** | `splitEdges()`, `flipEdges()`, `collapseEdges()`, `compact()`, `compactAndRenumberGids()` | conformity and manifoldness only | `Level` is **advisory** |

The remesh family is **complete**: split, flip and collapse are the three
edge-addressed edits a metric-driven remesher is built out of, and `compact()`
is what makes the third of them possible.

`refine()`'s whole design rests on the level model, and that model is coherent
only because `refine()` performs the uniform 1→4 red split. Bisecting *one* edge
of a triangle produces two children whose edges have mixed levels: no single
integer describes them, and a 2:1 level-difference invariant is not the right
statement about the result. So a `splitEdges()` child **inherits its parent's
level**, and the mesh is thereafter not 2:1-level-meaningful.

Interleaving the two on one mesh is therefore **unsupported, and enforced rather
than documented**. The mesh carries an `EditFamily` tag (`mesh.editFamily()`),
`None` until its first topological edit and then fixed; each entry point throws a
`std::runtime_error` naming **both** families when the tag disagrees:

> `Tessera::refine: this mesh belongs to the Remesh
> (splitEdges/collapseEdges/flipEdges/compact) editing family, and refine belongs
> to the Hierarchical (refine/refineLocal) family. The two are DISJOINT and must
> not be interleaved on one mesh: the hierarchical family maintains the 2:1 level
> balance and the conforming closure with Level authoritative, while the remesh
> family maintains conformity and manifoldness only and leaves Level advisory (a
> child inherits its parent's level). Build a fresh mesh for the other family.`

A one-line check that turns a subtle wrong answer into an immediate abort.
Extending the level model to anisotropic bisection (per-edge levels with a
compatible balance rule) is a much larger design that no known consumer needs; it
is recorded under *Future Optimizations* below, not attempted. See
`docs/design.md` → *Edge-addressed splitting* and
[tasks/completed/edge-split.md](tasks/completed/edge-split.md).

### Edge flip

`flipEdges()` (`Tessera_EdgeFlip.hpp`) replaces the diagonal of the quad formed
by the two faces incident on each marked edge. It is the cheapest of the remesh
operations and the one that does the most for element quality per unit cost:
**V, E and F are all unchanged and only connectivity moves.**

```cpp
#include <Tessera.hpp>   // Tessera_EdgeFlip.hpp

std::vector<char> edgeMask( mesh.numOwnedEdges(), 0 );  // OWNED-edge indexing
// ... the caller decides which edges to flip; see "Choosing the edges" below

DefaultFlipPolicy policy;      // maxNormalDeviation = 0.35 rad, minQuality = 0.05
FlipResult r = flipEdges( mesh, halo, edgeMask, policy );

r.requested;              // marked owned edges, globally
r.accepted;               // edges actually flipped
r.rejectedBoundary;       // other than exactly two incident faces
r.rejectedDuplicateEdge;  // the edge the flip would create already exists
r.rejectedGeometric;      // the policy's normal-deviation or quality test failed
r.rejectedConflict;       // lost the independent-set round
r.flipped;                // (old EdgeKey, new EdgeKey) for every flip this rank touches
```

The five verdict counters **partition** `requested`. `DefaultFlipPolicy` has two
knobs: `maxNormalDeviation` (radians) rejects a flip either of whose new faces
points more than that far from the area-weighted average of the two old normals,
which is what stops a nearly-flat pair being folded; and `minQuality` rejects a
flip either of whose new faces has a radius ratio (inradius/circumradius,
**scaled to 1 for an equilateral triangle**) below it. Note the scaling — the
same quantity is conventionally quoted as 0.5 for an equilateral triangle, so a
`minQuality` of 0.05 is a floor of 0.025 in the unscaled convention that
`tests/MeshInvariants.hpp`'s `minRadiusRatio()` reports.

Three decisions a caller has to know about:

**Gids are preserved, so `gid ↔ key` is no longer a bijection.** The flipped edge
keeps its gid and gets new endpoints; the two faces keep theirs and get new
corners. Nothing is created or destroyed, so there is nothing to assign, and no
peer's reference to any entity is invalidated by a flip — that is what makes the
operation cheap. The price is that an edge's `EdgeKey` **changes** while its gid
does not, so a caller that cached "the edge whose key is K has gid G" across a
`flipEdges()` call is wrong afterwards. Nothing inside Tessera assumes the
bijection: the `edgeKeys()`/`faceKeys()` side tables are rebuilt from `Verts` by
the halo rebuild, and every gid-keyed path is keyed on gid alone.

**Choosing the edges is the caller's, not Tessera's.** The dominant use of
flipping is valence equalization, which needs the full valence of all four
corners of the quad. A vertex's valence is a local quantity at its owner, so the
caller computes it and encodes it in the mask; keeping it out of `flipEdges()` is
what makes the operation safe at **halo depth 1** — every test it performs is
evaluable from the two incident faces, which the edge's coordinator supplies.
Pulling valence inside would force depth 2 and hard-code one selection criterion
into a general operation.

**At most an independent set is applied per call.** Two flips sharing a *face*
conflict — that face would be rewritten twice — and only the higher-priority one
survives; two sharing only a vertex do not conflict. Iterating internally would
hide an unbounded number of collectives behind one call, so **the caller loops**:

```cpp
for ( int round = 0; round < maxRounds; ++round )
{
    auto mask = myValenceSelection( mesh );   // re-derived every round
    if ( flipEdges( mesh, halo, mask ).accepted == 0 ) break;
}
```

which is also what lets the caller re-derive valences between rounds. The
priority is `(squared length descending, EdgeKey ascending)` and contains **no
gid**, so the accepted set is a pure function of the global mesh and is identical
at every rank count.

Because the independent set is not a serial shortest-first sweep, **the flip set
differs from a serial pass and both are valid** — a consumer must compare quality
*statistics*, not edit sets. `flipEdges()` ends by calling `rebuildHalo()` at the
halo's recorded depth, so the halo is valid on return. An empty mask is a no-op
fast path. See `docs/design.md` → *Edge flip* and
[tasks/completed/edge-flip.md](tasks/completed/edge-flip.md).

### Edge collapse

`collapseEdges()` (`Tessera_EdgeCollapse.hpp`) merges the two endpoints of each
marked edge into one vertex. It is the **only operation that removes degrees of
freedom**: without it a remesher is monotone — it can refine where a metric
demands resolution but never coarsen where it does not, so the mesh grows without
bound and a long run dies on memory or on timestep.

Each accepted collapse deletes the edge's two incident faces, the edge itself,
and one of each pair of edges that become coincident: **V−1, E−3, F−2**, so the
Euler number is preserved.

```cpp
#include <Tessera.hpp>   // Tessera_EdgeCollapse.hpp

// REQUIRES halo depth >= 2 -- see "The depth 2 precondition" below.
distribute( mesh, halo, faceOwner, 2 );

std::vector<char> edgeMask( mesh.numOwnedEdges(), 0 );  // OWNED-edge indexing
// ... the caller decides which edges to collapse, e.g. every edge shorter than
//     4/5 of a target length

DefaultCollapsePolicy policy;   // t = 0.5, maxNormalRotation = 0.5 rad,
                                // minQuality = 0.05
CollapseResult r = collapseEdges( mesh, halo, edgeMask, policy );

r.requested;               // marked owned edges, globally
r.accepted;                // edges actually collapsed
r.rejectedBoundary;        // other than exactly two incident faces
r.rejectedLinkCondition;   // link(a) ∩ link(b) != {c,d}, or the merge would
                           // fuse two faces into one
r.rejectedNormalFlip;      // a surviving incident face's normal would rotate
                           // more than policy.maxNormalRotation
r.rejectedQuality;         // a surviving incident face would fall below
                           // policy.minQuality
r.rejectedConflict;        // lost the independent-set round
r.verticesRemoved;         // == accepted
r.edgesRemoved;            // == 3 * accepted
r.facesRemoved;            // == 2 * accepted
r.collapsed;               // EdgeKey of every accepted collapse this rank touches
```

The five verdict counters **partition** `requested`. `DefaultCollapsePolicy` has
three knobs and two hooks:

```cpp
struct DefaultCollapsePolicy
{
    double t = 0.5;                  // 0 keeps a, 1 keeps b, a = the LOWER-gid endpoint
    double maxNormalRotation = 0.5;  // radians; the fold guard
    double minQuality = 0.05;        // radius ratio, 1 for an equilateral triangle

    void interpolatePosition( double* out, const double* a, const double* b,
                              double t, int dim ) const;         // (1-t)a + t b
    template <std::size_t M>
    double interpolateVertexField( double a, double b, double t ) const;
};
```

`minQuality` is in the convention where an equilateral triangle scores **1**, the
same as `DefaultFlipPolicy::minQuality`, so 0.05 is a floor of 0.025 in the
unscaled convention `tests/MeshInvariants.hpp`'s `minRadiusRatio()` reports.

**The hooks carry `t`, and that is why this is a separate policy from
`DefaultRefinePolicy`.** `DefaultRefinePolicy::interpolatePosition( mid, a, b,
dim )` and `interpolateVertexField<M>( a, b )` are hard-coded 0.5 averages with
no `t` argument, so a collapse at `t != 0.5` is not expressible through them. The
per-field override pattern is otherwise identical — derive, shadow
`interpolateVertexField`, dispatch on `M` with `if constexpr` — so a consumer's
existing refine policy transfers by inspection:

```cpp
struct MyPolicy : Tessera::DefaultCollapsePolicy
{
    template <std::size_t M>
    double interpolateVertexField( double a, double b, double t ) const
    {
        if constexpr ( M == Tessera::VertexField::UserBegin + 0 )
            return conserve_vorticity( a, b, t );
        else
            return Tessera::DefaultCollapsePolicy::
                template interpolateVertexField<M>( a, b, t );
    }
};
```

**The surviving vertex is `min(gid(a), gid(b))`** — deterministic, rank-count
independent, and derivable on every rank with no agreement round. Because an
`EdgeKey` is canonical `(lo, hi)`, the survivor is always `key.id[0]`, which is
also what orients `t`.

**The depth 2 precondition.** The link condition below is a statement about the
full one-ring of *both* endpoints — a two-ring around the edge — so
`collapseEdges()` **throws** `std::runtime_error` naming the required and the
actual depth if `0 < mesh.haloDepth() < 2`. Depth `0` means *never distributed* —
a replicated mesh straight out of the builder, where every entity is local and no
ring is missing — and is accepted, the same convention `buildVertexStencil()`
uses. Correctness itself does **not** rest on local ring residency: the owner of
an edge may own neither endpoint, so both one-rings are assembled from
vertex-coordinator advertisements, which are exact at any depth.

**The link condition is what stops a collapse corrupting the mesh silently.** A
collapse of `(a,b)` with incident faces `(a,b,c)` and `(b,a,d)` is topologically
valid iff `link(a) ∩ link(b) == {c, d}`. If some other vertex `x` is adjacent to
both endpoints, the edges `(a,x)` and `(b,x)` become coincident and the result is
non-manifold — two distant parts of the surface welded together, which no later
check untangles. The vertex link test alone misses one case (a mesh that is a
single tetrahedron, where the merge fuses two triangles into one face), so the
coordinator also rejects a candidate whose rewritten face set would contain a
duplicate face key; both are counted as `rejectedLinkCondition`.

**At most an independent set is applied per call, and it is not the serial
shortest-first set.** The priority is `(squared length ascending, EdgeKey
ascending)` — shortest first, matching the serial *preference* — and a candidate
is accepted only if it holds the strictly best priority among all candidates
incident on either endpoint's one-ring. The priority contains **no gid** and every
length goes through `edgeLen2Canonical()`, so the accepted set is a pure function
of the global mesh and is identical at every rank count. **The consequence: the
accepted set is a strict subset of what a serial shortest-first pass with
re-evaluation would take.** On the subdivision-2 icosphere with every edge marked,
8 of 480 are accepted in one call, because accepting one excludes every candidate
in its two-ring. **Compare statistics — face count, quality distribution,
edge-length histogram — never edit sets.** The caller loops:

```cpp
while ( collapseEdges( mesh, halo, shortEdges( mesh ) ).accepted > 0 ) {}
```

**Vertex positions and vertex user fields must be halo-consistent on entry.** The
merged values are blended by the surviving vertex's owner from its local copies of
the two endpoints, one of which is in general a ghost, so call `haloExchange()`
after mutating them and before collapsing. `splitEdges()` reads its endpoints the
same way and carries the same requirement.

`collapseEdges()` ends by calling `compact()`, which itself ends with a halo
rebuild at the recorded depth — so the mesh is compact and fully haloed on return,
a caller never sees a tombstone, and a second `collapseEdges()` may follow
immediately. An empty mask, **and a call in which every candidate is rejected**,
are both genuine no-ops: no rewrite, no compaction, and every gid, key and local
ordering unchanged. See `docs/design.md` → *Edge collapse* and
[tasks/completed/edge-collapse.md](tasks/completed/edge-collapse.md).

### Compaction: removing entities

Every other editor only *adds* entities. `compact()` is the one call that removes
them, and it is what lets `collapseEdges()` put its dead entities somewhere
instead of growing a private half-compaction.

```cpp
#include <Tessera.hpp>   // Tessera_Compact.hpp

// Mark entities dead. Purely local, non-collective. The convention is uniform
// across the three kinds: Gid == invalid_gid marks a dead entity.
tombstoneFace  ( mesh, f );   // a face is removed ONLY if it is marked
tombstoneEdge  ( mesh, e );   // optional -- see below
tombstoneVertex( mesh, v );   // optional -- see below

CompactStats st = compact( mesh, halo );          // collective; rebuilds `halo`
// st.verticesRemoved / edgesRemoved / facesRemoved -- GLOBAL owned-count drops
// st.gidSpaceBefore / gidSpaceAfter                -- GLOBAL gid-space size

// Periodically, instead of compact():
CompactStats st2 = compactAndRenumberGids( mesh, halo );
```

**What is removed.** Every tombstoned **owned face**, plus every vertex and edge
that no surviving face references — *whether or not it was tombstoned*. So
marking a vertex or an edge is optional for removal; what the mark buys is the
**check**. The tombstone set must **close**: a live face may not reference an
entity its owner marked dead. `compact()` verifies this before mutating anything
and throws `std::runtime_error` naming the offending live face and the dead gid,
rather than producing a corrupt mesh. Deadness is **owner-scoped** — a mark on a
ghost copy says nothing about the entity and is silently repaired — so the check
is collective, and so is the throw (every rank throws, so a caller bug cannot
deadlock the ranks that do not see it).

**Gids are preserved, and renumbering is a separate, periodic call.** A surviving
entity keeps the gid it had, so every cross-rank reference a peer holds stays
valid, no communication is needed to agree on a renaming, and — since connectivity
fields hold gids — there is **no connectivity rewrite at all**. That is what makes
`compact()` cheap enough to call every step: a local face filter plus one
`rebuildHalo()`, which does the owned-first reordering, both CSRs, both key side
tables and the three halo plans.

The cost of preservation is that the gid **space** only ever grows: gids are
assigned by `MPI_Exscan` onto a monotonically rising global count, so a long run
of split-and-collapse rounds inflates the space without bound even while the mesh
stays the same size — and several code paths index by gid into a **dense** host
array sized to a max gid. `compactAndRenumberGids()` is the sanctioned fix: it
renumbers gids contiguously to `[0, N)` per kind, at the price of invalidating
every gid a peer holds, rewriting every connectivity field, and a **second** halo
rebuild. Call it every few hundred steps, not every step. See `docs/design.md` →
*Compaction* for the hazard and the affected paths.

The new gid of an entity is *the number of live entities of its kind with a
smaller old gid* — an order statistic rather than an exscan over owned counts, so
the resulting global gid→entity map is identical at every rank count.

**`CompactStats` is entirely global** and identical on every rank: a *local* count
is not a statement about the mesh, since a rank's local count also moves when the
ghost set changes. `gidSpaceBefore`/`gidSpaceAfter` are filled by both calls; for
`compact()` they are equal — that equality *is* the statement that gids were
preserved — unless the removal happened to include the globally maximal gid of
some kind, in which case the space shrinks by exactly the vacated tail.

A rank may legitimately end with **zero owned entities**; its peers drop it from
their plans and the collectives still complete. That is a real load-balance state,
not a pathological one.

**Family consequence.** Both calls claim `EditFamily::Remesh`, so compacting a
`refine()`d mesh throws, and compacting a freshly built mesh **tags** it Remesh —
a later `refine()` on it is then refused. See *Editing families* above.

Not provided: deciding *what* to tombstone (that is the caller's, or
`collapseEdges()`'s, job), shrinking AoSoA capacity, and repairing a non-closed
tombstone set. `EdgeField::Faces` is neither checked nor repaired by `compact()` —
it is best-effort by design, and a surviving boundary edge whose second incidence
was removed is exactly what a legitimate compaction produces —
but `compactAndRenumberGids()` *does* map it, to `invalid_gid` where the named
face is not held locally, because a gid left in the old space would silently alias
a different live face.

### Example programs

| Example | Directory | Arguments | Description |
|---|---|---|---|
| `hello_tessera` | `examples/01_hello_tessera/` | *(none)* | Prints the active Kokkos backend and MPI rank count. Build verification only. |
| `mesh_pipeline` | `examples/02_mesh_pipeline/` | `--subdiv N` `--axis N` `--balance` `--iters N` `--frac F` `--seed N` `--refine-mode {hanging,conforming}` `--out STEM` | End-to-end demo: build an icosphere, partition + distribute it, then run `N` iterations of random-percent refinement, writing each frame (frame 0 plus one per iteration) through a `MeshSeries`: per-frame `.h5`/`.xmf` files plus one master `<out>_<tag>_np<size>.xmf` (and its `.xmfindex`) per execution space, with the frame index as each timestep's time. Open the master in Paraview to step through the mesh evolution as one timestepped dataset. Runs the pipeline once per available Kokkos execution space (Serial, plus the platform default and OpenMP where distinct); `<tag>` is `<space>_<mode>`, e.g. `Serial_conforming`. `--refine-mode` selects the `RefinementMode` the mesh type is instantiated with (default `conforming`); the mode tag is part of the master and frame stems, so a `hanging` and a `conforming` run of the same `--out` can be compared side by side in Paraview. Also demonstrates the profiling API (see below): each iteration is one reporting window. |

### Profiling

Tessera carries an optional, level-gated wall-time profiler
(`Tessera_Profiling.hpp`, header-only). Every instrumented region is an RAII scoped
timer that (1) pushes a `Kokkos::Profiling` region — so external Kokkos-aware tools
(Kokkos Tools, `rocprof`, Nsight) see the named region for free — and (2)
accumulates `MPI_Wtime` wall time into a process-local registry. It is **compiled
out entirely** at level 0; the default build enables nothing.

**Enable it (CMake):**

| Option | Default | Meaning |
|---|---|---|
| `Tessera_ENABLE_PROFILING` | `OFF` | Master switch. `OFF` is an authoritative kill switch — it forces the effective level to 0 regardless of `Tessera_PROFILING_LEVEL`. |
| `Tessera_PROFILING_LEVEL` | *(empty)* | `0`/`1`/`2`/`3`. Empty resolves to `1` when profiling is enabled, else `0`. |

When the effective level is `> 0`, `TESSERA_ENABLE_PROFILING` and
`TESSERA_PROFILING_LEVEL=<n>` are added as compile definitions on the `Tessera`
interface target (inherited by every consumer). The `run_cmake*.sh` scripts default
to level 0; override on the command line, e.g.
`bash ../run_cmake_toulumne.sh -DTessera_PROFILING_LEVEL=2`.

**Verbosity levels** — higher levels are strictly additive (a level-*n* build emits
every region at levels ≤ *n*):

| Level | Adds | Example regions |
|---|---|---|
| 1 | Top-level phases | `build_icosphere`, `build_icosphere_distributed`, `build_soup_distributed`, `partition`, `distribute`, `halo_exchange`, `refine`, `migrate`, `load_balance`, `write_mesh`, `read_mesh`, `mark_*` |
| 2 | Major sub-phases | `refine_2to1_balance`, `refine_local_rebuild`, `migrate_round_*`, `halo_round_*`, `distribute_csr_rebuild`, `dbuild_*`, `write_datasets`, `lb_zoltan2_solve` |
| 3 | Comm rounds / device kernels | `refine_advertise_alltoallv`, `mark_edge_length_kernel`, `write_hyperslabs`, `read_hyperslabs` |

**Reporting is caller-driven — Tessera has no timestep loop.** The library owns the
mechanism; the downstream application decides the cadence. Two registries back every
region: a resettable **window** and a monotonic **lifetime**. The caller-facing
macros (all no-ops at level 0):

```cpp
TESSERA_RESET_TIMERS();               // clear the window registry
TESSERA_PRINT_TIMERS( comm );         // collective: rank 0 prints the window
                                      //   min/max/mean/imbalance per region
TESSERA_PRINT_TIMERS_TOTAL( comm );   // collective: rank 0 prints the lifetime total
```

A simulation prints and resets every `ts` steps for a per-window breakdown, and
prints the lifetime total once at shutdown:

```cpp
for ( int step = 0; step < nsteps; ++step ) {
    // ... Tessera refine / migrate / haloExchange / writeMesh ...
    if ( step % ts == 0 ) {
        TESSERA_PRINT_TIMERS( comm );   // aggregated over this window
        TESSERA_RESET_TIMERS();         // start the next window
    }
}
TESSERA_PRINT_TIMERS_TOTAL( comm );     // whole-run aggregate
```

`examples/02_mesh_pipeline/` wires exactly this pattern, treating each adaptive
refinement iteration as one window. All three print/reset macros are collective on
the passed communicator, so every rank must call them.

### Geometry & stencil operators

A narrow, physics-free surface for building local surface operators (surface
gradient, Laplace–Beltrami, curvature, vertex normals/areas) on top of the mesh.
**Tessera owns the traversal, the assembly, and the raw geometry; the caller owns
the weights and every convention** — the weight scheme (cotangent vs RBF-FD vs
uniform), the normal orientation, the area definition (barycentric/Voronoi/⅓),
the curvature sign. Tessera never sees a physics parameter. This split is what
lets one interface hold both operator families; when the discretization is
settled it is one weight-builder in the caller, and *zero* changes in Tessera.

Headers: `Tessera_Geometry.hpp`, `Tessera_Stencil.hpp`, `Tessera_FieldReduce.hpp`,
`Tessera_Reduction.hpp` (all folded into `<Tessera.hpp>`).

**Raw geometric primitives** — pure geometry, no convention. Because a `Mesh` is
not capturable into a device kernel and its connectivity stores vertex *global
ids*, the primitives read a lightweight, device-capturable accessor built once
from the mesh:

```cpp
auto geom = buildMeshGeometry( mesh );   // captures the position slice + per-face /
                                         // per-edge LOCAL vertex indices
// inside a KOKKOS_LAMBDA, per face f / edge e / corner c (0,1,2):
Scalar A   = faceArea( geom, f );              // ½‖(p1−p0)×(p2−p0)‖
Scalar n[3]; faceNormalRaw( geom, f, n );      // unnormalized (p1−p0)×(p2−p0); the
                                               //   outward sign is the caller's choice
Scalar v[3]; edgeVector( geom, e, v );         // p[v1]−p[v0]
Scalar cot = cotangentAtCorner( geom, f, c );  // (u·v)/‖u×v‖ (triangle geometry only)
```

**k-ring stencil topology** — a CSR of the k-ring vertex neighbours (`k=1` and
`k=2` both supported), built by edge BFS over the existing connectivity. Rows are
complete for every owned vertex provided the mesh's **halo depth is ≥ k**, and
that is checked: `k > mesh.haloDepth()` throws `std::invalid_argument` naming both
numbers, rather than returning silently short rows on a partition boundary.

```cpp
distribute( mesh, halo, faceOwner, /*depth=*/2 );     // once, at setup
auto stencil = buildVertexStencil( mesh, /*k=*/2 );   // VertexStencil<MemSpace>
```

**Weighted-stencil apply** — halo-correct, GPU-resident. Applies a caller-built
weight View aligned to the stencil CSR: `out(i) = Σ_j w(i,j)·in(j)` over owned
vertices. **The caller builds `w`** (that is the convention) and must
`haloExchange` `in` first so ghost neighbour values are current:

```cpp
Kokkos::View<Scalar*, MemSpace> w( "w", stencil.csr.get().numEntries() );
buildMyWeights( mesh, geom, stencil, w );   // caller's cotangent / RBF-FD / … fill
haloExchange( mesh, halo );                  // refresh ghost `in` before apply
applyStencil( mesh, stencil, w, in_slice, out_slice );   // owned vertices only
```

**Face→vertex reduce** — Tessera iterates a vertex's incident faces; the caller's
device functor `op(v, f, geom, faceSlice, vertSlice)` defines the accumulation
(no atomics — one thread per owned vertex). This is the primitive behind vertex
normals and vertex areas, whose conventions stay in the caller:

```cpp
reduceVertexFromFaces( mesh, geom, faceSlice, vertSlice, MyAreaOrNormalOp{} );
```

**Face→face adjacency through shared edges** — a CSR of every local face's
edge-neighbours, built collectively. `Tessera_FaceAdjacency.hpp`.

```cpp
auto adj = buildFaceAdjacency( mesh );   // FaceAdjacency<MemSpace>; COLLECTIVE
```

This cannot be derived locally, for two independent reasons, which is why it is a
collective builder rather than a walk over the connectivity the mesh already
holds. `EdgeField::Faces` holds the **global ids** of an edge's incident faces,
and `migrate()` carries them verbatim without repair — a gid is not a usable
neighbour handle. And a vertex-incidence walk only finds a neighbour that happens
to be co-resident: the local face set is *owned faces plus faces incident on an
owned vertex*, so an owned face all three of whose corners are ghosts can have
edge-neighbours that are not held locally at all. The 1-deep **vertex** halo does
not guarantee edge-neighbour co-residency. `buildFaceAdjacency` therefore reuses
`refine()`'s **edge coordinator** (`edgeCoordRank` + `allToAllV`) rather than the
halo, and adds no new communication mechanism.

**The return type has two halves, and which one you may use is the precondition
split to read first:**

| Half | Always valid? | For |
|---|---|---|
| `nbrGid`, `nbrOwner` — flat Views parallel to `csr.get().neighbors` | **Yes**, resident or not | A **topological** consumer: mark growth by neighbour rings, an independent set of edge flips, any conflict-resolution pass that *communicates* with the neighbour's owner. Works at halo depth 1 with no further precondition. |
| `csr` — `GenerationHandle<CsrAdjacency>` of local face indices | Only where the neighbour is resident; `invalid_local` otherwise | A **geometric** consumer, one that reads the neighbour's vertex positions. It must **check `numNonResident == 0`**, not assume it. |

`numNonResident` is the count of owned-row entries that are `invalid_local`,
summed over this rank. Zero means every owned face's neighbours are co-resident,
so `csr` may be used directly; a deeper halo (`distribute(..., depth >= 2)`)
makes that likely but does not guarantee it in general. Ghost rows are excluded
from the count, because they are best-effort by contract.

```cpp
// Rows for OWNED faces are complete: every true edge-neighbour appears, as a
// local index if resident and as invalid_local if not. Rows for GHOST faces are
// best-effort and may be SHORT -- do not iterate them.
const auto& csr = adj.csr.get();
if ( adj.numNonResident == 0 ) { /* geometric consumer may use csr */ }
```

Rows are sorted ascending by neighbour **global id**, not by local index, so the
row order is rank-count invariant and two runs at different rank counts compare
as exact lists rather than as sets. Non-manifold input (three or more distinct
faces on one edge, which `buildFromTriangleSoup` will accept) throws
`std::runtime_error` naming the offending `EdgeKey` — collectively, so every rank
throws rather than one deadlocking the rest. Generation-guarded exactly like
`VertexStencil`: rebuild after any
`distribute`/`migrate`/`refine`/`splitEdges`/`loadBalance`.

Two mode notes, both contract rather than defect. On a `Conforming` mesh the face
AoSoA stores **only the visible (closed) faces** — retired red parents exist only
inside `refine()` and are never stored — so adjacency over the local faces *is*
adjacency over the visible faces, with nothing to skip, and every face's degree
is exactly 3 on a closed surface. Under `HangingNode2to1` a **T-junction is not a
shared edge**: the coarse side holds `(a,b)` while the fine side holds `(a,m)` and
`(m,b)`, three distinct edges with one incidence each, so a face on either side
of a T-junction has degree **< 3**. That mode keeps no record of which edges carry
a hanging node (see *Known Issues*), so there is nothing to match `(a,m)` against
`(a,b)` with. A consumer needing neighbours across a refinement front wants a
`Conforming` mesh.

Not provided, deliberately: growing a refinement mask by neighbour rings (that is
the consumer's loop over this CSR plus a mark exchange it already has), vertex→
vertex or face→vertex adjacency (both already exist or are derivable), and
incremental maintenance across a topology edit.

**Global scalar reductions** — single-sourced `MPI_Allreduce` wrappers over
`mesh.comm()`. Each takes a per-rank scalar and returns the global result on
every rank: **Tessera owns the collective, the caller owns the local value.** All
are collective — every rank must call them.

```cpp
Scalar dt     = globalMin( mesh, local_dt_estimate );   // adaptive timestep
Scalar cfl    = globalMax( mesh, local_cfl_estimate );  // CFL-style bound
Scalar volume = globalSum( mesh, local_volume );        // enclosed volume
bool   ok     = globalAllFinite( mesh, local_verdict ); // NaN/Inf tripwire
```

`globalAllFinite` is an `MPI_Allreduce(MPI_LAND)` returning true iff *every* rank
passed true, so every rank aborts on the step that produced the NaN rather than
one rank diverging silently. It takes a **verdict, not data** — the local
"everything I hold is finite" sweep is the caller's Kokkos reduction over
whichever fields it cares about, which is knowledge Tessera does not have. There
is deliberately no device-side all-finite helper.

> **`globalSum` floating-point reproducibility — read this before writing a
> cross-rank test.** `MPI_SUM` is not associative in floating point, so a
> `double` result is **not** bitwise reproducible across rank counts, nor across
> runs on a GPU partial-sum path. A quantity carried for the whole run (an
> enclosed volume, a reduced minimum edge length) will differ in its low bits
> between a 4-rank and a 5-rank run, and anything scaling off it — an adaptive
> timestep — diverges from there. **Integer sums are exact and reproducible.**
> Do not write a cross-rank bitwise comparison over a floating-point `globalSum`.
> Fixed-order/compensated summation is out of scope.

The scalar type is mapped to its `MPI_Datatype` by a specialization set; an
arithmetic type with no mapping (e.g. `long double`) is a **compile** error, not
a runtime MPI error.

**Global entity counts** — owned entities partition the global mesh, so a SUM of
the owned counts is the global count. Reduced as `long long`, exact since
integer:

```cpp
long long V = globalOwnedVertices( mesh );
long long E = globalOwnedEdges( mesh );
long long F = globalOwnedFaces( mesh );
long long X = globalOwnedEuler( mesh );   // V - E + F; 2 for a closed conforming surface
```

**Validity.** `MeshGeometry`, `VertexStencil` and `FaceAdjacency` are
generation-guarded like mesh slices (see *Slice/handle validity*): each is stamped
with the mesh generation at build time and aborts on use after a topology op.
**Rebuild them with
`buildMeshGeometry`/`buildVertexStencil`/`buildFaceAdjacency` after any
`distribute`/`migrate`/`refine`/`loadBalance`**; they survive `haloExchange`
(topology-preserving). `FaceAdjacency`'s `nbrGid`/`nbrOwner` are bare Views
parallel to the guarded CSR and are only meaningful together with it, so the guard
on the CSR is the guard on all three. Note also that a
solver's *operator consistency* at irregular (non-valence-6) vertices is a property
of the weights, not the apply — document and convergence-test that on the caller's
weight-builder; `applyStencil` is exact arithmetic and is correctness-tested against
an analytic field.

---

## Dependencies and Build Notes

### Dependencies

| Dependency | Notes |
|---|---|
| [Kokkos](https://github.com/kokkos/kokkos) | ≥ 4.0; Serial, OpenMP, and HIP execution spaces |
| [Cabana](https://github.com/ECP-copa/Cabana) | MPI-enabled build required |
| [Trilinos](https://github.com/trilinos/Trilinos) | Zoltan2 (geometric MultiJagged) for internal load balancing |
| [HDF5](https://www.hdfgroup.org/) | Parallel (`+mpi`) build, for mesh I/O *(added to the spack env in Step 0)* |
| MPI | Cray-MPICH on Tuolumne (GPU-aware); OpenMPI on local workstations |
| CMake | ≥ 3.21 |
| clang-format | For `make format` / `make format-check` |

### Build (Tuolumne — AMD MI300A)

```bash
spack env activate ~/spack_envs/tuolumne_trilinos/
mkdir build-tuolumne && cd build-tuolumne
bash ../run_cmake_toulumne.sh
make -j $(nproc)
```

`run_cmake_toulumne.sh` resolves and passes `-DHDF5_ROOT=<prefix>` automatically
(via `spack location -i hdf5`, falling back to the known Cray path). This is
required on Tuolumne because the Cray parallel HDF5 is a spack **external** and
is not view-linked onto `CMAKE_PREFIX_PATH`; a bare `find_package(HDF5)` would
otherwise silently resolve the OS serial `/usr/lib64` build. CMake fails loudly
(`HDF5_IS_PARALLEL` guard) rather than configuring against a non-parallel HDF5.

### Build (local workstation)

```bash
mkdir build-local && cd build-local
bash ../run_cmake.sh -DCMAKE_PREFIX_PATH=<deps-install>
make -j $(nproc)
```

### Design limitations

- Milestone 1 implements **split-based refinement only** (no edge collapse/flip).
- Topology surgery (pinch-off / merger of surfaces) is not implemented; the data
  model is component-agnostic to leave room for it.

---

## Future Optimizations

- Incrementally patch the `HaloExchangePlan` index maps after refinement instead of
  a full `rebuild()` each 2:1-balance iteration.
- Compress the conforming-mode closure bookkeeping. A closure child currently
  stores its red parent outright — `ClosureParent` (one `GlobalId`) plus
  `ClosureParentVerts` (three more), so 4 × 8 B on a ~78 B face core, in memory
  and on disk. A (parent gid, pattern + child-index byte) pair would halve that
  and reconstruct the parent's corners from the sibling group instead; the cost
  is that un-close stops being local per child, which is what makes the current
  encoding cheap in `migrate()`. Worth doing only if face memory becomes the
  binding constraint.
- Unify the two editing families by extending the level model to **anisotropic
  bisection**. Today `refine()` and `splitEdges()` are disjoint families (see
  *Editing families*) because a single integer `Level` cannot describe a face
  whose edges were bisected unevenly: `refine()`'s 2:1 invariant is stated in
  level differences and is only coherent because it performs the uniform 1→4 red
  split. Making them interleavable means **per-edge levels plus a compatible
  balance rule**, which then has to be maintained by the mark-propagation
  fixpoint, the closure patterns, and the HDF5 format alike — a much larger design
  than the guard it would replace, and one no known consumer needs (the driving
  consumer, Beatnik's z-model remesher, is entirely edge-addressed and never calls
  `refine()`). Until then the guard turns the mistake into an immediate abort,
  which is the cheap 95% of the value. Design notes:
  [tasks/completed/edge-split.md](tasks/completed/edge-split.md) → Decision 1's alternative.

---

## Known Issues

- **`collapseEdges()` has three accepted limits, all deliberate.** *(Not defects —
  recorded because each one is a case where the operation declines to do something
  a caller might reasonably expect.)*
  1. **Collapse is undefined on a `refine()`d mesh**, in either refinement mode,
     and the `EditFamily` guard throws rather than trying. Under
     `RefinementMode::Conforming` the visible layer is a *transient* closure whose
     children reference retired red parents, and collapsing across it would have
     to un-close first; under `HangingNode2to1` there are T-junctions and the link
     condition is not even well posed at one. The alternative — a true inverse to
     `refine()`, un-refining a red split back to its parent — is a different and
     much larger feature (it needs the parent's identity retained, sibling
     co-residency, and an inverse to the closure) and is **out of scope**; it is
     also not what a metric remesher wants, since it can only coarsen along the
     refinement tree. Reproduce: `refine()` then `collapseEdges()` — throws naming
     both families (`tests/test_collapse_edges.cpp` check 13, both modes).
  2. **A boundary edge is always rejected** (`rejectedBoundary`), where "boundary"
     means anything other than exactly two incident faces. A boundary-preserving
     collapse needs its own rule for what happens to the boundary polyline and is
     a separate feature. Note the consequence for an *interior* edge one of whose
     endpoints lies on a boundary: that collapse is **allowed** if it clears the
     link condition and the geometric tests, and it moves the boundary.
  3. **One call applies at most an independent set**, so a single
     `collapseEdges()` on a mesh with every edge marked coarsens by only a few
     percent — 8 accepted of 480 marked on the subdivision-2 icosphere, because
     accepting one candidate excludes every candidate in its two-ring. This is not
     a throughput bug to be fixed by loosening the conflict relation; the relation
     is what makes the accepted set rank-count invariant and pairwise
     non-conflicting. **The caller loops on `accepted`** — twenty rounds take that
     same mesh from 320 faces to 174.
- **One distributed MultiJagged solve (`LoadBalanceMode::Distributed`) is not
  run-to-run reproducible, which is why `Sampled` is the default.** *(Not a
  Tessera defect — a measured property of Zoltan2, recorded because the
  reasonable expectation is that one solve gives one answer every time.)* Moving
  the partition solve off rank 0 raised a question the old rank-0-only path never
  had to answer: is a **single** distributed solve repeatable at a fixed rank
  count? Cross-rank *agreement* is not the issue — there is one solve, so there
  is one answer — but run-to-run *reproducibility* is a separate property and
  Zoltan2 does not promise it. Measured by
  `tests/test_loadbalance_distributed.cpp` check 4, which balances two
  identically-built subdivision-4 icospheres (5120 faces) in the same run and
  compares the two `dest` arrays element-wise: over two ctest invocations × both
  backend registrations × both execution spaces, **np1–np4 agreed in every
  invocation** and **np5 disagreed on 0, 4, 8, 16 or 18 faces** depending on the
  invocation. `GatherRoot` is no better — its own solve is a Kokkos-parallel
  MultiJagged, and a second `loadBalance()` of an already-balanced mesh moves
  anywhere from 0 to 42 faces from one invocation to the next (case 9) — so
  `Sampled` is not merely the safer default, it is the only mode measured
  reproducible: its `dest` checksum was **bit-identical in all eight invocations
  at every rank count 1–5**. The mechanism is the expected one: MJ's cut coordinates come from
  floating-point reductions whose partial-sum order is not fixed, and an
  icosphere is symmetric enough that many centroids sit on or adjacent to a cut,
  so a last-bit difference in the cut flips them. The consequence is not a wrong
  partition — every invariant, the balance bound and the topology checksum hold
  in every run — but a `dest` that is not a function of the mesh alone. So
  `LoadBalanceMode::Sampled` is the default: its cuts come from a gid-order
  sample solved once and broadcast, and the per-face classification is exact
  local arithmetic, so it agrees with itself at every rank count, across runs,
  and across two different starting partitions of the same mesh (check 7). `Distributed` remains
  fully supported and gathers nothing to rank 0 either; choose it when partition
  quality matters more than bitwise repeatability. Check 4 reports both modes'
  mismatch counts and prints per-mode checksums; only the default's zero is
  asserted, because asserting `Distributed`'s would pin a property it does not
  have.
- **`markByQuality` on an anisotropic surface marks the *equator* of a lat/lon
  sphere, not the poles.** *(Not a defect — recorded here because the naive
  expectation is inverted, and the inversion is easy to mistake for a bug.)*
  Intuition says the pole region of a UV sphere is the "bad" one, because that is
  where the triangles degenerate as `nLon` grows. It depends entirely on which
  way the mesh is anisotropic. At `(nLat=33, nLon=8)` the meridional step is
  `dθ = π/32` (chord `0.0981`, latitude-independent) while the in-ring step is
  `dφ = 45°`, giving a ring-edge chord of `0.7654·sin θ`. So the ring edges — and
  with them the triangle areas — are **longest at the equator** and shrink to
  nothing at the poles: the polar triangles are the small, nearly *isotropic*
  ones and the equatorial ones carry the 7.8:1 stretch. `EdgeLengthCriterion`
  marks long edges, so it marks the equator, exactly per its documented contract.
  Measured at `(33,8)` with `maxLen = 0.4`, identical at ranks 1–5 on both
  backends: 352 of 496 faces marked; of the 128 faces entirely inside the polar
  caps (`|z| ≥ cos 28.125°`) **none** are marked, and of the 192 entirely inside
  the equatorial band (`|z| ≤ cos 56.25°`) **all** are.
  `CurvatureCriterion` behaves the same way round: at `maxAngle = 40°` it marks
  160 faces, all in the equatorial band and none in the polar caps; at `20°` it
  marks 384, of which only 16 (the two pole fans themselves, 8 faces each) are
  polar. The lesson for a consumer driving AMR from a lat/lon mesh is that
  "polar" and "badly shaped" are not the same set, and which one a criterion
  selects is decided by the `dθ`/`dφ` ratio. Pinned by
  `tests/test_latlon_sphere.cpp` against closed-form latitude cut points rather
  than measured output.
- **Conforming refinement is the `Mesh` default.** *(Not a defect — recorded here
  because it changes what a default-spelled `Mesh` does.)* The whole
  `RefinementMode::Conforming` path — closure kernel, distributed `refine()`,
  `migrate()`/`loadBalance()`, HDF5 round-trip, `markByQuality` — is implemented,
  registered across the suite, and **verified**: the ship gate is 230/230 (180/180
  when this was written; the ten `latlon_sphere`, ten `halo_scatter_add`, ten
  `face_adjacency`, ten `loadbalance_distributed` and ten `distributed_build`
  entries came later) and the
  diagnostic tier 62/62 on SERIAL and HIP at **ranks 1–5**, over multiple successive
  adaptive rounds, and the shape-quality bounds are measured rather than assumed (the
  worst radius ratio saturates by round 11 and is flat through round 16 while the mesh
  grows 5.7×). Because it is the default, a `Mesh<...>` spelled with seven template
  arguments gets conforming refinement; a consumer that wants the previous behaviour
  must spell `RefinementMode::HangingNode2to1` explicitly, which every pre-existing
  test in the gate now does. Under `HangingNode2to1` a partial refine mask leaves
  T-junctions bounded to a 2:1 level jump, so the owned-only Euler number equals 2
  only for a uniform refine; that is the *contract* of the mode, not a defect. The
  visible mesh is now rank-count reproducible as well: the blue closure diagonal is
  chosen **geometrically** (shorter diagonal, i.e. the midpoint of the longer split
  edge joins its opposite corner), so nothing about the closure reads an
  `MPI_Exscan`-derived gid. This closed the last recorded limit of conforming mode.
  Design: [tasks/completed/conforming-refinement.md](tasks/completed/conforming-refinement.md) and
  `docs/design.md` → *Adaptive refinement*; verification evidence:
  [tasks/completed/conforming-refinement-debug.md](tasks/completed/conforming-refinement-debug.md).
- **`RefinementMode::HangingNode2to1` does not track hanging nodes across refine
  rounds.** Two consequences, both long-standing and neither visible to that mode's
  own tests (which assert non-conformity anyway), found while fixing the conforming
  closure. First, the 2:1 mark propagation cannot see across a hanging node: its
  coordinator rule needs an edge to have two incident faces, and a hanging node
  leaves the coarse side holding `(a,b)` while the fine side holds `(a,m)`/`(m,b)`
  — different keys, one incidence each — so a sequence of *adaptive* rounds can
  drive a level jump larger than 2:1. Second, when the coarse face itself refines,
  `refine()` mints a fresh midpoint for `(a,b)` rather than reusing the `m` that is
  already there, leaving two coincident vertices. A uniform mask hits neither.
  `RefinementMode::Conforming` fixes both, because it can recover the persistent
  split-edge map locally from the closure bookkeeping; `HangingNode2to1` keeps no
  such record and would need a new face field or an extra message round. The same
  missing record is why `buildFaceAdjacency()` reports a face on either side of a
  T-junction as having **fewer than 3** edge-neighbours in that mode — `(a,b)`,
  `(a,m)` and `(m,b)` are three distinct edges with one incidence each, and there
  is nothing to match them against each other with. Measured on a subdivision-2
  icosphere refined with the mask `gid % 3 == 0`, identical at ranks 1–5 on both
  backends: of 1266 edges, 657 have two incidences and **609 are T-junction
  edges**. Pinned by `tests/test_face_adjacency.cpp` case 7, which deliberately
  does not assert degree 3.
- **Edge user fields are reset by `refine()`/`refineLocal()`/`splitEdges()`.** Edges
  are re-derived from the new face connectivity, so any per-edge user data is
  re-initialized (M1 carries no edge user state through AMR or remeshing). Vertex and
  face user fields are preserved (interpolated / inherited).
