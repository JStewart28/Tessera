# Lat/lon sphere generator

**Status:** DONE. Smallest and most self-contained of the eleven gap tasks
alongside [global-reductions.md](global-reductions.md); a good first one.

**Verified against `08dd346`** (branch `conforming-refinement`) — the code this task
cites was re-read at that commit.

## Problem

`generateIcosphere()` is Tessera's only mesh generator. There is no lat/lon
(UV-sphere) generator, so a consumer that wants one builds the triangle soup itself
and hands it to `buildFromTriangleSoup()`.

That is not a workaround for a missing capability — the soup interface exists
precisely so callers can supply their own geometry, and generation is neither
haloing nor partitioning. But it is the wrong home for the code. A lat/lon sphere is
a canonical test surface with a handful of easy-to-get-wrong details (duplicate pole
vertices, a duplicated seam meridian, inconsistent winding, degenerate pole
triangles), and every consumer that writes it writes the same bugs. It belongs
beside `buildIcosphere` for exactly the reason `buildIcosphere` belongs in the
library.

**Why it matters, concretely.** An icosphere is *too good* a test surface: it is
nearly isotropic, its triangles are nearly equilateral, and its vertex valences are
almost uniformly 6. A lat/lon sphere is anisotropic by construction — triangles
stretch toward the poles, and the two pole vertices have valence `nLon` — so it
exercises code paths an icosphere never reaches: quality-based marking, the
cotangent weights at a high-valence vertex, stencil rows of very different lengths,
and the poles as valence outliers. The driving consumer (the Beatnik z-model) offers
a `latlon` mesh option and is not on any regression path today partly because the
generator does not exist.

## Approach

Add to `src/Tessera_Icosphere.hpp` (rename it? no — leave the file name alone and
add there, or add `src/Tessera_LatLonSphere.hpp` if the file is getting long; either
is fine, but `TriangleSoup` lives in the former so prefer adding there).

```cpp
//! Generate a unit lat/lon (UV) sphere triangle soup.
//!
//! nLat is the number of latitude RINGS INCLUDING both poles (nLat >= 3);
//! nLon is the number of meridians (nLon >= 3). Throws std::invalid_argument
//! otherwise.
//!
//! Vertices:
//!   theta_j = pi * j / (nLat - 1),  j = 0 .. nLat-1   (0 = north pole)
//!   phi_i   = 2*pi * i / nLon,      i = 0 .. nLon-1   (NOT nLon+1 — no seam
//!                                                      duplicate; i wraps)
//!   position = ( sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta) )
//!   The poles (j = 0 and j = nLat-1) are ONE vertex each, exactly (0,0,+1) and
//!   (0,0,-1), not nLon coincident copies.
//!
//! Counts (closed surface, V - E + F = 2):
//!   V = 2 + (nLat - 2) * nLon
//!   F = 2 * nLon * (nLat - 2)        // nLon per pole fan + 2 per interior quad
//!   E = V + F - 2 = 3 * nLon * (nLat - 2)
//!
//! Ordering: index 0 = north pole, then ring j = 1 .. nLat-2 each contributing
//! nLon vertices in ascending i, then the south pole last. Deterministic and
//! documented so a consumer can address a vertex arithmetically.
//!
//! Winding: CCW seen from OUTSIDE, matching generateIcosphere().
//! Quad diagonal: each interior quad (i,j)-(i+1,j)-(i+1,j+1)-(i,j+1) is split
//! along the (i, j+1)-(i+1, j) diagonal, chosen so the split is consistent around
//! the whole sphere. Fixed, not adaptive — a caller wanting a different
//! triangulation flips edges.
template <class Scalar>
TriangleSoup<Scalar> generateLatLonSphere( int nLat, int nLon );

//! Convenience: generate and build full connectivity, mirroring buildIcosphere().
template <class MeshT>
void buildLatLonSphere( MeshT& mesh, int nLat, int nLon );
```

### Details worth pinning, because they are the bugs

- **Angles in `double` regardless of `Scalar`.** Compute `theta` and `phi` and their
  `sin`/`cos` in `double`, then cast the products. `generateIcosphere` already does
  its arithmetic in `double` and casts (`detail::normalize3`); match it.
- **The poles are exact.** Do not compute `sin(0)*cos(phi)`; write `(0, 0, 1)` and
  `(0, 0, -1)` literally. `sin(pi)` is not zero in floating point, so the south pole
  computed from the formula is `(±1.2e-16, ∓1.5e-32, -1)` — off the unit sphere in
  the last bits and, worse, *different* for different `phi`, which is how a
  duplicate-pole bug hides.
- **No seam duplicate.** `phi_i` runs `i = 0 .. nLon-1` and the last quad wraps to
  `i = 0`. Generating `nLon + 1` meridians produces a coincident seam ring, a
  non-manifold mesh, and a `buildFromTriangleSoup` edge map that silently disagrees
  with itself. This is the single most common UV-sphere bug.
- **`nLat == 3`** is the degenerate-but-legal case: two pole fans and no interior
  quads, i.e. a bipyramid with `2*nLon` faces. It must work, since it is the
  boundary of the formula.
- **Reproducibility caveat.** `sin`/`cos` at computed angles may differ in the last
  bit across libm implementations and platforms, so positions are **not** guaranteed
  bit-reproducible across machines the way the icosphere's (rational base table plus
  `sqrt`) nearly are. Document it in the header — a consumer comparing against a
  gold file generated elsewhere needs to know.

### Non-goals

- A distributed lat/lon generator. Once
  [distributed-coarse-build.md](distributed-coarse-build.md) lands, its
  `buildFromTriangleSoupDistributed` takes patches plus canonical keys, and a
  distributed lat/lon generator is a natural follow-on — the canonical key is just
  `{j * nLon + i, invalid_gid}`. Note the follow-on; do not build it here.
- Other generators (torus, plane, cylinder, cube-sphere). Add on demand.
- Adaptive or Delaunay triangulation of the quads.

### Tests

New `tests/test_latlon_sphere.cpp`, registered at **TIER `regression`**, backends
**SERIAL and HIP**, ranks **`${TESSERA_TEST_MPI_RANKS}` (1–5)**. Gate promotion is
**pre-authorized for this task**.

Parameterize the checks over several `(nLat, nLon)`: `(3,3)`, `(3,8)`, `(5,4)`,
`(9,12)`, `(4,17)` — including the `nLat == 3` degenerate case and a prime `nLon`
so no check accidentally relies on divisibility.

1. **Closed-form counts.** After `buildLatLonSphere` at rank 1: `numVertices`,
   `numEdges`, `numFaces` equal the formulas above, for every parameter pair. Then
   after `distribute` at ranks 1–5: `globalOwnedVertices/Edges/Faces` match and
   `globalOwnedEuler == 2`.
2. **Manifoldness.** Every edge has exactly two incident faces —
   `EdgeField::Faces[1] != invalid_gid` for every edge. This is the check that
   catches the seam-duplicate bug, and it catches it loudly.
3. **No duplicate vertices.** The position multiset has no two entries within
   `1e-14` of each other. Catches the duplicated-pole bug, which check 2 can miss
   when the duplicates happen to pair up consistently.
4. **On the unit sphere.** Every vertex position has `|x| == 1` to `1e-15`; the two
   poles are **exactly** `(0,0,±1)` (bitwise). The bitwise part pins the exact-pole
   decision.
5. **Outward winding.** For every face, `faceNormalRaw · centroid > 0`. And the
   enclosed volume (summed over faces as the signed tetrahedron volume to the
   origin) is **positive** and converges to `4π/3` from below as `nLat, nLon` rise —
   check `(9,12)` is within 5% and `(33,64)` within 0.5%. A sign error here inverts
   every normal downstream and nothing else in the suite would catch it.
6. **Pole fan structure.** Exactly `nLon` faces are incident on each pole vertex,
   and each pole vertex has valence `nLon`. Verified via `mesh.vertexFaces()` /
   `vertexEdges()` at rank 1.
7. **Valence histogram.** Interior vertices have valence 6 except along the quad
   diagonal convention; assert the histogram matches the closed form derived from
   the fixed diagonal choice (compute it in the test from `nLat`/`nLon` rather than
   hardcoding), and that no interior vertex has valence below 4. Pins the diagonal
   convention.
8. **Anisotropy is real** — the reason the surface is worth having. Report the ratio
   of maximum to minimum triangle area and of maximum to minimum edge length at
   `(33,64)`, and assert the area ratio exceeds 10. If it does not, the generator is
   not producing the pole stretching the test is meant to exercise.
9. **Degenerate arguments throw.** `nLat < 3` and `nLon < 3`, each polarity, throw
   `std::invalid_argument`.
10. **Downstream operations work.** On a `(9,12)` mesh: uniform `refine()`
    (`V' = V+E`, `E' = 2E+3F`, `F' = 4F`, `checkConforming`, `check21Balance`,
    `checkMidpointAgreement`); `migrate()`; `loadBalance()`; `haloExchange()` with
    ghost positions equal to owners'; `writeMesh`/`readMesh` round trip. The point
    is that the high-valence poles and the anisotropy do not break anything — which
    is a genuine test of the *rest* of the library, not just of the generator.
11. **`markByQuality` on an anisotropic mesh.** Run the existing quality marker at
    `(33,8)` — deliberately very anisotropic — and assert it marks the polar bands
    and not the equatorial ones. Exercises `Tessera_MarkQuality.hpp` on input the
    icosphere cannot produce.

## Exit criterion

- `test_latlon_sphere` green at **SERIAL and HIP, ranks 1–5**, and the full gate
  still green with nothing relabelled.
- Checks 2, 3, 4 and 5 pass for every parameter pair, including `nLat == 3` — those
  four are the four classic UV-sphere bugs.
- Check 8 confirms the anisotropy, with the measured area and length ratios recorded
  in the progress log.
- Check 11 passes, or if it does not, the `markByQuality` behaviour on an
  anisotropic surface is characterized and recorded in README *Known Issues*. Do not
  weaken the check to green it — finding something here is a legitimate outcome and
  is half the reason to add this generator.
- README API section documents `generateLatLonSphere` / `buildLatLonSphere`, the
  parameter meanings, the closed-form counts, the vertex ordering, the winding, the
  fixed quad diagonal, and the libm reproducibility caveat.
- `docs/design.md` mentions the generator beside the icosphere and states why it is
  worth having as a test surface (anisotropy, high-valence poles).

## Where this sits

Fully independent — no prerequisites and nothing depends on it. A distributed
version is a natural follow-on to
[distributed-coarse-build.md](distributed-coarse-build.md). See the ordering diagram
in [halo-depth.md](halo-depth.md).

## Progress log

- 2026-08-07 — Task written, then re-checked against `08dd346` after pulling
  `../tessera`. Nothing implemented.
- 2026-08-10 — **Implemented and green.** `generateLatLonSphere()` added to
  `src/Tessera_Icosphere.hpp` (beside `TriangleSoup`, as the task preferred) and
  `buildLatLonSphere()` to `src/Tessera_MeshBuilder.hpp` beside
  `buildIcosphere()`; one new level-1 profiling region,
  `build_latlon_sphere`. Purely additive — no existing code path was touched.
  New `tests/test_latlon_sphere.cpp` at TIER `regression`, SERIAL + HIP, ranks
  1–5: **10/10 green on the first run.** All eleven checks implemented; every
  printed figure is byte-identical at np1 and np5 on both backends and both
  execution spaces.

  **Check 8 — the anisotropy, measured at `(33,64)`:**
  max/min triangle area **10.2145** (`4.714122e-04` → `4.815259e-03`), max/min
  edge length **14.4108** (`9.618946e-03` → `1.386172e-01`). The assertion is
  `areaRatio > 10`, which this clears — but only by 2%, so the bound is tight
  rather than generous. That is a property of the parameter pair the task chose,
  not of the check: at `(33,64)` `dθ = dφ = π/32`, so the mesh is nearly square
  at the equator and all the anisotropy comes from the polar shrinkage.

  **Check 5 — the volume tolerance in the task text was wrong.** The measured
  enclosed volumes are `V(9,12) = 3.847759065` and `V(33,64) = 4.171995762`
  against `4π/3 = 4.188790205`, i.e. deficits of **8.1415%** and **0.4009%**.
  The task asked for `(9,12)` "within 5%", which is arithmetically unreachable —
  a UV polyhedron inscribed at that resolution cannot do better than its own
  vertices. The `(33,64)` figure does meet its 0.5%. Rather than drop the check
  or restate a number that happens to pass, the test asserts something stronger
  than either percentage: the summed signed tetrahedron volume telescopes in
  closed form to

      V = (nLon·sin(2π/nLon)/6) · 2 sin(dθ) · Σ_{j=1..nLat-2} sin(θ_j),  dθ = π/(nLat-1)

  (because `z_j·r_{j+1} − z_{j+1}·r_j ≡ sin dθ` and `r_1 = r_{nLat−2} = sin dθ`),
  and the measured volume is required to match that to `1e-12` relative **for
  every parameter pair**, plus `0 < V < 4π/3` and a converging deficit. The
  closed form pins the winding, the quad diagonal and both pole fans at once,
  where a percentage bound pins none of them. Sanity check on the formula:
  `(3,4)` is the regular octahedron and it gives exactly `4/3`.

  **Check 11 — the prediction was inverted, and the reason is geometric.** The
  task expected `markByQuality` to mark "the polar bands and not the equatorial
  ones" at `(33,8)`. It marks the equator, and it is right to. At `(33,8)` the
  meridional chord is `2 sin(dθ/2) = 0.0981` and latitude-independent, while the
  in-ring chord is `2 sin θ_j sin(π/8) = 0.7654 sin θ_j` — so the *equatorial*
  triangles are the long, 7.8:1-stretched ones and the *polar* ones are small and
  nearly isotropic. `EdgeLengthCriterion` marks long edges; therefore it marks
  the equator, exactly per its documented contract. Measured at `maxLen = 0.4`:
  352 of 496 faces marked; of the 128 faces entirely inside the polar caps
  (`|z| ≥ cos 28.125°`) **0** marked, of the 192 entirely inside the equatorial
  band (`|z| ≤ cos 56.25°`) **all 192** marked. `CurvatureCriterion` selects the
  same way round: at `40°` it marks 160 faces, all equatorial and none polar; at
  `20°` it marks 384, of which only 16 — the two pole fans themselves, 8 faces
  each — are polar. So the check was not weakened to green it: it asserts the
  *correct* direction, against closed-form latitude cut points computed in the
  test, with both bands asserted non-vacuous and the marked set asserted to be a
  strict non-trivial subset (so neither an all-marked nor an empty mask could
  pass). The behaviour is recorded in README *Known Issues* and in
  `docs/design.md` → *Quality-based refinement marking* as the task's exit
  criterion requires, since "polar" and "badly shaped" being different sets is a
  real trap for a consumer driving AMR off a lat/lon mesh.

  **Check 10** additionally verifies ghost positions against their *owners*
  through a gid coordinator (`gid % size`) rather than by re-reading the halo's
  own bookkeeping, after each of `distribute`, `refine`, `migrate`,
  `loadBalance` and `readMesh`.

  Documented in README (*Mesh generators*, with the parameter meanings, the
  closed-form counts, the vertex ordering, the winding, the fixed quad diagonal
  and the libm reproducibility caveat) and in `docs/design.md` (new *Mesh
  generation* section). The distributed lat/lon generator remains a noted
  follow-on to [distributed-coarse-build.md](distributed-coarse-build.md), not
  built here.
