# Milestone 1 — Distributed Unstructured Triangle Mesh

> **Handoff contract.** This library is built across multiple sessions with
> different models. This file is the source of truth for *what to build next, where
> it starts/ends, how it's accepted, who builds it (Sonnet/Opus), and what to report
> back* so the next session resumes cleanly. **At the start of any session touching
> this task, read this file first, then append to the Progress log as work lands.**
> The full design rationale lives in `README.md` (design reference) and the approved
> plan `plans/compiled-hopping-torvalds.md`.

## Problem

Tessera must provide the distributed, halo-able, refinable triangle mesh that the
Canopy FMM solver and the Beatnik/rocketrig rising-bubble problem build on. The
global Birkhoff–Rott/FMM velocity solve is **out of scope** (it lives in Canopy);
Tessera owns the local mesh machinery: entity storage, connectivity, 1-deep halo,
split-based AMR, optional migration/load-balancing, and parallel I/O.

## Approach

C++17, header-first, Kokkos + Cabana AoSoA, hand-built MPI. Mirror three proven
Canopy patterns (`~/spack_envs/tuolumne_beatnik/canopy/src/Canopy_{TreePartitioner,
MpiCoalescedExchange,RegisteredBufferPool}.hpp`) — do not modify Canopy. Key design
locks (see README for detail):

- `Mesh<Scalar, int Dim = 3, VertexFields, EdgeFields, FaceFields, MemorySpace,
  ExecSpace>` — **scalar precision AND embedding dimension are template parameters**
  (never hard-code `double`/`3`). `position` is `Scalar[Dim]`; `Dim=3` for the
  bubble, `Dim=2` expressible later.
- **Compile-time Cabana field pack** per entity kind for arbitrary user data
  (e.g. `VertexFields<Vort2>`); every field haloes/migrates with its entity.
- **Structured 128-bit gids** (ordered endpoint-gid pair), never a hash — guarantees
  cross-rank determinism with no comm and no collision risk at 1e8+ entities.
- **Lowest-rank ownership** of shared V/E; sharing set materialized by ghost build.
- **Conforming, 2:1-balanced** split refinement (hanging nodes bounded to ≤1 level).
  Midpoint placement via a **pluggable interpolation policy** (default linear
  midpoint for `position` + linear average for fields, per-field override hook).
- **Migration is the public contract** (`migrate(dest_rank_per_owned_face)`);
  internal Zoltan2 `loadBalance()` is a thin wrapper. LB is **optional** — Canopy
  can drive `migrate()` externally to avoid double migration.
- **Manual parallel HDF5 + XDMF** I/O with a full round-trip reader; dense global
  numbering via `MPI_Exscan` over owned-only counts.

### Gate / conventions (binding — see `CLAUDE.md`)

- Regression gate = label `regression` × backends {SERIAL, HIP} × ranks {1,2,3,4,5},
  single-sourced across `tests/CMakeLists.txt`, `run_regression_minset.flux`, CI,
  and `CLAUDE.md`. Unit/diagnostic tests use label `unit`.
- Every new source file carries the BSD-3-Clause SPDX header.
- README kept in sync with public API / example args. `--target format-check` clean.
- A failing test is never silently dropped from the gate (label `unit`, record in
  README Known Issues, report it).

## Step contract

Per-step model assignment: **Sonnet** = mechanical / lower-complexity (executed
against an Opus-written spec where the column says so); **Opus** = high design
complexity (distributed determinism, comm substrate, parallel AMR balance, Zoltan2
migration path).

| # | What to implement | Starts / ends | Acceptance (tests) | Model | Report back |
|---|---|---|---|---|---|
| 0 | Add `hdf5 +mpi` to the spack env; commit `systems/tuolumne/spack.yaml` snapshot; reconfirm full gate builds. | Start: edit spack env. End: gate builds clean with HDF5 discoverable by CMake. | Existing gate (Steps ≤7 once present) still green; `find_package(HDF5)` succeeds. | Sonnet | HDF5 version added; whether the Trilinos-adjacent rebuild perturbed anything; exact `spack.yaml` diff. Do **right before Step 8** to keep the env-mutation window small. |
| 1 | Design docs: this file + README design/data-model/API section. No code. | Start: TEMPLATE.md + README placeholder. End: README is a usable design reference; this contract complete. | N/A (docs). | Opus | Any design ambiguity found while writing; confirm README API matches intended template signature. **(DONE — see Progress log.)** |
| 2 | Core data model: `Mesh<Scalar, int Dim = 3, ...>` template skeleton; core topology AoSoAs + templated user-state AoSoA (field pack, `position` mandatory `Scalar[Dim]`); 128-bit structured-key utils (edge key, midpoint/child derivation); dense-local-index + canonical-key side table; CSR vertex-adjacency container. Convert INTERFACE target to real lib if needed. | Start: empty `src/`. End: headers under `src/mesh/`; mesh constructs empty; keys + CSR usable. | Unit (SERIAL+HIP), for **both `double` and `float`** (and a `Dim=2` compile check): structured-key determinism (ordering-invariant, same key regardless of endpoint order); AoSoA build; field-pack slice read/write. | Sonnet (keys Opus-spec'd) | Final gid/key struct + footprint decision (128-bit in AoSoA vs side table); field-pack declaration syntax; any Cabana template friction on HIP. |
| 3 | Serial mesh builder: coarse icosphere triangle soup → derive edges + both-direction connectivity (face→v/e, edge→f, vertex→e/f CSR). Single rank. | Start: Step 2 types. End: `buildIcosphere(subdiv)` yields a valid connected mesh. | Unit: Euler `V−E+F=2`; every edge has exactly 2 incident faces; every face has 3 edges + 3 verts; CSR 1-ring round-trips. | Sonnet | Icosphere generation choice (recursive subdivision vs lookup); base entity counts per subdiv level for test fixtures. |
| 4a | Comm primitive — migrate. Port `RegisteredBufferPool` verbatim; whole-tuple migrate generic over any AoSoA tuple (Alltoall counts + `MPI_Type_contiguous(sizeof tuple)` Isend/Irecv); self-peer guard. GPU-resident. | Start: Step 2. End: `src/comm/{RegisteredBufferPool,Migrate}.hpp`. | Round-trip migrate at ranks 1–5, SERIAL+HIP: send tuples to a known rank pattern, verify counts + payload; np1 self-peer no-op doesn't fault on HIP. | Opus | Confirm MI300A self-peer + signed-int-count guards reproduce; any Cabana tuple-copy gotchas on device. |
| 4b | Comm primitive — `HaloExchangePlan` (peer lists + gid→local-index maps + pools, `rebuild()`); in-place field-sync generalized to arbitrary value type / whole pack; variable-arity topology/connectivity exchange (for ghost build / sharing-set discovery). **Invariant: any local-count or ghost-set change → `rebuild()` before next sync.** | Start: 4a. End: `src/comm/{HaloExchangePlan,FieldSync}.hpp`. | Ranks 1–5: hand-build a plan, sync a multi-field pack, compare ghosts to owners; topology exchange reconstructs a known neighbor map. | Opus | The plan object's public shape (what Steps 5/6/7 call); how stale-plan invalidation is signaled; aligned pack-order rule used. |
| 5 | Distributed mesh + ownership + 1-deep halo. Replicated deterministic static partition (no Zoltan2); ghost build via 4b topology exchange; sharing-set → **lowest-rank** ownership; first halo field sync. | Start: 3 + 4b. End: `Mesh` distributes, owns, and halos. | Regression ranks 1–5 SERIAL+HIP: ownership is a partition (Σ owned = global; no gid double-/un-owned); every owned vertex's 1-ring local; ghost fields == owner; np1 = zero ghosts, no-op exchange. | Opus | Static-partition scheme used; the sharing-set discovery exchange; which invariants moved into `tests/MeshInvariants.hpp`. |
| 6a | Local 1→4 red refinement (single rank): midpoint vertices via structured keys; 4 child faces; connectivity rebuild. **Pluggable interpolation policy** (`src/mesh/RefinePolicy.hpp`): default linear midpoint for `position` + linear average for user fields, per-field override hook. | Start: 3. End: `refineLocal(face_mask)`. | Unit: Euler holds post-refine; child/parent incidence correct; midpoint shared between sibling faces; midpoint position == endpoint average under default policy; a custom policy overrides one field. | Sonnet | Child-face vertex/edge ordering convention; how `level` is propagated; the policy hook signature. |
| 6b | Parallel 2:1 balance + re-halo. Mark-and-propagate fixpoint (decision pure in gids+levels); sync `level` each iteration; `MPI_Allreduce` changed-count to fixpoint with hard cap + strictly-decreasing assert; rebuild halo plan each iteration. | Start: 6a + 5. End: `refine()` conforming across ranks. | Regression ranks 1–5: shared boundary edge midpoint gid bit-identical both sides (**key test**); 2:1 invariant; Euler over owned-only. | Opus | Iteration-count distribution observed; the cap value; any nondeterminism caught by the strictly-decreasing assert. |
| 7 | Migration API + optional Zoltan2 LB. Public `migrate(dest)` (faces + their V/E + whole field pack move) → recompute ownership → rebuild halo → sync (order explicit). Accessors `ownedFaceCentroids/Gids/Weights`. `loadBalance()` = Zoltan2 MultiJagged (SerialComm, solve-on-0+Bcast, never RCB) → `migrate()`. | Start: 4a + 5. End: `src/balance/Zoltan2Balancer.hpp` + `Mesh::migrate/loadBalance`. | Regression ranks 1–5: external path — hand-computed `dest` (and synthetic Canopy-style assignment) lands entities on requested ranks, invariants preserved; internal path — balance improves; rank-count-independent topology checksum. | Opus | The migrate→ownership→halo ordering that avoids phantom sends; Zoltan2 adapter/param specifics reused from Canopy; external-API surface Canopy will call. |
| 8 | Parallel HDF5 + XDMF writer + round-trip reader. Dense global numbering via `MPI_Exscan` over owned-only counts; XDMF references dense indices; write connectivity + per-V/E/F fields; reader reconstructs a mesh re-passing Step-5 invariants. (Needs Step 0 done.) | Start: 5 + Step 0. End: `src/io/{HDF5Writer,HDF5Reader}.hpp`. | Regression ranks 1–5: write→read→compare topology + field checksums; on-disk global checksum **independent of rank count**; opens in Paraview. | Sonnet (layout Opus-spec'd) | HDF5 dataset layout + XDMF schema used; the dense-numbering exscan; any collective-IO tuning needed on Tuolumne. |
| 9 | End-to-end example (icosphere→refine→rebalance→halo→write); separate unit vs regression runner scripts under `scripts/<system>/`; OPENMP smoke-build (non-gate); CI subset wiring; README/example-arg sync. | Start: 5–8. End: `examples/` + scripts + CI green. | E2E example runs at ranks 1–5; CI subset (`regression`/SERIAL ranks 1–2) green. | Sonnet | Example CLI args (mirror to README); runner script names; CI subset confirmation. |
| 10a | Geometric quality-based refinement marking — **edge-length**. `QualityCriterion` concept (a struct with `std::vector<char> mark(const MeshT&) const`, mirroring `RefinePolicy`); ship `EdgeLengthCriterion<Scalar>{maxLen}` (mark an owned face if **any** of its 3 edges exceeds `maxLen`). Free fn `markByQuality(mesh, crit)` + scalar convenience overload `markByQuality(mesh, Scalar maxLen)` returning `std::vector<char>` sized `numOwnedFaces()`, consumed by `refine()` unchanged. On-demand from existing `Position`/`Verts` — **no new stored fields**; device Kokkos `parallel_for` over owned faces (host-built face→vertex-local index view; reads device `Position` slice) → device char marks → host vector. **No comm** (pure per-face geometric function; marked set is partition-independent for free). | Start: `refine()` (Step 6b). End: `src/Tessera_MarkQuality.hpp` (edge-length); added to umbrella. | Regression 1–5 SERIAL+HIP: marked **set** is rank-count-independent; `markByQuality → refine` drops max owned edge length below threshold (monotone progress); refine invariants (2:1, midpoint agreement, owned Euler) still hold. | Sonnet (this spec) | Device-kernel triviality confirmed (or downgraded to future-opt with reason); the exact `QualityCriterion` method signature as built; edge-length test thresholds per subdiv level. |
| 10b | Geometric quality-based refinement marking — **curvature (dihedral)**. `CurvatureCriterion<Scalar>{maxAngle}` (radians): mark **both** faces incident to any edge whose dihedral bend exceeds `maxAngle`. Needs both incident faces' normals; the neighbour across a boundary edge is **not** guaranteed in the vertex-based 1-ring halo (Step 6b), so gather via **edge coordinators** — reuse `detail::edgeCoordRank` + `allToAllV`: device-compute per-owned-face unit normals, advertise `(EdgeKey, normal, faceGid, owner)` to the coordinator, coordinator receives exactly 2/edge on the closed surface, computes the dihedral, routes marks back to face owners. Extends `markByQuality` dispatch (criterion owns its own evaluation incl. comm). | Start: 10a + the refine() coordinator pattern (6b). End: `CurvatureCriterion` in `src/Tessera_MarkQuality.hpp`. | Regression 1–5 SERIAL+HIP: marked set rank-count-independent (identical faces marked regardless of partition, via the coordinator gather, **not** the halo); a synthetic sharp-fold fixture marks exactly the fold faces; `markByQuality → refine` invariants hold. | Opus (cross-rank determinism) | Coordinator round count/pattern reused from 6b; the normal-orientation/winding assumption relied on; dihedral threshold used in the test. |

## Progress log

- 2026-06-30 — **Step 1 started/landed (Opus).** Plan approved
  (`plans/compiled-hopping-torvalds.md`). Confirmed environment: Trilinos 16.2
  (+zoltan2), Cabana, Kokkos 4.7 (Serial/OpenMP/HIP gfx942), googletest present in
  `~/spack_envs/tuolumne_trilinos`; **HDF5 absent** (Trilinos `~hdf5`) → Step 0 will
  add `hdf5 +mpi`. Canopy reference confirmed at `~/spack_envs/tuolumne_beatnik/
  canopy` (the path in `background.md`, `~/spack_envs/beatnik/canopy`, does not
  exist). Wrote this handoff contract and the README design reference. **Next:**
  Step 2 (core data model) — Sonnet, against the gid/key spec in README.
- 2026-07-01 — **Design refinements (Opus, plan-mode).** Two user-requested changes,
  folded into the plan/README/this doc: (1) the mesh is now templated on embedding
  dimension too — `Mesh<Scalar, int Dim = 3, ...>`, `position : Scalar[Dim]` (`Dim=3`
  for the bubble; `Dim=2` expressible later). (2) Refinement midpoint placement is a
  **pluggable interpolation policy** (`src/mesh/RefinePolicy.hpp`): default linear
  midpoint for `position` + linear average for user fields, with a per-field override
  hook for a later curvature-aware/physics-correct rule. No code yet; Step 2 remains
  the next action.
- 2026-07-01 — **Step 2 landed (Opus; Sonnet delegation unavailable this env — the
  `sonnet` alias 403s for this subscription and raw model ids are rejected, so all
  steps run on Opus; the Sonnet/Opus column is advisory here).** Core data model
  implemented header-first under `src/` (flat `Tessera_*.hpp`, included as
  `<Tessera_Mesh.hpp>` per the README):
  - `Tessera_Types.hpp` — `GlobalId`/`LocalIndex`; structured `Key<N>` (EdgeKey=Key<2>,
    FaceKey=Key<3>) with order-invariant `makeEdgeKey`/`makeFaceKey`, device-callable.
    **Key-scheme decision resolved:** the 128-bit key is the cross-rank *matching
    identity* (side table); the 64-bit gid is the persistent identity; midpoint gid
    *assignment* (local exscan interior + owner-broadcast on boundary) is deferred to
    Step 6, which keeps keys bounded at Key<2>/Key<3> across refinement depth.
  - `Tessera_Fields.hpp` — core member layouts + `MemberTypesCat` to append the
    compile-time user field pack; named core slice indices (`VertexField::*` etc.) and
    `userVertexField<M>()`/`userEdgeField`/`userFaceField` for user slices.
  - `Tessera_CsrAdjacency.hpp` — CSR 1-ring container (built Step 3).
  - `Tessera_Mesh.hpp` — `Mesh<Scalar, int Dim=3, VUser, EUser, FUser, Mem, Exec>`
    skeleton: 3 AoSoAs, CSR members, key side tables, slice/resize/accessors.
  - Umbrella `Tessera.hpp`.
  Unit tests `tests/test_keys.cpp` + `tests/test_data_model.cpp` (self-contained, no
  gtest) run host **Serial** and device **Default(HIP)**, for **double and float**,
  incl. a **Dim=2** build; wired via `tessera_add_test` (unit tier, SERIAL+HIP, np1).
  All 4 pass. Build: `run_cmake_toulumne.sh` + make (SERIAL+HIP compile clean);
  `format-check` clean.
  - **Env fix (reach-Flux-tasks):** Cray `CC` links libsci and ROCm dlopens large-TLS
    libs → every binary aborted at load with *"libsci_cray_mp.so.6: cannot allocate
    memory in static TLS block"*. Fixed by `export
    GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8388608` in
    `scripts/tuolumne/runtime_env.sh` (LD_PRELOAD of libsci does NOT fix it and breaks
    flux itself; the tunable is the right remedy).
  - **Dev runner:** `scripts/tuolumne/run_unit_tests.flux` submits a ctest label to the
    pdebug queue via `flux batch` (login nodes can't `flux run` directly). Requires
    `export TESSERA_REPO=$(pwd)` before submit. Not a release artifact (Step 9
    formalizes runners). **Next:** Step 3 (serial icosphere builder + connectivity).
- 2026-07-01 — **Step 3 landed (Opus).** Serial builder + connectivity:
  - `Tessera_Icosphere.hpp` — `generateIcosphere<Scalar>(subdiv)` → watertight
    triangle soup (base icosahedron + 1→4 subdivision with shared-midpoint cache,
    projected to the unit sphere); deterministic/replicable.
  - `Tessera_MeshBuilder.hpp` — `buildFromTriangleSoup(mesh, soup)` /
    `buildIcosphere(mesh, subdiv)`: derive unique edges (dedup by EdgeKey), fill
    V/E/F AoSoAs (face convention `e[k]=edge(v[k],v[(k+1)%3])`), edge→face +
    face→v/e, vertex→faces and vertex→edges CSR, edge/face key side tables. Host
    derivation (ordered maps) then deep_copy into the mesh's (device) storage.
    Serial: gid == local index (Step 5 introduces the gid→local map).
  - Test `tests/test_connectivity.cpp` (unit, SERIAL+HIP, np1): exact V/E/F counts +
    Euler, every edge has 2 faces, faces have 3 valid distinct v/e, both vertex CSR
    relations round-trip (degrees 3F / 2E, membership), vertices on the unit sphere;
    subdiv 0–3. All 6 unit tests pass.
  - **Build/run env split fix (important, supersedes the Step-2 note):**
    `GLIBC_TUNABLES` **segfaults the Cray linker** if present at build time, but the
    run binaries need it. So it was REMOVED from `runtime_env.sh` (the resolver
    sources that for builds too) and is instead injected per test task via
    `MPIEXEC_PREFLAGS` `--env=GLIBC_TUNABLES=...` in `run_cmake_toulumne.sh` (flux
    run passes it to the task only; flux/compiler/linker env stays clean).
    `systems/tuolumne/claude.md` updated to match. **Reconfigure required** after
    pulling: re-run `run_cmake_toulumne.sh`. **Next:** Step 4a (comm migrate).
- 2026-07-01 — **Step 4a landed (Opus).** Migrate comm primitive:
  - `Tessera_RegisteredBufferPool.hpp` — verbatim port of Canopy's grow-only
    registered device pool (1.5x headroom, stable base address) into
    `Tessera::detail`; bounds CXI NIC registration churn.
  - `Tessera_Migrate.hpp` — `MigrateBuffers<Mem>` (persistent send/recv/idx pools)
    + `migrate(comm, aosoa, dest, bufs)`: generalizes Canopy migrate_particles to a
    caller-supplied per-element destination-rank array (dest(i)=owning rank).
    Alltoall(counts) → device pack into one registered send region → whole-tuple
    `MPI_Type_contiguous(sizeof(tuple))` Isend/Irecv → rebuild AoSoA as kept++recv.
    Self-peer never posted to MPI; np1/no-move fast path does zero MPI (MI300A
    self-send guard). Generic over any AoSoA tuple / Scalar, GPU-resident pack.
  - Test `tests/test_migrate.cpp` (unit, SERIAL+HIP, **np1–5**): scatter
    (dest=gid%size), shift (dest=(rank+1)%size, num_kept==0), identity (fast path,
    returns 0); verifies global conservation, correct landing rank, payload
    integrity, no duplicates. All 16 unit tests pass (incl. migrate np1–5 on HIP).
    **Next:** Step 4b (HaloExchangePlan + field-sync + topology-exchange).
- 2026-07-01 — **Step 4b landed (Opus).** Halo substrate:
  - `Tessera_AllToAllV.hpp` — `allToAllV<T>(comm, send)`: variable-arity neighbour
    topology exchange (Alltoall counts + Alltoallv bytes), host-side; the primitive
    Step 5 uses to advertise boundary gids/keys and build ghosts / discover sharing
    sets. Result groups received items by source rank. `T` trivially copyable.
  - `Tessera_HaloExchange.hpp` — `HaloExchangePlan<Mem>` (per-peer send/recv index
    maps grouped by peer + offsets, persistent registered pools, `clear()`,
    `setFromHost()` which drops self-peer; **alignment contract**: builder orders
    shared entities by gid on both sides so pack/unpack align) and
    `haloExchange(comm, aosoa, plan)` — whole-tuple field sync (syncs the entire
    field pack of every ghost from its owner), in place (no resize), GPU-resident,
    self-peer never posted. Invalidation invariant documented: any local-count/
    ghost-set change → rebuild before next sync.
  - Test `tests/test_halo.cpp` (unit, SERIAL+HIP, np1–5): allToAllV variable-length
    correctness; ring halo (send owned→rank+1, fill ghosts←rank-1) verifying every
    ghost gets the owner's whole tuple, owned untouched, np1 empty-plan no-op. All
    26 unit tests pass.
  - Gotcha fixed: `Kokkos::view_alloc(WithoutInitializing, label)` with a
    `const char*` variable is misread as a wrap-user-memory pointer; wrap in
    `std::string(label)`. **Next:** Step 5 (distributed mesh + ownership + 1-deep
    halo build).
- 2026-07-01 — **Step 5 landed (Opus). First regression-tier test.** Distributed
  mesh + ownership + 1-deep halo:
  - `Mesh`: added owned-count state (`numOwned{Vertices,Edges,Faces}`,
    `setOwnedCounts`); convention = entities stored owned-first, `[0,n_owned)` owned
    then ghosts. Serial builder marks all-owned. `fillCsr` moved to
    `Tessera_CsrAdjacency.hpp` (shared by builder + distribute).
  - `Tessera_Distribute.hpp`: `facePartitionByAxis` (deterministic geometric block
    partition — sort faces by centroid axis, tie-break gid, block into bands;
    replicated → identical on all ranks, no comm; Zoltan2 deferred to Step 7).
    `distribute(mesh, halo, faceOwner)`: lowest-rank ownership (vertex/edge = min
    incident faceOwner), 1-deep local closure (local faces = owned + incident-to-
    owned-vertex; local v/e = union over local faces), compacts to owned-first local
    AoSoAs, rebuilds local CSR, and builds the 3 `HaloExchangePlan`s. Send side of
    each plan discovered via `allToAllV` (advertise ghost gids to owners); recv/send
    aligned by ghoster gid order. `MeshHalo<Mem>` holds the 3 plans;
    `haloExchange(mesh, halo)` syncs V/E/F.
  - `tests/MeshInvariants.hpp` (shared, Steps 5–8): `checkOwnershipPartition`
    (Σowned==global + no gid double-owned via coordinator), `owned1RingLocal`,
    `topologyChecksum` (rank-count-independent BXOR). LOCAL-fail convention (sum ==
    global).
  - `tests/test_distribute.cpp` (**regression**, SERIAL+HIP, np1–5): partition →
    distribute → invariants → halo sync (corrupt ghosts, `haloExchange`, verify
    restored to owner's gid/owner; owned untouched; np1 no-op). All pass: regression
    10/10, unit 26/26.
  - Ran via `flux batch scripts/tuolumne/run_unit_tests.flux regression` (the dev
    runner takes a label arg). **Next:** Step 6a (local 1→4 refinement).
- 2026-07-01 — **Step 6a landed (Opus).** Local single-rank 1→4 red refinement:
  - `Tessera_RefinePolicy.hpp` — `DefaultRefinePolicy<Scalar>` with two hooks:
    `interpolatePosition(mid,a,b,dim)` (linear midpoint; **no** sphere projection —
    projection is icosphere-generation only) and `template<std::size_t M>
    interpolateVertexField(a,b)` (linear average of one scalar component of the vertex
    user field at ABSOLUTE member index `M`, called per component for `Scalar[N]`).
    **Per-field override pattern:** derive + shadow `interpolateVertexField`,
    dispatch on `M` with `if constexpr`, else fall through to the base — exactly one
    field's rule changes.
  - `Tessera_Refine.hpp` — `refineLocal(mesh, faceMask, policy=Default)`. Rebuild-
    from-face-soup: read current verts+faces (old edges ignored — re-derived),
    create one midpoint per split edge **deduped by `EdgeKey`** (endpoint-gid pair;
    the mechanism 6b extends across ranks — midpoint gid == new local index,
    preserving the single-rank `gid==index` invariant), emit kept + 4-child faces,
    then re-derive edges/CSR/edge+face key tables like the builder. **Child ordering
    `{a,ab,ca},{b,bc,ab},{c,ca,bc},{ab,bc,ca}`** matches the icosphere subdivision, so
    a uniform mask reproduces one subdivision level (V,E,F + Euler). **Level
    propagation:** child faces = parent+1; kept faces unchanged; derived edge level =
    min incident face level. Face user fields inherited from parent (compile-time
    member copy over the user pack, `MemberTypeAtIndex` + `std::rank`/`std::extent`,
    empty pack → no-op); edge user fields reset on refine (M1 carries no edge user
    state through AMR — documented). Partial masks leave bounded hanging nodes for
    Step 6b to balance.
  - Test `tests/test_refine.cpp` (unit, SERIAL+HIP, np1): (A) uniform refine of
    subdiv-0 → V42/E120/F80, Euler=2, every midpoint == plain endpoint average and
    strictly interior (no projection); (B) single-face refine → exactly 3 shared
    midpoints + 4 children spanning `{corners}∪{midpoints}`, each corner used once,
    each midpoint shared by ≥2 siblings; (C) default policy averages a user field vs
    a custom `if constexpr` policy overriding field 0 to a constant while position
    stays on the default. All 28 unit tests pass (refine np1 SERIAL+HIP);
    `format-check` clean.
  - **Report-back (per the contract):** child vertex/edge ordering = the subdivision
    convention above; `level` = parent+1 on child faces/edges (edge = min incident
    face level); policy hook signature = `interpolatePosition` + per-field
    `interpolateVertexField<M>` on absolute member index. Cabana note: an empty user
    pack makes `slice<UserBegin>` ill-formed, so any user-field access must be behind
    `if constexpr` on the member count (hit in the test helper; fixed). **Next:**
    Step 6b (parallel 2:1 balance + re-halo).
- 2026-07-01 — **Step 6b landed (Opus). Second regression-tier test.** Distributed
  2:1-balanced refinement:
  - **Scope decision (user-approved):** deliver + test the #1 risk (distributed 2:1
    balance and bit-identical cross-rank midpoint gids); **defer the post-refinement
    halo rebuild to Step 7**, where the general (non-replicated) ghost builder is
    shared with migration. All three acceptance invariants are owned-entity
    properties needing no halo, so this is a clean cut.
  - **Key architecture — edge coordinators, not the halo.** The Step-5 vertex-based
    1-ring halo does NOT guarantee a face sees its edge-neighbour across a partition
    boundary (verified by case analysis: the edge/vertex owner can be a *lower* rank
    than either incident face's owner at 3-way corners), so a halo-based level sync
    would be incomplete. Every cross-rank decision is instead routed to
    `edgeCoordRank(EdgeKey)=hash%nranks`, which gathers an edge's two incident faces
    from whichever ranks own them. `src/Tessera_RefineParallel.hpp`:
    - Phase 1 — 2:1 mark-propagation fixpoint: each round advertises (EdgeKey, face
      level+mark, gid, owner) to coordinators; coordinator flags the coarser face of
      any edge with final-level diff ≥2; mark-requests routed back to face owners;
      `MPI_Allreduce(changed)`; marks monotone → terminates; hard cap 256.
    - Phase 2 — midpoint gid agreement: refining faces advertise edges; coordinator
      sets midpoint owner = min incident refining-face owner and returns
      (EdgeKey→owner) to participants + cosharer lists to owners; owners `MPI_Exscan`
      a global vertex-count-based block, assign, and **send gids to cosharers** →
      bit-identical shared midpoints with no ordering assumption.
    - Phase 3 — local red split of owned faces (reuses 6a child ordering + policy via
      the new `detail::blendVertexUserCross` cross-AoSoA helper), then edges
      re-derived from owned faces with a coordinator round assigning edge owner = min
      incident child-face owner (keeps owned counts a global partition) + edge level.
      Leaves owned-only entities; clears the halo.
  - `refine()` returns `RefineResult{iterations, midpoints[(EdgeKey,gid)]}` for
    verification. New `MeshInvariants` helpers: `check21Balance`,
    `checkMidpointAgreement`, `ownedEulerGlobal`, `globalOwned{Vertices,Edges,Faces}`.
  - Test `tests/test_refine_parallel.cpp` (**regression**, SERIAL+HIP, np1–5):
    *uniform* refine of distributed subdiv-2 → global owned counts == one subdivision
    level (V+E=642, 2E+3F=1920, 4F=1280), owned Euler=2, midpoint agreement,
    iterations==1; *adaptive* (gid%7 predicate) → midpoint agreement + 2:1 invariant
    hold across boundaries, fixpoint terminates in cap. **Regression 20/20, unit
    28/28**, `format-check` clean.
  - **Report-back:** iteration counts observed — uniform=1 (nothing to propagate),
    adaptive terminated well under the 256 cap. The changed-count is monotone-progress
    (marks only turn on), not literally strictly-decreasing; termination is by mark
    saturation + Allreduce==0, cap is a backstop. **Next:** Step 7 (migration API +
    general halo rebuild + optional Zoltan2 LB).
- 2026-07-01 — **Step 7 core landed (Opus). Third regression-tier test.** Migration
  API + the general (non-replicated) 1-deep halo rebuild:
  - **Scope decision (user-approved):** land `migrate()` + the general ghost builder
    + read accessors now, tested at ranks 1–5; **defer the Zoltan2 internal
    `loadBalance()` to a 7b commit** so the Trilinos-link dependency-surface change is
    isolated from the core algorithmic work. `migrate()` is the actual public contract
    (Canopy drives it externally; Zoltan2 is the optional, thin wrapper).
  - `src/Tessera_MeshMigrate.hpp` — `migrate(mesh, halo, dest)` (dest indexed by
    owned face local index). Unlike `distribute()` (Step 5), it assumes **no
    replicated mesh**: every rank holds only its own entities, so ownership and the
    ghost set are discovered by communication. This is the **general ghost builder
    deferred from Step 6b**. Host-orchestrated over `allToAllV` (like `distribute`);
    entity payloads travel as whole Cabana tuples wrapped in a fixed-size
    `TupleBlob` (Cabana::Tuple is **not** `is_trivially_copyable`, so it can't go
    through `allToAllV` directly — memcpy'd byte image does, matching the assumption
    the device migrate primitive already makes with `MPI_Type_contiguous`). Four
    rounds: (A) move each owned face + its 3 vertices + 3 edges to `dest`; (B)
    ownership (lowest-rank) + ghost discovery via per-gid vertex/edge coordinators
    (`gid % size`) — the vertex coordinator returns to a vertex's owner the list of
    incident faces owned by *other* ranks; (C) each vertex owner fetches those remote
    ghost faces (+ their 3 vertex/edge tuples, owners stamped by the sender who knows
    them); (D) assemble owned-first local AoSoAs, rebuild the vertex CSR + key side
    tables + the three halo plans via the existing `buildKindPlan`. Read accessors
    `ownedFaceCentroids/Gids/Weights(mesh)` for external partitioners (weights = 1.0
    per leaf face for now).
  - **Ordering that avoids phantom sends (report-back):** the halo plan's alignment
    contract is satisfied by ordering owned-first then ghost, each **ascending by
    gid**, on every kind — `buildKindPlan`'s send side is discovered by the ghoster
    advertising gids to the owner (`allToAllV`), and both sides visit a peer pair's
    entities in ascending-gid order, so pack/unpack align with no extra metadata (same
    contract as Step 5). Ghost entity owners are resolved once at the vertex/edge
    coordinator and **stamped into the tuple by the sending owner** before the ghost
    fetch, so a rank that holds an entity only as a ghost still learns its true owner
    without a second query.
  - **Deviation from the plan's "faces move via 4a" wording:** the mesh-level migrate
    is host-orchestrated (multi-destination vertex/edge follow + ghost rebuild can't
    be expressed by the single-destination device primitive). The Step-4a device
    `migrate()` remains the tested building block for pure single-dest AoSoA moves.
  - Test `tests/test_migrate_mesh.cpp` (**regression**, SERIAL+HIP, np1–5): distribute
    subdiv-3, then two external assignments — (A) per-gid hash re-partition: every
    owned face lands on the hash-requested rank; (B) global rotation `(rank+1)%size`
    of the whole partition. After each: ownership is a partition (Σ owned == global
    V/E/F), every owned vertex's 1-ring is local, the rank-count-independent topology
    checksum is **unchanged** from the post-distribute mesh, and a corrupt→sync→verify
    halo exchange restores every ghost. **Regression 30/30, unit 28/28**,
    `format-check` clean.
  - **Next:** Step 7b (Zoltan2 `loadBalance()`: link Trilinos/Zoltan2, gather
    centroids to rank 0 → MultiJagged → `MPI_Bcast` → `migrate()`; internal-path
    balance-improvement test), then Step 0 (HDF5 spack env) right before Step 8.
- 2026-07-06 — **Step 7b landed (Opus). Fourth regression-tier test.** Internal
  Zoltan2 `loadBalance()`:
  - **CMake isolated first (per contract):** added `find_package(Trilinos
    REQUIRED)` + link/include wiring to the root `CMakeLists.txt` alone, then
    reconfigured + rebuilt the full existing gate clean before writing any Zoltan2
    code — confirms the dependency-surface change by itself doesn't perturb the
    existing build.
  - `src/Tessera_Zoltan2Balancer.hpp` — `computeLoadBalance(mesh, imbalanceTolerance
    =0.05)` + `loadBalance(mesh, halo, imbalanceTolerance=0.05)`. **Key deviation
    from the Canopy reference** (`Canopy_TreePartitioner.hpp`): Canopy's tree is
    replicated on every rank, so it can build the Zoltan2 adapter directly; Tessera's
    owned faces are **not** replicated, so `computeLoadBalance` first does an
    explicit `MPI_Gather` (counts) + `MPI_Gatherv` (centroids scaled by `Dim`,
    weights) to assemble the global geometric input on rank 0 before Zoltan2 ever
    runs. Reuses Canopy's core pattern otherwise: geometric **MultiJagged** (never
    RCB — breaks on Tuolumne), solved over a `Teuchos::SerialComm` on **rank 0
    only** (MultiJagged is not guaranteed deterministic across ranks), then
    `MPI_Scatterv`s the resulting per-face part assignment back to each rank in the
    same per-rank order its centroids/weights were gathered in. Single-rank (`size
    ==1`) is a fast-path no-op. Because `Mesh::Dim` is a template parameter (2 or
    3, not fixed at 3 like Canopy's tree), the adapter is built via Zoltan2's
    **generic multivector `BasicVectorAdapter` constructor** (per-dimension
    `std::vector<const scalar_t*>` after deinterleaving the row-major `[f*Dim+d]`
    gathered centroids) rather than Canopy's fixed x/y/z 3D constructor.
    `loadBalance()` itself is a thin wrapper: `computeLoadBalance()` → `migrate()`
    (Step 7 core) — no separate internal migration path, matching the design lock.
  - Test `tests/test_loadbalance.cpp` (**regression**, SERIAL+HIP, np1–5): builds a
    distributed icosphere, records its topology checksum, deliberately dumps every
    owned face onto rank 0 (`migrate()` with `dest=0` everywhere) to maximally
    imbalance it, then calls `loadBalance()` and asserts: max-owned-face count drops
    strictly below the pre-balance max and lands within `2×ideal` (generous
    MultiJagged-tolerance + coarse-icosphere geometric slack), every `migrate()`
    distribution invariant still holds (ownership partition, owned 1-ring local),
    and the topology checksum is **unchanged** (`loadBalance()` moves entities, it
    never alters the global mesh). Single rank is a checked no-op. **Regression
    40/40, unit 28/28** (unchanged unit count — confirms no regression),
    `format-check` clean after auto-fixing continuation-line wrapping in both new
    files via the `format` target.
  - **Report-back (per the contract):** the migrate→ownership→halo ordering is
    unchanged from Step 7 core (`loadBalance()` adds no new ordering — it only
    computes `dest` before handing off to the same `migrate()`); Zoltan2
    adapter/param specifics reused from Canopy verbatim (`algorithm=multijagged`,
    `imbalance_tolerance`, `debug_level=no_status`) except the adapter constructor,
    which had to switch to the Dim-generic multivector form; external-API surface
    Canopy will call is unchanged (`ownedFaceCentroids/Gids/Weights` + `migrate()`)
    — `loadBalance()` is purely an additional, optional internal convenience.
  - **Next:** Step 0 (add `hdf5 +mpi` to the Tuolumne spack env, commit the
    `systems/tuolumne/spack.yaml` snapshot, reconfirm the full gate still builds with
    HDF5 discoverable by CMake) right before Step 8 (parallel HDF5 + XDMF writer/
    reader).
- 2026-07-06 — **Step 0 landed (Sonnet). Env-only, no source changes.** HDF5 for
  the Tuolumne spack env:
  - `hdf5 +mpi` was **already present** as a root spec in
    `~/spack_envs/tuolumne_trilinos/spack.yaml` (added incidentally alongside an
    earlier env edit) and resolves as an **external**: `spack find -p hdf5` →
    `hdf5@1.14.3.7 +mpi` at `/opt/cray/pe/hdf5-parallel/1.14.3.7/crayclang/20.0`
    (Cray's parallel HDF5 module, not a from-source spack build). No install/
    reinstall was needed. Snapshot committed to
    [systems/tuolumne/spack.yaml](../../systems/tuolumne/spack.yaml) (verbatim copy of the live
    env file).
  - Existing gate untouched (no rebuild needed — nothing in the mesh library or its
    dependencies changed); Steps ≤7b remain green as last verified in the Step 7b
    entry.
  - **`find_package(HDF5)` gotcha for Step 8 (report-back, important):** a bare
    `find_package(HDF5 REQUIRED COMPONENTS C)` with **no `HDF5_ROOT`/
    `CMAKE_PREFIX_PATH` hint** silently resolves to the **wrong HDF5** — CMake
    picks up the OS-default `/usr/lib64/libhdf5.so` (**1.10.5, serial, not
    MPI-aware**, `HDF5_IS_PARALLEL=FALSE`) instead of the spack-external Cray
    parallel build, because `spack env activate` does **not** add the external
    HDF5's prefix to `CMAKE_PREFIX_PATH` (externals aren't view-linked) and the
    Cray `h5cc` compiler-wrapper probe fails (`"HDF5 C compiler wrapper is unable
    to compile a minimal HDF5 program"`, non-fatal, CMake falls back to searching
    default paths). Verified experimentally: passing
    `-DHDF5_ROOT=/opt/cray/pe/hdf5-parallel/1.14.3.7/crayclang/20.0` makes CMake
    resolve the correct `1.14.3`/`HDF5_IS_PARALLEL=TRUE` build. **Step 8 must**
    either set `HDF5_ROOT` (or `CMAKE_PREFIX_PATH`) explicitly — e.g. via
    `spack location -i hdf5` piped into `run_cmake_toulumne.sh` — or add an
    explicit check that `HDF5_IS_PARALLEL` is `TRUE` after `find_package`, so a
    silent fallback to the serial system HDF5 fails loudly instead of building
    a broken (non-parallel) I/O path. **Next:** Step 8 (parallel HDF5 + XDMF
    writer + round-trip reader) — needs an **Opus scoping pass first** (dataset
    layout + XDMF schema design per the Step-8 "Model" column: "Sonnet, layout
    Opus-spec'd"): Opus should design the HDF5 dataset layout, the XDMF schema,
    and the `HDF5_ROOT` discovery fix above, record it in this file, then a
    Sonnet session implements against that spec.
- 2026-07-06 — **Step 8 SPEC (Opus scoping pass; design only, no code).** Full
  implementation spec for the parallel HDF5 + XDMF writer/reader below — detailed
  enough for a Sonnet session to implement directly with no further design
  decisions. **Report-back for the "Model" column deferred to the Sonnet
  implementation session** (dataset layout + XDMF schema as-built, the dense-
  numbering exscan, any collective-IO tuning); the design of all three is fixed
  here.

  ### 8.0 — Guiding decisions (rationale)
  - **What is written = the OWNED entities of every rank, each exactly once.** The
    mesh is stored owned-first (`[0,nOwnedX)` owned, invariant from Steps 5/7), so
    "owned block" = the leading `nOwnedX` rows. Union of owned blocks = the global
    mesh with no duplication → the file is a clean partition, and a checksum over
    the on-disk persistent gids is **rank-count independent** (it is the same gid
    set no matter how the writer partitioned).
  - **Two identities per entity, both stored.** (1) The persistent 64-bit
    `GlobalId` (`Gid` member) is the cross-run identity and the checksum key — it is
    sparse/arbitrary after refine+migrate and is stored verbatim. (2) A **dense
    global index** in `[0,Nglobal)` per kind, assigned owned-only via `MPI_Exscan`,
    is what XDMF/Paraview connectivity must reference (0-based, contiguous). Both go
    in the file; connectivity datasets use dense indices, a parallel `gid` dataset
    carries the persistent identity.
  - **Partition-dependent state is NOT written** (owner rank, flags, ghost layer,
    CSR, key tables, edge→face incidence). It is *reconstructed* on read by re-
    running the tested `migrate()` ghost/halo builder, so the reader trivially
    re-passes the Step-5 invariants without duplicating that logic. Edge→face
    incidence is left `{invalid_gid,invalid_gid}` on read (unused by any invariant;
    Step 7 already documents it as carried-but-unused metadata).
  - **Flat header convention** (matches the realized `src/Tessera_*.hpp` layout, not
    the `src/io/` wording in the Step-8 contract row — same deviation already taken
    for `src/mesh/`). New headers, all header-only, added to the umbrella
    `src/Tessera.hpp`:
    `Tessera_IoCommon.hpp`, `Tessera_HDF5Writer.hpp`, `Tessera_HDF5Reader.hpp`,
    `Tessera_Xdmf.hpp`.

  ### 8.1 — HDF5 dataset layout (single file `<stem>.h5`)
  All datasets are 1-D or 2-D with **global** first dim = `N{v,e,f}` (the
  `MPI_Allreduce(SUM)` of owned counts). Each rank writes its owned block as a
  hyperslab `[off, off+nOwned)` where `off = MPI_Exscan(SUM, nOwned)` (exclusive
  scan; rank 0 offset 0). H5 native types via a `h5_type<T>()` trait in
  `Tessera_IoCommon.hpp` (`double→H5T_NATIVE_DOUBLE`, `float→…FLOAT`,
  `uint64→…UINT64`, `int16→…INT16`, `int32→…INT32`).

  Root attributes (scalars, written identically by all ranks): `format_version`
  (int=1), `dim` (int = `MeshT::dim`), `scalar_bytes` (int, 4 or 8),
  `Nv`,`Ne`,`Nf` (uint64), `n_user_v_fields`,`n_user_e_fields`,`n_user_f_fields`
  (int), and for each user field its extent (1 for scalar, N for `Scalar[N]`) in
  attrs `uv_ext_<j>`,`ue_ext_<j>`,`uf_ext_<j>`. The reader validates these against
  its compile-time template and **hard-fails on mismatch** (dim, scalar_bytes,
  field counts, extents).

  Group `/vertices`:
  | dataset | shape | H5 type | source (owned block) |
  |---|---|---|---|
  | `gid` | `Nv` | uint64 | `VertexField::Gid` |
  | `position` | `Nv × Dim` | Scalar | `VertexField::Position` |
  | `u<j>` (per user field j) | `Nv × ext_j` | field's scalar | `userVertexField<j>()` |

  Group `/edges`:
  | dataset | shape | H5 type | source |
  |---|---|---|---|
  | `gid` | `Ne` | uint64 | `EdgeField::Gid` |
  | `verts` | `Ne × 2` | uint64 | **dense** vertex indices of `EdgeField::Verts` |
  | `level` | `Ne` | int16 | `EdgeField::Level` |
  | `u<j>` | `Ne × ext_j` | field scalar | `userEdgeField<j>()` |

  Group `/faces`:
  | dataset | shape | H5 type | source |
  |---|---|---|---|
  | `gid` | `Nf` | uint64 | `FaceField::Gid` |
  | `verts` | `Nf × 3` | uint64 | **dense** vertex indices of `FaceField::Verts` (XDMF triangle connectivity) |
  | `edges` | `Nf × 3` | uint64 | **dense** edge indices of `FaceField::Edges` |
  | `level` | `Nf` | int16 | `FaceField::Level` |
  | `u<j>` | `Nf × ext_j` | field scalar | `userFaceField<j>()` |

  **Empty-user-pack guard (from Step 6a):** `slice<UserBegin>` is ill-formed when
  the pack is empty. Every user-field dataset loop MUST be behind
  `if constexpr (member_types::size > UserBegin)` and recurse member indices with
  the Step-6a introspection pattern
  (`Cabana::MemberTypeAtIndex<Mabs,MT>::type` + `std::rank`/`std::extent<…,0>` for
  the extent; `member_types::size` for the loop bound).

  ### 8.2 — Dense global numbering (writer) — the `MPI_Exscan` core
  1. `nOwned{V,E,F}` from `mesh.numOwned*()`. `N{v,e,f}=Allreduce(SUM)`;
     `{v,e,f}off = Exscan(SUM)` (rank 0 → 0). Store `N*` as attrs.
  2. Owned entity at owned-local-index `i∈[0,nOwned)` → dense index `off+i`
     (owned-first makes owned local indices exactly `[0,nOwned)`, no gather needed).
     Build host maps `denseV[gid]=voff+i`, `denseE[gid]=eoff+i` for **owned** V/E.
  3. **Ghost dense-index fetch** (faces/edges reference vertices/edges that may be
     ghosts owned by another rank; only those ranks know the dense index). Two
     `allToAllV` request/reply rounds, keyed by the ghost's `Owner` field (present
     in every tuple; the halo plan is NOT needed):
     - Requester groups its ghost vertex gids by `Owner`, sends gid lists; each
       owner looks each up in its owned `denseV` map, replies the dense value **in
       received order**; requester matches replies by position (same per-`(src,dst)`
       ordering guarantee `buildKindPlan` relies on). Merge into `denseV`.
     - Identical round for ghost edges → `denseE`.
     (Owned edges' endpoints are always vertices of an owned face — edge owner =
     min incident-face owner, so an owned edge has an incident owned face whose 3
     verts include both endpoints — hence the vertex set referenced by owned faces
     already covers owned-edge endpoints; still, a face may reference a **ghost
     edge**, so the ghost-edge round is required.)
  4. Faces reference no other faces → face dense index needs no exchange
     (`foff+i`). Translate each owned face's `Verts`/`Edges` gids → dense via
     `denseV`/`denseE`; each owned edge's `Verts` gids → dense via `denseV`. Pack
     the owned-block hyperslab arrays and write.

  ### 8.3 — Collective write mechanics
  - `fapl = H5Pcreate(H5P_FILE_ACCESS); H5Pset_fapl_mpio(fapl, mesh.comm(),
    MPI_INFO_NULL); H5Fcreate(<stem>.h5, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)`.
  - Per dataset: all ranks `H5Screate_simple` the **global** dims, `H5Dcreate2`
    (collective/identical on every rank), then each rank
    `H5Sselect_hyperslab(filespace, SET, start={off,0}, count={nOwned,width})`,
    memspace `H5Screate_simple({nOwned,width})`, `dxpl` with
    `H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE)`, `H5Dwrite`.
  - **Zero-owned rank gotcha (must handle):** a rank with `nOwned==0` (possible at
    high rank counts on a tiny mesh) MUST still call `H5Dcreate2`/`H5Dwrite`
    collectively but select **none** (`H5Sselect_none(filespace)` +
    `H5Sselect_none(memspace)`). Skipping the call deadlocks the collective. Call
    this out in a comment.
  - Root attributes written by all ranks with identical values (safe under MPIO).
  - `H5Pclose`/`H5Sclose`/`H5Dclose`/`H5Fclose` everywhere (disciplined close
    ordering; leaks exhaust the Lustre file-handle budget at rank 5).
  - **Tuolumne collective-IO note:** start with `MPI_INFO_NULL`. If Lustre
    write-hangs or slowness appear on larger meshes, pass an `MPI_Info` with
    `romio_cb_write=enable`, `cb_nodes`, `striping_factor`/`striping_unit`; not
    expected necessary for Milestone-1 coarse meshes. Record what was actually
    needed in the implementation report-back.

  ### 8.4 — XDMF sidecar (`<stem>.xmf`, written by rank 0 only)
  Plain-text XML (no HDF5 dependency — `Tessera_Xdmf.hpp` just emits a file), written
  once by rank 0 after the collective write completes (`MPI_Barrier` first).
  References the dense datasets, so it is byte-identical regardless of writer rank
  count. XDMF 3.0, single uniform `Unstructured` grid:
  ```xml
  <?xml version="1.0" ?>
  <Xdmf Version="3.0"><Domain>
    <Grid Name="Tessera" GridType="Uniform">
      <Topology TopologyType="Triangle" NumberOfElements="Nf">
        <DataItem Dimensions="Nf 3" NumberType="UInt" Precision="8" Format="HDF">
          <stem>.h5:/faces/verts</DataItem>
      </Topology>
      <Geometry GeometryType="XYZ">            <!-- XY when Dim==2 -->
        <DataItem Dimensions="Nv Dim" NumberType="Float" Precision="{4|8}" Format="HDF">
          <stem>.h5:/vertices/position</DataItem>
      </Geometry>
      <Attribute Name="v_gid" Center="Node" AttributeType="Scalar">
        <DataItem Dimensions="Nv" NumberType="UInt" Precision="8" Format="HDF">
          <stem>.h5:/vertices/gid</DataItem></Attribute>
      <Attribute Name="f_level" Center="Cell" AttributeType="Scalar">
        <DataItem Dimensions="Nf" NumberType="Int" Precision="2" Format="HDF">
          <stem>.h5:/faces/level</DataItem></Attribute>
      <!-- one <Attribute Center="Node"> per vertex user field u<j> (scalar or
           Vector when ext==Dim), one Center="Cell" per face user field -->
    </Grid>
  </Domain></Xdmf>
  ```
  - Use only the file **basename** (not the full path) in the `.h5:` reference so
    the pair is relocatable. `Precision` = `scalar_bytes` for Float, `8` for
    UInt gid/connectivity, `2` for int16 level.
  - Edges are round-trip data, not a standard surface cell type — **not** placed in
    XDMF (optional: a second `TopologyType="Polyline"` grid over `/edges/verts` if
    edge visualization is later wanted; out of scope for the gate).
  - `Dim∉{2,3}` → emit HDF5 only, skip the `<Geometry>`/`<Topology>` grid with a
    one-line warning (Milestone 1 is Dim=3).

  ### 8.5 — Reader reconstruction contract (re-passes Step-5 invariants by reuse)
  `readMesh(mesh, halo, <stem>)` produces a distributed `Mesh` + `MeshHalo` that
  passes `checkOwnershipPartition`, `owned1RingLocal`, and a corrupt→`haloExchange`
  →restored ghost check — **without re-deriving ownership/halo logic**, by handing a
  covering to the tested `migrate()`:
  1. Open with `H5Pset_fapl_mpio(comm)`. Read root attrs; **validate** dim/
     scalar_bytes/field-counts/extents against the compile-time template
     (runtime check → `MPI_Abort` with a clear message on mismatch). Read
     `Nv,Ne,Nf`.
  2. **Block-partition dense FACE indices**: rank R reads faces
     `[fs,fe) = [R*Nf/size, (R+1)*Nf/size)` (a fresh partition, deliberately
     unrelated to the writer's — this also exercises rank-count independence).
     Collectively read that hyperslab of `/faces/{gid,verts,edges,level,u*}`.
  3. **Fetch referenced vertex/edge records.** The dense vertex/edge indices this
     rank's faces reference may fall outside any contiguous local block, so map each
     dense index → owning reader-rank via the same `[k*N/size,(k+1)*N/size)` block
     arithmetic (no per-index gather needed). Every rank block-reads its own
     `/vertices` and `/edges` slice `[R*N/size,(R+1)*N/size)` up front; then
     `allToAllV`-request the needed dense V and E indices from their block-owners,
     which reply the full record (`/vertices/{gid,position,u*}`;
     `/edges/{gid,verts,level,u*}`) for each requested dense row.
  4. Build `denseV→gid`, `denseE→gid` from the fetched/owned records. Translate the
     face `verts`/`edges` (dense) and edge `verts` (dense) back to **persistent
     gids**. Assemble an **intermediate all-owned mesh**: vertices = unique fetched
     vertex records (dedup by gid; set `Gid`,`Position`,user fields;
     `Owner`=R placeholder), edges likewise (`Gid`,`Verts`=gids,
     `Faces`={invalid_gid,invalid_gid},`Level`,user; `Owner`=R), faces = this rank's
     block (`Gid`,`Verts`/`Edges`=gids,`Level`,user; `Owner`=R). `setOwnedCounts(
     nHeldV, nHeldE, blockFaces)`. **No CSR/keys needed here** — `migrate()` rebuilds
     them.
  5. `std::vector<Rank> dest(blockFaces, R); migrate(mesh, halo, dest);`
     — self-destination move (Round A loops to self), then `migrate()` recomputes
     lowest-rank ownership, discovers/fetches the 1-deep ghost layer, and builds the
     CSR + key tables + the three halo plans. **Postcondition = a valid distributed
     mesh identical in structure to the Step-5/7 output**, so all Step-5 invariants
     pass by construction. Final face ownership = the reader's contiguous block
     partition; V/E ownership = lowest-rank over the covering.
  - **np1**: one block = all faces; `migrate(dest=all-0)` → zero ghosts, no-op
     exchange (matches the Step-5 np1 acceptance).
  - **Why this satisfies "checksum independent of rank count":** the persistent gids
     read from `/*/gid` are the same set at any reader rank count; `migrate()` never
     invents or drops gids; so `topologyChecksum(mesh2)` equals the writer's in-memory
     checksum and equals the on-disk BXOR of `/*/gid` — at every np.

  ### 8.6 — Public API (both header-only, templated on the mesh type)
  ```cpp
  namespace Tessera {
  // Collective on mesh.comm(). Writes <stem>.h5 (parallel) + <stem>.xmf (rank 0).
  // Needs only the mesh: ghost dense indices are fetched via each ghost's Owner field.
  template <class MeshT>
  void writeMesh( const MeshT& mesh, const std::string& stem );

  // Collective. Fills an empty mesh (constructed on the same comm) + its halo by
  // block-reading <stem>.h5 and re-running migrate() to establish ownership+halo.
  template <class MeshT>
  void readMesh( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
                 const std::string& stem );
  }
  ```

  ### 8.7 — CMake / build fix (from the Step-0 report-back)
  **Portability split (important — the root `CMakeLists.txt` is shared by every
  system, so it may hold ONLY system-neutral logic):**
  - The `CMakeLists.txt` change is **general and correct on any system.** The
    `HDF5_IS_PARALLEL` guard is a portable *correctness* check, not a Tuolumne
    workaround: Step 8 uses collective MPI-IO, so every system genuinely requires a
    parallel HDF5, and failing loudly when only serial HDF5 is present is the right
    behavior everywhere. Keep its `FATAL_ERROR` message system-neutral (no Cray /
    `/usr/lib64` wording baked into the shared file).
  - The **`HDF5_ROOT` discovery is the only Tuolumne-specific piece** and lives in
    the per-system `run_cmake_toulumne.sh`, never in `CMakeLists.txt`. It is needed
    only on Tuolumne because the Cray parallel HDF5 is a spack **external**, so
    `spack env activate` does NOT view-link it onto `CMAKE_PREFIX_PATH` and a bare
    `find_package` silently resolves the OS serial `/usr/lib64` build
    (`HDF5_IS_PARALLEL=FALSE`); its `h5cc` probe also fails non-fatally. On a normal
    system a from-source `hdf5 +mpi` IS view-linked, so `CMAKE_PREFIX_PATH` carries
    it and the same `CMakeLists.txt` resolves the parallel build with no hint —
    `systems/local` needs no `HDF5_ROOT` injection. The guard is what makes any
    misconfigured system (serial-only HDF5) fail loudly rather than silently.

  Edits:
  - **`CMakeLists.txt`** — replace line 37 (`find_package(HDF5 REQUIRED)`) with a
    parallel-guarded resolve, and propagate link/include on the INTERFACE target
    (HDF5 is unconditionally pulled in via the umbrella header, same rationale as
    the Trilinos block). Message is system-neutral:
    ```cmake
    # I/O — parallel HDF5 (Step 8). Step 8 uses collective MPI-IO, so a *parallel*
    # HDF5 is required on every system. A bare find_package can silently resolve a
    # serial HDF5 (e.g. an OS /usr/lib64 build) when the parallel one is not on
    # CMAKE_PREFIX_PATH; the guard below turns that into a loud configure error.
    set(HDF5_PREFER_PARALLEL ON)
    find_package(HDF5 REQUIRED COMPONENTS C)
    if(NOT HDF5_IS_PARALLEL)
        message(FATAL_ERROR
            "Found HDF5 at ${HDF5_INCLUDE_DIRS} but it is NOT parallel "
            "(HDF5_IS_PARALLEL=FALSE). Tessera I/O (Step 8) requires an "
            "MPI-parallel HDF5. Point CMake at one via -DHDF5_ROOT=<prefix> or "
            "add its prefix to CMAKE_PREFIX_PATH.")
    endif()
    ```
    then extend the existing `target_link_libraries(Tessera INTERFACE …)` with
    `${HDF5_C_LIBRARIES}` (or `${HDF5_LIBRARIES}`) and add
    `target_include_directories(Tessera SYSTEM INTERFACE ${HDF5_INCLUDE_DIRS})`.
  - **`run_cmake_toulumne.sh`** (Tuolumne-only) — resolve and pass `HDF5_ROOT`
    before the `cmake` call:
    ```bash
    : "${HDF5_ROOT:=$(spack location -i hdf5 2>/dev/null || \
        echo /opt/cray/pe/hdf5-parallel/1.14.3.7/crayclang/20.0)}"
    ```
    and add `-DHDF5_ROOT="${HDF5_ROOT}"` to the `cmake` args.
  - **`systems/tuolumne/claude.md`** — mirror the `HDF5_ROOT` requirement into the
    Build-config args section (per the framework "keep docs in sync" invariant).
    No change to `systems/local` (view-linked HDF5 resolves without a hint).

  ### 8.8 — Test (`tests/test_io.cpp`, regression, SERIAL+HIP, ranks 1–5)
  Add via `tessera_add_test(NAME io … TIER regression RANKS ${TESSERA_TEST_MPI_RANKS})`
  for both SERIAL and HIP (gate rows in `tests/CMakeLists.txt`). Body:
  1. `buildIcosphere(mesh, subdiv=3)`; `facePartitionByAxis` + `distribute`. Use a
     **non-trivial user field**: `Mesh<Scalar, 3, VertexFields<Scalar>, EdgeFields<>,
     FaceFields<Scalar>>`; set the vertex user field to a deterministic function of
     the vertex gid and the face user field to a function of the face gid (so field
     round-trip is checkable).
  2. Record pre-write: `topologyChecksum(cv,ce,cf)`; a field checksum = BXOR over
     owned of the field bits, reduced `MPI_BXOR`.
  3. **Unique stem per executable+rankcount** to avoid concurrent-run collisions:
     `stem = basename(argv[0]) + "_np" + to_string(size)` (SERIAL/HIP exes differ,
     np differs → unique). Write to `$TESSERA_IO_TMPDIR` if set, else cwd.
  4. `writeMesh(mesh, stem); MPI_Barrier;` construct fresh `Mesh mesh2(comm)` +
     `MeshHalo halo2`; `readMesh(mesh2, halo2, stem)`.
  5. Assert (each `MPI_Allreduce(SUM)` of a LOCAL fail count, per the MeshInvariants
     convention): `checkOwnershipPartition(mesh2,Nv,Ne,Nf)==0`,
     `owned1RingLocal(mesh2)==0`, `ownedEulerGlobal(mesh2)==2` (uniform coarse
     icosphere), `topologyChecksum(mesh2)==(cv,ce,cf)`, field checksum unchanged.
  6. **On-disk rank-count independence**: rank 0 opens `<stem>.h5` (serial fapl),
     reads the whole `/vertices/gid` dataset, BXOR-reduces it, asserts `== cv`
     (and similarly `/edges/gid==ce`, `/faces/gid==cf`). Because `cv/ce/cf` are
     computed identically at every np and the file stores each owned gid once, this
     equals the in-memory checksum at *every* rank count → the on-disk checksum is
     rank-count independent (the gate running np1–5 confirms all five agree).
  7. Corrupt every ghost's gid on `mesh2`, `haloExchange(mesh2,halo2)`, verify each
     ghost restored to its owner's gid; owned untouched; np1 = no-op.
  8. Clean up the `<stem>.h5`/`<stem>.xmf` files at end (rank 0, after barrier).
  Manual Paraview open of one `<stem>.xmf` is the "opens in Paraview" acceptance
  (report-back; not automatable in the gate).

  ### 8.9 — Deliverables checklist for the Sonnet session
  - `src/Tessera_IoCommon.hpp` (`h5_type<T>()`, attr read/write helpers, block-bound
    arithmetic, member-field iteration helper).
  - `src/Tessera_HDF5Writer.hpp` (`writeMesh`), `src/Tessera_HDF5Reader.hpp`
    (`readMesh`), `src/Tessera_Xdmf.hpp` (`writeXdmf`, rank 0).
  - Add all four to `src/Tessera.hpp` (umbrella).
  - `CMakeLists.txt` + `run_cmake_toulumne.sh` + `systems/tuolumne/claude.md` HDF5 fix.
  - `tests/test_io.cpp` + the two gate rows in `tests/CMakeLists.txt`.
  - BSD-3-Clause SPDX header on every new file; `--target format-check` clean.
  - README: add an I/O section (public `writeMesh`/`readMesh` API, on-disk layout
    summary, the `HDF5_ROOT` build requirement) per the "README in sync" invariant.
  - Run the full gate (`flux batch scripts/tuolumne/run_regression_minset.flux` or
    the dev runner with the `regression` label); report the new regression count
    (expected 50/50) and unit count (unchanged 28/28).
- 2026-07-06 — **Step 8 landed (Sonnet). Fifth regression-tier test.** Parallel
  HDF5 + XDMF writer/reader, implemented against the Opus spec above with no
  design deviations:
  - `Tessera_IoCommon.hpp` — `h5_type<T>()` trait (double/float/uint64/int16/
    int32), `exscanCount()` (the dense-numbering `MPI_Exscan` core: global count
    via `Allreduce(SUM)`, this rank's offset via `Exscan(SUM)`, rank 0 → 0),
    `blockRange()`/`blockOwner()` (reader's fresh block-partition arithmetic —
    `blockOwner` inverts the floor-division block boundaries by direct
    adjustment rather than assuming a closed form), the `forEachUserField`
    compile-time iterator (the Step 6a empty-pack guard, reused verbatim: an
    empty user pack short-circuits via `if constexpr` before `Cabana::slice`
    would go ill-formed), `FieldInfo<AoSoA,Mabs>` (extent + scalar type per
    member), and `writeHyperslab`/`readHyperslab` (the collective owned-block
    hyperslab I/O core, with the zero-owned-rank `H5Sselect_none` guard on both
    file- and mem-space built in once rather than at every call site).
  - `Tessera_HDF5Writer.hpp` — `writeMesh()`: builds the dense `denseV`/`denseE`
    maps (owned entries dense directly via the exscan offset; ghost entries
    fetched from their `Owner` rank via two `allToAllV` rounds — request gids,
    owner replies dense indices in received order, matching the existing
    `buildKindPlan`/`migrate()` per-source-ordering convention), then writes
    `/vertices`, `/edges`, `/faces` (core fields + `u<j>` per user field) as
    collective MPI-IO hyperslabs, then the root attributes, then (rank 0, after
    `MPI_Barrier`) the XDMF sidecar.
  - `Tessera_HDF5Reader.hpp` — `readMesh()`: takes a **fresh** dense-FACE block
    partition (`blockRange(Nf,R,size)`, deliberately unrelated to the writer's
    partition), reads that face block, collects the referenced dense
    vertex/edge indices, block-reads this rank's own `/vertices`/`/edges`
    range up front and `allToAllV`-requests the remainder from their block
    owners (owner replies a `TupleBlob` of its local host tuple — reusing
    `Tessera_MeshMigrate.hpp`'s `TupleBlob`/`toBlob`/`fromBlob` directly, since
    `readMesh()` already depends on that header for the final `migrate()`
    call), translates dense→persistent-gid for `verts`/`edges` fields, and
    assembles an intermediate **all-owned** mesh (`Owner=R` on every held
    entity, edges' `Faces` left `{invalid_gid,invalid_gid}`) before handing off
    to `migrate(mesh, halo, dest=all-self)`. One simplification found during
    implementation (not in the original spec text but consistent with it): the
    held edge/vertex set only needs to be the union **referenced by this
    rank's face block** (not the full local block range) — since every edge's
    two endpoints are, by the codebase's own face→edge convention
    (`edge(v[k],v[(k+1)%3])`), always a subset of its incident face's three
    vertices, no separate vertex-fetch round is needed to resolve edge
    endpoints; they are always already covered by the face-referenced vertex
    set.
  - `Tessera_Xdmf.hpp` — `writeXdmf()`: XDMF 3.0 single `Unstructured` Triangle
    grid over `/faces/verts` (dense) + `/vertices/position`, `v_gid`/`f_level`
    attributes, and one `<Attribute>` per user field (`Vector` when
    `extent==Dim`, else `Scalar`); `Dim∉{2,3}` emits HDF5-only metadata with a
    one-line XML comment instead of a grid (Milestone 1 is always Dim=3, so
    untested in the gate). Edges are intentionally not placed in XDMF (matches
    the spec: not a standard surface cell type, out of scope for the gate).
  - **CMake fix, with one addition beyond the spec:** the `HDF5_IS_PARALLEL`
    guard + `HDF5_ROOT`/Tuolumne split landed exactly as specified in
    `CMakeLists.txt` / `run_cmake_toulumne.sh` / `systems/tuolumne/claude.md`.
    Discovered while reconfiguring: `FindHDF5.cmake`'s compiler-wrapper probe
    (used when no HDF5 CMake config package exists — true for Cray's parallel
    HDF5 module, which ships only `.pc` files) `try_compile`s a `.c` test
    program, which hard-errors ("Unknown extension .c ... currently these are:
    CXX") when the project has no C language enabled — this aborts
    configuration entirely (not merely a fallback warning), independent of and
    prior to the `HDF5_IS_PARALLEL` guard ever running. Fixed by adding `C` to
    `project(... LANGUAGES CXX C)` (system-neutral: harmless everywhere,
    required wherever HDF5 is discovered via the plain module rather than a
    config package). With that fix, HDF5_ROOT resolution + the parallel guard
    work as specified: `HDF5_IS_PARALLEL=TRUE`, resolved library
    `/opt/cray/pe/hdf5-parallel/1.14.3.7/crayclang/20.0/lib/libhdf5.so`.
  - `tests/test_io.cpp` (**regression**, SERIAL+HIP, np1–5): distributed
    subdiv-3 icosphere with `VertexFields<Scalar>`/`FaceFields<Scalar>` user
    fields set to a deterministic function of gid; write → read into a fresh
    `Mesh`/`MeshHalo` → assert ownership partition, owned-1-ring-local, owned
    Euler==2, topology checksum unchanged, vertex/face user-field BXOR
    checksums unchanged, on-disk `/*/gid` BXOR (read serially by rank 0) equals
    the in-memory checksum, and a corrupt→`haloExchange`→restored-ghost check
    on the round-tripped mesh; unique stem per executable+rankcount+exec-space
    (`argv[0] basename + "_np" + size + "_serial"/"_default"`), honoring
    `TESSERA_IO_TMPDIR` if set, cleaned up (rank 0) at the end of the test.
    **Regression 50/50** (40 previous + 10 new `io_{SERIAL,HIP}_np{1-5}`),
    **unit 28/28 unchanged**, `format-check` clean.
  - **Report-back (per the Step-8 contract):**
    - *Dataset layout as-built*: exactly the spec §8.1 table — `/vertices`
      {gid, position, u\*}, `/edges` {gid, verts(dense), level, u\*}, `/faces`
      {gid, verts(dense), edges(dense), level, u\*}, root attrs
      (`format_version`, `dim`, `scalar_bytes`, `Nv`/`Ne`/`Nf`,
      `n_user_{v,e,f}_fields`, `u{v,e,f}_ext_<j>`).
    - *XDMF schema as-built*: exactly the spec §8.4 template (Triangle topology
      + XYZ geometry over the dense datasets, `v_gid`/`f_level` attributes,
      per-user-field attributes), basename-only `.h5` reference for
      relocatability.
    - *Dense-numbering exscan*: `MPI_Allreduce(SUM)` for the global count +
      `MPI_Exscan(SUM)` for the offset (rank 0 forced to 0, since `MPI_Exscan`
      leaves rank 0's receive buffer undefined), exactly per spec §8.2.
    - *Collective-IO tuning*: **none needed** — `MPI_INFO_NULL` was sufficient
      for the Milestone-1 coarse-icosphere gate sizes on Tuolumne's Lustre; no
      write-hangs or slowness observed at any rank count 1–5.
  - **Next:** Step 9 (end-to-end example; unit/regression runner scripts;
    OPENMP smoke-build; CI subset wiring; README/example-arg sync).
- 2026-07-06 — **Step 10 SPEC (Opus scoping pass; design only, no code).** A
  geometric mesh-quality refinement criterion — a built-in that inspects face
  geometry and produces the `std::vector<char>` owned-face mask that `refine()`
  already consumes, so a caller can drive AMR from mesh quality instead of
  hand-authoring the mask. Two criteria, split by distributed complexity into
  **10a (edge-length, Sonnet)** and **10b (curvature, Opus)** per the user's
  design confirmation (both metrics requested). Detailed enough for the assigned
  session to implement with no further design decisions. **Report-back for the
  "Model" column is deferred to each implementation session.**

  ### 10.0 — Guiding decisions (rationale)
  - **Marking is a free algorithm over the mesh, like `refine`/`distribute`** —
    not a `Mesh` member (keeps `Mesh` lean; consistent with every other
    algorithm). It returns the exact object `refine()` expects: a
    `std::vector<char>` sized `numOwnedFaces()`, indexed by owned-face local
    index, so `markByQuality(mesh, crit)` → `refine(mesh, halo, mask, policy)` is
    a drop-in pipeline.
  - **Pluggable `QualityCriterion`, mirroring `RefinePolicy`.** A criterion is a
    small struct exposing one method — `std::vector<char> mark(const MeshT& mesh)
    const` — that owns its own evaluation *including any communication*. This is
    why the concept is a whole-mesh `mark()` and **not** a per-face
    `bool operator()(face)`: the edge-length criterion is embarrassingly local,
    but the curvature criterion needs a cross-rank gather, and a uniform
    `mark(mesh)` interface hides that difference from the caller.
    `markByQuality(mesh, crit)` is a one-line dispatcher: `return
    crit.mark(mesh);`. A later curvature-aware/physics criterion drops in with no
    API churn — exactly the extensibility argument used for `RefinePolicy`.
  - **On-demand, no new stored fields.** Both metrics are recomputed from the
    existing `VertexField::Position` + `FaceField::Verts`. A stored per-face
    quality field would have to halo/migrate/refine-propagate (a new invariant and
    cost) for a metric that is cheap and only queried at mark time. Recompute is
    stateless and always consistent.
  - **Flat header convention** (matches the realized `src/Tessera_*.hpp` layout):
    one new header `src/Tessera_MarkQuality.hpp`, header-only, added to the
    umbrella `src/Tessera.hpp`. Both criteria live in it (10b appends to the file
    10a creates).

  ### 10.1 — Public API (`src/Tessera_MarkQuality.hpp`)
  ```cpp
  namespace Tessera {

  // Concept (duck-typed, like RefinePolicy): a criterion is any struct with
  //   template<class MeshT> std::vector<char> mark(const MeshT& mesh) const;
  // returning a mask sized mesh.numOwnedFaces(), 1 = refine this owned face.

  // --- 10a: local, no communication -------------------------------------------
  template <class Scalar>
  struct EdgeLengthCriterion {
      Scalar maxLen;                       // absolute target edge length
      template <class MeshT> std::vector<char> mark( const MeshT& mesh ) const;
  };

  // --- 10b: cross-rank gather via edge coordinators ---------------------------
  template <class Scalar>
  struct CurvatureCriterion {
      Scalar maxAngle;                     // radians; mark if dihedral bend >
      template <class MeshT> std::vector<char> mark( const MeshT& mesh ) const;
  };

  // Uniform entry point + scalar convenience overload (builds EdgeLengthCriterion).
  template <class MeshT, class Criterion>
  std::vector<char> markByQuality( const MeshT& mesh, const Criterion& crit )
  { return crit.mark( mesh ); }

  template <class MeshT>
  std::vector<char> markByQuality( const MeshT& mesh,
                                   typename MeshT::scalar_type maxEdgeLength )
  { return EdgeLengthCriterion<typename MeshT::scalar_type>{ maxEdgeLength }
             .mark( mesh ); }
  }
  ```
  - **Threshold semantics = absolute.** `maxLen` is an absolute arc-length target
    (the vortex-sheet / interface-tracking convention: insert points when a
    segment exceeds ε). `maxAngle` is an absolute dihedral bend in radians. A
    relative-to-initial variant is a future criterion, not this step.
  - Marks are `1`/`0` `char`. A face flagged by *either* criterion is refined; if
    a caller wants the union of two criteria they OR the two masks
    element-wise (documented; no combinator shipped this step).

  ### 10.2 — Step 10a: `EdgeLengthCriterion::mark` (Sonnet)
  Pure per-owned-face geometry; **no MPI**. Every owned face's 3 vertices are
  local (owned or ghost — the 1-ring closure invariant), and a shared vertex's
  `Position` is bit-identical across ranks (same gid ⇒ same position, established
  by the builder + halo sync), so the marked **set is rank-count independent with
  no communication**.

  Device-kernel path (the "trivial device kernel" the user asked for — do this
  unless it proves non-trivial, in which case fall back to a host loop and record
  a Future Optimization in README with the reason):
  1. Host copy of **vertex gids only** (`nv` uint64, not positions):
     `gid2lv[gid] = localIndex`. (`refine()` builds the identical map — reuse the
     pattern.)
  2. Build a host `Kokkos::View<int*[3]>` `faceVertLocal("fvl", nOwnedF)` from a
     host copy of `FaceField::Verts` (owned block only), translating each face's 3
     vertex gids → local indices via `gid2lv`; `deep_copy` to device. This is the
     only host→device transfer of new data; it is small (`3·nOwnedF` ints).
  3. `Kokkos::parallel_for` over `[0,nOwnedF)` in `MeshT::execution_space`:
     read the device `VertexField::Position` slice at the 3 local indices, compute
     the 3 edge lengths (`sqrt(Σ_d (p[i][d]-p[j][d])²)` over `d∈[0,Dim)`), write
     `mark(f) = (anyEdge > maxLen) ? 1 : 0` into a
     `Kokkos::View<char*>` on device. Positions are **already device-resident**
     (`mesh.vertices()`), so the kernel avoids copying `nv·Dim` scalars to host —
     the actual win over a pure-host loop.
  4. `deep_copy` the device char view → `std::vector<char>` (via a host mirror);
     return it.
  - **Empty-mesh / `nOwnedF==0` guard:** return an empty vector; do not launch a
    zero-length kernel path that trips Cabana/Kokkos on some backends — early-out.
  - **`Scalar` templating:** works for `double` and `float` unchanged (lengths in
    `Scalar`); `Dim` from `MeshT::dim` (2 or 3). No sphere assumption — plain
    Euclidean edge length in the embedding dimension.

  ### 10.3 — Step 10b: `CurvatureCriterion::mark` (Opus)
  Dihedral bend across an edge needs **both** incident faces' normals. For an
  owned face and one of its edges, the neighbour face may be owned by another rank
  and — per the Step-6b analysis — is **not guaranteed present in the vertex-based
  1-ring halo** (at a 3-way corner the edge's vertices can both be ghosts owned by
  a lower rank, so the neighbour is incident to no owned vertex). Therefore the
  gather is routed through **edge coordinators**, exactly the Step-6b Phase-1
  pattern (`detail::edgeCoordRank(EdgeKey, size)` + `allToAllV`), **not** the halo.
  1. **Per-owned-face unit normal** (device kernel, same face→vertex-local view as
     10a): `n = normalize( (p1-p0) × (p2-p0) )` using the face's `Verts` winding
     (the builder/refine maintain consistent CCW winding via the
     `edge(v[k],v[(k+1)%3])` convention, so adjacent normals are comparably
     oriented and `n0·n1` measures the true bend). **`Dim==3` only** — the cross
     product is 3D; for `Dim==2` a surface has no dihedral, so `mark()` returns
     all-zero with a one-line note (Milestone 1 is `Dim=3`). `deep_copy` normals
     to host.
  2. **Advertise to coordinators** (host, `allToAllV`): for each owned face `f`
     and each of its 3 edges `keyOf(v[k], v[(k+1)%3])`, send
     `{ EdgeKey key, Scalar n[3], GlobalId faceGid, Rank owner }` to
     `edgeCoordRank(key, size)`. (Reuse the `PropMsg`-shaped advertisement idiom;
     define a local `NormalMsg` struct — trivially copyable, `T` for `allToAllV`.)
  3. **Coordinator verdict:** group received messages by `EdgeKey`; each edge has
     exactly **2** incident faces on the closed surface (assert/skip otherwise,
     matching the Step-6b `inc.size()!=2` guard). Compute
     `cosang = clamp(n0·n1, -1, 1)`; the edge is "sharp" iff
     `acos(cosang) > maxAngle` (equivalently `cosang < cos(maxAngle)` — prefer the
     `cos` form to avoid `acos` round-off near 0). For a sharp edge, route a
     mark-request `{ GlobalId faceGid }` back to **both** incident face owners
     (both faces adjacent to a sharp fold get resolved).
  4. **Apply marks** (host): `allToAllV` the mark-requests back; for each received
     `faceGid`, set `mask[gid2of[faceGid]] = 1` (owned-face-gid → owned index map,
     built as in `refine()`). Return the mask.
  - **Rank-count independence** holds because the verdict is computed at a single
    deterministic coordinator per edge from both true incident normals — never
    from partition-local halo state — so the same faces are marked at every np.
    This is the property the gate test pins.
  - **No new persistent state, no `refine()` change.** The criterion is
    self-contained; it only reads the mesh and returns a mask.

  ### 10.4 — Tests (both regression, SERIAL+HIP, ranks 1–5)
  Add gate rows via `tessera_add_test(... TIER regression RANKS
  ${TESSERA_TEST_MPI_RANKS})` for SERIAL and HIP in `tests/CMakeLists.txt`
  (expected new count **+20**: `markquality_edge_*` np1–5 and
  `markquality_curv_*` np1–5, each ×{SERIAL,HIP}).

  - **`tests/test_markquality_edge.cpp` (10a):** `buildIcosphere(subdiv=2)` +
    `facePartitionByAxis` + `distribute`. Compute a threshold `t` strictly between
    the coarse max edge length and the once-refined max (e.g.
    `t = 0.6 × maxOwnedEdgeLen` so a nontrivial proper subset is marked). Assert:
    (a) **rank-count independence** — collect the marked owned-face **gids** into a
    global set (BXOR checksum over marked `FaceField::Gid`, `MPI_BXOR`) and assert
    it is identical at np1–5 (the gate running all five confirms agreement);
    (b) **monotone progress** — `markByQuality(mesh,t)` → `refine(mesh,halo,mask)`
    → the new global max owned edge length is `< t` for every face that was marked
    (i.e. all previously-over-length edges are gone); (c) refine post-conditions
    (`check21Balance`, `checkMidpointAgreement`, `ownedEulerGlobal==2`) hold — reuse
    `MeshInvariants.hpp`. Also assert the all-pass degenerate cases: `t` above the
    global max ⇒ empty mask (no-op refine); `t` below the global min ⇒ full mask ⇒
    reproduces one uniform subdivision level (same counts as the Step-6b uniform
    case). `double` and `float` builds.
  - **`tests/test_markquality_curv.cpp` (10b):** a **synthetic sharp-fold
    fixture** — take an icosphere and displace one vertex outward (or build a small
    two-plane "roof" soup via `buildFromTriangleSoup`) so a known ring of edges has
    a large dihedral and the rest are near-flat; distribute. Assert:
    (a) `markByQuality(mesh, CurvatureCriterion{θ})` with θ between the flat and
    fold angles marks **exactly** the faces incident to the fold edges — checked as
    a rank-count-independent marked-gid set (BXOR at np1–5, same as 10a); (b) the
    marked set is identical whether or not the fold straddles a partition boundary
    (drive it by choosing `facePartitionByAxis` axis so the fold crosses a boundary
    at np≥2 — this is the property that fails if a halo-based gather were used
    instead of the coordinator); (c) `refine` invariants hold post-mark.

  ### 10.5 — Deliverables checklist
  - `src/Tessera_MarkQuality.hpp` (10a: `EdgeLengthCriterion`, `markByQuality`
    + scalar overload; 10b appends `CurvatureCriterion`). Add to `src/Tessera.hpp`.
  - `tests/test_markquality_edge.cpp` (10a) + `tests/test_markquality_curv.cpp`
    (10b) + their gate rows in `tests/CMakeLists.txt`.
  - BSD-3-Clause SPDX header on every new file; `--target format-check` clean.
  - **README:** add a "Quality-based refinement marking" subsection to the public
    API (the `markByQuality` free fn + both criteria + the absolute-threshold
    semantics), per the "README in sync" invariant.
  - Run the gate (`flux batch scripts/tuolumne/run_regression_minset.flux` or the
    dev runner with the `regression` label); report the new regression count
    (expected **60/60** after both 10a+10b: 50 + 10, or **55/55** after 10a alone)
    and unit count (unchanged **28/28**).

  ### 10.6 — Model assignment
  - **10a → Sonnet.** Local, no distributed determinism, a device kernel + a mask
    return + a straightforward test; mechanical against this spec.
  - **10b → Opus.** Cross-rank correctness: the edge-coordinator gather and its
    rank-count-independence guarantee are exactly the distributed-determinism class
    the contract reserves for Opus (same machinery as Step 6b Phase 1).
  - *(Env caveat, per prior entries: `sonnet` delegation 403s for this
    subscription, so both may run on Opus; the column is advisory.)*

  ### 10.7 — Implementation prompts (hand to the assigned session verbatim)

  **10a (Sonnet):**
  > Implement Milestone-1 Step 10a (edge-length quality-based refinement marking)
  > in Tessera. Read `tasks/completed/milestone1_mesh.md` in full first — the "Step 10 SPEC"
  > progress-log entry (§10.0–10.2, 10.4–10.5) is your complete spec; implement it
  > with no new design decisions. Create `src/Tessera_MarkQuality.hpp` with the
  > `QualityCriterion` concept, `EdgeLengthCriterion<Scalar>`, and the
  > `markByQuality(mesh, crit)` + scalar-overload free functions per §10.1;
  > evaluate via the device Kokkos kernel described in §10.2 (host-built
  > face→vertex-local index view, read the device `Position` slice, return
  > `std::vector<char>` sized `numOwnedFaces()`), with the `nOwnedF==0` early-out.
  > Add the header to `src/Tessera.hpp`. Write `tests/test_markquality_edge.cpp`
  > and its SERIAL+HIP regression gate rows exactly per §10.4 (rank-count-
  > independent marked-gid BXOR, monotone edge-length progress after `refine`,
  > refine invariants via `MeshInvariants.hpp`, empty/full degenerate cases;
  > `double` and `float`). BSD-3-Clause SPDX header on both new files; keep
  > `--target format-check` clean; add the README subsection (§10.5). Build
  > (`run_cmake_toulumne.sh` + make) and run the gate via the dev runner with the
  > `regression` label; report the new regression count (expect 55/55 for 10a
  > alone), unit count (unchanged 28/28), whether the device kernel proved trivial
  > (or was downgraded to a host loop + README Future-Optimization with the
  > reason), and the exact `QualityCriterion` method signature as built. Append a
  > Step-10a progress-log entry.

  **10b (Opus):**
  > Implement Milestone-1 Step 10b (curvature/dihedral quality-based refinement
  > marking) in Tessera, on top of a landed Step 10a. Read `tasks/completed/milestone1_mesh.md`
  > in full first — the "Step 10 SPEC" entry (§10.0–10.1, 10.3–10.5) is your spec.
  > Append `CurvatureCriterion<Scalar>` to `src/Tessera_MarkQuality.hpp` per §10.3:
  > device-compute per-owned-face unit normals (Dim==3; Dim==2 returns all-zero),
  > advertise `(EdgeKey, normal, faceGid, owner)` to `detail::edgeCoordRank` via
  > `allToAllV`, coordinator computes the dihedral over the exactly-two incident
  > faces (`cosang < cos(maxAngle)`, closed-surface `inc.size()==2` guard), routes
  > mark-requests back to both incident face owners, apply into the owned-face
  > mask. Reuse the Step-6b Phase-1 coordinator idiom (`Tessera_RefineParallel.hpp`)
  > — this must NOT use the 1-ring halo (Step 6b documents why it is incomplete
  > across boundaries). Write `tests/test_markquality_curv.cpp` and its SERIAL+HIP
  > regression rows per §10.4 (synthetic sharp-fold fixture; marked set is exactly
  > the fold faces and is rank-count-independent AND boundary-straddle-independent;
  > refine invariants hold). BSD-3-Clause SPDX + `format-check` clean; extend the
  > README subsection. Build and run the gate; report the new regression count
  > (expect 60/60), unit count (28/28), the coordinator round count/pattern, the
  > winding/orientation assumption relied on, and the dihedral threshold used.
  > Append a Step-10b progress-log entry.
- 2026-07-07 — **Step 10a landed (Sonnet). Sixth regression-tier test — crash from
  the prior BLOCKED entry diagnosed and fixed.** `src/Tessera_MarkQuality.hpp`
  and `tests/test_markquality_edge.cpp` (already written per the Step-10 SPEC)
  are now green at np1–5, SERIAL+HIP.
  - **Root cause (confirmed by reading `Tessera_RefineParallel.hpp`, not
    guessed):** `refine()`'s Phase 3 rebuilds the local vertex AoSoA as
    `nNewV = nOwnedV + myMid.size()` (pre-refine-owned vertices + new owned
    midpoints only) — any vertex that was only a *ghost* pre-refine (e.g. a
    shared corner owned by a lower rank) is dropped entirely, not merely
    marked non-owned. The test's `maxOwnedEdgeLength()` built a `gid2lv` map
    from post-refine `mesh.vertices()` and `.at()`'d every owned edge's two
    endpoints — exactly the vertices `refine()` can drop — reproducing only
    at np≥2 (np1 has no ghosts to begin with, so nothing is ever dropped).
    This confirmed the blocked entry's hypothesis in substance, but a
    **before/after position snapshot was not sufficient** on its own: a new
    midpoint's owner (min incident *refining-face* owner, Phase 2) and an
    edge incident to it (min incident *child-face* owner, Phase 3) can be
    *different* ranks, and `refine()` ships a midpoint's **gid** to its
    co-sharers but never its **position**
    (`Tessera_RefineParallel.hpp`'s Phase-2 `KeyGid` message carries no
    position field) — so a rank can legitimately own an edge whose midpoint
    endpoint it was never told the coordinates of, and no local or
    previously-held data can supply it. Verified empirically: a
    debug-instrumented run showed the exact missing gid (`162` on a subdiv-2
    np2 SERIAL run) was a midpoint gid (subdiv-2 has `V0=162` original
    vertices, so `162` is the first newly-created gid) owned by the *other*
    rank. **This is real, but confined to a test-only computation** (no
    stored invariant in `Tessera_MarkQuality.hpp`/`refine()`/`distribute()`
    is violated — `refine()`'s "owned-only, halo cleared until Step 7"
    contract is exactly as documented), so it stayed in scope for this
    session rather than escalating to Opus.
  - Also confirmed, and explicitly ruled out as a fix, that a **self-migrate**
    (`migrate(mesh, halo, dest=self)`, the pattern the Step-8 reader uses to
    rebuild ghosts) does **not** work here: `Tessera_MeshMigrate.hpp`'s own
    documented precondition is "a valid 1-deep halo (every owned face's 3
    vertices and 3 edges are locally present)" — exactly the invariant
    `refine()` just broke — so `migrate()`'s Round A hits the identical
    `.at()` crash internally when called directly on fresh `refine()` output.
    No other call site in the codebase chains `migrate()` directly after
    `refine()`, so this combination was untested and is not (yet) a supported
    pattern.
  - **Fix:** rewrote `maxOwnedEdgeLength()` in `tests/test_markquality_edge.cpp`
    to gather any locally-unknown edge-endpoint position from its true owner
    via a **gid coordinator** (`Tessera::detail::gidCoordRank`, `gid % size`) —
    the same idiom `MeshInvariants.hpp`'s `check21Balance`/
    `checkMidpointAgreement` already use for cross-rank edge decisions
    (Step 6b), applied here to raw position data instead of level/gid: every
    rank advertises its own owned vertices' positions to their coordinators;
    every rank then requests any edge-endpoint gid it doesn't hold locally;
    the coordinator replies with the position. Purely test-local (three
    `allToAllV` rounds using the existing `Tessera::allToAllV` primitive); no
    change to `src/Tessera_MarkQuality.hpp`, `refine()`, or `migrate()`.
  - Removed the temporary debug `fprintf(stderr, ...)` checkpoints
    (`"pre-mark"`, `"post-mark"`, `"post-checksum"`, `"post-refine"`,
    `"post-invariants"`, `"post-maxedge"`) added during the blocked session's
    diagnosis. `--target format-check` clean (the `format` target also
    reformatted one pre-existing long line in `Tessera_MarkQuality.hpp` that
    predated this session, plus this file's new code).
  - Added the README "Quality-based refinement marking" subsection (public
    `markByQuality`/`EdgeLengthCriterion` API, absolute-threshold semantics,
    the ambiguous-overload SFINAE note, and the post-refine position-gather
    gotcha above for any future code needing geometry immediately after
    `refine()`), placed after "### Adaptive refinement" and before
    "### Load balancing" per the hand-off.
  - **Full gate run:** `flux batch scripts/tuolumne/run_regression_minset.flux`
    → **regression 60/60** (50 prior + 10 new
    `markquality_edge_{SERIAL,HIP}_np{1-5}`) — confirms the blocked entry's
    flagged discrepancy: the correct count is **60/60**, not the Step-10
    SPEC's §10.5/§10.7 "55/55 for 10a alone" text (that arithmetic doesn't
    match the established +10-per-step pattern from Steps 5/7/7b/8, each
    registering 2 backends × 5 ranks). `flux batch
    scripts/tuolumne/run_unit_tests.flux unit` → **unit 28/28**, unchanged.
  - **Report-back (per the contract):**
    - Device kernel: **not downgraded** — `detail::markEdgeLength` in
      `src/Tessera_MarkQuality.hpp` is unchanged from the blocked hand-off,
      still a genuine device `Kokkos::parallel_for` over owned faces reading
      the device `Position` slice directly, with only the small
      `3·nOwnedF`-int face→vertex-local view transferred host→device. It
      proved trivial as specced; no Future Optimization needed.
    - `QualityCriterion`/`EdgeLengthCriterion::mark` signature as built:
      ```cpp
      template <class Scalar>
      struct EdgeLengthCriterion {
          Scalar maxLen;
          template <class MeshT> std::vector<char> mark( const MeshT& mesh ) const;
      };
      // dispatcher (SFINAE-guarded against a bare arithmetic Criterion):
      template <class MeshT, class Criterion,
                class = std::enable_if_t<!std::is_arithmetic<Criterion>::value>>
      std::vector<char> markByQuality( const MeshT& mesh, const Criterion& crit );
      // scalar convenience overload:
      template <class MeshT>
      std::vector<char> markByQuality( const MeshT& mesh,
                                       typename MeshT::scalar_type maxEdgeLength );
      ```
    - Root cause + fix: see above — a test-only gap in computing global
      max owned edge length immediately post-`refine()` (a rank can own an
      edge whose midpoint endpoint's position was never communicated to it,
      since `refine()` ships midpoint gids to co-sharers but not positions),
      fixed with an in-test gid-coordinator position gather; no changes to
      `Tessera_MarkQuality.hpp`, `Tessera_Refine*.hpp`, or
      `Tessera_MeshMigrate.hpp`.
  - **Next:** Step 10b (curvature/dihedral criterion) — Opus, per the
    Step-10 SPEC §10.7 prompt above (unchanged).
- 2026-07-06 — **Step 10a in progress (Sonnet), BLOCKED on a distributed-run
  crash — handing off mid-task.** `src/Tessera_MarkQuality.hpp` (concept +
  `EdgeLengthCriterion<Scalar>` + `markByQuality` free fn/scalar overload, per
  §10.1/§10.2) is written and added to the umbrella `src/Tessera.hpp`; the
  device kernel path (host-built face->vertex-local index view, device
  `Position` slice, `nOwnedF==0` early-out) builds clean and **passes at
  np1** for both `double`/`float` and Serial/HIP. One real design gap found
  and fixed versus the literal §10.1 pseudocode: `markByQuality(mesh, crit)`
  and the scalar overload `markByQuality(mesh, Scalar)` are ambiguous for a
  bare scalar argument (template deduction accepts `Criterion=Scalar` for the
  generic overload too) — fixed with
  `class = std::enable_if_t<!std::is_arithmetic<Criterion>::value>` on the
  generic overload.
  - `tests/test_markquality_edge.cpp` + its SERIAL+HIP regression gate rows
    are wired into `tests/CMakeLists.txt` (`markquality_edge_{SERIAL,HIP}_np{1-5}`).
    Test design: builds an independent, un-partitioned reference mesh via
    `buildIcosphere` alone (gid==index at that stage) to compute `minEdge`/
    `maxEdge` and a partition-free expected-mark array + reference BXOR
    checksum with **no MPI at all**; separately builds+distributes the same
    icosphere, calls `markByQuality`, and compares the owned marked set (by
    gid) against that reference, plus `refine()` + `MeshInvariants` checks
    and the empty/full-mask degenerate cases.
  - **BLOCKED:** at `np==1` all 4 (Scalar x backend) variants pass. At
    `np>=2` (both SERIAL and HIP) the test aborts with
    `terminate called after throwing an instance of 'std::out_of_range'
    what(): unordered_map::at`. Debug instrumentation (temporary `fprintf
    (stderr, ...)` checkpoints, still in the file — clean up once fixed)
    narrowed it: at np2, both ranks print `post-mark ok` (so
    `markByQuality`/`detail::markEdgeLength` itself completes without
    throwing on either rank), so the throw is **downstream** of marking —
    most likely in the test's own `maxOwnedEdgeLength()` helper
    (`tests/test_markquality_edge.cpp`, computes global max edge length
    **after `refine()`** by building a `gid2lv` map from `mesh.vertices()`
    (owned+ghost) and doing `.at()` on every owned edge's two endpoint
    vertex gids). Leading hypothesis (unverified — next session should
    confirm before changing code): Step 6b's `refine()` **clears the halo**
    and leaves the mesh holding essentially only owned-closure entities
    ("Leaves owned-only entities; clears the halo... the 1-deep halo is
    rebuilt in Step 7", per the Step-6b log entry above) — so
    `mesh.vertices()` post-refine may no longer contain every vertex an
    owned edge references, and a naive gid2lv-over-`mesh.vertices()` lookup
    (the pattern used by `refine()`/`refineLocal()` internals themselves,
    but *before* clearing) is not safe to reuse verbatim on a
    freshly-refined mesh. `check21Balance`/`checkMidpointAgreement` (used
    unmodified from `MeshInvariants.hpp`) do NOT hit this because they route
    through edge coordinators (`allToAllV`) instead of a local gid2lv
    lookup — that is probably the pattern `maxOwnedEdgeLength` needs to
    follow instead, or it needs to restrict itself to data provably local
    post-refine (e.g. derive edge length from OWNED FACE vertices via the
    same per-face kernel `markEdgeLength` uses, not via `EdgeField::Verts`
    + a full-mesh vertex map). **Confirm the actual root cause before
    picking a fix** — this is a hypothesis, not a confirmed diagnosis.
  - **Next (hand-off prompt below):** diagnose + fix the np>=2 crash, get
    `markquality_edge_{SERIAL,HIP}_np{1-5}` green, remove the temporary debug
    `fprintf` instrumentation, add the README subsection (§10.5, not yet
    written), run the full gate, and append a proper Step-10a completion
    entry (report the actual new regression count — note the §10.5/§10.7
    text's "55/55 for 10a alone" appears arithmetically inconsistent with
    the established "+10 per step" pattern from Steps 5/7/7b/8 registering
    2 backends x 5 ranks each; 10a's 10 new rows should bring regression
    from 50 to **60**, not 55 — flag this discrepancy explicitly in the
    report-back rather than silently reconciling to either number).

  > **Hand-off prompt (verbatim) — recommended Sonnet, escalate to Opus only
  > if warranted:**
  >
  > Diagnose and fix a distributed-run crash in Tessera's in-progress Step
  > 10a (edge-length quality-based refinement marking). Read
  > `tasks/completed/milestone1_mesh.md` in full first, especially the "Step 10 SPEC"
  > entry (§10.0-10.2, 10.4-10.5) and the "Step 10a in progress... BLOCKED"
  > entry immediately above this prompt, which has the full symptom,
  > repro steps, and a leading (unconfirmed) hypothesis.
  >
  > Summary of state: `src/Tessera_MarkQuality.hpp` and
  > `tests/test_markquality_edge.cpp` exist and are wired into
  > `tests/CMakeLists.txt` as regression rows
  > (`markquality_edge_{SERIAL,HIP}_np{1-5}`) but are **not yet green** — do
  > not report Step 10a complete until they are. The build directory
  > `build-tuolumne` is already configured (spack env
  > `~/spack_envs/tuolumne_trilinos/` active, `run_cmake_toulumne.sh` already
  > run) and builds clean. Repro: `export TESSERA_REPO=$(pwd); flux batch
  > scripts/tuolumne/run_unit_tests.flux regression -R markquality_edge`
  > (or a smaller flux job running `tessera_test_markquality_edge_SERIAL`
  > directly at `--ntasks 2`, as the blocked entry's debug job did) — np1
  > passes, np>=2 aborts with `unordered_map::at` / `std::out_of_range`
  > immediately after both ranks print the temporary debug line
  > `post-mark ok`, i.e. after `markByQuality()` itself succeeds.
  >
  > Confirm the actual throw site (add/adjust the temporary `fprintf(stderr,
  > ...)` checkpoints already in the test, or a debugger/core dump, or
  > targeted try/catch around candidate `.at()` calls) before changing any
  > code — do not guess-fix. The blocked entry's hypothesis is that
  > `maxOwnedEdgeLength()` (a test-only helper in
  > `tests/test_markquality_edge.cpp`) is unsafe to call on a
  > freshly-`refine()`-d mesh because Step 6b's `refine()` clears the halo
  > (ghosts are only rebuilt in Step 7's `migrate()`), so a `gid2lv` map
  > built from `mesh.vertices()` may not cover every vertex an **owned
  > edge** references post-refine. If confirmed, fix by making
  > `maxOwnedEdgeLength()` derive lengths only from data guaranteed local
  > post-refine (e.g. per-owned-face vertex lookups, the same closure
  > `detail::markEdgeLength` relies on, rather than `EdgeField::Verts` +
  > a full-mesh vertex map) — or by using an edge-coordinator gather like
  > `MeshInvariants.hpp`'s `check21Balance`/`checkMidpointAgreement` already
  > do for exactly this reason. If the actual root cause is instead
  > something in `Tessera_MarkQuality.hpp` itself, or a genuine
  > halo/ownership invariant bug in `refine()`/`distribute()` rather than a
  > test-only issue, **stop and escalate to Opus** — that would mean this is
  > a distributed-correctness design question, not a mechanical fix, and is
  > out of scope for a Sonnet session per this repo's Sonnet/Opus contract
  > (see the Step-contract table near the top of `tasks/completed/milestone1_mesh.md`).
  >
  > Once green at np1-5 for both SERIAL and HIP: remove the temporary debug
  > `fprintf(stderr, ...)` lines from `tests/test_markquality_edge.cpp`
  > (search for `"pre-mark"`, `"post-mark"`, `"post-checksum"`,
  > `"post-refine"`, `"post-invariants"`, `"post-maxedge"`), confirm
  > `--target format-check` is clean, add the still-missing README
  > "Quality-based refinement marking" subsection per §10.5 (a home for it:
  > right after the existing "### Adaptive refinement" section and before
  > "### Load balancing", matching the doc's existing section order), run
  > the full regression gate (`flux batch
  > scripts/tuolumne/run_regression_minset.flux`, or the dev runner with the
  > `regression` label), and append a **new** Step-10a completion entry to
  > `tasks/completed/milestone1_mesh.md` (do not edit this blocked entry — append
  > below it) reporting: the actual new regression count (expect 60/60,
  > per the discrepancy noted above — call it out either way), the unit
  > count (expect unchanged 28/28), whether the device kernel proved
  > trivial or was downgraded to a host loop (it was NOT downgraded as of
  > this hand-off — record if that changes), the exact
  > `QualityCriterion`/`EdgeLengthCriterion::mark` signature as built, and
  > the root cause + fix for this crash.
- 2026-07-07 — **Step 10b landed (Opus). Seventh regression-tier test —
  Milestone-1 marking suite complete.** Curvature/dihedral criterion appended to
  `src/Tessera_MarkQuality.hpp`; `tests/test_markquality_curv.cpp` + its
  SERIAL+HIP gate rows green at np1–5.
  - `CurvatureCriterion<Scalar>{maxAngle}` (radians) marks **both** faces
    incident to any edge whose dihedral bend exceeds `maxAngle`, evaluated by
    `detail::markCurvature`. The dihedral needs both incident faces' normals and
    the neighbour across a partition boundary is **not** in the vertex-based
    1-ring halo (Step-6b analysis), so the gather is routed through **edge
    coordinators** (`detail::edgeCoordRank` + `allToAllV`), reusing the Step-6b
    Phase-1 idiom — **not** the halo. Implementation:
    - Device kernel (same host-built face→vertex-local view as 10a) computes the
      per-owned-face outward unit normal `n = normalize((p1−p0)×(p2−p0))`,
      reading the device `Position` slice directly; copied to host.
    - Each owned face advertises `{EdgeKey, Scalar n[3], faceGid, owner}`
      (`detail::NormalMsg<Scalar>`, trivially copyable) for each of its 3 edges to
      `edgeCoordRank(key, size)`.
    - Coordinator groups by `EdgeKey`, uses the **exactly-two-incident** guard
      (`inc.size()!=2` skipped, matching the closed-surface assumption the whole
      codebase relies on), and flags an edge sharp iff `clamp(n0·n1,−1,1) <
      cos(maxAngle)` (the `cos` form avoids `acos` round-off near 0). A sharp
      edge routes a `GlobalId faceGid` mark-request back to **both** incident
      face owners.
    - Owners `set mask[gid2of[faceGid]] = 1`. No new stored state, no `refine()`
      change; `Dim==2` returns an all-zero mask via `if constexpr` (a planar
      surface has no dihedral; Milestone 1 is `Dim=3`). `markByQuality(mesh,
      crit)`'s existing non-arithmetic-SFINAE generic overload dispatches it
      unchanged — no dispatch edit needed.
  - `tests/test_markquality_curv.cpp` (**regression**, SERIAL+HIP, np1–5,
    double+float): synthetic sharp-fold fixture = subdiv-2 icosphere with
    original vertex gid 0 displaced radially ×3 (a 5-face spike). An independent
    replicated reference (gid==index, **no MPI**) computes per-face normals /
    per-edge dihedrals with the *same* formula as the device kernel and derives
    the threshold `theta` as the **midpoint of the largest gap in the sorted
    dihedral distribution** (so `theta` sits far from every actual dihedral →
    the marked set is insensitive to host/device float rounding, and no
    hard-coded angle can drift stale). That same `theta` drives the distributed
    `CurvatureCriterion`. Asserts: marked owned-face gid set (BXOR) matches the
    partition-free reference at every np (rank-count independence), global marked
    count == reference (exactly the fold faces, a nontrivial proper subset
    `0<count<320`), and `refine()` post-conditions (`check21Balance`,
    `checkMidpointAgreement`). Euler==2 is deliberately **not** asserted
    (adaptive marking leaves bounded hanging nodes, per the Step-6b adaptive
    case). **Boundary-straddle independence** is exercised by
    `facePartitionByAxis(axis=z)`: vertex 0's fold ring spans the full z-extent,
    so the fold straddles partition bands at np≥2 and the fold-edge neighbour is
    frequently cross-rank and absent from the halo — a halo-based gather would
    disagree across np, which the cross-np BXOR agreement pins.
  - **Gate:** `ctest -L regression -R "SERIAL|HIP" --output-on-failure` — **all
    pass, regression 70/70** (60 prior + 10 new `markquality_curv_{SERIAL,HIP}_
    np{1-5}`). **Unit 28/28 unchanged.** `--target format-check` clean (the
    `format` target reflowed a few continuation lines in both new files).
  - **Report-back (per the Step-10b contract):**
    - *Regression count:* **70/70**, not the prompt's "expect 60/60". That
      number is stale: the Step-10 SPEC §10.5/§10.7 ("55/55 for 10a alone" /
      "60/60 after both") mis-assumed 10a+10b together add only 10 rows, but the
      established pattern is **+10 per step** (2 backends × 5 ranks). Step 10a
      alone already reached 60/60 (its completion entry flagged the same
      arithmetic error); 10b's 10 rows bring it to **70/70**. Flagging
      explicitly rather than reconciling to the stale figure.
    - *Coordinator round count/pattern:* **one non-iterative round-trip** = two
      `allToAllV` calls (advertise normals → coordinator verdict → mark-request
      reply). This is *simpler* than the Step-6b Phase-1 it reuses: 6b's 2:1
      mark-propagation iterates to an `MPI_Allreduce`-guarded fixpoint because a
      mark can induce further marks, whereas the curvature verdict is a **pure
      geometric function** of the two incident normals, so a single gather
      suffices — no fixpoint, no Allreduce loop. Same `edgeCoordRank` routing and
      `allToAllV` substrate, same exactly-2-incident guard.
    - *Winding/orientation assumption:* the builder and `refine()` maintain a
      **consistent CCW-seen-from-outside winding** with the edge convention
      `e[k]=(v[k],v[(k+1)%3])`, so `n = normalize((p1−p0)×(p2−p0))` is the
      **outward** normal on every face and adjacent faces' normals are comparably
      oriented ⇒ `n0·n1 = cos(dihedral bend)` (= 1 flat, decreasing as the fold
      sharpens). Relies on the closed genus-0 surface giving exactly two incident
      faces per edge (the coordinator skips any edge without exactly two).
    - *Dihedral threshold used in the test:* **runtime-derived**, not
      hard-coded — the midpoint of the largest gap in the reference mesh's sorted
      per-edge dihedral distribution (with vertex 0 displaced ×3 on subdiv-2 this
      lands cleanly between the shallow flat cluster and the steep fold cluster),
      passed verbatim to `CurvatureCriterion`.
  - **Next:** Step 10 (both a+b) complete → Milestone-1 quality-marking suite
    done. Remaining deferred items: Step 9 (end-to-end example + runner scripts +
    OPENMP smoke-build + CI subset wiring).

- 2026-07-07 — **Step 9 landed (Sonnet).** New example
  `examples/02_mesh_pipeline/mesh_pipeline.cpp` chains the full pipeline:
  `buildIcosphere` → `facePartitionByAxis`+`distribute` → baseline `writeMesh`
  frame 0 → `--iters` iterations of (uniform-random `--frac` of owned faces
  marked → `refine` → `loadBalance` or self-destination `migrate` halo rebuild
  → `haloExchange` → `writeMesh` frame N), run once per available Kokkos
  execution space (`Serial` always; `DefaultExecutionSpace` if distinct — HIP
  on Tuolumne, satisfying the gate backend; `OpenMP` if `KOKKOS_ENABLE_OPENMP`
  and distinct from both, satisfying the non-gate OPENMP smoke-build). CLI
  args: `--subdiv`, `--axis`, `--balance`, `--iters`, `--frac`, `--seed`,
  `--out` (mirrored into README's Example programs table). Every pipeline step
  in the source carries a comment explaining what's called, why, and what the
  arguments mean, for a first-time downstream reader.
  - Wired into `examples/CMakeLists.txt` (`add_subdirectory(02_mesh_pipeline)`).
  - New runner scripts under `scripts/tuolumne/`:
    - `run_unit_minset.flux` — release-quality unit runner (mirrors
      `run_regression_minset.flux`'s structure, hardcoded `label=unit`; the
      two release runner names are now `run_regression_minset.flux` (existing)
      and `run_unit_minset.flux` (new)).
    - `run_mesh_pipeline_example.flux` — DEV example runner (not a ctest gate
      entry — mirrors `run_unit_tests.flux`'s dev-tool style), submits on the
      **pdebug** queue, runs `mesh_pipeline` at **4 MPI ranks**; since
      `Kokkos::DefaultExecutionSpace` is HIP on Tuolumne, this exercises the
      HIP backend's `writeMesh` I/O path in addition to the always-run Serial
      pass. Requires `export TESSERA_REPO=$(pwd)` before submit (same
      constraint as `run_unit_tests.flux`).
  - CI subset: no change needed — `.github/workflows/ci.yml` already builds
    `examples/` (`-DTessera_ENABLE_EXAMPLES=ON`) and runs
    `regression`/`SERIAL` at ranks 1–2; the new example adds no ctest entries,
    so it's smoke-compiled by CI but not executed there (matches the Step-9
    contract; no new CI dependency needed since the example only uses APIs
    already linked via `Tessera::Tessera`).
  - README updated: "Usage / API" pseudocode block replaced with the real
    free-function pipeline (`buildIcosphere`, `facePartitionByAxis`,
    `distribute`, `haloExchange`, `refine`, `loadBalance`/`migrate`,
    `writeMesh`) and the stale "Planned" disclaimer removed; "Example
    programs" table gained a `mesh_pipeline` row, placeholder line removed.
  - **Per user instruction this session**, the build/format-check/manual-run/
    gate-run verification steps (CLAUDE.md §"Build & verify") were **skipped
    for this entry** since no library source was touched — only new
    example/script/doc files were added, so the existing gate (**regression
    70/70, unit 28/28** as of Step 10b) is unaffected. The user will build,
    run `mesh_pipeline` manually, and verify the output `.h5`/`.xmf` frame
    sequence in Paraview themselves; the `run_mesh_pipeline_example.flux`
    script above is the submission path for that verification.
  - **Next:** Milestone 1 step contract (Steps 0–10b) is now fully landed.
    Start a new milestone/task file for the next phase of work.

- 2026-07-08 — **Step 9 example crash diagnosed + fixed (Opus).** The first
  manual run of `mesh_pipeline` (default args, no `--balance`) aborted at
  iteration 1 with `std::out_of_range` / `map::at` on 3 of 4 ranks
  (`tessera-mesh-pipeline-example.f3Js8vX8yyBm.out`). Two distinct latent
  **library** bugs, both in the `refine()`→halo-rebuild composition that no gate
  test exercised (the refine tests only check owned-entity invariants; the
  migrate/loadbalance tests only run post-`distribute()` meshes):
  - **Bug 1 — `migrate()` could not rebuild the halo of a freshly-refined
    mesh.** `refine()` leaves each rank owning faces whose vertices/edges may be
    owned elsewhere and held nowhere locally (a new midpoint's *position* is
    never shipped to co-sharers, only its gid — see README post-refine gotcha),
    but `migrate()` round A assumed every owned face's 3 verts/3 edges were
    locally present (`gid2lv.at`/`gid2le.at`). `readMesh()`→`migrate()` only
    worked because `readMesh` pre-fetches all referenced records first. Fix: a
    new **round G** in `migrate()` gathers referenced-but-non-held vertex/edge
    tuples from their true owner via a gid coordinator (`gid % size`) before
    round A — the same idiom `maxOwnedEdgeLength()`/the invariant checks use;
    a no-op (guarded by an `MPI_Allreduce` need-count) for the halo-consistent
    callers (`distribute()`+`haloExchange`, `readMesh`). `ownedFaceCentroids()`
    (driven by `loadBalance()`) had the identical missing-vertex flaw and got
    the same position-gather. `migrate()`'s precondition doc updated: a valid
    1-deep halo is no longer required.
  - **Bug 2 — `refine()` produced colliding face gids across rounds.** Child
    faces were numbered from the global owned-face **count**, but a refined
    parent's gid is retired (replaced by 4 children), so after a round the live
    gids are sparse and the max exceeds the count; the next round's children
    then collided with the previous round's high-numbered children. Invisible in
    the self-destination `migrate()` path (colliding gids never meet on one
    rank) but fatal once `--balance`'s real Zoltan2 redistribution brought two
    ranks' same-gid faces together (`faceById` dedup dropped one, stranding its
    verts). Fix: base child-face gids on the global **max** existing face gid
    `+ 1` (via `MPI_Allreduce(MAX)`), not the count. Vertices are unaffected
    (never retired → contiguous); edges are unaffected (fully renumbered from 0
    each round).
  - **Verified:** `mesh_pipeline` now completes rc=0 at 4 ranks across Serial +
    HIP (DefaultExecutionSpace) + OpenMP for both default args (frames 0–3) and
    `--balance --iters 3`. Library source changed, so the full gate was rerun:
    **regression 70/70, unit 28/28** green. Fix is `src/Tessera_MeshMigrate.hpp`
    + `src/Tessera_RefineParallel.hpp`; the example source itself was not
    modified.
