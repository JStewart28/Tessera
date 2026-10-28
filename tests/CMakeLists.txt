############################################################################
# Copyright (c) 2024, JStewart28                                           #
# All rights reserved.                                                     #
#                                                                          #
# This file is part of the Tessera library. Tessera is distributed under a #
# BSD 3-Clause license. For the licensing terms see the LICENSE file in   #
# the top-level directory.                                                 #
#                                                                          #
# SPDX-License-Identifier: BSD-3-Clause                                    #
############################################################################
#
# Tessera test suite — tiered by label, selectable by backend.
#
# TIER LABELS
# -----------
#   regression  Full end-to-end tests composing the whole pipeline.
#               These form the ship gate — must pass before merging.
#               Run: ctest -L regression -R SERIAL          (Serial backend)
#                    ctest -L regression -R HIP              (HIP backend)
#                    ctest -L regression -R "SERIAL|HIP"     (both, gate)
#
#   unit        Single-component or utility tests. Diagnostic; not ship-gated.
#               Run: ctest -L unit
#
# BACKEND SUFFIXES
# ----------------
#   Test names follow the convention: <name>_<BACKEND>_np<N>
#     SERIAL  — Kokkos::Serial (CPU, single-threaded)
#     OPENMP  — Kokkos::OpenMP (CPU, multi-threaded)
#     HIP     — Kokkos::HIP (AMD GPU, MI300A on Tuolumne)
#   Select a backend: ctest -R SERIAL  or  ctest -R HIP
#
# RANK PARAMETERIZATION
# ---------------------
#   TESSERA_TEST_MPI_RANKS defaults to "1;2;3;4;5" (the gate rank set).
#   Override: cmake -DTESSERA_TEST_MPI_RANKS="1;2"
#
# GATE DEFINITION (must match CLAUDE.md, run_regression_minset.flux, and CI)
#   Label:    regression
#   Backends: SERIAL, HIP
#   Ranks:    1, 2, 3, 4, 5
#
# NOTE: a failing test is never silently excluded from the gate.
#   If a test is unstable, label it 'unit', record it in README Known Issues,
#   and report it — never promote a failing test to the gate.

set(TESSERA_TEST_MPI_RANKS "1;2;3;4;5" CACHE STRING
    "MPI rank counts over which to parameterize tests (gate uses 1-5)")

# ── Helper macro ──────────────────────────────────────────────────────────────
# tessera_add_test(
#   NAME     <base-name>           e.g. mesh_partition
#   BACKEND  <SERIAL|OPENMP|HIP>
#   TIER     <regression|unit>
#   RANKS    <rank-list>           default: "1" (override per call)
#   SOURCES  <source files>
# )
#
# Creates one executable and one CTest entry per (NAME, BACKEND, rank) triple.
# Test name: <NAME>_<BACKEND>_np<N>
# Labels:    <TIER> and <BACKEND> (multiple labels supported by CTest)
macro(tessera_add_test)
    cmake_parse_arguments(TAT "" "NAME;BACKEND;TIER" "RANKS;SOURCES" ${ARGN})

    if(NOT TAT_RANKS)
        set(TAT_RANKS "1")
    endif()
    if(NOT TAT_TIER)
        message(FATAL_ERROR "tessera_add_test: TIER is required (regression or unit)")
    endif()
    if(NOT TAT_BACKEND)
        message(FATAL_ERROR "tessera_add_test: BACKEND is required (SERIAL, OPENMP, or HIP)")
    endif()

    set(_exe tessera_test_${TAT_NAME}_${TAT_BACKEND})

    add_executable(${_exe} ${TAT_SOURCES})
    target_link_libraries(${_exe} PRIVATE Tessera::Tessera)

    if(Tessera_INSTALL_TEST_EXECUTABLES)
        install(TARGETS ${_exe} DESTINATION ${CMAKE_INSTALL_DATADIR}/Tessera/tests)
    endif()

    foreach(_np IN LISTS TAT_RANKS)
        set(_test_name ${TAT_NAME}_${TAT_BACKEND}_np${_np})
        add_test(
            NAME    ${_test_name}
            COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} ${_np}
                    ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:${_exe}>
        )
        set_tests_properties(${_test_name} PROPERTIES
            LABELS "${TAT_TIER};${TAT_BACKEND}"
        )
    endforeach()
endmacro()

# ── Tests ─────────────────────────────────────────────────────────────────────
#
# Step 2 (core data model) unit tests. Each source internally exercises BOTH the
# host (Kokkos::Serial) and device (Kokkos::DefaultExecutionSpace, = HIP on
# Tuolumne) execution spaces, so the SERIAL- and HIP-labelled registrations run
# the same coverage; the label selects which the gate/ctest filter picks up.
# Single rank (no MPI decomposition exercised until Step 5).

# Structured-key (EdgeKey/FaceKey) determinism.
tessera_add_test(
    NAME     keys
    BACKEND  SERIAL
    TIER     unit
    RANKS    "1"
    SOURCES  test_keys.cpp
)
tessera_add_test(
    NAME     keys
    BACKEND  HIP
    TIER     unit
    RANKS    "1"
    SOURCES  test_keys.cpp
)

# Core data model: Mesh construction + core/user field-pack slices (double+float,
# Dim=3 and Dim=2).
tessera_add_test(
    NAME     data_model
    BACKEND  SERIAL
    TIER     unit
    RANKS    "1"
    SOURCES  test_data_model.cpp
)
tessera_add_test(
    NAME     data_model
    BACKEND  HIP
    TIER     unit
    RANKS    "1"
    SOURCES  test_data_model.cpp
)

# Serial icosphere builder + connectivity: Euler, edge/face incidence, vertex
# 1-ring CSR round-trip.
tessera_add_test(
    NAME     connectivity
    BACKEND  SERIAL
    TIER     unit
    RANKS    "1"
    SOURCES  test_connectivity.cpp
)
tessera_add_test(
    NAME     connectivity
    BACKEND  HIP
    TIER     unit
    RANKS    "1"
    SOURCES  test_connectivity.cpp
)

# Whole-tuple AoSoA migration primitive: round-trip scatter/shift/identity over
# 1-5 ranks (conservation + payload integrity). Multi-rank, so parameterized over
# the full rank set.
tessera_add_test(
    NAME     migrate
    BACKEND  SERIAL
    TIER     unit
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_migrate.cpp
)
tessera_add_test(
    NAME     migrate
    BACKEND  HIP
    TIER     unit
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_migrate.cpp
)

# Halo primitives: allToAllV neighbour topology exchange + HaloExchangePlan
# whole-tuple field sync (ring halo) over 1-5 ranks.
tessera_add_test(
    NAME     halo
    BACKEND  SERIAL
    TIER     unit
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_halo.cpp
)
tessera_add_test(
    NAME     halo
    BACKEND  HIP
    TIER     unit
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_halo.cpp
)

# Local 1->4 red refinement (Step 6a): uniform Euler/counts, partial shared
# midpoints + child incidence, default vs custom per-field interpolation policy.
# Single rank (parallel 2:1 balance is Step 6b).
tessera_add_test(
    NAME     refine
    BACKEND  SERIAL
    TIER     unit
    RANKS    "1"
    SOURCES  test_refine.cpp
)
tessera_add_test(
    NAME     refine
    BACKEND  HIP
    TIER     unit
    RANKS    "1"
    SOURCES  test_refine.cpp
)

# RefinementMode plumbing (conforming refinement, Task 1): a Conforming-typed
# mesh instantiates and its conditional closure face members round-trip; the
# public face user-field indices (FaceField::UserBegin, userFaceField<M>()) are
# bit-identical in both modes, and the Conforming face tuple is exactly two
# members larger. Type-layout property -> single rank.
tessera_add_test(
    NAME     refinement_mode
    BACKEND  SERIAL
    TIER     unit
    RANKS    "1"
    SOURCES  test_refinement_mode.cpp
)
tessera_add_test(
    NAME     refinement_mode
    BACKEND  HIP
    TIER     unit
    RANKS    "1"
    SOURCES  test_refinement_mode.cpp
)

# Serial closure kernel (conforming refinement, Task 2): the green/blue/red
# closure patterns and their inverse. Pins the hand-computed child triangles for
# each |S| in {0,1,2,3}, CCW winding, the blue lower-gid diagonal tie-break in
# both directions, rotation invariance of the emitted triangle set (why the
# tie-break is partition-independent), unclose o close == identity on a random
# partial mask over buildIcosphere(3), and -- for refineLocal() in Conforming
# mode with a PARTIAL mask -- Euler V-E+F == 2, exactly two faces per edge, no
# T-junction, outward winding, and the same vertex count as HangingNode2to1 mode.
# The same conformity checks are run on the hanging-node result and must fail
# there, so the case cannot go vacuous. Local by construction -> single rank.
tessera_add_test(
    NAME     refine_closure
    BACKEND  SERIAL
    TIER     unit
    RANKS    "1"
    SOURCES  test_refine_closure.cpp
)
tessera_add_test(
    NAME     refine_closure
    BACKEND  HIP
    TIER     unit
    RANKS    "1"
    SOURCES  test_refine_closure.cpp
)

# Stale-handle generation guard: a slice held across a count-changing op and
# then copied (as it would be when captured into a KOKKOS_LAMBDA) must abort;
# a slice held across haloExchange() (topology-preserving) must not. Also
# covers the CONFORMING refine path, which un-closes and re-closes and so
# changes the local face count even when nothing refines -- a face slice, a
# MeshGeometry, and a VertexStencil held across it must all abort. This
# exercises host-side generation bookkeeping, not device kernels, so it is
# registered SERIAL-only (no HIP duplicate, unlike the other unit tests).
tessera_add_test(
    NAME     staleslice_guard
    BACKEND  SERIAL
    TIER     unit
    RANKS    "1"
    SOURCES  test_staleslice_guard.cpp
)

# Raw geometric primitives (Milestone-1 operators): faceArea, faceNormalRaw,
# edgeVector, cotangentAtCorner on an analytic two-triangle mesh with exact
# hand-computed answers, checked host-side after device evaluation. Single rank.
tessera_add_test(
    NAME     geometry
    BACKEND  SERIAL
    TIER     unit
    RANKS    "1"
    SOURCES  test_geometry.cpp
)
tessera_add_test(
    NAME     geometry
    BACKEND  HIP
    TIER     unit
    RANKS    "1"
    SOURCES  test_geometry.cpp
)

# k-ring vertex stencil topology (buildVertexStencil): k=1 row equals the
# edge-derived 1-ring; k=2 row is the exact distance-<=2 set and a superset of
# k=1; self excluded. Single rank (topology derivation, gid == local).
tessera_add_test(
    NAME     stencil_topology
    BACKEND  SERIAL
    TIER     unit
    RANKS    "1"
    SOURCES  test_stencil_topology.cpp
)
tessera_add_test(
    NAME     stencil_topology
    BACKEND  HIP
    TIER     unit
    RANKS    "1"
    SOURCES  test_stencil_topology.cpp
)

# Weighted-stencil apply (applyStencil): analytic field f(p)=p_x, uniform
# weights, reference computed from (haloed) positions independent of the
# in-field halo path -- so a failure is unambiguously an apply/halo bug. Over
# 1-5 ranks to exercise partition boundaries.
tessera_add_test(
    NAME     apply_stencil
    BACKEND  SERIAL
    TIER     unit
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_apply_stencil.cpp
)
tessera_add_test(
    NAME     apply_stencil
    BACKEND  HIP
    TIER     unit
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_apply_stencil.cpp
)

# Face -> vertex reduction (reduceVertexFromFaces): a one-third-area op whose
# global owned-vertex sum equals the total surface area (global owned-face
# faceArea sum) -- a partition-independent identity. Over 1-5 ranks.
tessera_add_test(
    NAME     reduce_faces
    BACKEND  SERIAL
    TIER     unit
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_reduce_faces.cpp
)
tessera_add_test(
    NAME     reduce_faces
    BACKEND  HIP
    TIER     unit
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_reduce_faces.cpp
)

# ── Regression tier (ship gate: label regression x {SERIAL,HIP} x ranks 1-5) ────

# Global scalar collectives (globalMin/globalMax/globalSum/globalAllFinite) and
# the owned-count reductions. Extrema and the dissenting rank are placed away
# from rank 0 so a rank-0 shortcut cannot pass; the owned-count checks run on a
# real distributed icosphere before and after a uniform refine.
tessera_add_test(
    NAME     global_reduce
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_global_reduce.cpp
)
tessera_add_test(
    NAME     global_reduce
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_global_reduce.cpp
)

# Distributed mesh + deterministic ownership + 1-deep halo (Step 5). First
# end-to-end gate test: partition -> distribute -> ownership/1-ring invariants ->
# halo sync.
tessera_add_test(
    NAME     distribute
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_distribute.cpp
)
tessera_add_test(
    NAME     distribute
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_distribute.cpp
)

# Mesh migration + general (non-replicated) 1-deep halo rebuild (Step 7):
# external face assignment lands entities on requested ranks, ownership is a
# partition, every owned vertex's 1-ring is local, rank-count-independent topology
# checksum preserved, ghost halo sync. The general ghost builder deferred from 6b.
tessera_add_test(
    NAME     migrate_mesh
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_migrate_mesh.cpp
)
tessera_add_test(
    NAME     migrate_mesh
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_migrate_mesh.cpp
)

# Distributed 2:1-balanced refinement (Step 6b): cross-rank midpoint-gid
# agreement, 2:1 level invariant, owned-only Euler (uniform), fixpoint
# termination (adaptive). Owned-entity invariants; halo rebuild is Step 7.
tessera_add_test(
    NAME     refine_parallel
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_refine_parallel.cpp
)
tessera_add_test(
    NAME     refine_parallel
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_refine_parallel.cpp
)

# Distributed split-edge discovery (conforming refinement, Task 3): refine()'s
# Phase 2 advertises EVERY owned face's edges, so RefineResult::midpoints is a
# complete split-edge map -- a rank holding only the KEPT side of an edge a
# neighbour bisected still learns its midpoint gid, which is what the conforming
# closure (Task 4) needs. Pins round-1 exactness against a partition-free
# reference, the midpoint gids forming the contiguous block above the pre-refine
# global vertex count, non-vacuity (kept-side discovery actually occurs at
# ranks >= 2), and coordinator-decided soundness+completeness over three further
# adaptive rounds where the 2:1 fixpoint really propagates.
tessera_add_test(
    NAME     refine_splitedges
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_refine_splitedges.cpp
)
tessera_add_test(
    NAME     refine_splitedges
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_refine_splitedges.cpp
)

# Distributed CONFORMING refinement (conforming refinement, Task 4) -- the
# milestone test for the feature. Over three successive ADAPTIVE rounds: every
# global edge has exactly two incident faces, owned-only Euler V-E+F == 2 (the
# criterion the hanging-node mode fails under an adaptive mask), no vertex
# strictly inside an edge, ownership still a partition, midpoint gids still
# agreed cross-rank, 2:1 balance on the RED layer, and unclose o close reproducing
# the visible and red layers bit-for-bit. The same mask is run on a
# HangingNode2to1 mesh as a non-vacuity control and MUST fail the conformity
# checks there. Plus the degenerate empty-mask (identity) and full-mask (no kept
# faces, so the closure is inert and the counts equal the hanging-node mode's)
# cases. Prints the closure-face fraction and |S| histogram per round.
tessera_add_test(
    NAME     refine_conforming
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_refine_conforming.cpp
)
tessera_add_test(
    NAME     refine_conforming
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_refine_conforming.cpp
)

# migrate() / loadBalance() / halo rebuild on a CONFORMING mesh (conforming
# refinement, Task 5). A closure child names its retired red parent outright, so
# splitting a sibling group across ranks would have each holding rank restore
# the same red face. migrate() repairs `dest` so siblings follow the lowest-gid
# sibling; this drives that with a deliberately sibling-splitting assignment and
# with Zoltan2 (which scatters siblings on its own, partitioning by centroid),
# then re-checks every conforming invariant, the owned 1-ring, the ghost
# corrupt-resync over the wider conforming face tuple, and the topology
# checksum. Prints the fixup counts and the pre/post imbalance figures.
tessera_add_test(
    NAME     conforming_migrate
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_conforming_migrate.cpp
)
tessera_add_test(
    NAME     conforming_migrate
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_conforming_migrate.cpp
)

# Internal Zoltan2 loadBalance() (Step 7b): rebalancing a deliberately
# maximally-imbalanced mesh reduces the max-owned-face imbalance while
# preserving every migrate() distribution invariant and the rank-count-
# independent topology checksum.
tessera_add_test(
    NAME     loadbalance
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_loadbalance.cpp
)
tessera_add_test(
    NAME     loadbalance
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_loadbalance.cpp
)

# Distributed Zoltan2 load-balance solve (tasks/completed/distributed-loadbalance-solve.md):
# LoadBalanceMode::{GatherRoot,Distributed,Sampled}. The deliverable is that rank 0
# receives ZERO faces for its solve in the default Distributed mode (case 6), plus
# validity/balance/conforming/reproducibility/quality/single-rank/idempotence checks
# and printed scaling evidence at subdivision 5. test_loadbalance above stays as the
# GatherRoot regression guard.
tessera_add_test(
    NAME     loadbalance_distributed
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_loadbalance_distributed.cpp
)
tessera_add_test(
    NAME     loadbalance_distributed
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_loadbalance_distributed.cpp
)

# Parallel HDF5 + XDMF writer/reader round-trip (Step 8): write -> read ->
# compare topology + user-field checksums, on-disk gid checksum independent
# of rank count, and a post-read corrupt/haloExchange ghost-restoration check.
tessera_add_test(
    NAME     io
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_io.cpp
)
tessera_add_test(
    NAME     io
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_io.cpp
)

# XDMF time-series grouping (T2 of tasks/fix-file-grouping-io.md): three frames
# written through one MeshSeries produce ONE master .xmf holding a temporal
# collection -- exactly one CollectionType="Temporal", three fully independent
# child grids with their own <Time>/<Topology>/<Geometry>, each referencing only
# its own frame's .h5 basename, and an identically ordered <Attribute Name= set
# across all three (risk R1's diagnostic). Also asserts the .xmf.tmp rename-over
# leaves no temp behind, that python3's xml.etree parses the master, and that a
# non-increasing time and an out-of-directory frame stem each throw
# std::runtime_error leaving the master byte-unchanged. The master is rank-0
# text with no device involvement, so SERIAL only; unit tier, so the ship gate
# definition is untouched.
tessera_add_test(
    NAME     xdmf_series
    BACKEND  SERIAL
    TIER     unit
    RANKS    "1;2;3"
    SOURCES  test_xdmf_series.cpp
)

# Edge-length quality-based refinement marking (Step 10a): markByQuality's
# marked owned-face gid set matches an independent, partition-free reference
# (rank-count independence), monotone edge-length progress after refine(),
# refine() post-conditions (2:1 balance, midpoint agreement), and the
# empty-mask/full-mask degenerate cases.
tessera_add_test(
    NAME     markquality_edge
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_markquality_edge.cpp
)
tessera_add_test(
    NAME     markquality_edge
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_markquality_edge.cpp
)

# Curvature/dihedral quality-based refinement marking (Step 10b): CurvatureCriterion
# marks both faces incident to any sharp-fold edge. The dihedral gather is routed
# through edge coordinators (allToAllV), NOT the 1-ring halo, so the marked set is
# rank-count- AND boundary-straddle-independent -- pinned by matching a partition-
# free geometric reference at ranks 1-5. refine() post-conditions still hold.
tessera_add_test(
    NAME     markquality_curv
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_markquality_curv.cpp
)
tessera_add_test(
    NAME     markquality_curv
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_markquality_curv.cpp
)

# Surface OPERATORS on a conforming mesh (Task 7 of
# tasks/completed/conforming-refinement.md) -- the payoff test for the whole feature. A
# hanging node's neighbourhood is self-consistent but is a HALF-disc, so every
# operator assembled over it silently misses the kept face it touches. This
# identifies the closure vertices (former hanging nodes) from the
# ClosureParent/ClosureParentVerts bookkeeping and checks, at those vertices
# specifically: the 1-ring is a closed fan, buildVertexStencil's k=1 row equals
# the independently face-derived 1-ring, applyStencil matches an analytic
# reference to the SAME tolerance as at interior vertices (asserted separately),
# and reduceVertexFromFaces accumulates the true one-third area over the FULL
# incident-face set. Fails loudly if no closure vertex exists.
tessera_add_test(
    NAME     conforming_operators
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_conforming_operators.cpp
)
tessera_add_test(
    NAME     conforming_operators
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_conforming_operators.cpp
)

# DETERMINISM of the conforming closure (Task 7 of
# tasks/completed/conforming-refinement.md). Three senses: (a) rank-count independence,
# against a reference the same run recomputes on MPI_COMM_SELF and compares by
# quantised POSITION (new gids come from an MPI_Exscan, so nothing gid-keyed is
# comparable across rank counts) -- this is what pins the blue-diagonal
# tie-break, and it prints a per-component breakdown plus a direct count of blue
# parents whose diagonal differs; (b) closure idempotence -- an empty-mask
# refine on an already-closed mesh is un-close -> re-close and must reproduce
# the visible topology; (c) cross-mode equivalence under a uniform mask, where
# there are no kept faces so the closure must be inert and the two modes must
# agree on face gids, corner gids, and levels -- not merely on counts.
tessera_add_test(
    NAME     conforming_determinism
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_conforming_determinism.cpp
)
tessera_add_test(
    NAME     conforming_determinism
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_conforming_determinism.cpp
)

# Triangle SHAPE QUALITY over refinement depth (Task 7 of
# tasks/completed/conforming-refinement.md): the guarantee that justifies making the
# closure transient. SIXTEEN adaptive rounds over a shrinking geodesic cap,
# tracking global min angle, max radius ratio and closure-face fraction per
# round -- plus each of the four closure patterns separately and the red layer
# that feeds them -- each asserted against a FIXED bound, and the worst radius
# ratio additionally required to have STOPPED GROWING over the final rounds.
# Fixed is the point: a persistent closure would degrade without bound.
#
# TIER regression, ranks 1-5, both backends (promoted in Task 8 sub-task D7 with
# explicit user confirmation, per CLAUDE.md). The bounds were provisional until
# D7 measured them at 16 rounds; the round count is 16 because 8 rounds could
# not distinguish a saturating radius ratio from a growing one. Stability
# evidence for the promotion: 320 printed round lines over np1-5 x {SERIAL, HIP}
# x {Serial, Default} reduce to exactly 16 distinct lines, i.e. every measured
# quantity is byte-identical at every rank count on both backends; 11-16 s per
# instance.
tessera_add_test(
    NAME     conforming_quality
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_conforming_quality.cpp
)
tessera_add_test(
    NAME     conforming_quality
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_conforming_quality.cpp
)

# markByQuality() driving CONFORMING refinement (Task 6 of
# tasks/completed/conforming-refinement.md): the conforming counterpart of the two
# markquality tests above. Pins the criteria COMPOSED WITH the closure -- the
# visible-to-red mask translation, and the fact that CurvatureCriterion's edge
# coordinator now sees exactly two incident faces where a hanging-node mesh
# leaves a T-junction it silently skips. A HangingNode2to1 control run with the
# same criterion must fail conformity, or the case is vacuous.
tessera_add_test(
    NAME     markquality_conforming
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_markquality_conforming.cpp
)
tessera_add_test(
    NAME     markquality_conforming
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_markquality_conforming.cpp
)

# refine() rebuilds the halo itself, so the NAIVE sequence works (follow-up 1 of
# tasks/completed/halo-rebuild-split-edge-design.md). refine() used to drop every ghost and
# CLEAR the three halo plans, so haloExchange() silently no-opped on an empty
# plan and a second refine() threw std::out_of_range from inside Phase 3a's
# endpoint lookup, on a non-zero rank -- the midpoint owner is the lowest
# incident refining-face owner, which can hold one endpoint only as a ghost. The
# workaround was an identity migrate(), an idiom that reads like a no-op and that
# every multi-round driver had to know. This test therefore contains NO migrate()
# and NO rebuildHalo() call: four refine() rounds back-to-back with nothing in
# between, in both refinement modes, checking the ownership partition, the owned
# 1-ring, conformity/Euler, the 2:1 balance and cross-rank midpoint agreement
# after each. NON-VACUITY at ranks >= 2: the plans must be non-empty and a ghost
# corrupt-resync must restore the owners' positions -- every structural check
# above passes on the old EMPTY plan, so those two are what detect a regression.
tessera_add_test(
    NAME     refine_rehalo
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_refine_rehalo.cpp
)
tessera_add_test(
    NAME     refine_rehalo
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_refine_rehalo.cpp
)

# Configurable halo depth (tasks/completed/halo-depth.md). The ghost layer used to be
# 1-deep and not configurable, so buildVertexStencil(mesh, 2) was SILENTLY SHORT
# within one hop of a partition boundary -- a short CSR row looks exactly like a
# correct one, and the resulting operator error moves when the rank count
# changes. This test pins the fix end to end: a depth-2 distribute() gives every
# owned vertex its complete 2-ring measured against the global reference
# assembled from every rank's owned edges; depth 1 still reproduces
# distribute()'s own closure exactly (the depth loop lives in the
# finishHaloAndAssemble() that migrate() and refine() share, so that agreement is
# what protects the rest of the gate); ownership is depth-invariant;
# refine() x4, migrate() and loadBalance() all PRESERVE the depth; and
# buildVertexStencil(mesh, 2) throws on a depth-1 mesh instead of quietly
# returning short rows. Non-vacuity guards as in refine_rehalo.
tessera_add_test(
    NAME     halo_depth
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_halo_depth.cpp
)
tessera_add_test(
    NAME     halo_depth
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_halo_depth.cpp
)

# Caller-driven edge split (tasks/completed/edge-split.md) -- the first operation of the
# REMESH editing family, and the one refine() could not express: refine() takes
# a FACE mask and bisects all three edges of every marked face, so "bisect these
# particular edges" was inexpressible. Every count assertion is a closed-form
# identity checked through the global owned-count reductions, so the test checks
# arithmetic rather than checking Tessera against itself: one edge -> V+1/E+3/F+2,
# one face's three edges -> V+3/E+9/F+6 (Euler delta 0 in both), and ALL edges ->
# exactly a uniform refine(), asserted not just on V/E/F but on the vertex
# position and face corner-position multisets BITWISE. Plus two-edge diagonal
# determinism against an MPI_COMM_SELF reference (with all three |S| patterns
# asserted non-vacuous), midpoint agreement and coverage, exact off-sphere
# midpoints, linear user-field transfer, five length-driven rounds with a
# measured radius-ratio floor, the empty-mask identity, halo validity on return,
# and the editing-family guard in both directions.
#
# TIER regression, ranks 1-5, both backends: gate promotion is pre-authorized
# for this task by tasks/completed/edge-split.md.
tessera_add_test(
    NAME     split_edges
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_split_edges.cpp
)
tessera_add_test(
    NAME     split_edges
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_split_edges.cpp
)

# Caller-driven EDGE FLIP (tasks/completed/edge-flip.md) -- the second remesh-family
# operation. A flip changes connectivity only: V, E and F are unchanged and every
# gid is preserved, so nearly every assertion is an IDENTITY rather than a
# comparison against a hardcoded number.
#
# The two that make the operation trustworthy are INVOLUTION -- flip an edge,
# flip the same edge GID back, and the multiset of face corner-GID triples and
# the edge key set must return EXACTLY to the original, which pins the winding,
# the corner assignment and which of the two old face gids lands on which new
# face -- and RANK-COUNT INVARIANCE of the accepted set against an
# MPI_COMM_SELF reference the same run computes, which is what the gid-free
# independent-set priority (squared length descending, EdgeKey ascending) buys.
#
# Also: duplicate-edge rejection on a hand-built SUBDIVIDED TETRAHEDRON, whose
# four original corners keep valence 3 so that flipping one of their edges would
# create an edge that already exists (an icosphere has no such vertex); the
# independent set (no face rewritten twice); a policy nothing can satisfy
# rejecting everything and moving nothing; the valence-equalization use case
# written out caller-side; the edge key set moving by exactly the accepted flips
# while the gid set does not move at all; halo validity on return; the empty-mask
# identity; the editing-family guard both ways plus flip-after-split; and five
# rounds of split composed with valence flips against a measured radius-ratio
# floor.
#
# TIER regression, ranks 1-5, both backends: gate promotion is pre-authorized
# for this task by tasks/completed/edge-flip.md.
tessera_add_test(
    NAME     flip_edges
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_flip_edges.cpp
)
tessera_add_test(
    NAME     flip_edges
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_flip_edges.cpp
)

# Mesh compaction (tasks/completed/mesh-compaction.md) -- tombstone removal, and the
# periodic gid renumbering that closes the gid-space leak. The prerequisite of
# edge collapse: without it a coarsening operation has nowhere to put its dead
# entities, and would grow its own private half-compaction.
#
# The fixture is CLOSED SUB-MESH REMOVAL against a replicated MPI_COMM_SELF
# reference mesh -- remove one base-icosahedron patch (face gids [16i, 16i+16)
# at subdivision 2) plus exactly the vertices and edges no surviving face
# references -- so every expected count, gid set and position is computed rather
# than hardcoded, and the result is a surface WITH BOUNDARY whose Euler number
# is 2 minus its boundary-component count. Checks: the no-op and its
# idempotence; gid PRESERVATION through a removal (against a pre-edit gid ->
# position snapshot) and gid RENUMBERING to a contiguous [0, N) with the
# rank-count determinism the order-statistic definition buys, asserted against
# an MPI_COMM_SELF reference; the halo, ownership partition and owned 1-ring
# after a removal; a rank emptied to zero owned AND zero local entities that no
# peer names in any plan; the collective throw on a non-closing tombstone set;
# the editing-family guard both ways; and ten remove-and-compact rounds showing
# the gid space pinned while the mesh shrinks, then returned to exactly the live
# count by one compactAndRenumberGids().
#
# TIER regression, ranks 1-5, both backends: gate promotion is pre-authorized
# for this task by tasks/completed/mesh-compaction.md.
tessera_add_test(
    NAME     compact
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_compact.cpp
)
tessera_add_test(
    NAME     compact
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_compact.cpp
)

# Caller-driven EDGE COLLAPSE (tasks/completed/edge-collapse.md) -- the FOURTH and last
# remesh-family operation, and the only one that removes degrees of freedom:
# each accepted collapse merges an edge's two endpoints for V-1, E-3, F-2, so
# the Euler number is preserved. Without it a remesher is monotone and a long
# run dies on memory or on timestep.
#
# It is also the only one of the four that can silently corrupt a mesh in three
# distinct ways, so each is detected and reported through a NAMED COUNTER: the
# LINK CONDITION (link(a) INTERSECT link(b) == {c,d}, answered from
# vertex-coordinator advertisements rather than local residency, since the edge's
# owner may own neither endpoint) tested both ways -- three edges of one face
# accepting at most one, and a hand-built soup whose two endpoints share a third
# neighbour giving rejectedLinkCondition == 1; the NORMAL-FLIP guard on a closed
# decagonal bipyramid with one apex pulled into a spike, with every surviving
# face's normal checked against its own pre-collapse normal by face gid; and the
# CONNECTIVITY REWRITE, whose failure mode is a tombstone set that does not
# close -- caught by compact()'s global closure check and by asserting that no
# tombstone survives after every single call.
#
# Also: the exact one-edge deltas with the surviving vertex at min(a,b) and at
# the midpoint; the DEPTH GUARD (throws iff 0 < haloDepth() < 2 -- depth 0 means
# "never distributed", which is what lets the soup fixtures run); the
# independent set (no two accepted collapses share a vertex) and its RANK-COUNT
# INVARIANCE against an MPI_COMM_SELF reference the same run computes, which is
# what the gid-free priority (squared length ASCENDING, EdgeKey ascending) buys;
# a policy nothing can satisfy rejecting everything and leaving even the local
# gid ORDERING unmoved; the t-aware user-field blend on a double and a
# double[3]; halo validity, depth preservation and a second call with nothing in
# between; the empty-mask identity; the family guard in BOTH refinement modes;
# twenty rounds of decimation to the quality floor; a split-then-collapse round
# trip landing within a factor of two of the original face count; and TWENTY
# COMPOSED { split, flip, collapse } ROUNDS -- the test that proves the four
# remesh operations compose, which is the deliverable of the family.
#
# TIER regression, ranks 1-5, both backends: gate promotion is pre-authorized
# for this task by tasks/completed/edge-collapse.md.
tessera_add_test(
    NAME     collapse_edges
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_collapse_edges.cpp
)
tessera_add_test(
    NAME     collapse_edges
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_collapse_edges.cpp
)

# splitEdges() SHAPE QUALITY OVER SPLIT DEPTH (tasks/completed/edge-split.md, risk R12).
# The DIAGNOSTIC behind test_split_edges case 8's floor. Case 8 runs five
# length-driven rounds; five are consistent with a bound but do not establish
# one -- test_conforming_quality records that eight rounds could not distinguish
# saturation from a maximum being discovered slowly and sixteen could, and a
# downstream remesher refines far more than five times. Nor can the answer be
# inherited from test_conforming_quality: THAT bound rests on the closure being
# transient (every visible face is one of finitely many retriangulations of a red
# face), and splitEdges() has no such reset -- its children persist and can be
# cut again, so whether shape degrades is a property of the MASK.
#
# Four mask families bracket a consumer's behaviour: above-mean length (case 8's
# rule, a coarse relative of Rivara longest-edge bisection and self-correcting),
# below-mean length (the anti-Rivara rule -- cut the short edges of a stretched
# triangle), a length-blind hash of the midpoint POSITION (rank-count invariant,
# unlike a gid hash), and that hash restricted to a fixed geodesic cap so growth
# is bounded and the drive can reach 25+ rounds. Per round it prints the global
# min r/R, the min angle, and the POPULATION below five r/R thresholds -- the
# tail being the diagnostic a minimum cannot give.
#
# TIER unit: it is run to whatever depth the face budget allows and its output is
# a table to read, not a bound to assert. The bound it establishes is asserted in
# test_split_edges case 8. Knobs: TESSERA_SPLIT_DEPTH_ROUNDS,
# TESSERA_SPLIT_DEPTH_FACES.
tessera_add_test(
    NAME     split_edges_depth
    BACKEND  SERIAL
    TIER     unit
    RANKS    "1;4"
    SOURCES  test_split_edges_depth.cpp
)
tessera_add_test(
    NAME     split_edges_depth
    BACKEND  HIP
    TIER     unit
    RANKS    "1;4"
    SOURCES  test_split_edges_depth.cpp
)

# Lat/lon (UV) sphere generator (tasks/completed/latlon-sphere.md). generateIcosphere()
# was Tessera's only generator, and an icosphere is TOO GOOD a test surface:
# nearly isotropic, nearly equilateral, almost every vertex of valence 6. A
# lat/lon sphere is anisotropic by construction and its two poles have valence
# nLon, so it reaches code paths the icosphere never does. Half the test is the
# generator -- the closed-form V/E/F and Euler at five (nLat,nLon) pairs
# including the nLat==3 bipyramid and a prime nLon, plus the four classic
# UV-sphere bugs one check each: manifoldness (the seam-duplicate detector),
# no-duplicate-vertices (the duplicate-pole detector), bitwise-exact poles, and
# outward winding backed by the enclosed volume matched to a CLOSED-FORM
# polyhedron volume rather than a percentage bound. The other half is the rest
# of the library on that input: uniform refine(), migrate(), loadBalance(),
# haloExchange() with ghost positions verified against their owners through a
# gid coordinator, an HDF5 round trip, and markByQuality() on a deliberately
# 7.8:1-stretched (33,8) surface where the marked band is asserted against the
# closed-form latitude cut points.
#
# TIER regression, ranks 1-5, both backends: gate promotion is pre-authorized
# for this task by tasks/completed/latlon-sphere.md.
tessera_add_test(
    NAME     latlon_sphere
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_latlon_sphere.cpp
)
tessera_add_test(
    NAME     latlon_sphere
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_latlon_sphere.cpp
)

# Ghost scatter-add -- the reverse halo (tasks/completed/halo-scatter-add.md). haloExchange()
# is a pure gather (owner -> ghost, overwrite); haloScatterAdd() is the missing
# reverse (ghost -> owner, +=) without which any per-vertex quantity assembled by
# iterating OWNED faces -- vertex areas, vertex normals, a per-element residual --
# leaves the owner of a boundary vertex holding a PARTIAL sum. Ground truth for
# every value check comes from the REPLICATED reference mesh plus distribute()'s
# own faceOwner input, re-deriving ownership and each rank's local set
# independently, so a self-consistent wrong answer cannot pass; the reference's
# predicted per-rank local counts are themselves asserted against the real mesh.
# Covers the exact integer ghost multiplicity, the face-loop assembly (the case
# that fails with a plain haloExchange), a double[3] field componentwise, the
# three contract properties (ghosts untouched, non-idempotence pinned to a
# computed double-counted value, bitwise determinism within a run), the
# gather/scatter round trip that catches a swapped send_idx/recv_idx, the
# kind-named edge and face wrappers, the empty single-rank plan, and depth 2.
#
# TIER regression, ranks 1-5, both backends: gate promotion is pre-authorized
# for this task by tasks/completed/halo-scatter-add.md.
tessera_add_test(
    NAME     halo_scatter_add
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_halo_scatter_add.cpp
)
tessera_add_test(
    NAME     halo_scatter_add
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_halo_scatter_add.cpp
)

# Distributed initial mesh construction (tasks/completed/distributed-coarse-build.md). The
# initial mesh used to be built in FULL ON EVERY RANK before it was cut --
# buildFromTriangleSoup() is serial and takes a REPLICATED soup, and distribute()
# can compute ownership locally only *because* the mesh is replicated -- so peak
# memory per rank scaled with the GLOBAL mesh size, on every rank at once.
# buildFromTriangleSoupDistributed() takes per-rank local patches plus a canonical
# key per local vertex; buildIcosphereDistributed() partitions by the SUBDIVISION
# TREE (an axis sort would need every centroid, hence the global mesh).
#
# The definitive check is BITWISE EQUIVALENCE with the replicated path at
# subdivisions 1-3 -- compared by the gid-independent identity (the vertex
# position multiset and the face corner-position triple multiset), since gid
# numbering legitimately differs between two partitions. Because the replicated
# reference is partition-free, that same assertion holding at np1-5 IS rank-count
# reproducibility. The other headline check is that NOBODY HOLDS THE GLOBAL MESH:
# at subdivision 5 every rank's local counts are strictly below the global ones and
# are printed, since that measurement is the deliverable. Plus counts/invariants,
# halo correctness at depth 1 AND 2 against a global-adjacency k-ring reference,
# refine()/migrate()/loadBalance()/HDF5 on the result, a hand-built octahedron in
# two OVERLAPPING patches (which the icosphere's disjoint ranges never reach), a
# canonical-key collision that must throw naming the key rather than silently
# welding two vertices, and the degenerate case where a rank owns zero faces.
#
# TIER regression, ranks 1-5, both backends: gate promotion is pre-authorized
# for this task by tasks/completed/distributed-coarse-build.md.
tessera_add_test(
    NAME     distributed_build
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_distributed_build.cpp
)
tessera_add_test(
    NAME     distributed_build
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_distributed_build.cpp
)

# Face->face adjacency through shared edges (tasks/completed/face-adjacency.md). "Which
# faces share an edge with this face" is not derivable locally: EdgeField::Faces
# holds GIDS that migrate() carries verbatim and that may name faces this rank
# does not hold, and a vertex-incidence walk only finds a co-resident neighbour,
# which a 1-deep VERTEX halo does not guarantee. buildFaceAdjacency() reuses
# refine()'s edge coordinator and returns two halves -- a local-index CSR for a
# GEOMETRIC consumer (valid only when numNonResident == 0) and always-valid
# nbrGid/nbrOwner arrays for a TOPOLOGICAL one.
#
# Ground truth for the headline case is derived from the REPLICATED icosphere
# soup with a plain std::map<EdgeKey, vector<gid>> -- not from Tessera -- so a
# self-consistent wrong answer cannot pass. Covers exact ordered row equality
# against that reference, degree 3 and the closed-surface identity
# (3F == 2E == 960 at subdiv 2), symmetry, rank-count invariance (rows are
# gid-sorted, so the comparison is an exact list comparison), numNonResident
# being exact in BOTH directions, the Conforming post-refine case including that
# no retired red parent appears anywhere, the HangingNode2to1 case where a
# T-junction is deliberately NOT a shared edge, a hand-built non-manifold soup
# that must throw naming the EdgeKey, and the generation guard.
#
# TIER regression, ranks 1-5, both backends: gate promotion is pre-authorized
# for this task by tasks/completed/face-adjacency.md.
tessera_add_test(
    NAME     face_adjacency
    BACKEND  SERIAL
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_face_adjacency.cpp
)
tessera_add_test(
    NAME     face_adjacency
    BACKEND  HIP
    TIER     regression
    RANKS    ${TESSERA_TEST_MPI_RANKS}
    SOURCES  test_face_adjacency.cpp
)
