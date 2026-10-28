/****************************************************************************
 * Copyright (c) 2024, JStewart28                                           *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Tessera library. Tessera is distributed under a *
 * BSD 3-Clause license. For the licensing terms see the LICENSE file in   *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

// Regression test: distributed split-edge discovery (conforming refinement,
// Task 3 of tasks/completed/conforming-refinement.md).
//
// refine()'s Phase 2 now advertises the edges of EVERY owned face — refining and
// kept — so that RefineResult::midpoints is a complete SPLIT-EDGE MAP: for each
// rank, the (EdgeKey -> midpoint gid) pairs of every edge of any of its owned
// faces that some incident face bisected. That completeness is what lets the
// conforming closure (Task 4) close a kept face whose neighbour across a
// partition boundary refined. This test pins it:
//
//   (a) ROUND-1 EXACTNESS vs a partition-free reference. On a level-0 subdiv-2
//       icosphere the 2:1 fixpoint cannot propagate (all levels equal, so no
//       final-level gap can reach 2), hence the refining set is exactly the
//       caller's mask and the split-edge set is exactly the edges of the marked
//       faces — computable from a replicated, un-partitioned mesh with no MPI
//       and no partition information. Each rank's key set must equal that
//       reference restricted to the edges of its own pre-refine owned faces, at
//       every rank count.
//   (b) MIDPOINT GID BLOCK. The globally distinct midpoint gids must be exactly
//       the contiguous block [V0, V0 + |S|) above the pre-refine global vertex
//       count — checked by count, sum, and XOR against the closed forms, so both
//       a duplicate and a gap fail.
//   (c) NON-VACUITY. At ranks >= 2 some rank must hold a split edge that belongs
//       to NONE of its refining owned faces — i.e. an edge it learned about only
//       because a neighbour rank refined across the boundary. That is precisely
//       the case the pre-Task-3 code missed; if the count is zero the test would
//       prove nothing, so it fails loudly.
//   (d) MULTI-ROUND SOUNDNESS + COMPLETENESS. Over three further adaptive rounds
//       (where levels DO differ and the fixpoint really propagates, so no cheap
//       serial reference exists) the map is verified against a globally-decided
//       ground truth: an edge is split iff SOME rank reports it, and then every
//       rank touching that edge must report it, with no rank reporting an edge
//       it does not touch. Plus checkMidpointAgreement and check21Balance.
//
// Also prints the Phase-2a message volume (total vs the pre-Task-3 refining-only
// subset) for the Task-8 measurement table.
//
// Runs on host (Serial) and device (default, HIP), ranks 1-5.

#include "MeshInvariants.hpp"

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

using namespace Tessera;

//! Corner vertex gids and gid of every OWNED face (host snapshot).
template <class MeshT>
void ownedFaceSnapshot( MeshT& mesh,
                        std::vector<std::array<GlobalId, 3>>& verts,
                        std::vector<GlobalId>& gids )
{
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto g = Cabana::slice<FaceField::Gid>( hf );
    auto v = Cabana::slice<FaceField::Verts>( hf );
    const std::size_t n = mesh.numOwnedFaces();
    verts.resize( n );
    gids.resize( n );
    for ( std::size_t f = 0; f < n; ++f )
    {
        for ( int k = 0; k < 3; ++k )
            verts[f][k] = v( f, k );
        gids[f] = g( f );
    }
}

// edgeSetOf() and the globally-decided ground truth checkSplitEdgeCoverage()
// now live in MeshInvariants.hpp, shared with test_split_edges (the
// edge-addressed splitEdges() publishes a split-edge map with the identical
// contract, so it wants the identical check).
using TesseraTest::checkSplitEdgeCoverage;
using TesseraTest::edgeSetOf;

template <class Exec>
int run( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    // Pinned to the hanging-node mode. Phase 2 is shared by both modes, and
    // this test isolates it: with no closure the VISIBLE faces ARE the red
    // faces, so "the edges of my owned faces" -- the reference set (a), the
    // ground truth (d), and RefineResult::midpoints' own contract -- are one
    // and the same set. In Conforming mode the map is keyed by the RED layer
    // while mesh.faces() shows the closure, and the reference would have to
    // un-close first; that composition is what refine_conforming covers.
    using MeshT = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                       mem, Exec, RefinementMode::HangingNode2to1>;

    int fails = 0;

    // Coarse subdiv-2 icosphere entity counts (matches the Step-6b fixture).
    const long long V0 = 162, F0 = 320;

    // ---- partition-free reference for round 1 -------------------------------
    // Replicated (un-partitioned) mesh: gid == index at this stage, so the
    // marked set { f : f % 7 == 0 } and hence the split-edge set are computed
    // with no MPI and no partition information whatsoever. The 2:1 fixpoint
    // cannot fire here: every face is level 0, so |(lvl+mark) - (lvl+mark)| <= 1
    // on every edge and no face is ever force-marked.
    std::set<EdgeKey> refSplit;
    long long refRefiningFaces = 0;
    {
        MeshT ref( MPI_COMM_WORLD );
        buildIcosphere( ref, 2 );
        Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
            "hf", ref.numFaces() );
        Cabana::deep_copy( hf, ref.faces() );
        auto fv = Cabana::slice<FaceField::Verts>( hf );
        for ( std::size_t f = 0; f < ref.numFaces(); ++f )
        {
            if ( f % 7 != 0 )
                continue;
            ++refRefiningFaces;
            for ( int k = 0; k < 3; ++k )
                refSplit.insert(
                    makeEdgeKey( fv( f, k ), fv( f, ( k + 1 ) % 3 ) ) );
        }
        if ( static_cast<long long>( ref.numFaces() ) != F0 ||
             static_cast<long long>( ref.numVertices() ) != V0 )
            ++fails; // fixture drifted
    }
    const long long nSplitRef = static_cast<long long>( refSplit.size() );

    // ---- round 1: exact match against the reference -------------------------
    {
        MeshT mesh( MPI_COMM_WORLD );
        buildIcosphere( mesh, 2 );
        auto faceOwner = facePartitionByAxis( mesh );
        MeshHalo<mem> halo;
        distribute( mesh, halo, faceOwner );

        std::vector<std::array<GlobalId, 3>> preVerts;
        std::vector<GlobalId> preGids;
        ownedFaceSnapshot( mesh, preVerts, preGids );

        std::vector<char> mask( mesh.numOwnedFaces(), 0 );
        std::vector<std::array<GlobalId, 3>> refiningVerts;
        for ( std::size_t f = 0; f < mask.size(); ++f )
            if ( preGids[f] % 7 == 0 )
            {
                mask[f] = 1;
                refiningVerts.push_back( preVerts[f] );
            }

        auto res = refine( mesh, halo, mask );

        // (a) exactness: my key set == refSplit restricted to my faces' edges.
        std::set<EdgeKey> mine;
        for ( const auto& kv : res.midpoints )
            mine.insert( kv.first );
        std::set<EdgeKey> expect;
        for ( const EdgeKey& k : edgeSetOf( preVerts ) )
            if ( refSplit.count( k ) )
                expect.insert( k );
        int local = ( mine == expect ) ? 0 : 1;

        // (c) non-vacuity: split edges learned only via a kept face, i.e. not on
        //     any refining face I own. Zero at np1 by construction; must be
        //     positive once the mesh is actually partitioned.
        const std::set<EdgeKey> myRefiningEdges = edgeSetOf( refiningVerts );
        long long keptOnly = 0;
        for ( const EdgeKey& k : mine )
            if ( myRefiningEdges.find( k ) == myRefiningEdges.end() )
                ++keptOnly;

        // (b) the globally distinct midpoint gids form the block
        //     [V0, V0 + nSplitRef). Dedup at the edge coordinator, then compare
        //     count / sum / XOR against the closed forms.
        struct KG
        {
            EdgeKey key;
            GlobalId gid;
        };
        std::vector<std::vector<KG>> send( size );
        for ( const auto& kv : res.midpoints )
            send[Tessera::detail::edgeCoordRank( kv.first, size )].push_back(
                { kv.first, kv.second } );
        auto got = allToAllV( MPI_COMM_WORLD, send );
        std::set<EdgeKey> seen;
        long long myCount = 0, mySum = 0;
        unsigned long long myXor = 0;
        for ( const auto& m : got.data )
        {
            if ( !seen.insert( m.key ).second )
                continue;
            ++myCount;
            mySum += static_cast<long long>( m.gid );
            myXor ^= static_cast<unsigned long long>( m.gid );
        }
        long long redCount = 0, redSum = 0;
        unsigned long long redXor = 0;
        long long redKeptOnly = 0;
        MPI_Allreduce( &myCount, &redCount, 1, MPI_LONG_LONG, MPI_SUM,
                       MPI_COMM_WORLD );
        MPI_Allreduce( &mySum, &redSum, 1, MPI_LONG_LONG, MPI_SUM,
                       MPI_COMM_WORLD );
        MPI_Allreduce( &myXor, &redXor, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR,
                       MPI_COMM_WORLD );
        MPI_Allreduce( &keptOnly, &redKeptOnly, 1, MPI_LONG_LONG, MPI_SUM,
                       MPI_COMM_WORLD );

        long long expectSum = 0;
        unsigned long long expectXor = 0;
        for ( long long g = V0; g < V0 + nSplitRef; ++g )
        {
            expectSum += g;
            expectXor ^= static_cast<unsigned long long>( g );
        }

        local += TesseraTest::checkMidpointAgreement( MPI_COMM_WORLD, size,
                                                      res.midpoints ); // (d)
        local += TesseraTest::check21Balance( mesh );
        int glob = 0;
        MPI_Allreduce( &local, &glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

        if ( glob != 0 )
            ++fails;
        if ( redCount != nSplitRef || redSum != expectSum ||
             redXor != expectXor )
            ++fails; // midpoint gids are not the contiguous block above V0
        if ( size >= 2 && redKeptOnly <= 0 )
            ++fails; // vacuous: no kept-side discovery was exercised
        if ( res.iterations != 1 )
            ++fails; // level-0 mesh: the fixpoint cannot propagate

        // Phase-2a message volume, for the Task-8 measurement table.
        long long advTot = 0, advRef = 0;
        MPI_Allreduce( &res.phase2Adverts, &advTot, 1, MPI_LONG_LONG, MPI_SUM,
                       MPI_COMM_WORLD );
        MPI_Allreduce( &res.phase2AdvertsRefining, &advRef, 1, MPI_LONG_LONG,
                       MPI_SUM, MPI_COMM_WORLD );

        if ( rank == 0 )
        {
            std::printf( "  [%s] round1 %s (refining=%lld splitEdges=%lld "
                         "keptOnlyDiscovered=%lld phase2a: %lld total vs %lld "
                         "refining-only, x%.2f)\n",
                         tag, ( glob == 0 && fails == 0 ) ? "ok" : "FAIL",
                         refRefiningFaces, nSplitRef, redKeptOnly, advTot,
                         advRef,
                         advRef > 0 ? static_cast<double>( advTot ) /
                                          static_cast<double>( advRef )
                                    : 0.0 );
            std::fflush( stdout );
        }
    }

    // ---- rounds 2-4: globally-decided soundness + completeness --------------
    // Levels now differ across the mesh, so the 2:1 fixpoint really propagates
    // and there is no cheap serial reference; the coordinator supplies the
    // ground truth instead.
    {
        MeshT mesh( MPI_COMM_WORLD );
        buildIcosphere( mesh, 2 );
        auto faceOwner = facePartitionByAxis( mesh );
        MeshHalo<mem> halo;
        distribute( mesh, halo, faceOwner );

        for ( int round = 0; round < 4; ++round )
        {
            std::vector<std::array<GlobalId, 3>> preVerts;
            std::vector<GlobalId> preGids;
            ownedFaceSnapshot( mesh, preVerts, preGids );

            std::vector<char> mask( mesh.numOwnedFaces(), 0 );
            for ( std::size_t f = 0; f < mask.size(); ++f )
                mask[f] = ( preGids[f] % ( round == 0 ? 7 : 5 ) == 0 ) ? 1 : 0;

            auto res = refine( mesh, halo, mask );

            int local = checkSplitEdgeCoverage( MPI_COMM_WORLD, size, preVerts,
                                                res.midpoints );
            local += TesseraTest::checkMidpointAgreement( MPI_COMM_WORLD, size,
                                                          res.midpoints );
            local += TesseraTest::check21Balance( mesh );
            int glob = 0;
            MPI_Allreduce( &local, &glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
            if ( glob != 0 )
                ++fails;
            if ( res.iterations <= 0 || res.iterations >= 256 )
                ++fails;

            long long myMids = static_cast<long long>( res.midpoints.size() );
            long long advTot = 0, advRef = 0, midTot = 0;
            MPI_Allreduce( &res.phase2Adverts, &advTot, 1, MPI_LONG_LONG,
                           MPI_SUM, MPI_COMM_WORLD );
            MPI_Allreduce( &res.phase2AdvertsRefining, &advRef, 1,
                           MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD );
            MPI_Allreduce( &myMids, &midTot, 1, MPI_LONG_LONG, MPI_SUM,
                           MPI_COMM_WORLD );
            // Collective: every rank must call it, so it cannot live inside
            // the rank-0 print below.
            const long long gFaces = TesseraTest::globalOwnedFaces( mesh );

            if ( rank == 0 )
            {
                std::printf( "  [%s] round%d %s (it=%d faces=%lld "
                             "localMidsSum=%lld phase2a: %lld vs %lld, "
                             "x%.2f)\n",
                             tag, round + 1, glob == 0 ? "ok" : "FAIL",
                             res.iterations, gFaces, midTot, advTot, advRef,
                             advRef > 0 ? static_cast<double>( advTot ) /
                                              static_cast<double>( advRef )
                                        : 0.0 );
                std::fflush( stdout );
            }

            // Nothing between rounds: refine() rebuilds the 1-deep halo itself
            // (rebuildHalo()), so its own Phase 3a finds the POSITIONS of both
            // endpoints of every midpoint this rank owns even when one of them
            // is a ghost across a partition boundary. This used to need an
            // identity migrate() here; refine_rehalo is the test for that.
        }
    }

    return fails;
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int fails = 0;
    {
        int rank = 0, size = 1;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        MPI_Comm_size( MPI_COMM_WORLD, &size );
        if ( rank == 0 )
            std::printf( "test_refine_splitedges: distributed split-edge "
                         "discovery (size %d)\n",
                         size );
        std::fflush( stdout );

        fails += run<Kokkos::Serial>( rank, size, "Serial" );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            fails +=
                run<Kokkos::DefaultExecutionSpace>( rank, size, "Default" );
    }

    Kokkos::finalize();

    int global = 0;
    MPI_Allreduce( &fails, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    MPI_Finalize();
    return global == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
