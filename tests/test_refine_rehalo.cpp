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

// Regression test: refine() rebuilds the halo itself, so the NAIVE sequence
// works (follow-up 1 of tasks/completed/halo-rebuild-split-edge-design.md).
//
// This is the acceptance test for factoring the 1-deep halo rebuild out of
// migrate() into rebuildHalo(), which refine() now calls. Before that, refine()
// dropped every ghost and cleared the three halo plans, and nothing rebuilt
// them, so:
//
//   * haloExchange() on a freshly-refined mesh was a SILENT NO-OP on an empty
//     plan, not a synced ghost layer; and
//   * a second refine() THREW. refineImpl() needs the POSITIONS of both
//     endpoints of every midpoint the rank owns in order to interpolate it, and
//     midpoint ownership is "lowest incident refining-face owner", so a rank can
//     own the midpoint of an edge one of whose endpoints it holds only as a
//     ghost -- which the previous refine() had already dropped. The failure was
//     std::out_of_range from unordered_map::at, on a non-zero rank, from INSIDE
//     the next refine() rather than at the call the caller got wrong.
//
// The only way out was an idiom that reads like a no-op: an identity migrate()
// (dest[f] == rank) followed by haloExchange(), which worked only because
// migrate()'s rounds G/B/C/D ARE the halo rebuild. Every multi-round driver had
// to know it. This test asserts it is no longer needed, so it must contain NO
// migrate() call and no rebuildHalo() call -- that absence is the whole point.
//
// Cases, each in both refinement modes:
//
//   A. TWO ROUNDS BACK-TO-BACK. refine(); refine(); haloExchange(); with nothing
//      in between. The second refine() must not throw. Then:
//        checkOwnershipPartition   no duplicated / lost entity
//        owned1RingLocal           the rebuilt halo closes every owned 1-ring,
//                                  which is exactly what refine() alone did not
//                                  deliver before
//        checkConforming + Euler   (Conforming mode) the visible layer is sound
//        check21Balance/Red        the level balance survived
//        checkMidpointAgreement    midpoint gids still agree cross-rank
//      NON-VACUITY at ranks >= 2: the halo plans must be NON-EMPTY after
//      refine() alone. An empty plan is precisely the old silent no-op, and
//      every structural check above would still pass with one, so without this
//      assertion the case would not detect a regression to the old behaviour.
//
//   B. FOUR ROUNDS. Same, iterated, to confirm the property is not a one-shot
//      accident of the first round's topology: at round 3+ a Conforming mesh has
//      PERSISTENT split edges and closure children whose midpoint corner is
//      owned by the refining neighbour, which is the case round G of the rebuild
//      exists for. Also checks a haloExchange() after each round is a re-sync
//      and not a corruption: the ghost values already equal the owners', so the
//      topology checksum must be unchanged across it.
//
//   C. HALO IS MEANINGFUL. Corrupt every ghost vertex position on every rank,
//      haloExchange(), and require the owners' values to come back. On an empty
//      plan (the old behaviour) the corruption would survive, so this pins that
//      the plan refine() leaves behind actually carries data.
//
// Runs on host (Serial) and device (default, HIP), ranks 1-5.

#include "MeshInvariants.hpp"

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstdio>
#include <type_traits>
#include <vector>

using namespace Tessera;

//! Gid of every OWNED face (host snapshot).
template <class MeshT>
std::vector<GlobalId> ownedFaceGidList( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto g = Cabana::slice<FaceField::Gid>( hf );
    std::vector<GlobalId> out( mesh.numOwnedFaces() );
    for ( std::size_t f = 0; f < out.size(); ++f )
        out[f] = g( f );
    return out;
}

//! Adaptive mask: refine face gids divisible by `m`. Gid-derived, so it is the
//! same global face set at any rank count.
template <class MeshT>
std::vector<char> gidMask( MeshT& mesh, int m )
{
    const std::vector<GlobalId> g = ownedFaceGidList( mesh );
    std::vector<char> mask( g.size(), 0 );
    for ( std::size_t f = 0; f < g.size(); ++f )
        mask[f] = ( g[f] % static_cast<GlobalId>( m ) == 0 ) ? 1 : 0;
    return mask;
}

//! Total entries the three halo plans will actually send, summed globally. Zero
//! at size 1 (no peers); must be positive at size > 1 on a connected mesh, and
//! is zero on the CLEARED plans refine() used to leave behind.
template <class MeshT>
long long globalHaloPlanSize( MeshT& mesh,
                              MeshHalo<typename MeshT::memory_space>& halo )
{
    long long local = 0;
    for ( const auto* p : { &halo.vplan, &halo.eplan, &halo.fplan } )
        local += static_cast<long long>( p->totalSend() ) +
                 static_cast<long long>( p->totalRecv() );
    long long global = 0;
    MPI_Allreduce( &local, &global, 1, MPI_LONG_LONG, MPI_SUM, mesh.comm() );
    return global;
}

//! Every conforming/hanging-node invariant that holds on a refined mesh with a
//! valid halo. Returns LOCAL fails.
template <class MeshT>
int checkAll( MeshT& mesh, const RefineResult& res )
{
    int fails = 0;
    const long long NvG = TesseraTest::globalOwnedVertices( mesh );
    const long long NeG = TesseraTest::globalOwnedEdges( mesh );
    const long long NfG = TesseraTest::globalOwnedFaces( mesh );
    fails += TesseraTest::checkOwnershipPartition( mesh, NvG, NeG, NfG );
    fails += TesseraTest::owned1RingLocal( mesh );
    fails += TesseraTest::checkMidpointAgreement( mesh.comm(), mesh.commSize(),
                                                  res.midpoints );
    if constexpr ( MeshT::refinement_mode == RefinementMode::Conforming )
    {
        fails += TesseraTest::checkConforming( mesh );
        fails += TesseraTest::checkNoInteriorVertex( mesh );
        fails += TesseraTest::check21BalanceRed( mesh );
        if ( TesseraTest::ownedEulerGlobal( mesh ) != 2 )
            ++fails;
    }
    else
    {
        fails += TesseraTest::check21Balance( mesh );
    }
    return fails;
}

//! Overwrite every GHOST vertex position with a value no owner would produce.
//! A following haloExchange() must restore the owners' values.
template <class MeshT>
void corruptGhostPositions( MeshT& mesh )
{
    using Scalar = typename MeshT::scalar_type;
    constexpr int Dim = MeshT::dim;
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto pos = Cabana::slice<VertexField::Position>( hv );
    for ( std::size_t i = mesh.numOwnedVertices(); i < mesh.numVertices(); ++i )
        for ( int d = 0; d < Dim; ++d )
            pos( i, d ) = Scalar( -1234.5 );
    Cabana::deep_copy( mesh.vertices(), hv );
}

//! Ghost vertices whose position is still the corruption sentinel.
template <class MeshT>
long long countCorruptGhosts( MeshT& mesh )
{
    using Scalar = typename MeshT::scalar_type;
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto pos = Cabana::slice<VertexField::Position>( hv );
    long long bad = 0;
    for ( std::size_t i = mesh.numOwnedVertices(); i < mesh.numVertices(); ++i )
        if ( pos( i, 0 ) == Scalar( -1234.5 ) )
            ++bad;
    long long global = 0;
    MPI_Allreduce( &bad, &global, 1, MPI_LONG_LONG, MPI_SUM, mesh.comm() );
    return global;
}

//! Sum a per-rank local fail count into a global one.
inline int globalFails( MPI_Comm comm, int local )
{
    int g = 0;
    MPI_Allreduce( &local, &g, 1, MPI_INT, MPI_SUM, comm );
    return g;
}

// ===========================================================================
// The run: `rounds` successive refine() calls with NOTHING in between.
// ===========================================================================
template <class ExecSpace, RefinementMode Mode>
int run( int rank, int size, const char* tag, const char* modeTag, int rounds )
{
    using mem = typename ExecSpace::memory_space;
    using MeshT = Mesh<double, 3, Cabana::MemberTypes<>, Cabana::MemberTypes<>,
                       Cabana::MemberTypes<>, mem, ExecSpace, Mode>;
    MPI_Comm comm = MPI_COMM_WORLD;
    int fails = 0;

    MeshT mesh( comm );
    buildIcosphere( mesh, 2 );
    MeshHalo<mem> halo;
    {
        auto faceOwner = facePartitionByAxis( mesh );
        distribute( mesh, halo, faceOwner );
    }

    // THE POINT OF THIS TEST: no migrate(), no rebuildHalo(), nothing at all
    // between the refine() calls below. Before follow-up 1 the second one threw
    // std::out_of_range at size >= 2.
    const int masks[4] = { 7, 5, 3, 4 };
    for ( int r = 0; r < rounds; ++r )
    {
        auto res = refine( mesh, halo, gidMask( mesh, masks[r % 4] ) );

        fails += globalFails( comm, checkAll( mesh, res ) );

        // Non-vacuity: the plans refine() left must actually carry data at
        // size > 1. An EMPTY plan is the old silent no-op, and every check
        // above passes with one, so this is what detects a regression.
        const long long planSize = globalHaloPlanSize( mesh, halo );
        if ( size > 1 && planSize <= 0 )
            ++fails;

        // A haloExchange() straight after refine() is a re-sync: ghost values
        // already equal the owners', so nothing may change.
        unsigned long long cv0, ce0, cf0, cv1, ce1, cf1;
        TesseraTest::topologyChecksum( mesh, cv0, ce0, cf0 );
        haloExchange( mesh, halo );
        TesseraTest::topologyChecksum( mesh, cv1, ce1, cf1 );
        if ( cv0 != cv1 || ce0 != ce1 || cf0 != cf1 )
            ++fails;
        fails += globalFails( comm, checkAll( mesh, res ) );

        // Case C: the plan carries data, not just metadata.
        corruptGhostPositions( mesh );
        const long long corrupted = countCorruptGhosts( mesh );
        haloExchange( mesh, halo );
        const long long surviving = countCorruptGhosts( mesh );
        if ( surviving != 0 )
            ++fails; // haloExchange did not restore the owners' positions
        if ( size > 1 && corrupted <= 0 )
            ++fails; // vacuous: there were no ghost vertices to corrupt
        fails += globalFails( comm, checkAll( mesh, res ) );

        // Reduced on EVERY rank, then printed on rank 0 -- these are collective,
        // so evaluating them inside the rank-0 guard would deadlock.
        const long long gV = TesseraTest::globalOwnedVertices( mesh );
        const long long gE = TesseraTest::globalOwnedEdges( mesh );
        const long long gF = TesseraTest::globalOwnedFaces( mesh );
        if ( rank == 0 )
            printf( "[refine_rehalo] %-6s %-14s np%d round %d: V=%lld E=%lld "
                    "F=%lld plan=%lld ghostsCorrupted=%lld fails=%d\n",
                    tag, modeTag, size, r + 1, gV, gE, gF, planSize, corrupted,
                    fails );
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

        // Case A (2 rounds) is subsumed by case B (4 rounds): the per-round
        // assertions are identical and round 2 is checked on the way to round 4.
        fails += run<Kokkos::Serial, RefinementMode::Conforming>(
            rank, size, "Serial", "Conforming", 4 );
        fails += run<Kokkos::Serial, RefinementMode::HangingNode2to1>(
            rank, size, "Serial", "HangingNode2to1", 4 );
        if constexpr ( !std::is_same<Kokkos::DefaultExecutionSpace,
                                     Kokkos::Serial>::value )
        {
            fails +=
                run<Kokkos::DefaultExecutionSpace, RefinementMode::Conforming>(
                    rank, size, "Default", "Conforming", 4 );
            fails += run<Kokkos::DefaultExecutionSpace,
                         RefinementMode::HangingNode2to1>(
                rank, size, "Default", "HangingNode2to1", 4 );
        }

        if ( rank == 0 )
            printf( "[refine_rehalo] TOTAL fails=%d\n", fails );
    }
    Kokkos::finalize();
    MPI_Finalize();
    return fails == 0 ? 0 : 1;
}
