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

// Regression test: distributed CONFORMING refinement (Task 4 of
// tasks/completed/conforming-refinement.md). This is the milestone test for the feature.
//
// refine() on a RefinementMode::Conforming mesh un-closes the transient closure
// layer, runs the same 2:1-balanced red engine as the hanging-node mode, and
// re-closes every kept red face whose edges a neighbour bisected. The result is
// a mesh with NO hanging nodes under an ARBITRARY (adaptive) mask. Cases:
//
//   A. THREE SUCCESSIVE ADAPTIVE ROUNDS. After each round:
//        checkConforming        every global edge has exactly two incident faces
//        checkOwnedEuler == 2   the headline criterion -- this is what the
//                               hanging-node mode fails under an adaptive mask
//        checkNoInteriorVertex  geometric: no vertex strictly inside an edge
//        checkOwnershipPartition / checkMidpointAgreement  (unchanged contracts)
//        check21BalanceRed      the 2:1 invariant, applied to the RED layer --
//                               closure children carry their parent's level, so
//                               the balance is a property of the red faces
//        checkClosureInverse    unclose o close reproduces the visible layer and
//                               then the red layer bit-for-bit
//      Plus NON-VACUITY: closure children must actually be emitted at every rank
//      count, and the same three conformity checks are run on a HangingNode2to1
//      mesh with the SAME mask and must FAIL there -- otherwise a mask too weak
//      to create a hanging node would make the whole case prove nothing.
//
//   B. EMPTY MASK. Nothing splits, so nothing needs closing: the visible
//      topology is unchanged and |S| = 0 everywhere.
//
//   C. FULL (UNIFORM) MASK. Every face refines, so there are no kept faces and
//      the closure is inert -- |S| = 0 everywhere, no closure children, and the
//      global V/E/F counts must equal the hanging-node mode's exactly.
//
// Prints the closure-face fraction and the |S|-pattern histogram per round for
// the Task-8 measurement table.
//
// Runs on host (Serial) and device (default, HIP), ranks 1-5.

#include "MeshInvariants.hpp"

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <type_traits>
#include <vector>

using namespace Tessera;

//! Gid of every OWNED face (host snapshot).
template <class MeshT>
std::vector<GlobalId> ownedFaceGids( MeshT& mesh )
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

//! The deterministic adaptive mask used by every case: refine face gids
//! divisible by `m`. Non-uniform, crosses partition boundaries, and (unlike a
//! local-index predicate) rank-count independent.
template <class MeshT>
std::vector<char> gidMask( MeshT& mesh, int m )
{
    const std::vector<GlobalId> g = ownedFaceGids( mesh );
    std::vector<char> mask( g.size(), 0 );
    for ( std::size_t f = 0; f < g.size(); ++f )
        mask[f] = ( g[f] % m == 0 ) ? 1 : 0;
    return mask;
}

//! Globally summed |S| histogram + closure/visible counts for one refine call.
struct ClosureTotals
{
    long long pattern[4] = { 0, 0, 0, 0 };
    long long visible = 0;
    long long closureChildren = 0;
    //! Blue quads resolved to each diagonal by the GEOMETRIC tie-break, and
    //! those whose two split edges were exactly equal in length so the
    //! midpoint-gid fallback decided.
    long long blueQ0C = 0, blueQ1A = 0, blueTie = 0;
};

inline ClosureTotals reduceClosure( const ClosureStats& s )
{
    long long local[9] = {
        s.patternCount[0], s.patternCount[1], s.patternCount[2],
        s.patternCount[3], s.nVisible,        s.nClosureChildren,
        s.nBlueDiagQ0C,    s.nBlueDiagQ1A,    s.nBlueDiagTie };
    long long g[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    MPI_Allreduce( local, g, 9, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD );
    ClosureTotals t;
    for ( int i = 0; i < 4; ++i )
        t.pattern[i] = g[i];
    t.visible = g[4];
    t.closureChildren = g[5];
    t.blueQ0C = g[6];
    t.blueQ1A = g[7];
    t.blueTie = g[8];
    return t;
}

//! Every owned gid of one entity kind, gathered and sorted globally. Used to
//! report WHICH gids a checksum mismatch is about, not merely that there is one.
inline std::vector<GlobalId>
globalSortedGids( const std::vector<GlobalId>& mine, int size )
{
    int n = static_cast<int>( mine.size() );
    std::vector<int> counts( size, 0 );
    MPI_Allgather( &n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD );
    std::vector<int> displs( size, 0 );
    int total = 0;
    for ( int r = 0; r < size; ++r )
    {
        displs[r] = total;
        total += counts[r];
    }
    std::vector<GlobalId> all( total );
    MPI_Allgatherv( mine.data(), n, MPI_UINT64_T, all.data(), counts.data(),
                    displs.data(), MPI_UINT64_T, MPI_COMM_WORLD );
    std::sort( all.begin(), all.end() );
    return all;
}

//! Print the first few gids present in `b` but not `a` and vice versa.
inline void reportGidDelta( const char* tag, const char* kind,
                            const std::vector<GlobalId>& a,
                            const std::vector<GlobalId>& b )
{
    if ( a == b )
        return;
    auto diff = [&]( const std::vector<GlobalId>& x,
                     const std::vector<GlobalId>& y, const char* label )
    {
        std::vector<GlobalId> d;
        std::set_difference( x.begin(), x.end(), y.begin(), y.end(),
                             std::back_inserter( d ) );
        std::printf( "  [%s] %s %s (%zu):", tag, kind, label, d.size() );
        for ( std::size_t i = 0; i < d.size() && i < 12; ++i )
            std::printf( " %llu", static_cast<unsigned long long>( d[i] ) );
        std::printf( "\n" );
    };
    diff( a, b, "lost" );
    diff( b, a, "gained" );
    std::fflush( stdout );
}

//! The three mode-agnostic conformity criteria, as one global figure each.
struct Conformity
{
    long long badIncidence = 0; //!< edges without exactly two incident faces
    long long euler = 0;
    long long interiorVerts = 0;
};

template <class MeshT>
Conformity measureConformity( MeshT& mesh )
{
    Conformity c;
    long long local[2] = {
        static_cast<long long>( TesseraTest::checkConforming( mesh ) ),
        static_cast<long long>( TesseraTest::checkNoInteriorVertex( mesh ) ) };
    long long g[2] = { 0, 0 };
    MPI_Allreduce( local, g, 2, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD );
    c.badIncidence = g[0];
    c.interiorVerts = g[1];
    c.euler = TesseraTest::checkOwnedEuler( mesh );
    return c;
}

//! Sum a LOCAL fail count into a global one.
inline int globalFails( int local )
{
    int g = 0;
    MPI_Allreduce( &local, &g, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    return g;
}

// ===========================================================================
// Case A -- three successive adaptive rounds
// ===========================================================================

template <class Exec, class ConfMesh, class HangMesh>
int case_adaptive( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    int fails = 0;

    ConfMesh mesh( MPI_COMM_WORLD );
    buildIcosphere( mesh, 2 );
    MeshHalo<mem> halo;
    {
        auto faceOwner = facePartitionByAxis( mesh );
        distribute( mesh, halo, faceOwner );
    }

    // The same mask sequence on a HangingNode2to1 mesh, as the non-vacuity
    // control: the conformity checks must FAIL there.
    HangMesh ctrl( MPI_COMM_WORLD );
    buildIcosphere( ctrl, 2 );
    MeshHalo<mem> ctrlHalo;
    {
        auto faceOwner = facePartitionByAxis( ctrl );
        distribute( ctrl, ctrlHalo, faceOwner );
    }

    long long totalClosure = 0;
    long long ctrlBad = 0;

    for ( int round = 0; round < 3; ++round )
    {
        const int m = ( round == 0 ) ? 7 : 5;

        auto res = refine( mesh, halo, gidMask( mesh, m ) );
        auto ctrlRes = refine( ctrl, ctrlHalo, gidMask( ctrl, m ) );
        (void)ctrlRes;

        int local = 0;
        local += TesseraTest::checkConforming( mesh );
        local += TesseraTest::checkNoInteriorVertex( mesh );
        local += TesseraTest::checkMidpointAgreement( MPI_COMM_WORLD, size,
                                                      res.midpoints );
        local += TesseraTest::check21BalanceRed( mesh );
        local += TesseraTest::checkClosureInverse( mesh, res.midpoints );
        local += TesseraTest::checkOwnershipPartition(
            mesh, TesseraTest::globalOwnedVertices( mesh ),
            TesseraTest::globalOwnedEdges( mesh ),
            TesseraTest::globalOwnedFaces( mesh ) );
        const int glob = globalFails( local );
        if ( glob != 0 )
            ++fails;

        const long long euler = TesseraTest::checkOwnedEuler( mesh );
        if ( euler != 2 )
            ++fails; // THE criterion: conforming under an adaptive mask
        if ( res.iterations <= 0 || res.iterations >= 256 )
            ++fails;

        const ClosureTotals ct = reduceClosure( res.closure );
        totalClosure += ct.closureChildren;
        if ( ct.visible != TesseraTest::globalOwnedFaces( mesh ) )
            ++fails; // the visible count refine() reported is the mesh's

        // Non-vacuity control: the same mask on a hanging-node mesh must leave
        // T-junctions, or the conforming result above proves nothing.
        const Conformity cc = measureConformity( ctrl );
        ctrlBad += cc.badIncidence + cc.interiorVerts;

        if ( rank == 0 )
        {
            const double frac =
                ct.visible > 0 ? static_cast<double>( ct.closureChildren ) /
                                     static_cast<double>( ct.visible )
                               : 0.0;
            std::printf(
                "  [%s] adaptive round%d %s (it=%d F=%lld euler=%lld "
                "closure=%lld/%lld=%.3f |S| hist=[%lld,%lld,%lld,%lld]"
                " blue q0C/q1A/tie=%lld/%lld/%lld | control "
                "euler=%lld badInc=%lld tjunc=%lld)\n",
                tag, round + 1, ( glob == 0 && euler == 2 ) ? "ok" : "FAIL",
                res.iterations, ct.visible, euler, ct.closureChildren,
                ct.visible, frac, ct.pattern[0], ct.pattern[1], ct.pattern[2],
                ct.pattern[3], ct.blueQ0C, ct.blueQ1A, ct.blueTie, cc.euler,
                cc.badIncidence, cc.interiorVerts );
        }

        // refine() drops every ghost and clears the halo plans, but its own
        // Nothing between rounds: refine() rebuilds the 1-deep halo itself, so
        // its Phase 3a finds the positions of BOTH endpoints of every midpoint
        // the rank owns even when one is a ghost across a partition boundary.
        // This used to need an identity migrate() on each mesh here.
    }

    if ( totalClosure <= 0 )
        ++fails; // vacuous: the closure never fired
    if ( ctrlBad <= 0 )
        ++fails; // vacuous: the mask never created a hanging node at all

    return fails;
}

// ===========================================================================
// Case B -- empty mask: closure is the identity
// ===========================================================================

template <class Exec, class ConfMesh>
int case_empty( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    int fails = 0;

    ConfMesh mesh( MPI_COMM_WORLD );
    buildIcosphere( mesh, 2 );
    MeshHalo<mem> halo;
    {
        auto faceOwner = facePartitionByAxis( mesh );
        distribute( mesh, halo, faceOwner );
    }

    const long long V0 = TesseraTest::globalOwnedVertices( mesh );
    const long long E0 = TesseraTest::globalOwnedEdges( mesh );
    const long long F0 = TesseraTest::globalOwnedFaces( mesh );
    unsigned long long cv0, ce0, cf0;
    TesseraTest::topologyChecksum( mesh, cv0, ce0, cf0 );
    const std::vector<GlobalId> gv0 = globalSortedGids(
        TesseraTest::ownedGids( mesh.vertices(), 0, mesh.numOwnedVertices() ),
        size );
    const std::vector<GlobalId> ge0 = globalSortedGids(
        TesseraTest::ownedGids( mesh.edges(), 0, mesh.numOwnedEdges() ), size );
    const std::vector<GlobalId> gf0 = globalSortedGids(
        TesseraTest::ownedGids( mesh.faces(), 0, mesh.numOwnedFaces() ), size );

    std::vector<char> mask( mesh.numOwnedFaces(), 0 );
    auto res = refine( mesh, halo, mask );

    // Every condition gets its own bit, OR-reduced over ranks, so a failure
    // names itself instead of only incrementing a count.
    enum Why
    {
        kConforming = 1 << 0,
        kInverse = 1 << 1,
        kCounts = 1 << 2,
        kChecksum = 1 << 3,
        kClosureInert = 1 << 4,
        kEuler = 1 << 5,
        kMidpoints = 1 << 6
    };
    int why = 0;
    if ( TesseraTest::checkConforming( mesh ) != 0 )
        why |= kConforming;
    if ( TesseraTest::checkClosureInverse( mesh, res.midpoints ) != 0 )
        why |= kInverse;

    const ClosureTotals ct = reduceClosure( res.closure );
    unsigned long long cv1, ce1, cf1;
    TesseraTest::topologyChecksum( mesh, cv1, ce1, cf1 );

    const long long V1 = TesseraTest::globalOwnedVertices( mesh );
    const long long E1 = TesseraTest::globalOwnedEdges( mesh );
    const long long F1 = TesseraTest::globalOwnedFaces( mesh );
    const long long euler = TesseraTest::checkOwnedEuler( mesh );

    if ( V1 != V0 || E1 != E0 || F1 != F0 )
        why |= kCounts; // an empty mask must not change any count
    const std::vector<GlobalId> gv1 = globalSortedGids(
        TesseraTest::ownedGids( mesh.vertices(), 0, mesh.numOwnedVertices() ),
        size );
    const std::vector<GlobalId> ge1 = globalSortedGids(
        TesseraTest::ownedGids( mesh.edges(), 0, mesh.numOwnedEdges() ), size );
    const std::vector<GlobalId> gf1 = globalSortedGids(
        TesseraTest::ownedGids( mesh.faces(), 0, mesh.numOwnedFaces() ), size );
    if ( cv1 != cv0 || ce1 != ce0 || cf1 != cf0 || gv1 != gv0 || ge1 != ge0 ||
         gf1 != gf0 )
        why |= kChecksum; // ... nor any gid
    if ( ct.closureChildren != 0 || ct.pattern[1] || ct.pattern[2] ||
         ct.pattern[3] )
        why |= kClosureInert; // nothing was split, so nothing may be closed
    if ( euler != 2 )
        why |= kEuler;
    if ( !res.midpoints.empty() )
        why |= kMidpoints; // no edge was bisected

    int gWhy = 0;
    MPI_Allreduce( &why, &gWhy, 1, MPI_INT, MPI_BOR, MPI_COMM_WORLD );
    if ( gWhy != 0 )
        ++fails;

    (void)size;
    if ( rank == 0 )
    {
        std::printf( "  [%s] empty-mask %s (V=%lld->%lld E=%lld->%lld "
                     "F=%lld->%lld euler=%lld closureChildren=%lld"
                     " |S| hist=[%lld,%lld,%lld,%lld])\n",
                     tag, gWhy == 0 ? "ok" : "FAIL", V0, V1, E0, E1, F0, F1,
                     euler, ct.closureChildren, ct.pattern[0], ct.pattern[1],
                     ct.pattern[2], ct.pattern[3] );
        if ( gWhy != 0 )
        {
            static const char* names[7] = {
                "conforming",      "closureInverse", "counts",   "checksum",
                "closureNotInert", "euler",          "midpoints" };
            std::printf( "  [%s] empty-mask failed:", tag );
            for ( int b = 0; b < 7; ++b )
                if ( gWhy & ( 1 << b ) )
                    std::printf( " %s", names[b] );
            std::printf( "\n" );
            reportGidDelta( tag, "vertex", gv0, gv1 );
            reportGidDelta( tag, "edge", ge0, ge1 );
            reportGidDelta( tag, "face", gf0, gf1 );
        }
        std::fflush( stdout );
    }
    return fails;
}

// ===========================================================================
// Case C -- full mask: no kept faces, so the closure must be inert
// ===========================================================================

template <class Exec, class ConfMesh, class HangMesh>
int case_uniform( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    int fails = 0;

    ConfMesh mesh( MPI_COMM_WORLD );
    buildIcosphere( mesh, 2 );
    MeshHalo<mem> halo;
    {
        auto faceOwner = facePartitionByAxis( mesh );
        distribute( mesh, halo, faceOwner );
    }
    HangMesh ref( MPI_COMM_WORLD );
    buildIcosphere( ref, 2 );
    MeshHalo<mem> refHalo;
    {
        auto faceOwner = facePartitionByAxis( ref );
        distribute( ref, refHalo, faceOwner );
    }

    std::vector<char> mask( mesh.numOwnedFaces(), 1 );
    auto res = refine( mesh, halo, mask );
    std::vector<char> refMask( ref.numOwnedFaces(), 1 );
    refine( ref, refHalo, refMask );

    int local = TesseraTest::checkConforming( mesh );
    local += TesseraTest::checkNoInteriorVertex( mesh );
    local += TesseraTest::checkMidpointAgreement( MPI_COMM_WORLD, size,
                                                  res.midpoints );
    local += TesseraTest::checkClosureInverse( mesh, res.midpoints );
    if ( globalFails( local ) != 0 )
        ++fails;

    const ClosureTotals ct = reduceClosure( res.closure );
    if ( ct.closureChildren != 0 || ct.pattern[1] || ct.pattern[2] ||
         ct.pattern[3] )
        ++fails; // no kept faces => |S| = 0 everywhere => closure is inert
    if ( TesseraTest::checkOwnedEuler( mesh ) != 2 )
        ++fails;

    const long long gV = TesseraTest::globalOwnedVertices( mesh );
    const long long gE = TesseraTest::globalOwnedEdges( mesh );
    const long long gF = TesseraTest::globalOwnedFaces( mesh );
    if ( gV != TesseraTest::globalOwnedVertices( ref ) ||
         gE != TesseraTest::globalOwnedEdges( ref ) ||
         gF != TesseraTest::globalOwnedFaces( ref ) )
        ++fails; // an inert closure must reproduce the hanging-node counts

    if ( rank == 0 )
        std::printf(
            "  [%s] uniform-mask %s (V=%lld E=%lld F=%lld closureChildren=%lld "
            "|S| hist=[%lld,%lld,%lld,%lld])\n",
            tag, fails == 0 ? "ok" : "FAIL", gV, gE, gF, ct.closureChildren,
            ct.pattern[0], ct.pattern[1], ct.pattern[2], ct.pattern[3] );
    return fails;
}

template <class Exec>
int run( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    using ConfMesh = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                          mem, Exec, RefinementMode::Conforming>;
    using HangMesh = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                          mem, Exec, RefinementMode::HangingNode2to1>;

    int fails = 0;
    fails += case_adaptive<Exec, ConfMesh, HangMesh>( rank, size, tag );
    fails += case_empty<Exec, ConfMesh>( rank, size, tag );
    fails += case_uniform<Exec, ConfMesh, HangMesh>( rank, size, tag );
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
            std::printf( "test_refine_conforming: distributed conforming "
                         "refinement (size %d)\n",
                         size );

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
