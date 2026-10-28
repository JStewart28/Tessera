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

// Regression test: migrate() / loadBalance() / halo rebuild on a CONFORMING
// mesh (Task 5 of tasks/completed/conforming-refinement.md).
//
// The hazard the closure introduces to redistribution is sibling splitting. A
// closure child names its retired red parent outright, so un-closing is local
// per child -- but if two ranks each end up holding a child of the SAME parent,
// each restores that parent and the red layer gains a duplicated face.
// migrate() therefore repairs `dest` so every sibling follows the lowest-gid
// sibling's destination. Cases:
//
//   A. ADVERSARIAL DEST. Refine adaptively, then migrate with the deliberately
//      sibling-splitting assignment dest = faceGid % size -- siblings have
//      distinct gids, so at size > 1 essentially every group is scattered and
//      the fixup must fire. After the move + haloExchange:
//        checkSiblingCoresidency  no parent's children on two ranks
//        checkOwnershipPartition  no duplicated / lost entity
//        owned1RingLocal          the rebuilt halo closes every owned 1-ring
//        checkConforming / Euler == 2 / checkNoInteriorVertex / 21BalanceRed
//        topology checksum        unchanged: migrate moves, it does not alter
//      Plus a ghost corrupt-resync over the FACE plan, which is what exercises
//      the two extra closure members of the wider conforming face tuple.
//      NON-VACUITY: at size > 1 the global fixup count must be positive, else
//      the sibling hazard was never actually created and the case proves
//      nothing.
//
//   B. LOAD BALANCE. Same pipeline, then deliberately dump every owned face on
//      rank 0 and call loadBalance(). Zoltan2 partitions by face CENTROID, so
//      it scatters siblings on its own -- the fixup fires here without any help
//      from the test. Asserts the balance improves, that the WEIGHTED load
//      (the quantity ownedFaceWeights() defines and Zoltan2 optimizes -- a
//      closure child counts 1/nsiblings so a red parent is one unit) lands
//      within tolerance, and that every conforming invariant above still holds.
//
// Prints the sibling-fixup counts and the pre/post imbalance figures for the
// Task-8 measurement table.
//
// Runs on host (Serial) and device (default, HIP), ranks 1-5.

#include "MeshInvariants.hpp"

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

using namespace Tessera;

//! Gid of every OWNED face (host snapshot).
template <class MeshT>
std::vector<GlobalId> ownedGidsOfFaces( MeshT& mesh )
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

//! The deterministic adaptive mask: refine face gids divisible by `m`. Not a
//! local-index predicate, so it is the same set of faces at any rank count.
template <class MeshT>
std::vector<char> gidMask( MeshT& mesh, int m )
{
    const std::vector<GlobalId> g = ownedGidsOfFaces( mesh );
    std::vector<char> mask( g.size(), 0 );
    for ( std::size_t f = 0; f < g.size(); ++f )
        mask[f] = ( g[f] % m == 0 ) ? 1 : 0;
    return mask;
}

inline long long commSum( long long v, MPI_Comm comm )
{
    long long g = 0;
    MPI_Allreduce( &v, &g, 1, MPI_LONG_LONG, MPI_SUM, comm );
    return g;
}
inline long long commMax( long long v, MPI_Comm comm )
{
    long long g = 0;
    MPI_Allreduce( &v, &g, 1, MPI_LONG_LONG, MPI_MAX, comm );
    return g;
}
inline double globalSumD( double v, MPI_Comm comm )
{
    double g = 0;
    MPI_Allreduce( &v, &g, 1, MPI_DOUBLE, MPI_SUM, comm );
    return g;
}
inline double globalMaxD( double v, MPI_Comm comm )
{
    double g = 0;
    MPI_Allreduce( &v, &g, 1, MPI_DOUBLE, MPI_MAX, comm );
    return g;
}

//! Corrupt every ghost entry of `a`, haloExchange, and verify the owner's
//! values came back -- over the WHOLE tuple, so on a conforming face AoSoA this
//! covers the two closure members the mode appends after the user pack.
template <class AoSoAType, class PlanT>
int corruptResyncTuples( MPI_Comm comm, AoSoAType& a, std::size_t n_owned,
                         PlanT& plan )
{
    using host_aosoa =
        Cabana::AoSoA<typename AoSoAType::member_types, Kokkos::HostSpace>;
    const std::size_t n = a.size();

    host_aosoa before( "before", n );
    Cabana::deep_copy( before, a );

    // Overwrite the ghost region byte-wise with a value no gid/owner can take.
    host_aosoa corrupt( "corrupt", n );
    Cabana::deep_copy( corrupt, a );
    {
        auto g = Cabana::slice<0>( corrupt ); // gid is member 0 for every kind
        auto o = Cabana::slice<1>( corrupt ); // owner is member 1
        for ( std::size_t i = n_owned; i < n; ++i )
        {
            g( i ) = ~static_cast<GlobalId>( 0 );
            o( i ) = -1;
        }
    }
    Cabana::deep_copy( a, corrupt );

    haloExchange( comm, a, plan );

    host_aosoa after( "after", n );
    Cabana::deep_copy( after, a );
    auto gb = Cabana::slice<0>( before );
    auto ob = Cabana::slice<1>( before );
    auto ga = Cabana::slice<0>( after );
    auto oa = Cabana::slice<1>( after );

    int fails = 0;
    for ( std::size_t i = n_owned; i < n; ++i )
    {
        if ( ga( i ) != gb( i ) || oa( i ) != ob( i ) )
            ++fails; // the owner's gid/owner did not come back
        if ( ga( i ) == ~static_cast<GlobalId>( 0 ) || oa( i ) == -1 )
            ++fails; // still corrupt: this ghost was never synced
    }
    return fails;
}

//! Every conforming invariant that must survive a redistribution, as one LOCAL
//! fail count (sum across ranks == global). `pcv/pce/pcf` are the pre-move
//! topology checksums -- migrate() moves entities, it never alters the global
//! entity set, so they must be reproduced exactly.
template <class MeshT>
int checkConformingDistributed( MeshT& mesh, long long NvG, long long NeG,
                                long long NfG, unsigned long long pcv,
                                unsigned long long pce, unsigned long long pcf )
{
    int fails = 0;
    fails += TesseraTest::checkSiblingCoresidency( mesh );
    fails += TesseraTest::checkOwnershipPartition( mesh, NvG, NeG, NfG );
    fails += TesseraTest::owned1RingLocal( mesh );
    fails += TesseraTest::checkConforming( mesh );
    fails += TesseraTest::checkNoInteriorVertex( mesh );
    fails += TesseraTest::check21BalanceRed( mesh );

    unsigned long long cv, ce, cf;
    TesseraTest::topologyChecksum( mesh, cv, ce, cf );
    if ( cv != pcv || ce != pce || cf != pcf )
        ++fails;
    return fails;
}

//! Build + distribute an icosphere and run two adaptive conforming rounds.
template <class MeshT, class Mem>
void refinedConformingMesh( MeshT& mesh, MeshHalo<Mem>& halo )
{
    buildIcosphere( mesh, 2 );
    {
        auto faceOwner = facePartitionByAxis( mesh );
        distribute( mesh, halo, faceOwner );
    }
    // Two rounds back-to-back with nothing in between: refine() rebuilds the
    // 1-deep halo itself, so its Phase 3a finds both endpoint positions of every
    // midpoint the rank owns even when one is a ghost across a partition
    // boundary. This used to need an identity migrate() between the rounds.
    refine( mesh, halo, gidMask( mesh, 7 ) );
    refine( mesh, halo, gidMask( mesh, 5 ) );
}

// ===========================================================================
// Case A -- an adversarial, sibling-splitting dest
// ===========================================================================

template <class Exec, class MeshT>
int case_adversarial_dest( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    MPI_Comm comm = MPI_COMM_WORLD;
    int fails = 0;

    MeshT mesh( comm );
    MeshHalo<mem> halo;
    refinedConformingMesh( mesh, halo );

    const long long NvG = TesseraTest::globalOwnedVertices( mesh );
    const long long NeG = TesseraTest::globalOwnedEdges( mesh );
    const long long NfG = TesseraTest::globalOwnedFaces( mesh );
    const long long groups =
        commSum( TesseraTest::closureSiblingGroups( mesh ), comm );
    unsigned long long pcv, pce, pcf;
    TesseraTest::topologyChecksum( mesh, pcv, pce, pcf );

    // Deliberately scatter siblings: children of one parent hold consecutive
    // gids, so gid % size sends them to different ranks whenever size > 1.
    std::vector<Rank> dest;
    for ( GlobalId g : ownedGidsOfFaces( mesh ) )
        dest.push_back(
            static_cast<Rank>( g % static_cast<GlobalId>( size ) ) );

    const MigrateStats st = migrate( mesh, halo, dest );
    const long long fixups = commSum( st.siblingFixups, comm );

    int local =
        checkConformingDistributed( mesh, NvG, NeG, NfG, pcv, pce, pcf );
    local += corruptResyncTuples( comm, mesh.faces(), mesh.numOwnedFaces(),
                                  halo.fplan );
    local += corruptResyncTuples( comm, mesh.vertices(),
                                  mesh.numOwnedVertices(), halo.vplan );
    local += corruptResyncTuples( comm, mesh.edges(), mesh.numOwnedEdges(),
                                  halo.eplan );
    const long long glob = commSum( local, comm );
    if ( glob != 0 )
        ++fails;

    const long long euler = TesseraTest::checkOwnedEuler( mesh );
    if ( euler != 2 )
        ++fails;

    // Non-vacuity: if the fixup never fired, this case exercised nothing.
    if ( size > 1 && fixups <= 0 )
        ++fails;
    if ( groups <= 0 )
        ++fails; // no closure at all -- the mask failed to create one

    if ( rank == 0 )
        std::printf( "  [%s] adversarial-dest %s (F=%lld siblingGroups=%lld "
                     "destFixups=%lld euler=%lld inv=%lld)\n",
                     tag, fails == 0 ? "ok" : "FAIL", NfG, groups, fixups,
                     euler, glob );
    std::fflush( stdout );
    return fails;
}

// ===========================================================================
// Case B -- loadBalance() on a conforming mesh
// ===========================================================================

template <class Exec, class MeshT>
int case_load_balance( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    MPI_Comm comm = MPI_COMM_WORLD;
    int fails = 0;

    MeshT mesh( comm );
    MeshHalo<mem> halo;
    refinedConformingMesh( mesh, halo );

    const long long NvG = TesseraTest::globalOwnedVertices( mesh );
    const long long NeG = TesseraTest::globalOwnedEdges( mesh );
    const long long NfG = TesseraTest::globalOwnedFaces( mesh );
    unsigned long long pcv, pce, pcf;
    TesseraTest::topologyChecksum( mesh, pcv, pce, pcf );

    // Maximal imbalance: every owned face onto rank 0. Siblings all follow the
    // same destination here, so this move needs no fixup.
    {
        std::vector<Rank> dest( mesh.numOwnedFaces(), static_cast<Rank>( 0 ) );
        migrate( mesh, halo, dest );
    }
    auto weightSum = [&]( MeshT& m )
    {
        double s = 0.0;
        for ( double w : ownedFaceWeights( m ) )
            s += w;
        return s;
    };
    const long long maxFacesBefore =
        commMax( static_cast<long long>( mesh.numOwnedFaces() ), comm );
    const double totalWeight = globalSumD( weightSum( mesh ), comm );
    const double maxWeightBefore = globalMaxD( weightSum( mesh ), comm );

    const MigrateStats st = loadBalance( mesh, halo );
    const long long fixups = commSum( st.siblingFixups, comm );

    const long long maxFacesAfter =
        commMax( static_cast<long long>( mesh.numOwnedFaces() ), comm );
    const double maxWeightAfter = globalMaxD( weightSum( mesh ), comm );
    const double idealWeight = totalWeight / static_cast<double>( size );

    if ( size == 1 )
    {
        if ( maxFacesAfter != NfG )
            ++fails; // nothing to balance
    }
    else
    {
        if ( maxFacesAfter >= maxFacesBefore )
            ++fails; // no improvement over the maximally imbalanced start
        // The WEIGHTED load is what ownedFaceWeights() defines and Zoltan2
        // optimizes; the sibling fixup perturbs it, but the per-parent weight
        // is precisely what keeps that perturbation load-neutral, so a
        // generous factor of two over ideal must still hold.
        if ( maxWeightAfter > 2.0 * idealWeight )
            ++fails;
    }

    int local =
        checkConformingDistributed( mesh, NvG, NeG, NfG, pcv, pce, pcf );
    const long long glob = commSum( local, comm );
    if ( glob != 0 )
        ++fails;
    if ( TesseraTest::checkOwnedEuler( mesh ) != 2 )
        ++fails;

    if ( rank == 0 )
        std::printf( "  [%s] loadBalance %s (F=%lld maxFaces %lld->%lld "
                     "maxWeight %.1f->%.1f ideal=%.1f destFixups=%lld "
                     "inv=%lld)\n",
                     tag, fails == 0 ? "ok" : "FAIL", NfG, maxFacesBefore,
                     maxFacesAfter, maxWeightBefore, maxWeightAfter,
                     idealWeight, fixups, glob );
    std::fflush( stdout );
    return fails;
}

template <class Exec>
int run( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                       mem, Exec, RefinementMode::Conforming>;

    int fails = 0;
    fails += case_adversarial_dest<Exec, MeshT>( rank, size, tag );
    fails += case_load_balance<Exec, MeshT>( rank, size, tag );
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
            std::printf( "test_conforming_migrate: migrate / loadBalance / "
                         "halo on a conforming mesh (size %d)\n",
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
