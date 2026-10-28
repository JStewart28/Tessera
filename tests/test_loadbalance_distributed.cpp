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

// Regression test: the DISTRIBUTED load-balance solve
// (tasks/completed/distributed-loadbalance-solve.md).
//
// computeLoadBalance() used to be rank-0-bound in the GLOBAL face count: it
// gathered every rank's centroids/weights to rank 0, solved the whole problem
// there over a Teuchos::SerialComm, and scattered the assignment back. The
// determinism rationale for that ("MultiJagged is not deterministic across
// ranks") is about every rank solving the same problem INDEPENDENTLY, which a
// single distributed solve makes moot. LoadBalanceMode now selects between:
//
//   GatherRoot   the old path, kept as the reference and the fallback
//   Distributed  one MultiJagged solve over a Teuchos::MpiComm; rank 0 gets
//                nothing, but it is NOT run-to-run reproducible (case 4)
//   Sampled      rank 0 solves an O(size) gid-order sample, broadcasts the cut
//                structure, every rank classifies its own faces locally --
//                the DEFAULT, because case 4's verdict went against Distributed
//
// The fixture is a DELIBERATELY SKEWED start: a subdivision-4 icosphere (5120
// faces) with every face migrated onto rank 0, so the balancer has real work to
// do rather than being handed an already-good partition.
//
// Cases (numbered as in the task):
//   1  VALIDITY.       Every mode: dest sized numOwnedFaces(), entries in
//                      [0,size); after migrate(), checkOwnershipPartition,
//                      globalOwnedEuler == 2, owned1RingLocal, checkConforming,
//                      and a ghost corrupt/haloExchange resync.
//   2  BALANCE.        Distributed from the skewed start lands within
//                      (1 + tol + slack) x mean, and far below the pre-balance
//                      max (the whole mesh on rank 0), so it cannot pass
//                      vacuously. slack is the MEASURED value, recorded in the
//                      task's progress log.
//   3  CONFORMING.     1 and 2 on a refine()d Conforming mesh. siblingFixups
//                      may be nonzero -- expected and documented, reported not
//                      asserted.
//   4  REPRODUCIBILITY -- the deciding measurement, and its verdict. Two
//                      identical meshes in one run are balanced in BOTH modes
//                      and the two dest arrays compared element-wise.
//                      MEASURED: Distributed disagrees with itself on 8-20 of
//                      5120 faces at np5 (the count varies run to run) and
//                      agrees at np1-np4; Sampled agrees exactly at every rank
//                      count. So Sampled is the default and Distributed's
//                      mismatch count is REPORTED, not asserted -- asserting it
//                      zero would pin a property Zoltan2 does not have. Both
//                      modes' checksums are printed for the cross-RUN
//                      comparison (two ctest invocations, logs compared); the
//                      test asserts against values it recomputes, never
//                      hardcoded ones.
//   5  QUALITY.        Same input, all three modes: the resulting imbalance
//                      ratios are reported and the new default is asserted no
//                      worse than GatherRoot plus a margin. The partitions are
//                      NOT asserted identical -- different comms legitimately
//                      give different valid partitions.
//   6  ROOT INPUT SIZE -- the point of the task. GatherRoot receives the global
//                      count on rank 0; Distributed receives ZERO; Sampled
//                      receives O(size), asserted < global/4.
//   7  SAMPLED.        1, 2 and 5 in Sampled mode, plus: the same mesh in two
//                      DIFFERENT starting partitions yields bit-identical cut
//                      coordinates. That is what gid-order sampling buys.
//   8  SINGLE RANK.    All three modes take the fast path, return all-zeros,
//                      and loadBalance is a no-op migrate.
//   9  IDEMPOTENCE.    loadBalance twice: the second call's imbalance is no
//                      worse, and few faces change owner (reported; asserted
//                      below a measured fraction).
//  10  SCALING.        Subdivision 5 (20480 faces): the solve-phase wall time
//                      of each mode is PRINTED, never asserted (machine
//                      dependent).
//
// Runs on host (Serial) and device (default, HIP), ranks 1-5.

#include "MeshInvariants.hpp"

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <set>
#include <type_traits>
#include <vector>

using namespace Tessera;

namespace
{

// ---------------------------------------------------------------------------
// Small collective helpers
// ---------------------------------------------------------------------------

long long allSum( long long v, MPI_Comm comm )
{
    long long g = 0;
    MPI_Allreduce( &v, &g, 1, MPI_LONG_LONG, MPI_SUM, comm );
    return g;
}
long long allMax( long long v, MPI_Comm comm )
{
    long long g = 0;
    MPI_Allreduce( &v, &g, 1, MPI_LONG_LONG, MPI_MAX, comm );
    return g;
}
double allMaxD( double v, MPI_Comm comm )
{
    double g = 0;
    MPI_Allreduce( &v, &g, 1, MPI_DOUBLE, MPI_MAX, comm );
    return g;
}

const char* modeName( LoadBalanceMode m )
{
    switch ( m )
    {
    case LoadBalanceMode::GatherRoot:
        return "GatherRoot";
    case LoadBalanceMode::Distributed:
        return "Distributed";
    default:
        return "Sampled";
    }
}

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

//! Refine-face mask by gid divisibility: the same face set at any rank count.
template <class MeshT>
std::vector<char> gidMask( MeshT& mesh, int m )
{
    const std::vector<GlobalId> g = ownedFaceGidList( mesh );
    std::vector<char> mask( g.size(), 0 );
    for ( std::size_t f = 0; f < g.size(); ++f )
        mask[f] = ( g[f] % m == 0 ) ? 1 : 0;
    return mask;
}

//! max owned-face count over ranks, divided by the mean. 1.0 is perfect.
template <class MeshT>
double postImbalance( MeshT& mesh )
{
    MPI_Comm comm = mesh.comm();
    const long long mine = static_cast<long long>( mesh.numOwnedFaces() );
    const long long mx = allMax( mine, comm );
    const long long tot = allSum( mine, comm );
    const double mean =
        static_cast<double>( tot ) / static_cast<double>( mesh.commSize() );
    return mean > 0 ? static_cast<double>( mx ) / mean : 1.0;
}

//! Imbalance a `dest` array WOULD produce: max part face count over the mean.
//! Measured from dest itself, so the three modes are compared on one input
//! without migrating three times.
double destImbalance( const std::vector<Rank>& dest, int size, MPI_Comm comm )
{
    std::vector<long long> local( size, 0 ), global( size, 0 );
    for ( Rank r : dest )
        ++local[static_cast<int>( r )];
    MPI_Allreduce( local.data(), global.data(), size, MPI_LONG_LONG, MPI_SUM,
                   comm );
    long long mx = 0, tot = 0;
    for ( int r = 0; r < size; ++r )
    {
        mx = std::max( mx, global[r] );
        tot += global[r];
    }
    const double mean =
        static_cast<double>( tot ) / static_cast<double>( size );
    return mean > 0 ? static_cast<double>( mx ) / mean : 1.0;
}

//! dest must be one entry per owned face, each naming a real rank.
template <class MeshT>
int checkDestValid( MeshT& mesh, const std::vector<Rank>& dest )
{
    int fails = 0;
    if ( dest.size() != mesh.numOwnedFaces() )
        ++fails;
    for ( Rank r : dest )
        if ( static_cast<int>( r ) < 0 ||
             static_cast<int>( r ) >= mesh.commSize() )
            ++fails;
    return fails;
}

//! Order-independent checksum of (face gid -> destination rank), reduced over
//! all ranks. Two dest arrays over the same mesh agree iff their checksums do
//! (up to the astronomically unlikely collision a 64-bit mix permits), and
//! unlike an element-wise compare it is comparable ACROSS RUNS from stdout.
template <class MeshT>
unsigned long long destChecksum( MeshT& mesh, const std::vector<Rank>& dest )
{
    const std::vector<GlobalId> g = ownedFaceGidList( mesh );
    unsigned long long local = 0;
    for ( std::size_t f = 0; f < dest.size() && f < g.size(); ++f )
    {
        unsigned long long h =
            static_cast<unsigned long long>( g[f] ) * 1000003ULL +
            static_cast<unsigned long long>( dest[f] );
        h ^= h >> 29;
        h *= 0xbf58476d1ce4e5b9ULL;
        h ^= h >> 32;
        local += h;
    }
    unsigned long long global = 0;
    MPI_Allreduce( &local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                   mesh.comm() );
    return global;
}

//! Corrupt every ghost tuple, haloExchange, verify the owner's values return.
template <class AoSoAType, class PlanT>
int corruptResyncTuples( MPI_Comm comm, AoSoAType& a, std::size_t n_owned,
                         PlanT& plan )
{
    using host_aosoa =
        Cabana::AoSoA<typename AoSoAType::member_types, Kokkos::HostSpace>;
    const std::size_t n = a.size();

    host_aosoa before( "before", n );
    Cabana::deep_copy( before, a );

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
            ++fails;
        if ( ga( i ) == ~static_cast<GlobalId>( 0 ) || oa( i ) == -1 )
            ++fails;
    }
    return fails;
}

//! Every post-migrate invariant case 1 requires, as a LOCAL fail count.
template <class MeshT, class Mem>
int checkPostMigrate( MeshT& mesh, MeshHalo<Mem>& halo, long long NvG,
                      long long NeG, long long NfG )
{
    int fails = 0;
    fails += TesseraTest::checkOwnershipPartition( mesh, NvG, NeG, NfG );
    fails += TesseraTest::owned1RingLocal( mesh );
    fails += TesseraTest::checkConforming( mesh );
    if ( TesseraTest::ownedEulerGlobal( mesh ) != 2 )
        ++fails;
    fails += corruptResyncTuples( mesh.comm(), mesh.vertices(),
                                  mesh.numOwnedVertices(), halo.vplan );
    fails += corruptResyncTuples( mesh.comm(), mesh.faces(),
                                  mesh.numOwnedFaces(), halo.fplan );
    return fails;
}

//! Build a distributed icosphere and then DUMP EVERY FACE ONTO RANK 0, so the
//! balancer starts from the worst possible partition.
template <class MeshT, class Mem>
void skewedMesh( MeshT& mesh, MeshHalo<Mem>& halo, int subdiv )
{
    buildIcosphere( mesh, subdiv );
    auto faceOwner = facePartitionByAxis( mesh );
    distribute( mesh, halo, faceOwner );
    std::vector<Rank> dest( mesh.numOwnedFaces(), static_cast<Rank>( 0 ) );
    migrate( mesh, halo, dest );
}

// The slack over (1 + tol) that case 2 allows, MEASURED on the first
// implementation run rather than guessed. MultiJagged optimizes the WEIGHTED
// load with a rectilinear multisection over a coarse geodesic surface, so the
// face-count imbalance it lands at is looser than its own imbalance_tolerance,
// and Sampled fits its cuts to a sample so it is looser again. Worst observed
// over ranks 2-5 on both backends at subdivision 4: 1.0615 (Sampled, np5;
// GatherRoot and Distributed both hit 1.0000), i.e. an excess of 0.0115 over
// 1+tol. 0.10 is that with room, and still far tighter than "balanced at all".
constexpr double kBalanceSlack = 0.10;
// Case 9: fraction of owned faces allowed to change owner on a second
// loadBalance() of an already-balanced mesh. MEASURED: 0.0000 for Sampled (its
// cuts are a function of the mesh, so the second pass reproduces the first
// exactly) and 0.0053 for Distributed at np5.
constexpr double kIdempotenceFrac = 0.05;

} // namespace

// ===========================================================================
// Cases 1, 5, 6: validity, quality and rank-0 input size, one mode each
// ===========================================================================
template <class Exec>
int caseValidityQualityRootSize( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                       mem, Exec, RefinementMode::HangingNode2to1>;
    MPI_Comm comm = MPI_COMM_WORLD;
    int fails = 0;

    const LoadBalanceMode modes[3] = { LoadBalanceMode::GatherRoot,
                                       LoadBalanceMode::Distributed,
                                       LoadBalanceMode::Sampled };
    double ratio[3] = { 0, 0, 0 };
    long long rootFaces[3] = { 0, 0, 0 };
    long long globalF = 0;

    // Case 5/6 first: all three modes solved against the SAME input, so the
    // quality comparison is apples to apples. computeLoadBalance() does not
    // move anything, so one mesh serves all three.
    {
        MeshT mesh( comm );
        MeshHalo<mem> halo;
        skewedMesh( mesh, halo, 4 );
        for ( int m = 0; m < 3; ++m )
        {
            LoadBalanceStats st;
            const std::vector<Rank> dest =
                computeLoadBalance( mesh, 0.05, modes[m], &st );
            fails += checkDestValid( mesh, dest );
            ratio[m] = destImbalance( dest, size, comm );
            rootFaces[m] = st.rootSolveFaces;
            globalF = st.globalFaces;
        }
    }

    if ( rank == 0 )
    {
        std::printf( "  [%s] case5 quality (max part / mean): GatherRoot=%.4f "
                     "Distributed=%.4f Sampled=%.4f\n",
                     tag, ratio[0], ratio[1], ratio[2] );
        std::printf( "  [%s] case6 rank-0 solve input faces (global=%lld): "
                     "GatherRoot=%lld Distributed=%lld Sampled=%lld\n",
                     tag, globalF, rootFaces[0], rootFaces[1], rootFaces[2] );
    }

    if ( size > 1 )
    {
        // Case 6 -- the task's deliverable.
        if ( rootFaces[0] != globalF )
            ++fails; // GatherRoot must gather everything
        if ( rootFaces[1] != 0 )
            ++fails; // Distributed must gather NOTHING
        if ( rootFaces[2] <= 0 || rootFaces[2] >= globalF / 4 )
            ++fails; // Sampled must be O(size), well under the global count
        // Case 5 -- no materially worse than the reference.
        if ( ratio[1] > ratio[0] + 0.15 )
            ++fails;
        if ( ratio[2] > ratio[0] + 0.15 )
            ++fails;
    }

    // Case 1: every mode's dest survives migrate() with every invariant intact.
    for ( int m = 0; m < 3; ++m )
    {
        MeshT mesh( comm );
        MeshHalo<mem> halo;
        buildIcosphere( mesh, 4 );
        const long long NvG = static_cast<long long>( mesh.numVertices() );
        const long long NeG = static_cast<long long>( mesh.numEdges() );
        const long long NfG = static_cast<long long>( mesh.numFaces() );
        {
            auto faceOwner = facePartitionByAxis( mesh );
            distribute( mesh, halo, faceOwner );
        }
        unsigned long long pcv, pce, pcf;
        TesseraTest::topologyChecksum( mesh, pcv, pce, pcf );
        {
            std::vector<Rank> d0( mesh.numOwnedFaces(),
                                  static_cast<Rank>( 0 ) );
            migrate( mesh, halo, d0 );
        }
        const long long maxBefore =
            allMax( static_cast<long long>( mesh.numOwnedFaces() ), comm );

        MigrateStats ms = loadBalance( mesh, halo, 0.05, modes[m] );

        int local = checkPostMigrate( mesh, halo, NvG, NeG, NfG );
        unsigned long long cv, ce, cf;
        TesseraTest::topologyChecksum( mesh, cv, ce, cf );
        if ( cv != pcv || ce != pce || cf != pcf )
            ++local; // loadBalance moves entities, it must not alter the mesh
        fails += allSum( local, comm ) > 0 ? 1 : 0;

        const double post = postImbalance( mesh );
        const long long maxAfter =
            allMax( static_cast<long long>( mesh.numOwnedFaces() ), comm );

        // Case 2 (and case 7's balance half): from the skewed start the max
        // owned count must land near the mean AND be dramatically better than
        // "everything on rank 0", so this cannot pass vacuously.
        if ( size > 1 )
        {
            if ( post > 1.0 + 0.05 + kBalanceSlack )
                ++fails;
            // Non-vacuity is carried by the imbalance bound above: maxBefore
            // is the WHOLE mesh, so landing within (1+tol+slack) of the mean is
            // the strongest correct statement at any rank count (at np2 a
            // perfect partition is exactly maxBefore/2, so no fixed fraction
            // of maxBefore would be a meaningful extra bound).
            if ( maxAfter >= maxBefore )
                ++fails;
        }
        else if ( maxAfter != NfG )
            ++fails; // single rank: no-op

        if ( rank == 0 )
            std::printf( "  [%s] case1/2 %-11s inv=%s maxBefore=%lld "
                         "maxAfter=%lld imbalance=%.4f fixups=%lld\n",
                         tag, modeName( modes[m] ), local == 0 ? "ok" : "FAIL",
                         maxBefore, maxAfter, post,
                         static_cast<long long>( ms.siblingFixups ) );
    }

    return fails;
}

// ===========================================================================
// Case 3: the same thing on a refine()d Conforming mesh
// ===========================================================================
template <class Exec>
int caseConforming( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                       mem, Exec, RefinementMode::Conforming>;
    MPI_Comm comm = MPI_COMM_WORLD;
    int fails = 0;

    MeshT mesh( comm );
    MeshHalo<mem> halo;
    buildIcosphere( mesh, 3 );
    {
        auto faceOwner = facePartitionByAxis( mesh );
        distribute( mesh, halo, faceOwner );
    }
    {
        auto mask = gidMask( mesh, 3 );
        refine( mesh, halo, mask );
    }
    const long long NvG = TesseraTest::globalOwnedVertices( mesh );
    const long long NeG = TesseraTest::globalOwnedEdges( mesh );
    const long long NfG = TesseraTest::globalOwnedFaces( mesh );
    unsigned long long pcv, pce, pcf;
    TesseraTest::topologyChecksum( mesh, pcv, pce, pcf );

    // Skew: everything onto rank 0.
    {
        std::vector<Rank> d0( mesh.numOwnedFaces(), static_cast<Rank>( 0 ) );
        migrate( mesh, halo, d0 );
    }
    const long long maxBefore =
        allMax( static_cast<long long>( mesh.numOwnedFaces() ), comm );

    LoadBalanceStats st;
    MigrateStats ms =
        loadBalance( mesh, halo, 0.05, LoadBalanceMode::Distributed, &st );

    int local = checkPostMigrate( mesh, halo, NvG, NeG, NfG );
    local += TesseraTest::checkSiblingCoresidency( mesh );
    local += TesseraTest::check21BalanceRed( mesh );
    local += TesseraTest::checkNoInteriorVertex( mesh );
    unsigned long long cv, ce, cf;
    TesseraTest::topologyChecksum( mesh, cv, ce, cf );
    if ( cv != pcv || ce != pce || cf != pcf )
        ++local;
    fails += allSum( local, comm ) > 0 ? 1 : 0;

    const double post = postImbalance( mesh );
    const long long maxAfter =
        allMax( static_cast<long long>( mesh.numOwnedFaces() ), comm );
    if ( size > 1 )
    {
        if ( post > 1.0 + 0.05 + kBalanceSlack )
            ++fails;
        if ( maxAfter >= maxBefore )
            ++fails;
        if ( st.rootSolveFaces != 0 )
            ++fails; // still no gather on a conforming mesh
    }

    if ( rank == 0 )
        std::printf( "  [%s] case3 conforming inv=%s NfG=%lld maxBefore=%lld "
                     "maxAfter=%lld imbalance=%.4f siblingFixups=%lld "
                     "rootFaces=%lld\n",
                     tag, local == 0 ? "ok" : "FAIL", NfG, maxBefore, maxAfter,
                     post, static_cast<long long>( ms.siblingFixups ),
                     st.rootSolveFaces );
    return fails;
}

// ===========================================================================
// Case 4: reproducibility -- the deciding measurement
// ===========================================================================
//
// Two meshes built identically in ONE run are balanced in the same mode, and the
// two dest arrays are compared entry for entry (the owned-face order is
// identical by construction, so this is well defined). The mode that survives
// this is the one that may be the default.
//
// VERDICT, measured: Sampled agrees exactly at every rank count; Distributed
// disagrees with itself on a handful of faces at np5 and the count itself varies
// between runs. Distributed's count is therefore PRINTED, not asserted -- a
// zero-assertion on it would pin a property a single distributed MultiJagged
// solve does not have, and the finding is recorded in README -> Known Issues.
// The checksums are printed for the cross-RUN comparison; nothing hardcoded.
template <class Exec>
int caseReproducibility( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                       mem, Exec, RefinementMode::HangingNode2to1>;
    MPI_Comm comm = MPI_COMM_WORLD;
    int fails = 0;

    const LoadBalanceMode modes[2] = { LoadBalanceMode::Sampled,
                                       LoadBalanceMode::Distributed };
    for ( int m = 0; m < 2; ++m )
    {
        unsigned long long sumA = 0, sumB = 0;
        std::vector<Rank> destA, destB;
        {
            MeshT m1( comm );
            MeshHalo<mem> h1;
            skewedMesh( m1, h1, 4 );
            destA = computeLoadBalance( m1, 0.05, modes[m] );
            sumA = destChecksum( m1, destA );

            MeshT m2( comm );
            MeshHalo<mem> h2;
            skewedMesh( m2, h2, 4 );
            destB = computeLoadBalance( m2, 0.05, modes[m] );
            sumB = destChecksum( m2, destB );
        }

        int local = 0;
        if ( destA.size() != destB.size() )
            ++local;
        else
            for ( std::size_t i = 0; i < destA.size(); ++i )
                if ( destA[i] != destB[i] )
                    ++local;
        const long long diff = allSum( local, comm );
        const bool agree = ( diff == 0 && sumA == sumB );

        // Asserted for the DEFAULT mode only; measured and reported for the
        // other. See the verdict note above.
        if ( modes[m] == LoadBalanceMode::Sampled && !agree )
            ++fails;

        if ( rank == 0 )
            std::printf( "  [%s] case4 reproducibility (np%d) %-11s "
                         "mismatches=%lld %s DEST_CHECKSUM_A=%llu "
                         "DEST_CHECKSUM_B=%llu\n",
                         tag, size, modeName( modes[m] ), diff,
                         modes[m] == LoadBalanceMode::Sampled
                             ? ( agree ? "ok" : "FAIL" )
                             : "(measured, not asserted)",
                         sumA, sumB );
    }
    return fails;
}

// ===========================================================================
// Case 7: Sampled cuts are a property of the mesh, not of the partition
// ===========================================================================
template <class Exec>
int caseSampledCuts( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                       mem, Exec, RefinementMode::HangingNode2to1>;
    MPI_Comm comm = MPI_COMM_WORLD;
    int fails = 0;

    LoadBalanceStats stSkewed, stAxis;
    {
        // Starting partition A: everything on rank 0.
        MeshT mesh( comm );
        MeshHalo<mem> halo;
        skewedMesh( mesh, halo, 4 );
        const std::vector<Rank> dest = computeLoadBalance(
            mesh, 0.05, LoadBalanceMode::Sampled, &stSkewed );
        fails += checkDestValid( mesh, dest );
    }
    {
        // Starting partition B: the axis partition, untouched.
        MeshT mesh( comm );
        MeshHalo<mem> halo;
        buildIcosphere( mesh, 4 );
        auto faceOwner = facePartitionByAxis( mesh );
        distribute( mesh, halo, faceOwner );
        const std::vector<Rank> dest =
            computeLoadBalance( mesh, 0.05, LoadBalanceMode::Sampled, &stAxis );
        fails += checkDestValid( mesh, dest );
    }

    int local = 0;
    if ( stSkewed.cuts.size() != stAxis.cuts.size() )
        ++local;
    else
        for ( std::size_t i = 0; i < stSkewed.cuts.size(); ++i )
            if ( stSkewed.cuts[i] != stAxis.cuts[i] )
                ++local; // BIT-identical is the claim, so == is the test
    if ( size > 1 && stSkewed.cuts.empty() )
        ++local; // no cuts broadcast at all: the check would be vacuous
    const long long diff = allSum( local, comm );
    if ( diff != 0 )
        ++fails;

    if ( rank == 0 )
        std::printf( "  [%s] case7 sampled cuts partition-independent %s "
                     "(cutDoubles=%zu stride=%lld/%lld sample=%lld/%lld "
                     "diff=%lld)\n",
                     tag, diff == 0 ? "ok" : "FAIL", stSkewed.cuts.size(),
                     stSkewed.sampleStride, stAxis.sampleStride,
                     stSkewed.rootSolveFaces, stAxis.rootSolveFaces, diff );
    return fails;
}

// ===========================================================================
// Case 8: single rank -- every mode takes the fast path
// ===========================================================================
template <class Exec>
int caseSingleRank( int rank, int size, const char* tag )
{
    if ( size != 1 )
        return 0;
    using mem = typename Exec::memory_space;
    using MeshT = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                       mem, Exec, RefinementMode::HangingNode2to1>;
    int fails = 0;

    const LoadBalanceMode modes[3] = { LoadBalanceMode::GatherRoot,
                                       LoadBalanceMode::Distributed,
                                       LoadBalanceMode::Sampled };
    for ( int m = 0; m < 3; ++m )
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        buildIcosphere( mesh, 3 );
        auto faceOwner = facePartitionByAxis( mesh );
        distribute( mesh, halo, faceOwner );
        const std::size_t nof = mesh.numOwnedFaces();

        LoadBalanceStats st;
        const std::vector<Rank> dest =
            computeLoadBalance( mesh, 0.05, modes[m], &st );
        if ( !st.fastPath )
            ++fails;
        if ( st.rootSolveFaces != 0 )
            ++fails;
        if ( dest.size() != nof )
            ++fails;
        for ( Rank r : dest )
            if ( r != 0 )
                ++fails;

        loadBalance( mesh, halo, 0.05, modes[m] );
        if ( mesh.numOwnedFaces() != nof )
            ++fails; // no-op migrate
    }
    if ( rank == 0 )
        std::printf( "  [%s] case8 single-rank fast path %s\n", tag,
                     fails == 0 ? "ok" : "FAIL" );
    return fails;
}

// ===========================================================================
// Case 9: idempotence
// ===========================================================================
//
// A balancer that reshuffles an already-balanced mesh is a performance bug no
// invariant check catches, so both the imbalance and the number of faces that
// change owner on a second call are measured, for every mode.
template <class Exec>
int caseIdempotence( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                       mem, Exec, RefinementMode::HangingNode2to1>;
    MPI_Comm comm = MPI_COMM_WORLD;
    int fails = 0;

    const LoadBalanceMode modes[3] = { LoadBalanceMode::GatherRoot,
                                       LoadBalanceMode::Distributed,
                                       LoadBalanceMode::Sampled };
    for ( int m = 0; m < 3; ++m )
    {
        MeshT mesh( comm );
        MeshHalo<mem> halo;
        skewedMesh( mesh, halo, 4 );

        loadBalance( mesh, halo, 0.05, modes[m] );
        const double first = postImbalance( mesh );
        const std::vector<GlobalId> before = ownedFaceGidList( mesh );
        const std::set<GlobalId> had( before.begin(), before.end() );

        loadBalance( mesh, halo, 0.05, modes[m] );
        const double second = postImbalance( mesh );
        const std::vector<GlobalId> after = ownedFaceGidList( mesh );
        const std::set<GlobalId> has( after.begin(), after.end() );

        // A face changed owner iff this rank held it and no longer does.
        long long left = 0;
        for ( GlobalId g : had )
            if ( has.find( g ) == has.end() )
                ++left;
        const long long moved = allSum( left, comm );
        const long long total =
            allSum( static_cast<long long>( after.size() ), comm );
        const double frac = total > 0 ? static_cast<double>( moved ) /
                                            static_cast<double>( total )
                                      : 0.0;

        int bad = 0;
        if ( size > 1 )
        {
            if ( second > first + 1e-9 )
                ++bad; // the second pass must not make the balance worse
            if ( frac > kIdempotenceFrac )
                ++bad; // reshuffling an already-balanced mesh
        }
        fails += bad;

        if ( rank == 0 )
            std::printf( "  [%s] case9 idempotence %-11s %s first=%.4f "
                         "second=%.4f moved=%lld/%lld (%.4f)\n",
                         tag, modeName( modes[m] ), bad == 0 ? "ok" : "FAIL",
                         first, second, moved, total, frac );
    }
    return fails;
}

// ===========================================================================
// Case 10: scaling evidence -- printed, never asserted
// ===========================================================================
template <class Exec>
int caseScaling( int rank, int size, const char* tag )
{
    if ( size < 2 )
        return 0;
    using mem = typename Exec::memory_space;
    using MeshT = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                       mem, Exec, RefinementMode::HangingNode2to1>;
    MPI_Comm comm = MPI_COMM_WORLD;

    MeshT mesh( comm );
    MeshHalo<mem> halo;
    skewedMesh( mesh, halo, 5 );
    const long long NfG = TesseraTest::globalOwnedFaces( mesh );

    const LoadBalanceMode modes[3] = { LoadBalanceMode::GatherRoot,
                                       LoadBalanceMode::Distributed,
                                       LoadBalanceMode::Sampled };
    for ( int m = 0; m < 3; ++m )
    {
        MPI_Barrier( comm );
        const double t0 = MPI_Wtime();
        LoadBalanceStats st;
        const std::vector<Rank> dest =
            computeLoadBalance( mesh, 0.05, modes[m], &st );
        const double dt = allMaxD( MPI_Wtime() - t0, comm );
        const double ratio = destImbalance( dest, size, comm );
        if ( rank == 0 )
            std::printf( "  [%s] case10 subdiv5 NfG=%lld np%d %-11s "
                         "solve=%.4f s imbalance=%.4f rootFaces=%lld\n",
                         tag, NfG, size, modeName( modes[m] ), dt, ratio,
                         st.rootSolveFaces );
    }
    return 0;
}

template <class Exec>
int run( int rank, int size, const char* tag )
{
    int fails = 0;
    fails += caseValidityQualityRootSize<Exec>( rank, size, tag );
    fails += caseConforming<Exec>( rank, size, tag );
    fails += caseReproducibility<Exec>( rank, size, tag );
    fails += caseSampledCuts<Exec>( rank, size, tag );
    fails += caseSingleRank<Exec>( rank, size, tag );
    fails += caseIdempotence<Exec>( rank, size, tag );
    fails += caseScaling<Exec>( rank, size, tag );

    int global = 0;
    MPI_Allreduce( &fails, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    if ( rank == 0 )
        std::printf( "  [%s] %s (%d)\n", tag, global == 0 ? "ok" : "FAIL",
                     global );
    return global;
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
            std::printf( "test_loadbalance_distributed: distributed Zoltan2 "
                         "load-balance solve (size %d)\n",
                         size );

        fails += run<Kokkos::Serial>( rank, size, "Serial" );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            fails +=
                run<Kokkos::DefaultExecutionSpace>( rank, size, "Default" );
    }

    Kokkos::finalize();
    MPI_Finalize();
    return fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
