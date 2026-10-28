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

// Test: SPLITEDGES TRIANGLE SHAPE QUALITY OVER SPLIT DEPTH (tasks/completed/edge-split.md,
// risk R12) -- how far a repeated splitEdges() drive can push triangle shape, and
// whether the answer depends on the round count or on the MASK.
//
// WHY THIS TEST EXISTS. test_split_edges case 8 runs FIVE length-driven rounds
// and asserts a measured radius-ratio floor. Five rounds are consistent with a
// bound but do not establish one: Tessera's own red-green quality test
// (test_conforming_quality) records that EIGHT rounds could not distinguish
// saturation from a maximum being discovered slowly, and sixteen could. A
// downstream consumer (the Beatnik z-model remesher) refines far more than five
// times, so "the shape is bounded" needs evidence at depth.
//
// WHY THE ANSWER CANNOT BE INHERITED FROM test_conforming_quality. That test's
// bound rests on the closure being TRANSIENT: un-close discards the whole
// closure layer every round, so every visible triangle is one of finitely many
// retriangulations of a RED triangle and the similarity classes are bounded by
// construction. splitEdges() has no such reset. Its children PERSIST: a |S|=1
// median-cut child is an ordinary face next round and can be cut again. The
// number of similarity classes reachable in n rounds is therefore NOT bounded a
// priori, and whether shape degrades is a property of the MASK, not of
// splitEdges().
//
// THE FOUR MASK FAMILIES, chosen to bracket a real consumer's behaviour:
//
//   above-mean   Split every edge longer than the global mean edge length.
//                Case 8's mask. LENGTH-AWARE and therefore self-correcting --
//                it is a coarse relative of Rivara longest-edge bisection, whose
//                shape stability is a classical result. This is the friendly end.
//   below-mean   Split every edge SHORTER than the global mean. The exact
//                anti-Rivara rule: it refuses to touch the long edge of a
//                stretched triangle and cuts the short ones, which is the
//                direction that makes slivers if any direction does.
//   hash-third   Split an edge iff a hash of its MIDPOINT POSITION is 0 mod 3.
//                LENGTH-BLIND and global. A metric-driven consumer whose metric
//                is uncorrelated with edge length looks like this.
//   cap-hash     hash-third restricted to a fixed geodesic cap around +z. The
//                same length-blind rule, but growth is bounded by the cap so it
//                can be driven MUCH deeper than the global families -- this is
//                the family that reaches 25+ rounds.
//
// The hash is taken over the raw IEEE bits of the midpoint POSITION, not over
// gids: gids come from an MPI_Exscan and are not rank-count invariant, positions
// are. So every family here selects the same global edge set at any rank count.
//
// WHAT IS MEASURED, per round: the global minimum radius ratio r/R (0.5 for an
// equilateral triangle, 0 for a degenerate one -- the same statistic case 8
// asserts), the global minimum angle, and the TAIL -- how many faces sit below
// each of five r/R thresholds. The tail is the diagnostic the minimum cannot
// give: a minimum that is flat while the count below 0.20 grows like the mesh is
// a different situation from one where both are flat.
//
// This is a DIAGNOSTIC (TIER unit), not a gate test: it is deliberately run to
// whatever depth the face budget allows, and its output is a table to read
// rather than a bound to assert. The bound it establishes is asserted in
// test_split_edges case 8. It does assert the invariants (conformity, Euler)
// after every round, and that no triangle becomes degenerate.
//
// Knobs (env): TESSERA_SPLIT_DEPTH_ROUNDS (max rounds per family, default 30),
// TESSERA_SPLIT_DEPTH_FACES (global face budget, default 2000000).
//
// Runs on host (Serial) and device (default, HIP).

#include "MeshInvariants.hpp"

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

using namespace Tessera;

static constexpr double kPi = 3.14159265358979323846;

//! r/R thresholds whose population is reported per round. The tail, not the
//! minimum, is what separates "one bad triangle exists" from "bad triangles are
//! becoming a fixed FRACTION of the mesh".
static constexpr int kNTail = 5;
static constexpr double kTail[kNTail] = { 0.30, 0.25, 0.20, 0.15, 0.10 };

//! Fixed geodesic cap for the cap-hash family: half-angle about +z.
static constexpr double kCapHalfAngle = 0.45;

static int envInt( const char* name, int dflt )
{
    const char* e = std::getenv( name );
    if ( e == nullptr )
        return dflt;
    const int v = std::atoi( e );
    return v > 0 ? v : dflt;
}

static inline unsigned long long mix64( unsigned long long z )
{
    z += 0x9e3779b97f4a7c15ULL;
    z = ( z ^ ( z >> 30 ) ) * 0xbf58476d1ce4e5b9ULL;
    z = ( z ^ ( z >> 27 ) ) * 0x94d049bb133111ebULL;
    return z ^ ( z >> 31 );
}

//! Hash of a position's raw IEEE bits. Positions are computed by the same
//! arithmetic on every rank (a midpoint is one 0.5*(a+b)), so this is a
//! rank-count-invariant function of the GLOBAL mesh.
static inline unsigned long long posHash( const std::array<double, 3>& p )
{
    unsigned long long h = 0xcbf29ce484222325ULL;
    for ( int d = 0; d < 3; ++d )
    {
        unsigned long long u = 0;
        std::memcpy( &u, &p[d], sizeof( u ) );
        h = mix64( h ^ u );
    }
    return h;
}

static inline long long gsumll( long long v )
{
    long long g = 0;
    MPI_Allreduce( &v, &g, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD );
    return g;
}
static inline double gminD( double v )
{
    double g = 0;
    MPI_Allreduce( &v, &g, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
    return g;
}

// ---------------------------------------------------------------------------
// Mesh readers (same idiom as test_split_edges)
// ---------------------------------------------------------------------------

template <class MeshT>
static std::unordered_map<GlobalId, std::array<double, 3>>
readPositions( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto vg = Cabana::slice<VertexField::Gid>( hv );
    auto vp = Cabana::slice<VertexField::Position>( hv );
    std::unordered_map<GlobalId, std::array<double, 3>> out;
    out.reserve( mesh.numVertices() * 2 );
    for ( std::size_t i = 0; i < mesh.numVertices(); ++i )
    {
        std::array<double, 3> p = { 0, 0, 0 };
        for ( int d = 0; d < MeshT::dim && d < 3; ++d )
            p[d] = static_cast<double>( vp( i, d ) );
        out[vg( i )] = p;
    }
    return out;
}

template <class MeshT>
static std::vector<std::array<GlobalId, 3>> ownedFaceVerts( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto fv = Cabana::slice<FaceField::Verts>( hf );
    std::vector<std::array<GlobalId, 3>> out( mesh.numOwnedFaces() );
    for ( std::size_t f = 0; f < mesh.numOwnedFaces(); ++f )
        for ( int k = 0; k < 3; ++k )
            out[f][k] = fv( f, k );
    return out;
}

template <class MeshT>
static std::vector<EdgeKey> ownedEdgeKeys( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
        "he", mesh.numEdges() );
    Cabana::deep_copy( he, mesh.edges() );
    auto ev = Cabana::slice<EdgeField::Verts>( he );
    std::vector<EdgeKey> out( mesh.numOwnedEdges() );
    for ( std::size_t e = 0; e < mesh.numOwnedEdges(); ++e )
        out[e] = makeEdgeKey( ev( e, 0 ), ev( e, 1 ) );
    return out;
}

// ---------------------------------------------------------------------------
// Per-round shape measurement
// ---------------------------------------------------------------------------

struct Shape
{
    double minRR = 1.0;      //!< global min inradius/circumradius
    double minAngleDeg = 180.0;
    long long tail[kNTail] = { 0, 0, 0, 0, 0 }; //!< faces below kTail[i]
    long long faces = 0;
};

//! Shape of every OWNED face, reduced globally.
template <class MeshT>
static Shape measure( MeshT& mesh, int& fails )
{
    const auto pos = readPositions( mesh );
    Shape s;
    double worst = 1.0, worstAng = 180.0;
    long long tail[kNTail] = { 0, 0, 0, 0, 0 };
    long long n = 0;

    for ( const auto& t : ownedFaceVerts( mesh ) )
    {
        std::array<double, 3> p[3];
        bool ok = true;
        for ( int k = 0; k < 3; ++k )
        {
            auto it = pos.find( t[k] );
            if ( it == pos.end() )
                ok = false;
            else
                p[k] = it->second;
        }
        if ( !ok )
        {
            ++fails; // every corner of an owned face must be held after halo
            continue;
        }
        ++n;

        // side[k] is the side from corner k to corner k+1.
        double side[3];
        for ( int k = 0; k < 3; ++k )
        {
            double d2 = 0.0;
            for ( int d = 0; d < 3; ++d )
            {
                const double dd = p[( k + 1 ) % 3][d] - p[k][d];
                d2 += dd * dd;
            }
            side[k] = std::sqrt( d2 );
        }
        double u[3], v[3];
        for ( int d = 0; d < 3; ++d )
        {
            u[d] = p[1][d] - p[0][d];
            v[d] = p[2][d] - p[0][d];
        }
        const double cx = u[1] * v[2] - u[2] * v[1];
        const double cy = u[2] * v[0] - u[0] * v[2];
        const double cz = u[0] * v[1] - u[1] * v[0];
        const double area =
            0.5 * std::sqrt( cx * cx + cy * cy + cz * cz );
        const double abc = side[0] * side[1] * side[2];
        if ( abc <= 0.0 || area <= 0.0 )
        {
            ++fails; // a degenerate triangle is a hard failure, not a datum
            continue;
        }
        const double semi = 0.5 * ( side[0] + side[1] + side[2] );
        const double rr = 4.0 * area * area / ( semi * abc );
        worst = std::min( worst, rr );
        for ( int i = 0; i < kNTail; ++i )
            if ( rr < kTail[i] )
                ++tail[i];

        // Min angle by the law of cosines. a=side[0] is opposite corner 2.
        for ( int k = 0; k < 3; ++k )
        {
            const double a = side[k], b = side[( k + 1 ) % 3],
                         c = side[( k + 2 ) % 3];
            double cosA = ( a * a + b * b - c * c ) / ( 2.0 * a * b );
            cosA = std::max( -1.0, std::min( 1.0, cosA ) );
            worstAng =
                std::min( worstAng, std::acos( cosA ) * 180.0 / kPi );
        }
    }

    s.minRR = gminD( worst );
    s.minAngleDeg = gminD( worstAng );
    for ( int i = 0; i < kNTail; ++i )
        s.tail[i] = gsumll( tail[i] );
    s.faces = gsumll( n );
    return s;
}

// ---------------------------------------------------------------------------
// Mask families
// ---------------------------------------------------------------------------

enum class Family
{
    AboveMean,
    BelowMean,
    HashThird,
    CapHash
};

static const char* familyName( Family f )
{
    switch ( f )
    {
    case Family::AboveMean:
        return "above-mean";
    case Family::BelowMean:
        return "below-mean";
    case Family::HashThird:
        return "hash-third";
    case Family::CapHash:
        return "cap-hash  ";
    }
    return "?";
}

//! Mask over this rank's OWNED edges, in owned-edge local index order. Every
//! rule below is a pure function of the GLOBAL mesh geometry, so the selected
//! global edge set is rank-count invariant.
template <class MeshT>
static std::vector<char> buildMask( MeshT& mesh, Family fam, int& fails )
{
    const auto pos = readPositions( mesh );
    const std::vector<EdgeKey> keys = ownedEdgeKeys( mesh );
    const std::size_t nE = keys.size();

    std::vector<std::array<double, 3>> mid( nE );
    std::vector<double> len( nE, 0.0 );
    double localSum = 0.0;
    for ( std::size_t e = 0; e < nE; ++e )
    {
        auto ia = pos.find( keys[e].id[0] );
        auto ib = pos.find( keys[e].id[1] );
        if ( ia == pos.end() || ib == pos.end() )
        {
            ++fails; // an owned edge's endpoints must be held locally
            continue;
        }
        double d2 = 0.0;
        for ( int d = 0; d < 3; ++d )
        {
            mid[e][d] = 0.5 * ( ia->second[d] + ib->second[d] );
            const double dd = ia->second[d] - ib->second[d];
            d2 += dd * dd;
        }
        len[e] = std::sqrt( d2 );
        localSum += len[e];
    }
    double totalLen = 0.0;
    MPI_Allreduce( &localSum, &totalLen, 1, MPI_DOUBLE, MPI_SUM, mesh.comm() );
    const long long gE = TesseraTest::globalOwnedEdges( mesh );
    const double mean = gE > 0 ? totalLen / static_cast<double>( gE ) : 0.0;

    std::vector<char> mask( nE, 0 );
    for ( std::size_t e = 0; e < nE; ++e )
    {
        bool take = false;
        switch ( fam )
        {
        case Family::AboveMean:
            take = len[e] > mean;
            break;
        case Family::BelowMean:
            take = len[e] > 0.0 && len[e] < mean;
            break;
        case Family::HashThird:
            take = ( posHash( mid[e] ) % 3ULL ) == 0ULL;
            break;
        case Family::CapHash:
        {
            double norm = 0.0;
            for ( int d = 0; d < 3; ++d )
                norm += mid[e][d] * mid[e][d];
            norm = std::sqrt( norm );
            const bool inCap =
                norm > 0.0 && mid[e][2] / norm > std::cos( kCapHalfAngle );
            take = inCap && ( posHash( mid[e] ) % 3ULL ) == 0ULL;
            break;
        }
        }
        mask[e] = take ? 1 : 0;
    }
    return mask;
}

// ---------------------------------------------------------------------------
// One family, driven to depth
// ---------------------------------------------------------------------------

//! Returns local fails. Prints one line per round and a verdict line.
template <class MeshT, class Exec>
static int driveFamily( int rank, const char* tag, Family fam, int maxRounds,
                        long long faceBudget )
{
    using mem = typename Exec::memory_space;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    buildIcosphere( mesh, 2 );
    auto faceOwner = facePartitionByAxis( mesh );
    distribute( mesh, halo, faceOwner );

    int fails = 0;
    std::vector<double> rr; // min r/R per round
    double worstEver = 1.0;
    int worstRound = 0;
    int rounds = 0;

    for ( int round = 1; round <= maxRounds; ++round )
    {
        const std::vector<char> mask = buildMask( mesh, fam, fails );
        const SplitResult res = splitEdges( mesh, halo, mask );
        if ( res.requested <= 0 )
        {
            if ( rank == 0 )
                std::printf( "  [%s] %s round%2d: mask empty -- family "
                             "exhausted\n",
                             tag, familyName( fam ), round );
            break;
        }

        fails += TesseraTest::checkConforming( mesh );
        fails += TesseraTest::owned1RingLocal( mesh );
        if ( TesseraTest::checkOwnedEuler( mesh ) != 2 )
            ++fails;

        const Shape s = measure( mesh, fails );
        rr.push_back( s.minRR );
        if ( s.minRR < worstEver )
        {
            worstEver = s.minRR;
            worstRound = round;
        }
        rounds = round;

        if ( rank == 0 )
            std::printf( "  [%s] %s round%2d: split=%-9lld F=%-9lld "
                         "minRR=%.4f minAng=%6.3f  tail<0.30/.25/.20/.15/.10 "
                         "= %lld/%lld/%lld/%lld/%lld\n",
                         tag, familyName( fam ), round, res.split, s.faces,
                         s.minRR, s.minAngleDeg, s.tail[0], s.tail[1],
                         s.tail[2], s.tail[3], s.tail[4] );
        std::fflush( stdout );

        if ( s.faces > faceBudget )
        {
            if ( rank == 0 )
                std::printf( "  [%s] %s stopping: F=%lld exceeds budget %lld\n",
                             tag, familyName( fam ), s.faces, faceBudget );
            break;
        }
    }

    // Verdict: where the worst was reached, and how many rounds have run since
    // WITHOUT it moving. A worst first reached in the last round is the
    // signature that cannot be distinguished from unbounded decline; a worst
    // reached early and flat since is saturation.
    if ( rank == 0 && rounds > 0 )
    {
        int flatSince = rounds - worstRound;
        double firstRR = rr.front(), lastRR = rr.back();
        std::printf( "  [%s] %s VERDICT: %d rounds, worst minRR=%.4f first "
                     "reached at round %d, flat for the %d rounds since; "
                     "r1=%.4f rN=%.4f\n",
                     tag, familyName( fam ), rounds, worstEver, worstRound,
                     flatSince, firstRR, lastRR );
        std::fflush( stdout );
    }
    return fails;
}

// ===========================================================================

template <class Exec>
static int run( int rank, const char* tag, int maxRounds, long long faceBudget )
{
    using mem = typename Exec::memory_space;
    using MeshT =
        Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>, mem, Exec>;

    int local = 0;
    local += driveFamily<MeshT, Exec>( rank, tag, Family::AboveMean, maxRounds,
                                       faceBudget );
    local += driveFamily<MeshT, Exec>( rank, tag, Family::BelowMean, maxRounds,
                                       faceBudget );
    local += driveFamily<MeshT, Exec>( rank, tag, Family::HashThird, maxRounds,
                                       faceBudget );
    local += driveFamily<MeshT, Exec>( rank, tag, Family::CapHash, maxRounds,
                                       faceBudget );
    std::fflush( stdout );
    int glob = 0;
    MPI_Allreduce( &local, &glob, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    return glob == 0 ? 0 : 1;
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
        const int maxRounds = envInt( "TESSERA_SPLIT_DEPTH_ROUNDS", 30 );
        const long long faceBudget =
            envInt( "TESSERA_SPLIT_DEPTH_FACES", 2000000 );
        if ( rank == 0 )
            std::printf( "test_split_edges_depth: splitEdges shape quality "
                         "over depth (size %d, maxRounds %d, faceBudget "
                         "%lld)\n",
                         size, maxRounds, faceBudget );
        std::fflush( stdout );

        fails += run<Kokkos::Serial>( rank, "Serial", maxRounds, faceBudget );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            fails += run<Kokkos::DefaultExecutionSpace>( rank, "Default",
                                                        maxRounds, faceBudget );
    }

    Kokkos::finalize();

    int global = 0;
    MPI_Allreduce( &fails, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    MPI_Finalize();
    return global == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
