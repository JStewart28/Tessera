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

// Regression test: the lat/lon (UV) sphere generator, tasks/completed/latlon-sphere.md.
//
// generateLatLonSphere()/buildLatLonSphere() exist because an icosphere is *too
// good* a test surface -- nearly isotropic, nearly equilateral triangles, almost
// every vertex of valence 6. A lat/lon sphere is anisotropic by construction and
// its two poles have valence nLon, so it reaches code paths the icosphere never
// does. Half of this test is therefore about the generator and half is about the
// REST of the library not breaking on that input.
//
// Every parameter-pair check runs over (3,3), (3,8), (5,4), (9,12) and (4,17):
// the nLat == 3 degenerate bipyramid (two pole fans, no interior quads) is the
// boundary of the closed-form counts, and the prime nLon = 17 keeps any check
// from accidentally relying on divisibility.
//
// The four classic UV-sphere bugs each have a dedicated check:
//   #2 manifoldness      catches the duplicated seam meridian (nLon+1 columns),
//                        loudly: the seam edges end up with one incident face.
//   #3 no duplicates     catches nLon coincident copies of a pole, which #2 can
//                        miss when the duplicates happen to pair up.
//   #4 exact poles       pins the (0,0,+-1)-written-literally decision BITWISE;
//                        sin(pi) != 0, so a computed south pole is off the
//                        sphere in the last bits AND phi-dependent.
//   #5 outward winding   a sign error inverts every normal downstream and
//                        nothing else in the suite would catch it.
//
// Registered at both SERIAL and HIP over ranks 1-5; gate promotion is
// pre-authorized by tasks/completed/latlon-sphere.md.

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
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace Tessera;

namespace
{

constexpr double kPi = 3.14159265358979323846;

struct Params
{
    int nLat;
    int nLon;
};

// nLat == 3 is the degenerate-but-legal bipyramid; 17 is prime.
const Params kParams[] = { { 3, 3 }, { 3, 8 }, { 5, 4 }, { 9, 12 }, { 4, 17 } };

// ---------------------------------------------------------------------------
// Closed forms -- computed here, never read back from Tessera.
// ---------------------------------------------------------------------------

long long expectV( int nLat, int nLon )
{
    return 2LL + static_cast<long long>( nLat - 2 ) * nLon;
}
long long expectF( int nLat, int nLon ) { return 2LL * nLon * ( nLat - 2 ); }
long long expectE( int nLat, int nLon ) { return 3LL * nLon * ( nLat - 2 ); }

//! Valence histogram, derived from the FIXED quad-diagonal convention rather
//! than hardcoded. Each interior quad (i,j)-(i+1,j)-(i+1,j+1)-(i,j+1) is cut
//! along (i,j+1)-(i+1,j), so between two rings it contributes one "vertical"
//! edge (i,j)-(i,j+1) and one diagonal (i,j+1)-(i+1,j). A ring vertex therefore
//! has exactly two edges to the ring above and two to the ring below, plus its
//! two in-ring neighbours:
//!   pole                        -> nLon
//!   ring 1 == ring nLat-2       -> 2 in-ring + 1 to N + 1 to S      = 4
//!   ring 1 or ring nLat-2       -> 2 in-ring + 1 to a pole + 2       = 5
//!   any other ring              -> 2 in-ring + 2 + 2                 = 6
//! Buckets ADD, so nLon in {4,5,6} correctly merges the pole bucket into a
//! ring one.
std::map<int, long long> expectValenceHistogram( int nLat, int nLon )
{
    std::map<int, long long> h;
    h[nLon] += 2; // the two poles
    if ( nLat == 3 )
        h[4] += nLon; // the single ring is adjacent to BOTH poles
    else
    {
        h[5] += 2LL * nLon; // rings 1 and nLat-2
        if ( nLat > 4 )
            h[6] += static_cast<long long>( nLat - 4 ) * nLon;
    }
    return h;
}

//! Exact enclosed volume of the lat/lon POLYHEDRON (not of the sphere).
//!
//! Summing the signed tetrahedron volumes collapses in closed form. With
//! r_j = sin(theta_j), z_j = cos(theta_j), dphi = 2pi/nLon, dtheta = pi/(nLat-1):
//! a pole-fan triangle contributes r_1^2 sin(dphi)/6, and an interior quad's two
//! triangles contribute sin(dphi)(r_j + r_{j+1})(z_j r_{j+1} - z_{j+1} r_j)/6
//! where z_j r_{j+1} - z_{j+1} r_j == sin(dtheta) identically. Since
//! r_1 == r_{nLat-2} == sin(dtheta), the telescoping sum is just
//!
//!   V = (nLon sin(dphi) / 6) * 2 sin(dtheta) * SUM_j sin(theta_j).
//!
//! Checking the measured volume against this is much stronger than a percentage
//! bound: it pins the winding, the diagonal, and the pole fans simultaneously.
//! Sanity: (3,4) is the regular octahedron and the formula gives 4/3.
double expectPolyVolume( int nLat, int nLon )
{
    const double dth = kPi / static_cast<double>( nLat - 1 );
    double s = 0.0;
    for ( int j = 1; j <= nLat - 2; ++j )
        s += std::sin( dth * static_cast<double>( j ) );
    return ( static_cast<double>( nLon ) *
             std::sin( 2.0 * kPi / static_cast<double>( nLon ) ) / 6.0 ) *
           2.0 * std::sin( dth ) * s;
}

// ---------------------------------------------------------------------------
// Soup-level geometry helpers (pure host arithmetic on the generator output).
// ---------------------------------------------------------------------------

struct Vec3
{
    double x, y, z;
};

Vec3 vertexOf( const TriangleSoup<double>& s, int i )
{
    return { s.positions[3 * i + 0], s.positions[3 * i + 1],
             s.positions[3 * i + 2] };
}

Vec3 cross( const Vec3& a, const Vec3& b )
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
             a.x * b.y - a.y * b.x };
}
double dot( const Vec3& a, const Vec3& b )
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
Vec3 sub( const Vec3& a, const Vec3& b )
{
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}
double norm( const Vec3& a ) { return std::sqrt( dot( a, a ) ); }

//! Enclosed volume as the sum of signed tetrahedron volumes to the origin.
double soupVolume( const TriangleSoup<double>& s )
{
    double v = 0.0;
    for ( std::size_t f = 0; f < s.numFaces(); ++f )
    {
        const Vec3 p0 = vertexOf( s, s.triangles[3 * f + 0] );
        const Vec3 p1 = vertexOf( s, s.triangles[3 * f + 1] );
        const Vec3 p2 = vertexOf( s, s.triangles[3 * f + 2] );
        v += dot( p0, cross( p1, p2 ) ) / 6.0;
    }
    return v;
}

double faceAreaOf( const TriangleSoup<double>& s, std::size_t f )
{
    const Vec3 p0 = vertexOf( s, s.triangles[3 * f + 0] );
    const Vec3 p1 = vertexOf( s, s.triangles[3 * f + 1] );
    const Vec3 p2 = vertexOf( s, s.triangles[3 * f + 2] );
    return 0.5 * norm( cross( sub( p1, p0 ), sub( p2, p0 ) ) );
}

//! faceNormalRaw . centroid, the outward-winding test statistic.
double outwardness( const TriangleSoup<double>& s, std::size_t f )
{
    const Vec3 p0 = vertexOf( s, s.triangles[3 * f + 0] );
    const Vec3 p1 = vertexOf( s, s.triangles[3 * f + 1] );
    const Vec3 p2 = vertexOf( s, s.triangles[3 * f + 2] );
    const Vec3 n = cross( sub( p1, p0 ), sub( p2, p0 ) );
    const Vec3 c = { ( p0.x + p1.x + p2.x ) / 3.0, ( p0.y + p1.y + p2.y ) / 3.0,
                     ( p0.z + p1.z + p2.z ) / 3.0 };
    return dot( n, c );
}

// ---------------------------------------------------------------------------
// Host mirrors of a (replicated, pre-distribute) mesh.
// ---------------------------------------------------------------------------

template <class MeshT>
Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace>
hostVertices( const MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> h(
        "hv", mesh.numVertices() );
    Cabana::deep_copy( h, mesh.vertices() );
    return h;
}

template <class MeshT>
Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace>
hostEdges( const MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> h(
        "he", mesh.numEdges() );
    Cabana::deep_copy( h, mesh.edges() );
    return h;
}

//! Row lengths of a CSR adjacency, host-side.
template <class Csr>
std::vector<int> csrRowLengths( const Csr& csr, std::size_t n )
{
    auto off =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), csr.offsets );
    std::vector<int> len( n );
    for ( std::size_t i = 0; i < n; ++i )
        len[i] = off( i + 1 ) - off( i );
    return len;
}

inline int globalFails( MPI_Comm comm, int local )
{
    int g = 0;
    MPI_Allreduce( &local, &g, 1, MPI_INT, MPI_SUM, comm );
    return g;
}

// ---------------------------------------------------------------------------
// Check 10 helper: every GHOST vertex position equals its OWNER's, verified
// through the same gid coordinator (gid % size) the rest of the suite uses --
// so the verdict never reads the halo's own bookkeeping back to itself.
// Returns LOCAL fails (sum across ranks == global).
// ---------------------------------------------------------------------------
template <class MeshT>
int checkGhostPositionsMatchOwners( MeshT& mesh )
{
    MPI_Comm comm = mesh.comm();
    const int size = mesh.commSize();
    constexpr int Dim = MeshT::dim;

    struct PosMsg
    {
        GlobalId gid;
        double p[3];
        unsigned char owned;
    };

    auto hv = hostVertices( mesh );
    auto vg = Cabana::slice<VertexField::Gid>( hv );
    auto vp = Cabana::slice<VertexField::Position>( hv );

    std::vector<std::vector<PosMsg>> send( size );
    for ( std::size_t i = 0; i < mesh.numVertices(); ++i )
    {
        PosMsg m;
        m.gid = vg( i );
        for ( int d = 0; d < 3; ++d )
            m.p[d] = ( d < Dim ) ? static_cast<double>( vp( i, d ) ) : 0.0;
        m.owned = ( i < mesh.numOwnedVertices() ) ? 1 : 0;
        send[m.gid % size].push_back( m );
    }
    auto got = allToAllV( comm, send );

    std::map<GlobalId, std::array<double, 3>> ownerPos;
    for ( const auto& m : got.data )
        if ( m.owned )
            ownerPos[m.gid] = { m.p[0], m.p[1], m.p[2] };

    int fails = 0;
    for ( const auto& m : got.data )
    {
        if ( m.owned )
            continue;
        auto it = ownerPos.find( m.gid );
        if ( it == ownerPos.end() )
        {
            ++fails; // a ghost whose gid no rank owns
            continue;
        }
        for ( int d = 0; d < 3; ++d )
            if ( it->second[d] != m.p[d] )
                ++fails; // ghost position != owner's, bitwise
    }
    return fails;
}

// ---------------------------------------------------------------------------
// Checks 1-9: the generator itself, per parameter pair.
// ---------------------------------------------------------------------------
template <class Exec>
int runParams( int rank, int size, const char* tag, const Params& p )
{
    using mem = typename Exec::memory_space;
    using MeshT =
        Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>, mem, Exec>;

    const int nLat = p.nLat;
    const int nLon = p.nLon;
    const long long V = expectV( nLat, nLon );
    const long long E = expectE( nLat, nLon );
    const long long F = expectF( nLat, nLon );

    int fails = 0;
    auto check = [&fails]( bool ok )
    {
        if ( !ok )
            ++fails;
    };

    // ---- 1a. Closed-form counts on the replicated (pre-distribute) mesh ----
    // buildLatLonSphere() replicates the whole mesh on every rank, so these
    // local counts are the global ones and every rank checks the same thing.
    MeshT mesh( MPI_COMM_WORLD );
    buildLatLonSphere( mesh, nLat, nLon );

    check( static_cast<long long>( mesh.numVertices() ) == V );
    check( static_cast<long long>( mesh.numEdges() ) == E );
    check( static_cast<long long>( mesh.numFaces() ) == F );
    check( V - E + F == 2 ); // arithmetic self-check on the formulas

    const TriangleSoup<double> soup =
        generateLatLonSphere<double>( nLat, nLon );
    check( static_cast<long long>( soup.numVertices() ) == V );
    check( static_cast<long long>( soup.numFaces() ) == F );

    // ---- 2. Manifoldness: every edge has exactly two incident faces --------
    // THE seam-duplicate detector. Generating nLon+1 meridians leaves a
    // coincident seam ring whose edges each carry a single incidence.
    {
        auto he = hostEdges( mesh );
        auto ef = Cabana::slice<EdgeField::Faces>( he );
        long long oneSided = 0;
        for ( std::size_t e = 0; e < mesh.numEdges(); ++e )
            if ( ef( e, 0 ) == invalid_gid || ef( e, 1 ) == invalid_gid )
                ++oneSided;
        check( oneSided == 0 );
    }

    // ---- 3. No duplicate vertices (the duplicated-pole detector) -----------
    {
        long long dup = 0;
        for ( std::size_t a = 0; a < soup.numVertices(); ++a )
            for ( std::size_t b = a + 1; b < soup.numVertices(); ++b )
                if ( norm( sub( vertexOf( soup, static_cast<int>( a ) ),
                                vertexOf( soup, static_cast<int>( b ) ) ) ) <
                     1e-14 )
                    ++dup;
        check( dup == 0 );
    }

    // ---- 4. On the unit sphere; the poles EXACTLY (0,0,+-1) bitwise --------
    {
        double worst = 0.0;
        for ( std::size_t i = 0; i < soup.numVertices(); ++i )
            worst = std::max(
                worst,
                std::abs( norm( vertexOf( soup, static_cast<int>( i ) ) ) -
                          1.0 ) );
        check( worst <= 1e-15 );

        const Vec3 north = vertexOf( soup, 0 );
        const Vec3 south =
            vertexOf( soup, static_cast<int>( soup.numVertices() ) - 1 );
        check( north.x == 0.0 && north.y == 0.0 && north.z == 1.0 );
        check( south.x == 0.0 && south.y == 0.0 && south.z == -1.0 );

        // The same two vertices in the BUILT mesh, so the builder is not
        // quietly perturbing them either.
        auto hv = hostVertices( mesh );
        auto vp = Cabana::slice<VertexField::Position>( hv );
        check( vp( 0, 0 ) == 0.0 && vp( 0, 1 ) == 0.0 && vp( 0, 2 ) == 1.0 );
        const std::size_t s = mesh.numVertices() - 1;
        check( vp( s, 0 ) == 0.0 && vp( s, 1 ) == 0.0 && vp( s, 2 ) == -1.0 );
    }

    // ---- 5. Outward winding, and a positive enclosed volume ---------------
    {
        long long inward = 0;
        for ( std::size_t f = 0; f < soup.numFaces(); ++f )
            if ( outwardness( soup, f ) <= 0.0 )
                ++inward;
        check( inward == 0 );

        const double vol = soupVolume( soup );
        const double want = expectPolyVolume( nLat, nLon );
        check( vol > 0.0 );
        check( std::abs( vol - want ) <= 1e-12 * want );
        // An inscribed polyhedron: strictly inside the sphere it interpolates.
        check( vol < 4.0 * kPi / 3.0 );
    }

    // ---- 6. Pole fan structure --------------------------------------------
    // Exactly nLon faces incident on each pole, and each pole has valence nLon.
    // gid == local index on the replicated mesh, so the poles are rows 0 and
    // numVertices()-1.
    const std::vector<int> vfLen =
        csrRowLengths( mesh.vertexFaces(), mesh.numVertices() );
    const std::vector<int> veLen =
        csrRowLengths( mesh.vertexEdges(), mesh.numVertices() );
    {
        const std::size_t s = mesh.numVertices() - 1;
        check( vfLen[0] == nLon );
        check( vfLen[s] == nLon );
        check( veLen[0] == nLon );
        check( veLen[s] == nLon );
    }

    // ---- 7. Valence histogram matches the closed form ---------------------
    {
        std::map<int, long long> got;
        for ( std::size_t i = 0; i < mesh.numVertices(); ++i )
            ++got[veLen[i]];
        check( got == expectValenceHistogram( nLat, nLon ) );

        // No interior (non-pole) vertex drops below 4 -- the nLat == 3 floor.
        int worst = 1 << 30;
        for ( std::size_t i = 1; i + 1 < mesh.numVertices(); ++i )
            worst = std::min( worst, veLen[i] );
        if ( mesh.numVertices() > 2 )
            check( worst >= 4 );

        // A closed triangle mesh: sum of valences == 2E, and 3F == 2E.
        long long sumVal = 0;
        for ( int l : veLen )
            sumVal += l;
        check( sumVal == 2 * E );
        check( 3 * F == 2 * E );
    }

    // ---- 1b. Distributed counts and Euler at the actual rank count --------
    {
        auto faceOwner = facePartitionByAxis( mesh );
        MeshHalo<mem> halo;
        distribute( mesh, halo, faceOwner );
        haloExchange( mesh, halo );

        check( globalOwnedVertices( mesh ) == V );
        check( globalOwnedEdges( mesh ) == E );
        check( globalOwnedFaces( mesh ) == F );
        check( globalOwnedEuler( mesh ) == 2 );

        fails += globalFails( mesh.comm(), TesseraTest::checkOwnershipPartition(
                                               mesh, V, E, F ) );
        fails +=
            globalFails( mesh.comm(), TesseraTest::owned1RingLocal( mesh ) );
        fails +=
            globalFails( mesh.comm(), TesseraTest::checkConforming( mesh ) );
    }

    if ( rank == 0 )
        std::printf( "  [%s] (nLat=%2d,nLon=%2d) V=%4lld E=%4lld F=%4lld "
                     "vol=%.9f (%.4f%% below 4pi/3) %s\n",
                     tag, nLat, nLon, V, E, F, soupVolume( soup ),
                     100.0 * ( 1.0 - soupVolume( soup ) / ( 4.0 * kPi / 3.0 ) ),
                     fails == 0 ? "ok" : "FAIL" );
    (void)size;
    return fails;
}

// ---------------------------------------------------------------------------
// Check 5 (continued) + check 8: volume convergence and the anisotropy that is
// the whole reason this surface is worth having.
// ---------------------------------------------------------------------------
int runAnisotropy( int rank )
{
    int fails = 0;
    auto check = [&fails]( bool ok )
    {
        if ( !ok )
            ++fails;
    };

    // Volume converges to 4pi/3 FROM BELOW as the resolution rises.
    const double target = 4.0 * kPi / 3.0;
    const double vCoarse = soupVolume( generateLatLonSphere<double>( 9, 12 ) );
    const double vFine = soupVolume( generateLatLonSphere<double>( 33, 64 ) );
    const double defCoarse = 1.0 - vCoarse / target;
    const double defFine = 1.0 - vFine / target;

    check( vCoarse > 0.0 && vFine > 0.0 );
    check( defCoarse > 0.0 && defFine > 0.0 ); // from below
    check( defFine < defCoarse );              // and converging
    // NOTE ON TOLERANCES. tasks/completed/latlon-sphere.md asked for (9,12) "within 5%".
    // That figure is arithmetically unreachable: the closed form above gives
    // V(9,12) = 3.84812443 against 4pi/3 = 4.18879020, an 8.13% deficit, and
    // an inscribed polyhedron cannot do better than its own vertices. The 5%
    // was a mis-estimate of the O(h^2) constant, not a defect, so the bound is
    // stated at the true value here rather than the check being dropped. The
    // exact closed-form identity asserted per parameter pair in check 5 is a
    // far stronger statement than either percentage anyway. (33,64) does meet
    // its 0.5%: the measured deficit is 0.406%.
    check( defCoarse < 0.10 );
    check( defFine < 0.005 );

    // Anisotropy at (33,64): triangles stretch toward the poles, so the
    // max/min area ratio must be well above 1. If it is not, the generator is
    // not producing the pole stretching this surface exists to exercise.
    const TriangleSoup<double> s = generateLatLonSphere<double>( 33, 64 );
    double aMin = 1e300, aMax = 0.0;
    for ( std::size_t f = 0; f < s.numFaces(); ++f )
    {
        const double a = faceAreaOf( s, f );
        aMin = std::min( aMin, a );
        aMax = std::max( aMax, a );
    }
    std::set<std::pair<int, int>> edges;
    for ( std::size_t f = 0; f < s.numFaces(); ++f )
        for ( int k = 0; k < 3; ++k )
        {
            const int a = s.triangles[3 * f + k];
            const int b = s.triangles[3 * f + ( k + 1 ) % 3];
            edges.insert( a < b ? std::make_pair( a, b )
                                : std::make_pair( b, a ) );
        }
    double lMin = 1e300, lMax = 0.0;
    for ( const auto& e : edges )
    {
        const double l =
            norm( sub( vertexOf( s, e.first ), vertexOf( s, e.second ) ) );
        lMin = std::min( lMin, l );
        lMax = std::max( lMax, l );
    }

    const double areaRatio = aMax / aMin;
    const double lenRatio = lMax / lMin;
    check( areaRatio > 10.0 );

    if ( rank == 0 )
        std::printf( "  [anisotropy] (33,64) areaRatio=%.4f lenRatio=%.4f "
                     "areaMin=%.6e areaMax=%.6e lenMin=%.6e lenMax=%.6e\n"
                     "  [volume]     V(9,12)=%.9f deficit=%.4f%%  "
                     "V(33,64)=%.9f deficit=%.4f%%\n",
                     areaRatio, lenRatio, aMin, aMax, lMin, lMax, vCoarse,
                     100.0 * defCoarse, vFine, 100.0 * defFine );
    return fails;
}

// ---------------------------------------------------------------------------
// Check 9: degenerate arguments throw, each polarity.
// ---------------------------------------------------------------------------
int runThrows( int rank )
{
    int fails = 0;
    auto throwsInvalid = [&fails]( int nLat, int nLon )
    {
        bool threw = false;
        try
        {
            (void)generateLatLonSphere<double>( nLat, nLon );
        }
        catch ( const std::invalid_argument& )
        {
            threw = true;
        }
        if ( !threw )
            ++fails;
    };

    // nLat too small, at several magnitudes and both signs.
    throwsInvalid( 2, 8 );
    throwsInvalid( 1, 8 );
    throwsInvalid( 0, 8 );
    throwsInvalid( -3, 8 );
    // nLon too small, likewise.
    throwsInvalid( 8, 2 );
    throwsInvalid( 8, 1 );
    throwsInvalid( 8, 0 );
    throwsInvalid( 8, -3 );
    // Both bad at once.
    throwsInvalid( 2, 2 );

    // And the boundary is INCLUSIVE: (3,3) must not throw.
    try
    {
        const auto s = generateLatLonSphere<double>( 3, 3 );
        if ( s.numVertices() != 5 || s.numFaces() != 6 )
            ++fails;
    }
    catch ( ... )
    {
        ++fails;
    }

    if ( rank == 0 )
        std::printf( "  [throws] degenerate arguments: %s\n",
                     fails == 0 ? "ok" : "FAIL" );
    return fails;
}

// ---------------------------------------------------------------------------
// Check 10: the REST of the library on an anisotropic, high-valence-pole mesh.
// refine() / migrate() / loadBalance() / haloExchange() / I/O round trip.
// ---------------------------------------------------------------------------
template <class Exec>
int runDownstream( int rank, int size, const char* tag,
                   const std::string& stem )
{
    using mem = typename Exec::memory_space;
    using MeshT =
        Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>, mem, Exec>;

    const int nLat = 9, nLon = 12;
    const long long V0 = expectV( nLat, nLon );
    const long long E0 = expectE( nLat, nLon );
    const long long F0 = expectF( nLat, nLon );

    int fails = 0;
    auto check = [&fails]( bool ok )
    {
        if ( !ok )
            ++fails;
    };

    MeshT mesh( MPI_COMM_WORLD );
    buildLatLonSphere( mesh, nLat, nLon );
    auto faceOwner = facePartitionByAxis( mesh );
    MeshHalo<mem> halo;
    distribute( mesh, halo, faceOwner );
    haloExchange( mesh, halo );
    MPI_Comm comm = mesh.comm();

    fails += globalFails( comm, checkGhostPositionsMatchOwners( mesh ) );

    // ---- uniform refine(): V+E / 2E+3F / 4F, and every refine invariant ----
    {
        std::vector<char> mask( mesh.numOwnedFaces(), 1 );
        const RefineResult res = refine( mesh, halo, mask );

        check( globalOwnedVertices( mesh ) == V0 + E0 );
        check( globalOwnedEdges( mesh ) == 2 * E0 + 3 * F0 );
        check( globalOwnedFaces( mesh ) == 4 * F0 );
        check( globalOwnedEuler( mesh ) == 2 );

        fails += globalFails( comm, TesseraTest::checkConforming( mesh ) );
        fails += globalFails( comm, TesseraTest::check21BalanceRed( mesh ) );
        fails += globalFails( comm, TesseraTest::checkMidpointAgreement(
                                        comm, size, res.midpoints ) );
        fails += globalFails( comm, TesseraTest::owned1RingLocal( mesh ) );
        fails +=
            globalFails( comm, TesseraTest::checkOwnershipPartition(
                                   mesh, V0 + E0, 2 * E0 + 3 * F0, 4 * F0 ) );
        fails += globalFails( comm, checkGhostPositionsMatchOwners( mesh ) );
    }

    unsigned long long cv0 = 0, ce0 = 0, cf0 = 0;
    TesseraTest::topologyChecksum( mesh, cv0, ce0, cf0 );
    const long long Vr = globalOwnedVertices( mesh );
    const long long Er = globalOwnedEdges( mesh );
    const long long Fr = globalOwnedFaces( mesh );

    // ---- migrate(): shift every face one rank to the right -----------------
    {
        std::vector<Rank> dest( mesh.numOwnedFaces(),
                                static_cast<Rank>( ( rank + 1 ) % size ) );
        migrate( mesh, halo, dest );
        haloExchange( mesh, halo );

        check( globalOwnedVertices( mesh ) == Vr );
        check( globalOwnedEdges( mesh ) == Er );
        check( globalOwnedFaces( mesh ) == Fr );
        check( globalOwnedEuler( mesh ) == 2 );
        unsigned long long cv = 0, ce = 0, cf = 0;
        TesseraTest::topologyChecksum( mesh, cv, ce, cf );
        check( cv == cv0 && ce == ce0 && cf == cf0 );
        fails += globalFails( comm, TesseraTest::owned1RingLocal( mesh ) );
        fails += globalFails( comm, checkGhostPositionsMatchOwners( mesh ) );
    }

    // ---- loadBalance(): Zoltan2 on a mesh whose poles are valence outliers -
    {
        loadBalance( mesh, halo );
        haloExchange( mesh, halo );

        check( globalOwnedVertices( mesh ) == Vr );
        check( globalOwnedEdges( mesh ) == Er );
        check( globalOwnedFaces( mesh ) == Fr );
        check( globalOwnedEuler( mesh ) == 2 );
        unsigned long long cv = 0, ce = 0, cf = 0;
        TesseraTest::topologyChecksum( mesh, cv, ce, cf );
        check( cv == cv0 && ce == ce0 && cf == cf0 );
        fails += globalFails( comm, TesseraTest::owned1RingLocal( mesh ) );
        fails +=
            globalFails( comm, TesseraTest::checkSiblingCoresidency( mesh ) );
        fails += globalFails( comm, checkGhostPositionsMatchOwners( mesh ) );
    }

    // ---- writeMesh / readMesh round trip -----------------------------------
    {
        writeMesh( mesh, stem );
        MeshT mesh2( MPI_COMM_WORLD );
        MeshHalo<mem> halo2;
        readMesh( mesh2, halo2, stem );
        haloExchange( mesh2, halo2 );

        check( globalOwnedVertices( mesh2 ) == Vr );
        check( globalOwnedEdges( mesh2 ) == Er );
        check( globalOwnedFaces( mesh2 ) == Fr );
        check( globalOwnedEuler( mesh2 ) == 2 );
        unsigned long long cv = 0, ce = 0, cf = 0;
        TesseraTest::topologyChecksum( mesh2, cv, ce, cf );
        check( cv == cv0 && ce == ce0 && cf == cf0 );
        fails += globalFails( comm, TesseraTest::checkConforming( mesh2 ) );
        fails += globalFails( comm, TesseraTest::owned1RingLocal( mesh2 ) );
        fails += globalFails( comm, checkGhostPositionsMatchOwners( mesh2 ) );

        if ( rank == 0 )
        {
            std::remove( ( stem + ".h5" ).c_str() );
            std::remove( ( stem + ".xmf" ).c_str() );
        }
    }

    if ( rank == 0 )
        std::printf( "  [%s] downstream (9,12): refined V=%lld E=%lld F=%lld "
                     "%s\n",
                     tag, Vr, Er, Fr, fails == 0 ? "ok" : "FAIL" );
    return fails;
}

// ---------------------------------------------------------------------------
// Check 11: markByQuality on a DELIBERATELY very anisotropic surface, (33,8).
//
// tasks/completed/latlon-sphere.md predicted "marks the polar bands and not the
// equatorial ones". The prediction is INVERTED, and the geometry says so
// unambiguously rather than the library misbehaving. At (33,8) the meridional
// step is dtheta = pi/32 (chord 0.0981, latitude-independent) while the in-ring
// step is dphi = 2pi/8 = 45 degrees, giving a ring-edge chord of
// 2 sin(theta_j) sin(pi/8) = 0.7654 sin(theta_j). So the ring edges -- and with
// them the triangle areas -- are LONGEST at the equator and shrink to nothing
// at the poles: at (33,8) the polar triangles are the small, nearly ISOTROPIC
// ones and the equatorial ones are the 7.8:1-stretched ones. EdgeLengthCriterion
// marks long edges, so it marks the equator. That is its documented contract
// applied to this surface, not a defect; the point of the check is that the
// marked set is exactly the closed-form band, which is a genuine test of the
// criterion on input the icosphere cannot produce.
//
// The band cut points are closed-form, not measured:
//   maxLen = 0.4  ->  a ring edge exceeds it iff sin(theta_j) > 0.52262,
//                     i.e. rings j = 6 .. 26 of 32.
//   POLAR CAP   all corners with |z| >= cos(28.125 deg) = 0.88192 (rings 0..5
//               and 27..32): the longest edge available is the j=5 ring edge
//               (0.3609) or the j=4/5 quad diagonal (0.3393). Must be UNMARKED.
//   EQUATORIAL  all corners with |z| <= cos(56.25 deg) = 0.55557 (rings 10..22):
//               every such face has an in-ring edge of at least 0.6363, since
//               each of the two children of a quad carries exactly one.
//               Must be MARKED.
// ---------------------------------------------------------------------------
template <class Exec>
int runMarkQuality( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT =
        Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>, mem, Exec>;

    int fails = 0;
    auto check = [&fails]( bool ok )
    {
        if ( !ok )
            ++fails;
    };

    MeshT mesh( MPI_COMM_WORLD );
    buildLatLonSphere( mesh, 33, 8 );
    auto faceOwner = facePartitionByAxis( mesh );
    MeshHalo<mem> halo;
    distribute( mesh, halo, faceOwner );
    haloExchange( mesh, halo );
    MPI_Comm comm = mesh.comm();

    const double kPolarZ = std::cos( 28.125 * kPi / 180.0 ); // 0.88192
    const double kEquatZ = std::cos( 56.25 * kPi / 180.0 );  // 0.55557

    // Per-owned-face min/max |z| over its corners, host-side from the local
    // positions (every owned face's corners are local -- the 1-ring invariant).
    const std::size_t nof = mesh.numOwnedFaces();
    std::vector<double> minAbsZ( nof, 1e300 ), maxAbsZ( nof, 0.0 );
    {
        auto hv = hostVertices( mesh );
        auto vg = Cabana::slice<VertexField::Gid>( hv );
        auto vp = Cabana::slice<VertexField::Position>( hv );
        std::map<GlobalId, std::size_t> lv;
        for ( std::size_t i = 0; i < mesh.numVertices(); ++i )
            lv.emplace( vg( i ), i );

        Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
            "hf", mesh.numFaces() );
        Cabana::deep_copy( hf, mesh.faces() );
        auto fv = Cabana::slice<FaceField::Verts>( hf );
        for ( std::size_t f = 0; f < nof; ++f )
            for ( int k = 0; k < 3; ++k )
            {
                auto it = lv.find( fv( f, k ) );
                if ( it == lv.end() )
                {
                    ++fails; // an owned face naming a non-local vertex
                    continue;
                }
                const double z =
                    std::abs( static_cast<double>( vp( it->second, 2 ) ) );
                minAbsZ[f] = std::min( minAbsZ[f], z );
                maxAbsZ[f] = std::max( maxAbsZ[f], z );
            }
    }

    // ---- EdgeLengthCriterion: the closed-form band, both directions --------
    {
        const std::vector<char> mask =
            markByQuality( mesh, EdgeLengthCriterion<double>{ 0.4 } );
        check( mask.size() == nof );

        long long nPolar = 0, nEquat = 0, badPolar = 0, badEquat = 0,
                  nMarked = 0;
        for ( std::size_t f = 0; f < nof && f < mask.size(); ++f )
        {
            if ( mask[f] )
                ++nMarked;
            if ( minAbsZ[f] >= kPolarZ )
            {
                ++nPolar;
                if ( mask[f] )
                    ++badPolar; // a polar face was marked
            }
            if ( maxAbsZ[f] <= kEquatZ )
            {
                ++nEquat;
                if ( !mask[f] )
                    ++badEquat; // an equatorial face was NOT marked
            }
        }
        long long loc[5] = { nPolar, nEquat, badPolar, badEquat, nMarked };
        long long glb[5] = { 0, 0, 0, 0, 0 };
        MPI_Allreduce( loc, glb, 5, MPI_LONG_LONG, MPI_SUM, comm );

        check( glb[2] == 0 ); // no polar face marked
        check( glb[3] == 0 ); // every equatorial face marked
        check( glb[0] > 0 );  // non-vacuous: polar faces exist
        check( glb[1] > 0 );  // non-vacuous: equatorial faces exist
        // And the marked set is a strict, non-trivial subset -- neither the
        // all-marked nor the empty mask would fail the two band checks alone.
        check( glb[4] > glb[1] );
        check( glb[4] < 2LL * 8 * 31 );

        if ( rank == 0 )
            std::printf( "  [%s] markByQuality(edgeLen=0.4) on (33,8): "
                         "marked %lld/%lld faces; polarBand=%lld (marked %lld) "
                         "equatBand=%lld (unmarked %lld)\n",
                         tag, glb[4], 2LL * 8 * 31, glb[0], glb[2], glb[1],
                         glb[3] );
    }

    // ---- CurvatureCriterion: characterized, not predicted ------------------
    // Reported rather than asserted beyond non-vacuity: the dihedral across a
    // meridional edge is set by dphi and is nearly latitude-independent, while
    // the dihedral across a ring edge is set by dtheta, so which band a given
    // maxAngle selects is a fact about the surface worth printing.
    for ( double deg : { 20.0, 40.0 } )
    {
        const std::vector<char> mask = markByQuality(
            mesh, CurvatureCriterion<double>{ deg * kPi / 180.0 } );
        long long polarMarked = 0, equatMarked = 0, nMarked = 0;
        for ( std::size_t f = 0; f < nof && f < mask.size(); ++f )
        {
            if ( !mask[f] )
                continue;
            ++nMarked;
            if ( minAbsZ[f] >= kPolarZ )
                ++polarMarked;
            if ( maxAbsZ[f] <= kEquatZ )
                ++equatMarked;
        }
        long long loc[3] = { nMarked, polarMarked, equatMarked };
        long long glb[3] = { 0, 0, 0 };
        MPI_Allreduce( loc, glb, 3, MPI_LONG_LONG, MPI_SUM, comm );
        if ( rank == 0 )
            std::printf( "  [%s] markByQuality(curvature=%.0fdeg) on (33,8): "
                         "marked %lld/%lld (polarBand %lld, equatBand %lld)\n",
                         tag, deg, glb[0], 2LL * 8 * 31, glb[1], glb[2] );
    }

    // The marked set must actually drive a refine() cleanly on this surface --
    // the point of check 11 is the composition, not just the mask.
    {
        const std::vector<char> mask =
            markByQuality( mesh, EdgeLengthCriterion<double>{ 0.4 } );
        const RefineResult res = refine( mesh, halo, mask );
        fails += globalFails( comm, TesseraTest::checkConforming( mesh ) );
        check( globalOwnedEuler( mesh ) == 2 );
        fails += globalFails( comm, TesseraTest::check21BalanceRed( mesh ) );
        fails += globalFails( comm, TesseraTest::checkMidpointAgreement(
                                        comm, size, res.midpoints ) );
        fails += globalFails( comm, TesseraTest::owned1RingLocal( mesh ) );
    }

    if ( rank == 0 )
        std::printf( "  [%s] markquality on (33,8): %s\n", tag,
                     fails == 0 ? "ok" : "FAIL" );
    return fails;
}

template <class Exec>
int run( int rank, int size, const char* tag, const std::string& stem )
{
    int fails = 0;
    for ( const Params& p : kParams )
        fails += runParams<Exec>( rank, size, tag, p );
    fails += runDownstream<Exec>( rank, size, tag, stem );
    fails += runMarkQuality<Exec>( rank, size, tag );
    return fails;
}

std::string basenameOf( const char* p )
{
    std::string s( p ? p : "" );
    const std::size_t slash = s.find_last_of( '/' );
    return slash == std::string::npos ? s : s.substr( slash + 1 );
}

} // namespace

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
            std::printf( "test_latlon_sphere (size=%d)\n", size );

        // Generator-only checks: pure host arithmetic, run once.
        fails += runThrows( rank );
        fails += runAnisotropy( rank );

        const std::string exe =
            argc > 0 ? basenameOf( argv[0] ) : "test_latlon_sphere";
        std::string stem = exe + "_np" + std::to_string( size );
        if ( const char* tmpdir = std::getenv( "TESSERA_IO_TMPDIR" ) )
            stem = std::string( tmpdir ) + "/" + stem;

        fails += run<Kokkos::Serial>( rank, size, "serial", stem + "_serial" );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            fails += run<Kokkos::DefaultExecutionSpace>( rank, size, "device",
                                                         stem + "_device" );
    }

    Kokkos::finalize();

    int global_fails = 0;
    MPI_Allreduce( &fails, &global_fails, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    if ( global_fails == 0 )
    {
        int rank = 0;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        if ( rank == 0 )
            std::printf( "test_latlon_sphere: ok\n" );
    }
    MPI_Finalize();
    return global_fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
