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

// Regression test: caller-driven edge flip (tasks/completed/edge-flip.md).
//
// flipEdges() replaces the diagonal of the quad formed by the two faces
// incident on each marked edge. V, E and F are UNCHANGED and every gid is
// preserved -- only connectivity moves -- which makes almost every assertion
// here an IDENTITY rather than a comparison against a hardcoded number:
//
//   1. COUNTS ARE INVARIANT. globalOwned{Vertices,Edges,Faces} unchanged, Euler
//      == 2, checkConforming and checkNoInteriorVertex pass. Asserted after
//      every flip in every case below -- the cheapest detector of a botched
//      rewrite.
//   2. INVOLUTION -- the sharpest check. Flip the globally smallest EdgeKey,
//      then flip THE SAME EDGE GID again. The multiset of face corner-GID
//      triples must return EXACTLY to the original and so must the edge key
//      set. Nothing else pins the rewrite this tightly: it constrains the
//      winding, the corner assignment AND which of the two old face gids lands
//      on which new face.
//   3. DUPLICATE-EDGE REJECTION. Every vertex of an icosphere has valence 5 or
//      6, so the case is built deliberately: a hand-built subdivided
//      TETRAHEDRON soup, whose four original corners keep valence 3. Flipping
//      the edge from such a corner to one of its three neighbours would create
//      an edge that already exists. Asserts rejectedDuplicateEdge == 1,
//      accepted == 0, the mesh unchanged, and no duplicate edge key afterwards.
//   4. INDEPENDENT SET. Mark EVERY owned edge. No face may be rewritten twice,
//      i.e. at most one of any pre-flip face's three edges is in the accepted
//      set -- checked against the flip map and the pre-flip face->edge map.
//   5. RANK-COUNT INVARIANCE of the accepted set: check 4 against a reference
//      the same run computes alone on MPI_COMM_SELF -- identical `accepted`,
//      identical verdict histogram, and an identical multiset of face
//      corner-POSITION triples. This is the payoff of the gid-free priority; if
//      it fails, the priority is not total or not rank-independent.
//   6. GEOMETRIC REJECTION. minQuality = 0.99 rejects everything: accepted == 0,
//      the verdict counters account for every request, and the mesh is
//      unchanged in gids AND in connectivity.
//   7. VALENCE USE CASE. Valences from mesh.vertexEdges()'s global degree, a
//      mask of the flips that reduce total deviation from 6, three rounds. The
//      subdivision-2 icosphere starts optimal (12 vertices of valence 5, 150 of
//      valence 6), so the assertion is that the histogram does NOT DEGRADE --
//      nothing below 4 or above 8 and no loss of valence-6 vertices. Minimum
//      radius ratio reported before and after. Statistics, not edit sets.
//   8. KEY AND GID BOOKKEEPING. After an accepted flip the edge GID set is
//      unchanged and the edge KEY set has changed by EXACTLY the accepted
//      flips; edgeKeys()/faceKeys() match the AoSoA connectivity entry for
//      entry and contain no duplicates globally. The side-table half is
//      asserted in every case, flipped or not -- including case 7, whose mesh
//      is exactly what distribute() produced.
//   9. HALO VALID ON RETURN. Corrupting every ghost position and exchanging
//      restores the owners' values; a SECOND flipEdges() with nothing in
//      between succeeds; checkOwnershipPartition passes.
//  10. EMPTY MASK is a no-op, all counters zero.
//  11. FAMILY GUARD. flipEdges() on a refine()d mesh throws naming both
//      families; flipEdges() after splitEdges() is allowed and passes checks 1
//      and 2.
//  12. COMPOSED WITH SPLIT. splitEdges() (length-threshold mask) then three
//      rounds of valence flips, five times over. Euler == 2 and conformity
//      after every operation, and the minimum radius ratio stays above a floor
//      MEASURED in the first implementation run (see kMinRadiusRatioFloor).
//
// Runs on host (Serial) and device (default, HIP), ranks 1-5.

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
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace Tessera;

// Coarse subdiv-2 icosphere entity counts (the Step-6b fixture).
static const long long V0 = 162, E0 = 480, F0 = 320;

// Floor on the global minimum radius ratio (inradius/circumradius, 0.5 exactly
// for an equilateral triangle) over case 12's five split+flip rounds.
//
// MEASURED IN THE FIRST IMPLEMENTATION RUN, not guessed:
//
//   round      1       2       3       4       5
//   min r/R  0.2452  0.0727  0.0727  0.0309  0.0330
//   min ang  30.382  14.744  14.744   8.666   5.968   (degrees)
//   F           800    1880    4520    9126   21284
//
// byte-identical at np1-5 on both backends and in both execution spaces -- all
// twenty instances print the same five lines -- so 0.0309 is a property of the
// DRIVE and not of a decomposition. (The per-round FLIP COUNTS do move with the
// rank count, by up to ~1.5%: the valence mask below skips an owned edge whose
// second incident face is not resident, and which edges those are is a property
// of the partition. That is the caller's mask making a legitimate depth-1
// choice, not flipEdges() being decomposition-dependent -- case 5 asserts the
// operation's own rank-count invariance directly.) The floor is set just below
// the measured worst.
//
// SCOPE -- READ THIS BEFORE QUOTING THE FLOOR. It is a statement about the
// DRIVE (a length-threshold split mask plus a valence-improving flip mask), not
// about either operation. splitEdges() makes no shape guarantee of its own
// (tasks/completed/edge-split.md Decision 3). flipEdges() guarantees only that no ACCEPTED
// flip produces a face below DefaultFlipPolicy::minQuality -- a bound on the
// faces a flip CREATES, not on the faces the mask never touched, and one that a
// later splitEdges() round can cut below. NOTE THE FACTOR OF TWO: minQuality is
// in the convention where an equilateral triangle scores 1, while the number
// tabulated above is TesseraTest::minRadiusRatio()'s, where it scores 0.5. The
// default minQuality of 0.05 is therefore a floor of 0.025 in these units --
// which is where the round-4 value of 0.0309 sits, just above it.
//
// A regression that starts folding quads or accepting sliver flips drops this
// immediately.
static const double kMinRadiusRatioFloor = 0.025;

// ---------------------------------------------------------------------------
// Order-independent multiset checksum (same idiom as test_split_edges)
// ---------------------------------------------------------------------------
//
// count + SUM + BXOR over a 64-bit hash per item. All three combiners are
// commutative, so the value depends on neither visit order, nor local index, nor
// how the items are spread over ranks. `remove()` is the exact inverse of
// `add()`, which is what lets check 8 predict the post-flip edge key checksum
// from the pre-flip one plus the flip map.
struct Chk
{
    long long n = 0;
    unsigned long long sum = 0;
    unsigned long long x = 0;

    void add( unsigned long long h )
    {
        ++n;
        sum += h;
        x ^= h;
    }
    void remove( unsigned long long h )
    {
        --n;
        sum -= h;
        x ^= h;
    }
    bool operator==( const Chk& o ) const
    {
        return n == o.n && sum == o.sum && x == o.x;
    }
    bool operator!=( const Chk& o ) const { return !( *this == o ); }

    void reduce( MPI_Comm comm )
    {
        long long gn = 0;
        unsigned long long gs = 0, gx = 0;
        MPI_Allreduce( &n, &gn, 1, MPI_LONG_LONG, MPI_SUM, comm );
        MPI_Allreduce( &sum, &gs, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm );
        MPI_Allreduce( &x, &gx, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, comm );
        n = gn;
        sum = gs;
        x = gx;
    }
};

static inline unsigned long long mix64( unsigned long long z )
{
    z += 0x9e3779b97f4a7c15ULL;
    z = ( z ^ ( z >> 30 ) ) * 0xbf58476d1ce4e5b9ULL;
    z = ( z ^ ( z >> 27 ) ) * 0x94d049bb133111ebULL;
    return z ^ ( z >> 31 );
}
static inline unsigned long long hashCombine( unsigned long long a,
                                              unsigned long long b )
{
    return mix64( a ^ ( b + 0x9e3779b97f4a7c15ULL + ( a << 6 ) + ( a >> 2 ) ) );
}

//! BITWISE position hash: the raw IEEE bit patterns, not a quantisation. A flip
//! moves no vertex, so exact equality is the right assertion everywhere here.
static inline unsigned long long posHashExact( const std::array<double, 3>& p )
{
    unsigned long long h = 0xcbf29ce484222325ULL;
    for ( int d = 0; d < 3; ++d )
    {
        unsigned long long u = 0;
        std::memcpy( &u, &p[d], sizeof( u ) );
        h = hashCombine( h, u );
    }
    return h;
}

//! Canonical hash of a triangle from its three corner hashes: sorted, then
//! folded, so it is independent of which corner is stored at v[0].
static inline unsigned long long
triHash( unsigned long long a, unsigned long long b, unsigned long long c )
{
    unsigned long long t[3] = { a, b, c };
    std::sort( t, t + 3 );
    return hashCombine( hashCombine( t[0], t[1] ), t[2] );
}

static inline unsigned long long keyHash( const EdgeKey& k )
{
    return hashCombine( mix64( k.id[0] ), mix64( k.id[1] ) );
}

static inline int gsum( MPI_Comm comm, int local )
{
    int g = 0;
    MPI_Allreduce( &local, &g, 1, MPI_INT, MPI_SUM, comm );
    return g;
}
static inline long long gsumll( MPI_Comm comm, long long local )
{
    long long g = 0;
    MPI_Allreduce( &local, &g, 1, MPI_LONG_LONG, MPI_SUM, comm );
    return g;
}

// ---------------------------------------------------------------------------
// Mesh readers
// ---------------------------------------------------------------------------

//! Vertex gid -> position, over every locally held vertex (owned + ghost).
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

//! Corner vertex gids of every OWNED face.
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

//! EdgeKey of every OWNED edge, in owned-edge local index order -- i.e. exactly
//! the indexing flipEdges()' mask uses.
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

//! Gid of every OWNED edge, in the same order.
template <class MeshT>
static std::vector<GlobalId> ownedEdgeGids( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
        "he", mesh.numEdges() );
    Cabana::deep_copy( he, mesh.edges() );
    auto eg = Cabana::slice<EdgeField::Gid>( he );
    std::vector<GlobalId> out( mesh.numOwnedEdges() );
    for ( std::size_t e = 0; e < mesh.numOwnedEdges(); ++e )
        out[e] = eg( e );
    return out;
}

//! The global mesh's V, E, F from the owned-count reductions.
template <class MeshT>
static void counts( MeshT& mesh, long long& V, long long& E, long long& F )
{
    V = TesseraTest::globalOwnedVertices( mesh );
    E = TesseraTest::globalOwnedEdges( mesh );
    F = TesseraTest::globalOwnedFaces( mesh );
}

//! Multiset signature of the OWNED face corner-POSITION triples, reduced over
//! mesh.comm(). Bitwise and gid-free, so two meshes agree iff they are the same
//! geometry however they are numbered or decomposed.
template <class MeshT>
static Chk posFaceSignature( MeshT& mesh, int& fails )
{
    const auto pos = readPositions( mesh );
    Chk faces;
    for ( const auto& t : ownedFaceVerts( mesh ) )
    {
        unsigned long long h[3];
        bool ok = true;
        for ( int k = 0; k < 3; ++k )
        {
            auto it = pos.find( t[k] );
            if ( it == pos.end() )
            {
                ok = false; // every corner of an owned face must be held
                h[k] = 0;
            }
            else
                h[k] = posHashExact( it->second );
        }
        if ( !ok )
            ++fails;
        faces.add( triHash( h[0], h[1], h[2] ) );
    }
    faces.reduce( mesh.comm() );
    return faces;
}

//! Multiset signature of the OWNED face corner-GID triples. Comparable only
//! within one rank count, and that is enough: check 2's involution is a
//! statement about one run.
template <class MeshT>
static Chk gidFaceSignature( MeshT& mesh )
{
    Chk faces;
    for ( const auto& t : ownedFaceVerts( mesh ) )
        faces.add( triHash( mix64( t[0] ), mix64( t[1] ), mix64( t[2] ) ) );
    faces.reduce( mesh.comm() );
    return faces;
}

//! Multiset signature of the OWNED edge KEYS, and of the OWNED edge GIDS.
template <class MeshT>
static void edgeSignatures( MeshT& mesh, Chk& keys, Chk& gids )
{
    keys = Chk{};
    gids = Chk{};
    for ( const EdgeKey& k : ownedEdgeKeys( mesh ) )
        keys.add( keyHash( k ) );
    for ( GlobalId g : ownedEdgeGids( mesh ) )
        gids.add( mix64( g ) );
    keys.reduce( mesh.comm() );
    gids.reduce( mesh.comm() );
}

// ---------------------------------------------------------------------------
// Rank-count-invariant edge selections, and mask builders
// ---------------------------------------------------------------------------

//! The globally smallest EdgeKey over all owned edges. A pure function of the
//! global mesh, so every rank count picks the same edge.
template <class MeshT>
static EdgeKey globalMinEdgeKey( MeshT& mesh )
{
    EdgeKey best;
    best.id[0] = ~0ULL;
    best.id[1] = ~0ULL;
    for ( const EdgeKey& k : ownedEdgeKeys( mesh ) )
        if ( k < best )
            best = k;
    const int size = mesh.commSize();
    std::vector<unsigned long long> all( 2 * size, 0 );
    unsigned long long mine[2] = { best.id[0], best.id[1] };
    MPI_Allgather( mine, 2, MPI_UNSIGNED_LONG_LONG, all.data(), 2,
                   MPI_UNSIGNED_LONG_LONG, mesh.comm() );
    for ( int r = 0; r < size; ++r )
    {
        EdgeKey k;
        k.id[0] = all[2 * r];
        k.id[1] = all[2 * r + 1];
        if ( k < best )
            best = k;
    }
    return best;
}

//! The gid of the owned edge whose key is `want`, agreed on every rank. Exactly
//! one rank owns it and contributes; an MPI_MAX carries the value everywhere.
template <class MeshT>
static GlobalId gidOfEdgeKey( MeshT& mesh, const EdgeKey& want, int& fails )
{
    const std::vector<EdgeKey> keys = ownedEdgeKeys( mesh );
    const std::vector<GlobalId> gids = ownedEdgeGids( mesh );
    long long mine = -1;
    for ( std::size_t e = 0; e < keys.size(); ++e )
        if ( keys[e] == want )
            mine = static_cast<long long>( gids[e] );
    long long best = -1;
    MPI_Allreduce( &mine, &best, 1, MPI_LONG_LONG, MPI_MAX, mesh.comm() );
    if ( best < 0 )
        ++fails;
    return static_cast<GlobalId>( best );
}

//! Mask marking this rank's owned edges whose key is in `want`.
template <class MeshT>
static std::vector<char> maskFromKeys( MeshT& mesh,
                                       const std::set<EdgeKey>& want )
{
    const std::vector<EdgeKey> keys = ownedEdgeKeys( mesh );
    std::vector<char> mask( keys.size(), 0 );
    for ( std::size_t e = 0; e < keys.size(); ++e )
        mask[e] = want.count( keys[e] ) ? 1 : 0;
    return mask;
}

//! Mask marking this rank's owned edges whose GID is in `want`. Check 2 needs
//! this rather than maskFromKeys(): the whole point of the involution is that
//! the edge's key CHANGED while its gid did not.
template <class MeshT>
static std::vector<char> maskFromGids( MeshT& mesh,
                                       const std::set<GlobalId>& want )
{
    const std::vector<GlobalId> gids = ownedEdgeGids( mesh );
    std::vector<char> mask( gids.size(), 0 );
    for ( std::size_t e = 0; e < gids.size(); ++e )
        mask[e] = want.count( gids[e] ) ? 1 : 0;
    return mask;
}

//! Mask marking every owned edge longer than the global mean owned edge length
//! (test_split_edges' case-8 mask, reused for case 12's split phase).
template <class MeshT>
static std::vector<char> aboveMeanLengthMask( MeshT& mesh, int& fails )
{
    const auto pos = readPositions( mesh );
    const std::vector<EdgeKey> keys = ownedEdgeKeys( mesh );
    std::vector<double> len( keys.size(), 0.0 );
    double localSum = 0.0;
    for ( std::size_t e = 0; e < keys.size(); ++e )
    {
        auto ia = pos.find( keys[e].id[0] );
        auto ib = pos.find( keys[e].id[1] );
        if ( ia == pos.end() || ib == pos.end() )
        {
            ++fails; // an owned edge's endpoints must be held locally
            continue;
        }
        double s = 0.0;
        for ( int d = 0; d < 3; ++d )
        {
            const double dd = ia->second[d] - ib->second[d];
            s += dd * dd;
        }
        len[e] = std::sqrt( s );
        localSum += len[e];
    }
    double totalLen = 0.0;
    MPI_Allreduce( &localSum, &totalLen, 1, MPI_DOUBLE, MPI_SUM, mesh.comm() );
    const long long nE = TesseraTest::globalOwnedEdges( mesh );
    const double mean = nE > 0 ? totalLen / static_cast<double>( nE ) : 0.0;

    std::vector<char> mask( keys.size(), 0 );
    for ( std::size_t e = 0; e < keys.size(); ++e )
        mask[e] = ( len[e] > mean ) ? 1 : 0;
    return mask;
}

// ---------------------------------------------------------------------------
// Global vertex valence -- the caller-side half of the valence use case
// ---------------------------------------------------------------------------
//
// A vertex's valence is the number of edges incident on it. Each rank counts
// its OWNED edges' two endpoints and sends the tallies to a gid coordinator
// (gid % size), which therefore holds the true global degree of every gid it
// owns and of no other. That is the whole of the communication: the histogram
// is a local reduction over the coordinator's map, and a lookup is one request
// round. Nothing here reads the vertex->edge CSR, which would only be complete
// on OWNED rows and would need the same round anyway for c and d.

namespace valence
{
struct GidMsg
{
    GlobalId g;
};
struct ValMsg
{
    GlobalId g;
    int val;
};

//! Global degree of every vertex gid, restricted to this rank's coordinator
//! share (gid % size). The shares partition the gid space, so summing a
//! per-share statistic over ranks gives the global one with no double counting.
template <class MeshT>
static std::map<GlobalId, int> coordinatorDegrees( MeshT& mesh )
{
    const int size = mesh.commSize();
    std::vector<std::vector<GidMsg>> inc( size );
    for ( const EdgeKey& k : ownedEdgeKeys( mesh ) )
        for ( int j = 0; j < 2; ++j )
            inc[k.id[j] % static_cast<GlobalId>( size )].push_back(
                { k.id[j] } );
    auto got = allToAllV( mesh.comm(), inc );
    std::map<GlobalId, int> deg;
    for ( const auto& m : got.data )
        ++deg[m.g];
    return deg;
}

//! The global valence histogram, as valence -> count. Collective; identical on
//! every rank.
template <class MeshT>
static std::map<int, long long> histogram( MeshT& mesh )
{
    const std::map<GlobalId, int> deg = coordinatorDegrees( mesh );
    const int kMax = 32;
    std::vector<long long> loc( kMax + 1, 0 ), glob( kMax + 1, 0 );
    for ( const auto& kv : deg )
        ++loc[std::min( kv.second, kMax )];
    MPI_Allreduce( loc.data(), glob.data(), kMax + 1, MPI_LONG_LONG, MPI_SUM,
                   mesh.comm() );
    std::map<int, long long> out;
    for ( int v = 0; v <= kMax; ++v )
        if ( glob[v] != 0 )
            out[v] = glob[v];
    return out;
}

//! The global valence of each gid in `want`, by one request/reply round against
//! the coordinators. Collective.
template <class MeshT>
static std::map<GlobalId, int> lookup( MeshT& mesh,
                                       const std::set<GlobalId>& want )
{
    const int size = mesh.commSize();
    const std::map<GlobalId, int> deg = coordinatorDegrees( mesh );
    std::vector<std::vector<GidMsg>> req( size );
    for ( GlobalId g : want )
        req[g % static_cast<GlobalId>( size )].push_back( { g } );
    auto reqGot = allToAllV( mesh.comm(), req );
    std::vector<std::vector<ValMsg>> rep( size );
    for ( int s = 0; s < size; ++s )
    {
        const GidMsg* p = reqGot.from( s );
        const int cnt = reqGot.count( s );
        for ( int i = 0; i < cnt; ++i )
        {
            auto it = deg.find( p[i].g );
            rep[s].push_back( { p[i].g, it == deg.end() ? 0 : it->second } );
        }
    }
    auto ans = allToAllV( mesh.comm(), rep );
    std::map<GlobalId, int> out;
    for ( const auto& m : ans.data )
        out[m.g] = m.val;
    return out;
}
} // namespace valence

//! THE VALENCE-EQUALIZATION MASK -- the intended caller-side pattern, written
//! out in full because it is what tasks/completed/edge-flip.md Decision 2 says belongs to
//! the caller rather than to flipEdges().
//!
//! A flip of (a,b) with opposite corners (c,d) takes one edge off each of a and
//! b and adds one to each of c and d, so it is an improvement iff it lowers
//! sum |valence - 6|. The opposite corners are read from the LOCALLY HELD faces
//! that reference the edge's gid; an owned edge whose second face is not
//! resident is simply not marked, which is a legitimate caller choice at halo
//! depth 1 and costs at most a few edges per partition boundary.
template <class MeshT>
static std::vector<char> valenceMask( MeshT& mesh, long long& consideredOut )
{
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto fv = Cabana::slice<FaceField::Verts>( hf );
    auto fe = Cabana::slice<FaceField::Edges>( hf );

    // edge gid -> the opposite corners of the locally held faces on it.
    std::unordered_map<GlobalId, std::vector<GlobalId>> opposites;
    opposites.reserve( mesh.numFaces() * 4 + 1 );
    for ( std::size_t f = 0; f < mesh.numFaces(); ++f )
        for ( int k = 0; k < 3; ++k )
            opposites[fe( f, k )].push_back( fv( f, ( k + 2 ) % 3 ) );

    const std::vector<EdgeKey> keys = ownedEdgeKeys( mesh );
    const std::vector<GlobalId> gids = ownedEdgeGids( mesh );

    std::set<GlobalId> want;
    for ( std::size_t e = 0; e < keys.size(); ++e )
    {
        auto it = opposites.find( gids[e] );
        if ( it == opposites.end() || it->second.size() != 2 )
            continue;
        want.insert( keys[e].id[0] );
        want.insert( keys[e].id[1] );
        want.insert( it->second[0] );
        want.insert( it->second[1] );
    }
    const std::map<GlobalId, int> val = valence::lookup( mesh, want );

    auto dev = []( int v ) { return std::abs( v - 6 ); };
    long long considered = 0;
    std::vector<char> mask( keys.size(), 0 );
    for ( std::size_t e = 0; e < keys.size(); ++e )
    {
        auto it = opposites.find( gids[e] );
        if ( it == opposites.end() || it->second.size() != 2 )
            continue;
        const GlobalId q[4] = { keys[e].id[0], keys[e].id[1], it->second[0],
                                it->second[1] };
        int v[4];
        bool ok = true;
        for ( int j = 0; j < 4; ++j )
        {
            auto iv = val.find( q[j] );
            if ( iv == val.end() || iv->second <= 0 )
                ok = false;
            else
                v[j] = iv->second;
        }
        if ( !ok )
            continue;
        ++considered;
        const int before = dev( v[0] ) + dev( v[1] ) + dev( v[2] ) + dev( v[3] );
        const int after = dev( v[0] - 1 ) + dev( v[1] - 1 ) + dev( v[2] + 1 ) +
                          dev( v[3] + 1 );
        mask[e] = ( after < before ) ? 1 : 0;
    }
    consideredOut = considered;
    return mask;
}

// ---------------------------------------------------------------------------
// The flip map, gathered globally
// ---------------------------------------------------------------------------

//! The global set of (old key, new key) pairs. FlipResult::flipped holds the
//! flips a rank TOUCHES, so the same pair appears on up to three ranks; an
//! allgather plus a std::set is the honest way to get the global set, and the
//! accepted set is at most E/3 so the cost is a fraction of the mesh.
static std::set<std::pair<EdgeKey, EdgeKey>>
globalFlipSet( MPI_Comm comm, const FlipResult& res )
{
    int size = 1;
    MPI_Comm_size( comm, &size );
    std::vector<unsigned long long> mine;
    mine.reserve( res.flipped.size() * 4 );
    for ( const auto& kv : res.flipped )
    {
        mine.push_back( kv.first.id[0] );
        mine.push_back( kv.first.id[1] );
        mine.push_back( kv.second.id[0] );
        mine.push_back( kv.second.id[1] );
    }
    int n = static_cast<int>( mine.size() );
    std::vector<int> cnt( size, 0 ), disp( size, 0 );
    MPI_Allgather( &n, 1, MPI_INT, cnt.data(), 1, MPI_INT, comm );
    int tot = 0;
    for ( int r = 0; r < size; ++r )
    {
        disp[r] = tot;
        tot += cnt[r];
    }
    std::vector<unsigned long long> all( tot > 0 ? tot : 1, 0 );
    MPI_Allgatherv( mine.data(), n, MPI_UNSIGNED_LONG_LONG, all.data(),
                    cnt.data(), disp.data(), MPI_UNSIGNED_LONG_LONG, comm );
    std::set<std::pair<EdgeKey, EdgeKey>> out;
    for ( int i = 0; i + 3 < tot; i += 4 )
    {
        EdgeKey a, b;
        a.id[0] = all[i];
        a.id[1] = all[i + 1];
        b.id[0] = all[i + 2];
        b.id[1] = all[i + 3];
        out.insert( { a, b } );
    }
    return out;
}

// ---------------------------------------------------------------------------
// Shared post-conditions
// ---------------------------------------------------------------------------

//! Check 1, in full: the three global owned counts are exactly what they were,
//! Euler is 2, the mesh is conforming with no interior vertex, ownership is a
//! partition and the owned 1-ring is local. Returns LOCAL fails.
template <class MeshT>
static int checkAll( MeshT& mesh, long long V, long long E, long long F )
{
    int fails = 0;
    long long v, e, f;
    counts( mesh, v, e, f );
    if ( v != V || e != E || f != F )
        ++fails;
    fails += TesseraTest::checkOwnershipPartition( mesh, v, e, f );
    fails += TesseraTest::owned1RingLocal( mesh );
    fails += TesseraTest::checkConforming( mesh );
    fails += TesseraTest::checkNoInteriorVertex( mesh );
    if ( TesseraTest::checkOwnedEuler( mesh ) != 2 )
        ++fails;
    return fails;
}

//! The five verdict counters must partition `requested` -- a flip is accepted or
//! rejected for exactly one reason.
static int checkCountersPartition( const FlipResult& r )
{
    return ( r.accepted + r.rejectedBoundary + r.rejectedDuplicateEdge +
                 r.rejectedGeometric + r.rejectedConflict ==
             r.requested )
               ? 0
               : 1;
}


//! Build -> partition -> distribute the subdiv-2 fixture on `comm`.
template <class MeshT, class Exec>
static void setupOn( MPI_Comm comm, MeshT& mesh,
                     MeshHalo<typename Exec::memory_space>& halo )
{
    (void)comm;
    buildIcosphere( mesh, 2 );
    auto faceOwner = facePartitionByAxis( mesh );
    distribute( mesh, halo, faceOwner );
}

template <class MeshT, class Exec>
static void setup( MeshT& mesh, MeshHalo<typename Exec::memory_space>& halo )
{
    setupOn<MeshT, Exec>( MPI_COMM_WORLD, mesh, halo );
}

// ===========================================================================
// Cases
// ===========================================================================

//! Case 2: a flip is its own inverse, down to the face corner-GID triples and
//! the edge key set.
template <class MeshT, class Exec>
static int caseInvolution( int rank, const char* tag )
{
    using mem = typename Exec::memory_space;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    setup<MeshT, Exec>( mesh, halo );

    int local = 0;
    long long V, E, F;
    counts( mesh, V, E, F );
    if ( V != V0 || E != E0 || F != F0 )
        ++local; // fixture drifted

    const EdgeKey k0 = globalMinEdgeKey( mesh );
    const GlobalId g0 = gidOfEdgeKey( mesh, k0, local );
    const Chk faces0 = gidFaceSignature( mesh );
    Chk keys0, gids0;
    edgeSignatures( mesh, keys0, gids0 );

    const FlipResult r1 = flipEdges( mesh, halo, maskFromKeys( mesh, { k0 } ) );
    local += checkAll( mesh, V, E, F );
    local += checkCountersPartition( r1 );
    if ( r1.requested != 1 || r1.accepted != 1 )
        ++local; // the case is vacuous unless this one flip is taken

    const Chk faces1 = gidFaceSignature( mesh );
    Chk keys1, gids1;
    edgeSignatures( mesh, keys1, gids1 );
    if ( faces1 == faces0 )
        ++local; // nothing moved: the "involution" would be trivial
    if ( keys1 == keys0 )
        ++local;
    if ( gids1 != gids0 )
        ++local; // gids are preserved by a flip

    // The SAME EDGE GID again -- its key changed, which is the whole point.
    const FlipResult r2 = flipEdges( mesh, halo, maskFromGids( mesh, { g0 } ) );
    local += checkAll( mesh, V, E, F );
    local += checkCountersPartition( r2 );
    if ( r2.requested != 1 || r2.accepted != 1 )
        ++local;

    const Chk faces2 = gidFaceSignature( mesh );
    Chk keys2, gids2;
    edgeSignatures( mesh, keys2, gids2 );
    if ( faces2 != faces0 )
        ++local; // NOT an involution on face corner gids
    if ( keys2 != keys0 )
        ++local; // NOT an involution on the edge key set
    if ( gids2 != gids0 )
        ++local;
    local += TesseraTest::checkKeyTables( mesh );

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case1+2 (counts invariant, involution) %s: "
                     "V=%lld E=%lld F=%lld, edge gid %llu flipped and "
                     "unflipped\n",
                     tag, glob == 0 ? "ok" : "FAIL", V, E, F,
                     static_cast<unsigned long long>( g0 ) );
    return glob == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Case 3's fixture: a SUBDIVIDED TETRAHEDRON.
// ---------------------------------------------------------------------------
//
// The four original corners keep valence 3 while the mesh is big enough (16
// faces, 10 vertices, 24 edges) to distribute over five ranks. With corner A at
// valence 3 and neighbours mAB, mAC, mAD, the edge (A, mAB) has opposite corners
// mAC and mAD -- and (mAC, mAD) IS ALREADY AN EDGE, of the child face
// (A, mAC, mAD). Flipping it would produce a non-manifold mesh, so it must be
// rejected. buildFromTriangleSoup assigns vertex gid == soup index, so the
// interesting edge's key is known up front.
static TriangleSoup<double> subdividedTetraSoup( EdgeKey& flipMe )
{
    const double P[4][3] = {
        { 1.0, 1.0, 1.0 },   // 0 = A
        { 1.0, -1.0, -1.0 }, // 1 = B
        { -1.0, 1.0, -1.0 }, // 2 = C
        { -1.0, -1.0, 1.0 }  // 3 = D
    };
    // Outward-CCW faces of the tetrahedron (each verified against the centroid).
    const int T[4][3] = { { 0, 1, 2 }, { 0, 2, 3 }, { 0, 3, 1 }, { 1, 3, 2 } };

    TriangleSoup<double> soup;
    for ( int i = 0; i < 4; ++i )
        for ( int d = 0; d < 3; ++d )
            soup.positions.push_back( P[i][d] );

    // Midpoints in a fixed order, so the gids below are deterministic:
    // (0,1)->4 (0,2)->5 (0,3)->6 (1,2)->7 (1,3)->8 (2,3)->9.
    std::map<std::pair<int, int>, int> mid;
    for ( int a = 0; a < 4; ++a )
        for ( int b = a + 1; b < 4; ++b )
        {
            mid[{ a, b }] = static_cast<int>( soup.positions.size() / 3 );
            for ( int d = 0; d < 3; ++d )
                soup.positions.push_back( 0.5 * ( P[a][d] + P[b][d] ) );
        }
    auto m = [&]( int a, int b )
    { return mid[{ std::min( a, b ), std::max( a, b ) }]; };

    for ( int t = 0; t < 4; ++t )
    {
        const int x = T[t][0], y = T[t][1], z = T[t][2];
        const int mxy = m( x, y ), myz = m( y, z ), mzx = m( z, x );
        const int child[4][3] = { { x, mxy, mzx },
                                  { y, myz, mxy },
                                  { z, mzx, myz },
                                  { mxy, myz, mzx } };
        for ( int c = 0; c < 4; ++c )
            for ( int k = 0; k < 3; ++k )
                soup.triangles.push_back( child[c][k] );
    }

    flipMe = makeEdgeKey( static_cast<GlobalId>( 0 ),
                          static_cast<GlobalId>( m( 0, 1 ) ) );
    return soup;
}

//! Case 3: flipping into an existing edge is detected, across the extra
//! coordinator round.
template <class MeshT, class Exec>
static int caseDuplicateEdge( int rank, const char* tag )
{
    using mem = typename Exec::memory_space;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    EdgeKey flipMe;
    const TriangleSoup<double> soup = subdividedTetraSoup( flipMe );
    buildFromTriangleSoup( mesh, soup );
    auto faceOwner = facePartitionByAxis( mesh );
    distribute( mesh, halo, faceOwner );

    int local = 0;
    long long V, E, F;
    counts( mesh, V, E, F );
    if ( V != 10 || E != 24 || F != 16 )
        ++local; // fixture drifted

    const Chk faces0 = gidFaceSignature( mesh );
    Chk keys0, gids0;
    edgeSignatures( mesh, keys0, gids0 );

    const FlipResult res =
        flipEdges( mesh, halo, maskFromKeys( mesh, { flipMe } ) );

    local += checkAll( mesh, V, E, F );
    local += checkCountersPartition( res );
    if ( res.requested != 1 )
        ++local; // the intended edge was not found
    if ( res.rejectedDuplicateEdge != 1 || res.accepted != 0 )
        ++local;
    if ( !res.flipped.empty() )
        ++local;

    const Chk faces1 = gidFaceSignature( mesh );
    Chk keys1, gids1;
    edgeSignatures( mesh, keys1, gids1 );
    if ( faces1 != faces0 || keys1 != keys0 || gids1 != gids0 )
        ++local; // a rejected flip must leave the mesh alone
    // no duplicate edge key afterwards
    local += TesseraTest::checkKeyTables( mesh );

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case3 (duplicate-edge rejection) %s: subdivided "
                     "tetra V=%lld E=%lld F=%lld, requested=%lld "
                     "rejectedDuplicateEdge=%lld accepted=%lld\n",
                     tag, glob == 0 ? "ok" : "FAIL", V, E, F, res.requested,
                     res.rejectedDuplicateEdge, res.accepted );
    return glob == 0 ? 0 : 1;
}

//! Cases 4, 5 and 8: mark EVERY owned edge. No face is rewritten twice; the
//! result is identical to an MPI_COMM_SELF reference; and the edge key set moved
//! by exactly the accepted flips while the gid set did not move at all.
template <class MeshT, class Exec>
static int caseIndependentSet( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    int local = 0;

    struct Out
    {
        Chk posFaces;
        Chk keysBefore, keysAfter, gidsBefore, gidsAfter;
        long long V = 0, E = 0, F = 0;
        FlipResult res;
        long long doubleWritten = 0;
        long long predictedMismatch = 0;
    };

    auto measure = [&]( MPI_Comm comm, Out& o )
    {
        MeshT mesh( comm );
        MeshHalo<mem> halo;
        setupOn<MeshT, Exec>( comm, mesh, halo );
        counts( mesh, o.V, o.E, o.F );

        const auto preVerts = ownedFaceVerts( mesh );
        edgeSignatures( mesh, o.keysBefore, o.gidsBefore );

        std::vector<char> mask( mesh.numOwnedEdges(), 1 );
        o.res = flipEdges( mesh, halo, mask );
        local += checkAll( mesh, o.V, o.E, o.F );
        local += checkCountersPartition( o.res );
        local += TesseraTest::checkKeyTables( mesh );

        const auto flips = globalFlipSet( comm, o.res );
        if ( static_cast<long long>( flips.size() ) != o.res.accepted )
            ++local; // the flip map is not complete/consistent

        // CHECK 4: no pre-flip face had two of its three edges accepted.
        std::set<EdgeKey> acceptedKeys;
        for ( const auto& kv : flips )
            acceptedKeys.insert( kv.first );
        for ( const auto& t : preVerts )
        {
            int hit = 0;
            for ( int k = 0; k < 3; ++k )
                if ( acceptedKeys.count( makeEdgeKey( t[k], t[( k + 1 ) % 3] ) ) )
                    ++hit;
            if ( hit > 1 )
                ++o.doubleWritten;
        }
        o.doubleWritten = gsumll( comm, o.doubleWritten );

        // CHECK 8: predict the post-flip edge key checksum from the pre-flip one.
        edgeSignatures( mesh, o.keysAfter, o.gidsAfter );
        Chk predicted = o.keysBefore;
        for ( const auto& kv : flips )
        {
            predicted.remove( keyHash( kv.first ) );
            predicted.add( keyHash( kv.second ) );
        }
        if ( predicted != o.keysAfter )
            ++o.predictedMismatch;
        if ( o.gidsAfter != o.gidsBefore )
            ++o.predictedMismatch;

        o.posFaces = posFaceSignature( mesh, local );
    };

    Out w, s;
    measure( MPI_COMM_WORLD, w );
    measure( MPI_COMM_SELF, s );

    if ( w.doubleWritten != 0 || s.doubleWritten != 0 )
        ++local; // a face would have been rewritten twice
    if ( w.predictedMismatch != 0 || s.predictedMismatch != 0 )
        ++local;
    if ( w.res.accepted <= 0 )
        ++local; // vacuous: the case proves nothing if nothing was flipped

    // CHECK 5: rank-count invariance.
    if ( w.V != s.V || w.E != s.E || w.F != s.F )
        ++local;
    if ( w.posFaces != s.posFaces )
        ++local;
    if ( w.res.accepted != s.res.accepted ||
         w.res.requested != s.res.requested ||
         w.res.rejectedBoundary != s.res.rejectedBoundary ||
         w.res.rejectedDuplicateEdge != s.res.rejectedDuplicateEdge ||
         w.res.rejectedGeometric != s.res.rejectedGeometric ||
         w.res.rejectedConflict != s.res.rejectedConflict )
        ++local;

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case4+5+8 (independent set, np%d vs SELF, key "
                     "bookkeeping) %s: requested=%lld accepted=%lld "
                     "boundary=%lld dup=%lld geom=%lld conflict=%lld "
                     "(SELF accepted=%lld)\n",
                     tag, size, glob == 0 ? "ok" : "FAIL", w.res.requested,
                     w.res.accepted, w.res.rejectedBoundary,
                     w.res.rejectedDuplicateEdge, w.res.rejectedGeometric,
                     w.res.rejectedConflict, s.res.accepted );
    return glob == 0 ? 0 : 1;
}

//! Case 6: a policy nothing can satisfy rejects everything and moves nothing.
template <class MeshT, class Exec>
static int caseGeometricRejection( int rank, const char* tag )
{
    using mem = typename Exec::memory_space;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    setup<MeshT, Exec>( mesh, halo );

    int local = 0;
    long long V, E, F;
    counts( mesh, V, E, F );
    unsigned long long cv0, ce0, cf0, cv1, ce1, cf1;
    TesseraTest::topologyChecksum( mesh, cv0, ce0, cf0 );
    const Chk faces0 = gidFaceSignature( mesh );
    Chk keys0, gids0;
    edgeSignatures( mesh, keys0, gids0 );

    DefaultFlipPolicy policy;
    policy.minQuality = 0.99; // no triangle but an equilateral one clears this
    std::vector<char> mask( mesh.numOwnedEdges(), 1 );
    const FlipResult res = flipEdges( mesh, halo, mask, policy );

    local += checkAll( mesh, V, E, F );
    local += checkCountersPartition( res );
    if ( res.accepted != 0 )
        ++local;
    if ( res.rejectedGeometric != res.requested - res.rejectedBoundary )
        ++local;
    if ( res.rejectedDuplicateEdge != 0 || res.rejectedBoundary != 0 )
        ++local; // a closed icosphere has no boundary and no valence-3 vertex

    TesseraTest::topologyChecksum( mesh, cv1, ce1, cf1 );
    const Chk faces1 = gidFaceSignature( mesh );
    Chk keys1, gids1;
    edgeSignatures( mesh, keys1, gids1 );
    if ( cv0 != cv1 || ce0 != ce1 || cf0 != cf1 )
        ++local;
    if ( faces1 != faces0 || keys1 != keys0 || gids1 != gids0 )
        ++local;

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case6 (geometric rejection, minQuality=0.99) %s: "
                     "requested=%lld rejectedGeometric=%lld\n",
                     tag, glob == 0 ? "ok" : "FAIL", res.requested,
                     res.rejectedGeometric );
    return glob == 0 ? 0 : 1;
}

//! Case 7: the valence use case. The fixture starts optimal, so the assertion
//! is that three rounds of valence-driven flips do not DEGRADE it.
template <class MeshT, class Exec>
static int caseValence( int rank, const char* tag )
{
    using mem = typename Exec::memory_space;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    setup<MeshT, Exec>( mesh, halo );

    int local = 0;
    long long V, E, F;
    counts( mesh, V, E, F );

    const std::map<int, long long> h0 = valence::histogram( mesh );
    double angle0 = 180.0;
    int qf = 0;
    const double q0 = TesseraTest::minRadiusRatio( mesh, qf, angle0 );
    local += qf;

    // The fixture: 12 vertices of valence 5, 150 of valence 6.
    if ( h0.size() != 2 || h0.count( 5 ) == 0 || h0.count( 6 ) == 0 ||
         h0.at( 5 ) != 12 || h0.at( 6 ) != 150 )
        ++local;

    long long accepted = 0, considered = 0;
    for ( int round = 0; round < 3; ++round )
    {
        long long cons = 0;
        const std::vector<char> mask = valenceMask( mesh, cons );
        considered += gsumll( MPI_COMM_WORLD, cons );
        const FlipResult res = flipEdges( mesh, halo, mask );
        accepted += res.accepted;
        const int fAll = checkAll( mesh, V, E, F );
        const int fCnt = checkCountersPartition( res );
        local += fAll + fCnt;
        if ( ( fAll || fCnt ) && rank == 0 )
            std::printf( "  [%s] case7 round%d CHECK FAILS: all=%d part=%d "
                         "(requested=%lld accepted=%lld boundary=%lld "
                         "dup=%lld geom=%lld conflict=%lld)\n",
                         tag, round + 1, fAll, fCnt, res.requested,
                         res.accepted, res.rejectedBoundary,
                         res.rejectedDuplicateEdge, res.rejectedGeometric,
                         res.rejectedConflict );
    }
    // Check 8 HERE is the strongest of the five places it is asserted: the
    // fixture starts valence-optimal, so the mask marks nothing, flipEdges()
    // takes its empty-mask fast path and the mesh is still exactly what
    // distribute() produced. That made it the one case that could not assert the
    // side tables while distribute() left them stale, and it is now the case
    // that proves distribute() rebuilds them.
    local += TesseraTest::checkKeyTables( mesh );

    const std::map<int, long long> h1 = valence::histogram( mesh );
    double angle1 = 180.0;
    const double q1 = TesseraTest::minRadiusRatio( mesh, qf, angle1 );
    local += qf;

    for ( const auto& kv : h1 )
        if ( kv.first < 4 || kv.first > 8 )
            ++local; // the histogram degraded past the stated bounds
    const long long six0 = h0.count( 6 ) ? h0.at( 6 ) : 0;
    const long long six1 = h1.count( 6 ) ? h1.at( 6 ) : 0;
    if ( six1 < six0 )
        ++local; // valence-6 vertices were lost
    if ( considered <= 0 )
        ++local; // vacuous: the mask never evaluated a single flip

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
    {
        std::printf( "  [%s] case7 (valence equalization, 3 rounds) %s: "
                     "considered=%lld accepted=%lld valence6 %lld -> %lld, "
                     "minRadiusRatio %.4f -> %.4f, minAngle %.3f -> %.3f\n",
                     tag, glob == 0 ? "ok" : "FAIL", considered, accepted, six0,
                     six1, q0, q1, angle0, angle1 );
    }
    return glob == 0 ? 0 : 1;
}

//! Case 9: the halo is valid on return -- a ghost corrupt/resync round-trips
//! and a SECOND flipEdges() succeeds with nothing in between. Case 10: an empty
//! mask is the identity.
template <class MeshT, class Exec>
static int caseEmptyAndHalo( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    int local = 0;

    // ---- case 10: empty mask -----------------------------------------------
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        setup<MeshT, Exec>( mesh, halo );
        long long V, E, F;
        counts( mesh, V, E, F );
        unsigned long long cv0, ce0, cf0, cv1, ce1, cf1;
        TesseraTest::topologyChecksum( mesh, cv0, ce0, cf0 );
        const Chk faces0 = gidFaceSignature( mesh );

        std::vector<char> mask( mesh.numOwnedEdges(), 0 );
        const FlipResult res = flipEdges( mesh, halo, mask );

        TesseraTest::topologyChecksum( mesh, cv1, ce1, cf1 );
        local += checkAll( mesh, V, E, F );
        if ( cv0 != cv1 || ce0 != ce1 || cf0 != cf1 )
            ++local;
        if ( gidFaceSignature( mesh ) != faces0 )
            ++local;
        if ( res.requested != 0 || res.accepted != 0 ||
             res.rejectedBoundary != 0 || res.rejectedDuplicateEdge != 0 ||
             res.rejectedGeometric != 0 || res.rejectedConflict != 0 ||
             !res.flipped.empty() )
            ++local;
    }

    // ---- case 9: halo valid on return --------------------------------------
    long long corrupted = 0, planSize = 0, accepted2 = 0;
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        setup<MeshT, Exec>( mesh, halo );
        long long V, E, F;
        counts( mesh, V, E, F );

        std::vector<char> mask( mesh.numOwnedEdges(), 1 );
        FlipResult res = flipEdges( mesh, halo, mask );
        local += checkAll( mesh, V, E, F );
        if ( res.accepted <= 0 )
            ++local;

        for ( const auto* p : { &halo.vplan, &halo.eplan, &halo.fplan } )
            planSize += static_cast<long long>( p->totalSend() ) +
                        static_cast<long long>( p->totalRecv() );
        planSize = gsumll( MPI_COMM_WORLD, planSize );

        // Corrupt every ghost position; haloExchange() must restore the owners'.
        {
            Cabana::AoSoA<typename MeshT::vertex_member_types,
                          Kokkos::HostSpace>
                hv( "hv", mesh.numVertices() );
            Cabana::deep_copy( hv, mesh.vertices() );
            auto pos = Cabana::slice<VertexField::Position>( hv );
            for ( std::size_t i = mesh.numOwnedVertices();
                  i < mesh.numVertices(); ++i )
            {
                ++corrupted;
                for ( int d = 0; d < MeshT::dim; ++d )
                    pos( i, d ) = -1234.5;
            }
            Cabana::deep_copy( mesh.vertices(), hv );
        }
        corrupted = gsumll( MPI_COMM_WORLD, corrupted );
        haloExchange( mesh, halo );
        {
            Cabana::AoSoA<typename MeshT::vertex_member_types,
                          Kokkos::HostSpace>
                hv( "hv", mesh.numVertices() );
            Cabana::deep_copy( hv, mesh.vertices() );
            auto pos = Cabana::slice<VertexField::Position>( hv );
            for ( std::size_t i = mesh.numOwnedVertices();
                  i < mesh.numVertices(); ++i )
                if ( pos( i, 0 ) == -1234.5 )
                    ++local; // haloExchange did not restore the owner's value
        }

        // A SECOND flipEdges() with nothing in between must work.
        std::vector<char> mask2( mesh.numOwnedEdges(), 1 );
        res = flipEdges( mesh, halo, mask2 );
        accepted2 = res.accepted;
        local += checkAll( mesh, V, E, F );
        local += TesseraTest::checkKeyTables( mesh );
        if ( res.requested <= 0 )
            ++local;

        if ( size > 1 && ( planSize <= 0 || corrupted <= 0 ) )
            ++local; // vacuous: an empty plan passes every structural check
    }

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case9+10 (empty mask, halo valid on return) %s: "
                     "plan=%lld ghostsCorrupted=%lld secondAccepted=%lld\n",
                     tag, glob == 0 ? "ok" : "FAIL", planSize, corrupted,
                     accepted2 );
    return glob == 0 ? 0 : 1;
}

//! Case 11: the editing families are disjoint, and flipEdges() lives in the same
//! one as splitEdges().
template <class MeshT, class Exec>
static int caseFamilyGuard( int rank, const char* tag )
{
    using mem = typename Exec::memory_space;
    int local = 0;

    auto namesBothFamilies = []( const std::string& what )
    {
        return what.find( "Hierarchical" ) != std::string::npos &&
               what.find( "Remesh" ) != std::string::npos;
    };

    // flipEdges() on a refine()d mesh throws.
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        setup<MeshT, Exec>( mesh, halo );
        std::vector<char> fmask( mesh.numOwnedFaces(), 1 );
        refine( mesh, halo, fmask );
        bool threw = false, named = false;
        try
        {
            std::vector<char> emask( mesh.numOwnedEdges(), 1 );
            flipEdges( mesh, halo, emask );
        }
        catch ( const std::exception& e )
        {
            threw = true;
            named = namesBothFamilies( e.what() );
        }
        if ( !threw || !named )
            ++local;
    }

    // refine() on a flipEdges()-edited mesh throws.
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        setup<MeshT, Exec>( mesh, halo );
        std::vector<char> emask( mesh.numOwnedEdges(), 1 );
        flipEdges( mesh, halo, emask );
        bool threw = false, named = false;
        try
        {
            std::vector<char> fmask( mesh.numOwnedFaces(), 1 );
            refine( mesh, halo, fmask );
        }
        catch ( const std::exception& e )
        {
            threw = true;
            named = namesBothFamilies( e.what() );
        }
        if ( !threw || !named )
            ++local;
    }

    // flipEdges() AFTER splitEdges() is allowed, and checks 1 and 2 hold on the
    // result -- the two operations are in the same family.
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        setup<MeshT, Exec>( mesh, halo );
        std::vector<char> emask( mesh.numOwnedEdges(), 1 );
        splitEdges( mesh, halo, emask );

        long long V, E, F;
        counts( mesh, V, E, F );
        const EdgeKey k0 = globalMinEdgeKey( mesh );
        const GlobalId g0 = gidOfEdgeKey( mesh, k0, local );
        const Chk faces0 = gidFaceSignature( mesh );

        FlipResult r = flipEdges( mesh, halo, maskFromKeys( mesh, { k0 } ) );
        local += checkAll( mesh, V, E, F );
        if ( r.accepted != 1 )
            ++local;
        r = flipEdges( mesh, halo, maskFromGids( mesh, { g0 } ) );
        local += checkAll( mesh, V, E, F );
        if ( r.accepted != 1 )
            ++local;
        if ( gidFaceSignature( mesh ) != faces0 )
            ++local; // involution must hold on a splitEdges() result too
    }

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case11 (editing-family guard, flip after split) "
                     "%s\n",
                     tag, glob == 0 ? "ok" : "FAIL" );
    return glob == 0 ? 0 : 1;
}

//! Case 12: composed with split. Five rounds of (length-driven split, then
//! three valence-driven flip passes), with Euler and conformity after every
//! operation and the minimum radius ratio reported and floored per round.
template <class MeshT, class Exec>
static int caseComposedWithSplit( int rank, const char* tag )
{
    using mem = typename Exec::memory_space;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    setup<MeshT, Exec>( mesh, halo );

    int local = 0;
    const int nRounds = 5;
    for ( int round = 0; round < nRounds; ++round )
    {
        int maskFails = 0;
        const SplitResult sres =
            splitEdges( mesh, halo, aboveMeanLengthMask( mesh, maskFails ) );
        local += maskFails;
        if ( sres.split <= 0 )
            ++local; // vacuous round
        local += TesseraTest::checkConforming( mesh );
        if ( TesseraTest::checkOwnedEuler( mesh ) != 2 )
            ++local;

        long long V, E, F;
        counts( mesh, V, E, F );
        long long accepted = 0;
        for ( int pass = 0; pass < 3; ++pass )
        {
            long long cons = 0;
            const FlipResult fres =
                flipEdges( mesh, halo, valenceMask( mesh, cons ) );
            accepted += fres.accepted;
            local += checkCountersPartition( fres );
            // A flip changes NOTHING about V, E, F, so the identity is
            // re-asserted after every pass rather than once per round.
            local += checkAll( mesh, V, E, F );
        }

        double minAngle = 180.0;
        int qFails = 0;
        const double q = TesseraTest::minRadiusRatio( mesh, qFails, minAngle );
        local += qFails;
        if ( q < kMinRadiusRatioFloor )
            ++local;
        if ( rank == 0 )
            std::printf( "  [%s] case12 round%d: split=%lld flipped=%lld "
                         "V=%lld E=%lld F=%lld minRadiusRatio=%.4f "
                         "minAngle=%.3f\n",
                         tag, round + 1, sres.split, accepted, V, E, F, q,
                         minAngle );
    }
    local += TesseraTest::checkNoInteriorVertex( mesh );

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case12 (%d split+flip rounds, floor %.2f) %s\n",
                     tag, nRounds, kMinRadiusRatioFloor,
                     glob == 0 ? "ok" : "FAIL" );
    return glob == 0 ? 0 : 1;
}

// ===========================================================================

template <class Exec>
static int run( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT =
        Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>, mem, Exec>;

    int fails = 0;
    fails += caseInvolution<MeshT, Exec>( rank, tag );
    fails += caseDuplicateEdge<MeshT, Exec>( rank, tag );
    fails += caseIndependentSet<MeshT, Exec>( rank, size, tag );
    fails += caseGeometricRejection<MeshT, Exec>( rank, tag );
    fails += caseValence<MeshT, Exec>( rank, tag );
    fails += caseEmptyAndHalo<MeshT, Exec>( rank, size, tag );
    fails += caseFamilyGuard<MeshT, Exec>( rank, tag );
    fails += caseComposedWithSplit<MeshT, Exec>( rank, tag );
    std::fflush( stdout );
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
            std::printf( "test_flip_edges: caller-driven edge flip (size %d)\n",
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
