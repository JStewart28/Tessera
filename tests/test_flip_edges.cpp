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
//  13. THE QUALITY HOOK. A flat-topped hexagonal bipyramid, one spoke marked,
//      and minQuality = kHookThreshold, which falls between the two new faces'
//      worst radius ratio and their worst mean ratio 4*sqrt(3)*area / sum(l^2).
//      The default policy rejects the flip as rejectedGeometric; a policy
//      deriving from it and shadowing quality() with the mean ratio accepts it,
//      on the same mesh and the same mask.
//  14. THE ENDPOINT-IN-QUAD CONFLICT RELATION. Under FlipConflict::
//      EndpointInQuad no accepted flip's endpoint is a corner of another
//      accepted flip -- checked from the gathered flip map, whose (old key,
//      new key) pairs give each flip's endpoints and opposite corners. On a
//      two-ring bipyramid: two flips sharing only an opposite corner are both
//      accepted; two pairs, sharing no face but each with one flip's endpoint
//      among the other's corners, are accepted whole by SharedFace and lose
//      exactly one flip to rejectedConflict under EndpointInQuad, one pair
//      through each branch of the endorsement rule. On the icosphere with every
//      edge marked, the accepted set equals the MPI_COMM_SELF one.
//  15. NO TWO ACCEPTED FLIPS CREATE THE SAME DIAGONAL. On the flat bipyramid,
//      ring edges (2,3) and (5,6) both have opposite corners 0 and 1, share no
//      face, and have no endpoint among each other's corners, so only the
//      claim on the new diagonal (0,1) separates them. Under both relations
//      exactly one is accepted and the other counts into rejectedConflict;
//      the winner is the priority's pick and equals the MPI_COMM_SELF one, and
//      checkKeyTables finds no duplicate key. On a copy with ring vertex 5
//      pushed outward, the higher-priority (5,6) fails geometry and does not
//      veto (2,3): one accepted, one rejectedGeometric, no conflict.
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

// ---------------------------------------------------------------------------
// Case 13's fixture: a FLAT-TOPPED HEXAGONAL BIPYRAMID.
// ---------------------------------------------------------------------------
//
// A unit hexagonal ring at z = 0, the top apex 0 at z = 0.1 and the bottom apex
// 1 at z = -1; ring vertex i is gid 2 + i. Flipping the spoke (0,2) replaces the
// two top faces on it with the shallow top face (0,7,3) and the ring triangle
// (2,3,7), whose 120-degree corner makes it an obtuse CAP -- the shape on which
// the radius ratio falls well below the mean ratio. Computed from these
// coordinates with the library's formulas, and printed by the case at full
// precision:
//
//   worst radius ratio  0.46410161513775444   (detail::flipRadiusRatio)
//   worst mean ratio    0.59999999999999987   (4*sqrt(3)*area / sum(l^2))
//   normal deviations   0.0977 and 0.0997 rad, under maxNormalDeviation = 0.35
//
// Both worsts are the ring triangle's. (3,7) is not an edge, so the duplicate
// test passes and the geometric test alone decides.
static const double kHookThreshold = 0.55;

static TriangleSoup<double> flatBipyramidSoup( EdgeKey& flipMe )
{
    const int n = 6;
    TriangleSoup<double> soup;
    auto push = [&]( double x, double y, double z )
    {
        soup.positions.push_back( x );
        soup.positions.push_back( y );
        soup.positions.push_back( z );
    };
    push( 0.0, 0.0, 0.1 );
    push( 0.0, 0.0, -1.0 );
    for ( int i = 0; i < n; ++i )
    {
        const double a = 2.0 * 3.14159265358979323846 * i / n;
        push( std::cos( a ), std::sin( a ), 0.0 );
    }
    for ( int i = 0; i < n; ++i )
    {
        // Top faces CCW seen from above, bottom faces the other way, so the
        // surface is consistently oriented.
        const int r0 = 2 + i, r1 = 2 + ( i + 1 ) % n;
        const int top[3] = { 0, r0, r1 };
        const int bot[3] = { 1, r1, r0 };
        for ( int k = 0; k < 3; ++k )
            soup.triangles.push_back( top[k] );
        for ( int k = 0; k < 3; ++k )
            soup.triangles.push_back( bot[k] );
    }
    flipMe = makeEdgeKey( static_cast<GlobalId>( 0 ),
                          static_cast<GlobalId>( 2 ) );
    return soup;
}

//! The mean ratio 4*sqrt(3)*area / (sum of squared sides): 1 for an equilateral
//! triangle, 0 for a degenerate one. Not monotone in the radius ratio, which is
//! what lets one threshold separate the two.
static inline double meanRatio( const double* p0, const double* p1,
                                const double* p2 )
{
    const double* p[3] = { p0, p1, p2 };
    double n[3], l2 = 0.0;
    for ( int k = 0; k < 3; ++k )
        for ( int d = 0; d < 3; ++d )
        {
            const double dd = p[( k + 1 ) % 3][d] - p[k][d];
            l2 += dd * dd;
        }
    const double u[3] = { p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] };
    const double v[3] = { p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2] };
    n[0] = u[1] * v[2] - u[2] * v[1];
    n[1] = u[2] * v[0] - u[0] * v[2];
    n[2] = u[0] * v[1] - u[1] * v[0];
    const double area =
        0.5 * std::sqrt( n[0] * n[0] + n[1] * n[1] + n[2] * n[2] );
    if ( !( l2 > 0.0 ) )
        return 0.0;
    return 4.0 * std::sqrt( 3.0 ) * area / l2;
}

//! The default policy with only its quality metric replaced -- the shape a
//! consumer's policy takes.
struct MeanRatioFlipPolicy : DefaultFlipPolicy
{
    double quality( const double* p0, const double* p1, const double* p2 ) const
    {
        return meanRatio( p0, p1, p2 );
    }
};

template <class MeshT, class Exec, class Policy>
static FlipResult hookFlip( const TriangleSoup<double>& soup, const EdgeKey& k,
                            const Policy& policy, int& local )
{
    using mem = typename Exec::memory_space;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    buildFromTriangleSoup( mesh, soup );
    auto faceOwner = facePartitionByAxis( mesh );
    distribute( mesh, halo, faceOwner );
    long long V, E, F;
    counts( mesh, V, E, F );
    if ( V != 8 || E != 18 || F != 12 )
        ++local; // fixture drifted
    const FlipResult res =
        flipEdges( mesh, halo, maskFromKeys( mesh, { k } ), policy );
    local += checkAll( mesh, V, E, F );
    local += checkCountersPartition( res );
    if ( res.requested != 1 )
        ++local; // the spoke was not found
    if ( res.rejectedBoundary != 0 || res.rejectedDuplicateEdge != 0 )
        ++local; // only the geometric test may decide
    local += TesseraTest::checkKeyTables( mesh );
    return res;
}

//! Case 13: the policy's quality() is the metric minQuality thresholds.
template <class MeshT, class Exec>
static int caseQualityHook( int rank, const char* tag )
{
    EdgeKey k;
    const TriangleSoup<double> soup = flatBipyramidSoup( k );
    int local = 0;

    // THE PREMISE, from the soup: the new faces (lo,x,w) = (0,7,3) and
    // (hi,w,x) = (2,3,7). Their worst mean ratio clears kHookThreshold and
    // their worst radius ratio does not.
    double worstRR = 1e300, worstMR = 1e300;
    {
        const double* P = soup.positions.data();
        const int nf[2][3] = { { 0, 7, 3 }, { 2, 3, 7 } };
        for ( const auto& f : nf )
        {
            const double* q0 = P + 3 * f[0];
            const double* q1 = P + 3 * f[1];
            const double* q2 = P + 3 * f[2];
            worstRR = std::min(
                worstRR, Tessera::detail::flipRadiusRatio( q0, q1, q2 ) );
            worstMR = std::min( worstMR, meanRatio( q0, q1, q2 ) );
        }
    }
    if ( !( worstMR >= kHookThreshold && worstRR < kHookThreshold ) )
        ++local; // the fixture no longer separates the two metrics

    DefaultFlipPolicy def;
    def.minQuality = kHookThreshold;
    MeanRatioFlipPolicy mr;
    mr.minQuality = kHookThreshold;

    const FlipResult rd = hookFlip<MeshT, Exec>( soup, k, def, local );
    if ( rd.accepted != 0 || rd.rejectedGeometric != 1 )
        ++local; // the default radius ratio fails the threshold
    const FlipResult rm = hookFlip<MeshT, Exec>( soup, k, mr, local );
    if ( rm.accepted != 1 || rm.rejectedGeometric != 0 )
        ++local; // the shadowed quality() must be the one thresholded

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case13 (quality hook, minQuality=%.17g) %s: worst "
                     "radius ratio %.17g, worst mean ratio %.17g; default "
                     "accepted=%lld rejectedGeometric=%lld, mean-ratio policy "
                     "accepted=%lld rejectedGeometric=%lld\n",
                     tag, kHookThreshold, glob == 0 ? "ok" : "FAIL", worstRR,
                     worstMR, rd.accepted, rd.rejectedGeometric, rm.accepted,
                     rm.rejectedGeometric );
    return glob == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Case 14's fixture: a TWO-RING BIPYRAMID.
// ---------------------------------------------------------------------------
//
// Top apex T = 0 at z = 1.5 and bottom apex B = 1 at z = -0.6; an upper ring
// R_i = 2 + i of radius 1 at z = 0.5 and a lower ring S_i = 10 + i of radius 0.6
// at z = -0.5, i = 0..7, both at angle 2*pi*i/8. T fans over R, B fans under S,
// and the band splits each trapezoid (R_i, R_i+1, S_i+1, S_i) on the diagonal
// (R_i+1, S_i). The solid is convex and contains the origin, so each triangle
// is wound outward by the sign of its normal against its centroid. V 18, E 48,
// F 32.
//
// The four edges the case marks, with endpoints | opposite corners and squared
// length:
//
//   ring  (R0,R1) = (2,3)   | T, S0 = 0, 10    2 - sqrt(2) = 0.586
//   ring  (R4,R5) = (6,7)   | T, S4 = 0, 14    0.586
//   spoke (T,R4)  = (0,6)   | R3, R5 = 5, 7    2
//   spoke (B,S0)  = (1,10)  | S7, S1 = 17, 11  0.37
//
// and the three pairs it marks, no pair sharing a face:
//
//   shared opposite -- (2,3) + (6,7) share only the opposite corner T. No
//     conflict under either relation, so both are accepted: EndpointInQuad is
//     not plain vertex disjointness.
//   endpoint wins   -- (2,3) + (0,6). T is an endpoint of the spoke and an
//     opposite corner of the ring edge. The best candidate at T is the longer
//     spoke, which names T as an endpoint, so it alone is endorsed there.
//   opposite wins   -- (2,3) + (1,10). S0 is an endpoint of the spoke and an
//     opposite corner of the ring edge. The best candidate at S0 is the longer
//     ring edge, which names S0 as opposite, so the spoke is not endorsed.
//
// No flip creates an existing edge, and the case uses the permissive policy
// {maxNormalDeviation = pi, minQuality = 0}, so the independent-set round alone
// decides. A Python model of both relations on this soup gives the same
// accepted sets the case asserts.
static TriangleSoup<double> twoRingBipyramidSoup()
{
    const int n = 8;
    const double twoPi = 2.0 * 3.14159265358979323846;
    TriangleSoup<double> soup;
    auto push = [&]( double x, double y, double z )
    {
        soup.positions.push_back( x );
        soup.positions.push_back( y );
        soup.positions.push_back( z );
    };
    push( 0.0, 0.0, 1.5 );
    push( 0.0, 0.0, -0.6 );
    for ( int i = 0; i < n; ++i )
        push( std::cos( twoPi * i / n ), std::sin( twoPi * i / n ), 0.5 );
    for ( int i = 0; i < n; ++i )
        push( 0.6 * std::cos( twoPi * i / n ), 0.6 * std::sin( twoPi * i / n ),
              -0.5 );

    auto tri = [&]( int a, int b, int c )
    {
        const double* p = soup.positions.data();
        double u[3], v[3], nrm[3], dot = 0.0;
        for ( int d = 0; d < 3; ++d )
        {
            u[d] = p[3 * b + d] - p[3 * a + d];
            v[d] = p[3 * c + d] - p[3 * a + d];
        }
        nrm[0] = u[1] * v[2] - u[2] * v[1];
        nrm[1] = u[2] * v[0] - u[0] * v[2];
        nrm[2] = u[0] * v[1] - u[1] * v[0];
        for ( int d = 0; d < 3; ++d )
            dot += nrm[d] * ( p[3 * a + d] + p[3 * b + d] + p[3 * c + d] );
        if ( dot < 0.0 )
            std::swap( b, c );
        soup.triangles.push_back( a );
        soup.triangles.push_back( b );
        soup.triangles.push_back( c );
    };
    for ( int i = 0; i < n; ++i )
    {
        const int r0 = 2 + i, r1 = 2 + ( i + 1 ) % n;
        const int s0 = 10 + i, s1 = 10 + ( i + 1 ) % n;
        tri( 0, r0, r1 );
        tri( 1, s1, s0 );
        tri( r0, r1, s0 );
        tri( r1, s1, s0 );
    }
    return soup;
}

//! The number of ORDERED pairs (E1, E2) of distinct flips in `flips` with an
//! endpoint of E1 among the four corners of E2 -- the pairs EndpointInQuad
//! forbids. A flip's corners are its old key's ids and its new key's ids.
static long long endpointInQuadViolations(
    const std::set<std::pair<EdgeKey, EdgeKey>>& flips )
{
    std::map<GlobalId, long long> endpointOf;
    for ( const auto& kv : flips )
        for ( int j = 0; j < 2; ++j )
            ++endpointOf[kv.first.id[j]];
    long long bad = 0;
    for ( const auto& kv : flips )
    {
        const GlobalId corner[4] = { kv.first.id[0], kv.first.id[1],
                                     kv.second.id[0], kv.second.id[1] };
        for ( int j = 0; j < 4; ++j )
        {
            auto it = endpointOf.find( corner[j] );
            if ( it != endpointOf.end() )
                bad += it->second - ( j < 2 ? 1 : 0 ); // not itself
        }
    }
    return bad;
}

//! One flipEdges() under `mode` with the permissive policy, on the case-14 soup
//! with `marked` marked, or (soup == nullptr) on the subdiv-2 icosphere with
//! every owned edge marked. Returns the result and the global flip set.
template <class MeshT, class Exec>
static FlipResult conflictRun( MPI_Comm comm, const TriangleSoup<double>* soup,
                               const std::set<EdgeKey>& marked,
                               FlipConflict mode,
                               std::set<std::pair<EdgeKey, EdgeKey>>& flips,
                               int& local )
{
    using mem = typename Exec::memory_space;
    MeshT mesh( comm );
    MeshHalo<mem> halo;
    if ( soup )
    {
        buildFromTriangleSoup( mesh, *soup );
        auto faceOwner = facePartitionByAxis( mesh );
        distribute( mesh, halo, faceOwner );
    }
    else
        setupOn<MeshT, Exec>( comm, mesh, halo );

    long long V, E, F;
    counts( mesh, V, E, F );
    if ( soup ? ( V != 18 || E != 48 || F != 32 )
              : ( V != V0 || E != E0 || F != F0 ) )
        ++local; // fixture drifted

    DefaultFlipPolicy policy;
    policy.maxNormalDeviation = std::acos( -1.0 );
    policy.minQuality = 0.0;
    policy.conflict = mode;
    const std::vector<char> mask =
        soup ? maskFromKeys( mesh, marked )
             : std::vector<char>( mesh.numOwnedEdges(), 1 );
    const FlipResult res = flipEdges( mesh, halo, mask, policy );

    local += checkAll( mesh, V, E, F );
    local += checkCountersPartition( res );
    local += TesseraTest::checkKeyTables( mesh );
    flips = globalFlipSet( comm, res );
    if ( static_cast<long long>( flips.size() ) != res.accepted )
        ++local; // the flip map is not complete/consistent
    return res;
}

static std::string keyStr( const EdgeKey& k )
{
    return "(" + std::to_string( static_cast<unsigned long long>( k.id[0] ) ) +
           "," + std::to_string( static_cast<unsigned long long>( k.id[1] ) ) +
           ")";
}

//! Case 14: FlipConflict::EndpointInQuad accepts no pair in which an endpoint
//! of one flip is a corner of another, accepts two flips sharing only an
//! opposite corner, and rejects exactly one of a pair that SharedFace accepts
//! whole. The accepted sets are rank-count invariant.
template <class MeshT, class Exec>
static int caseEndpointInQuad( int rank, int size, const char* tag )
{
    using FlipSet = std::set<std::pair<EdgeKey, EdgeKey>>;
    int local = 0;
    const TriangleSoup<double> soup = twoRingBipyramidSoup();
    auto key = []( int a, int b )
    {
        return makeEdgeKey( static_cast<GlobalId>( a ),
                            static_cast<GlobalId>( b ) );
    };

    struct Pair
    {
        const char* name;
        EdgeKey a, b;
        std::set<EdgeKey> eiqAccepted; // expected under EndpointInQuad
    };
    const Pair pairs[3] = {
        { "shared opposite", key( 2, 3 ), key( 6, 7 ),
          { key( 2, 3 ), key( 6, 7 ) } },
        { "endpoint wins", key( 2, 3 ), key( 0, 6 ), { key( 0, 6 ) } },
        { "opposite wins", key( 2, 3 ), key( 1, 10 ), { key( 2, 3 ) } } };

    for ( const Pair& p : pairs )
    {
        const std::set<EdgeKey> marked = { p.a, p.b };
        FlipSet sfSet, eiqSet;
        const FlipResult sf = conflictRun<MeshT, Exec>(
            MPI_COMM_WORLD, &soup, marked, FlipConflict::SharedFace, sfSet,
            local );
        const FlipResult eiq = conflictRun<MeshT, Exec>(
            MPI_COMM_WORLD, &soup, marked, FlipConflict::EndpointInQuad,
            eiqSet, local );

        std::set<EdgeKey> sfKeys, eiqKeys;
        for ( const auto& kv : sfSet )
            sfKeys.insert( kv.first );
        for ( const auto& kv : eiqSet )
            eiqKeys.insert( kv.first );

        for ( const FlipResult* r : { &sf, &eiq } )
            if ( r->requested != 2 || r->rejectedBoundary != 0 ||
                 r->rejectedDuplicateEdge != 0 || r->rejectedGeometric != 0 )
                ++local; // only the independent-set round may decide
        // No pair shares a face, so SharedFace accepts both.
        if ( sf.accepted != 2 || sf.rejectedConflict != 0 || sfKeys != marked )
            ++local;
        if ( eiqKeys != p.eiqAccepted )
            ++local;
        if ( eiq.accepted != static_cast<long long>( p.eiqAccepted.size() ) ||
             eiq.rejectedConflict != 2 - eiq.accepted )
            ++local;
        const long long sfBad = endpointInQuadViolations( sfSet );
        const long long eiqBad = endpointInQuadViolations( eiqSet );
        if ( eiqBad != 0 )
            ++local;
        // The fixture must be a real conflict exactly when EIQ rejects one.
        if ( ( sfBad > 0 ) != ( p.eiqAccepted.size() == 1 ) )
            ++local;

        if ( rank == 0 )
        {
            std::string acc;
            for ( const EdgeKey& k : eiqKeys )
                acc += ( acc.empty() ? "" : " " ) + keyStr( k );
            std::printf( "  [%s] case14 %s %s+%s: SharedFace accepted=%lld "
                         "conflict=%lld violations=%lld; EndpointInQuad "
                         "accepted=%lld conflict=%lld violations=%lld "
                         "{%s}\n",
                         tag, p.name, keyStr( p.a ).c_str(),
                         keyStr( p.b ).c_str(), sf.accepted,
                         sf.rejectedConflict, sfBad, eiq.accepted,
                         eiq.rejectedConflict, eiqBad, acc.c_str() );
        }
    }

    // Every owned edge of the icosphere marked: no accepted pair conflicts,
    // and the accepted set equals the one the same run computes alone.
    FlipSet sfW, eiqW, eiqS;
    const std::set<EdgeKey> none;
    const FlipResult sf = conflictRun<MeshT, Exec>(
        MPI_COMM_WORLD, nullptr, none, FlipConflict::SharedFace, sfW, local );
    const FlipResult w = conflictRun<MeshT, Exec>(
        MPI_COMM_WORLD, nullptr, none, FlipConflict::EndpointInQuad, eiqW,
        local );
    const FlipResult s = conflictRun<MeshT, Exec>(
        MPI_COMM_SELF, nullptr, none, FlipConflict::EndpointInQuad, eiqS,
        local );
    const long long sfBad = endpointInQuadViolations( sfW );
    const long long wBad = endpointInQuadViolations( eiqW );
    const long long sBad = endpointInQuadViolations( eiqS );
    if ( wBad != 0 || sBad != 0 )
        ++local;
    if ( w.accepted <= 0 )
        ++local; // vacuous: the case proves nothing if nothing was flipped
    if ( eiqW != eiqS )
        ++local; // the accepted set depends on the decomposition
    if ( w.accepted != s.accepted || w.requested != s.requested ||
         w.rejectedBoundary != s.rejectedBoundary ||
         w.rejectedDuplicateEdge != s.rejectedDuplicateEdge ||
         w.rejectedGeometric != s.rejectedGeometric ||
         w.rejectedConflict != s.rejectedConflict )
        ++local;

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
    {
        std::printf( "  [%s] case14 icosphere all marked: requested=%lld; "
                     "SharedFace accepted=%lld conflict=%lld violations=%lld; "
                     "EndpointInQuad accepted=%lld dup=%lld geom=%lld "
                     "conflict=%lld violations=%lld (SELF accepted=%lld "
                     "violations=%lld, set %s)\n",
                     tag, w.requested, sf.accepted, sf.rejectedConflict,
                     sfBad, w.accepted, w.rejectedDuplicateEdge,
                     w.rejectedGeometric, w.rejectedConflict, wBad,
                     s.accepted, sBad, eiqW == eiqS ? "identical" : "DIFFERS" );
        std::printf( "  [%s] case14 (endpoint-in-quad conflict, np%d) %s\n",
                     tag, size, glob == 0 ? "ok" : "FAIL" );
    }
    return glob == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Case 15's fixtures: the flat bipyramid, and a copy with ring vertex 5 pushed
// out to radius kPushOut.
// ---------------------------------------------------------------------------
//
// Ring edges (2,3) and (5,6) both have opposite corners 0 and 1, so both flips
// create the axis (0,1), which is not an edge. They share no face, and neither
// has an endpoint among the other's corners, so neither FlipConflict relation
// separates them; only the claim on the new diagonal does. The new faces are
// (lo, x, w) = (lo, 1, 0) and (hi, w, x) = (hi, 0, 1).
//
// On flatBipyramidSoup itself the two ring edges have the same length up to
// rounding (a Python transcription of the soup gives squared lengths
// 0.99999999999999978 and 0.99999999999999922), so the priority picks one of
// them and the case asserts only that the pick is rank-count invariant and is
// the one the soup coordinates predict.
//
// On the perturbed copy (5,6) is longer -- squared length 2.9999999999999987
// against 0.99999999999999978 -- and so has the higher priority, and its new
// face through vertex 5 is elongated: worst radius ratio 0.73625936037355844,
// against 0.87969658092961811 for (2,3). minQuality = kDiagThreshold sits
// between them, so (5,6) fails geometry and must not veto (2,3).
static const double kPushOut = 2.0;
static const double kDiagThreshold = 0.8;

static TriangleSoup<double> pushedBipyramidSoup()
{
    EdgeKey unused;
    TriangleSoup<double> soup = flatBipyramidSoup( unused );
    soup.positions[3 * 5 + 0] *= kPushOut;
    soup.positions[3 * 5 + 1] *= kPushOut;
    return soup;
}

//! One flipEdges() under `mode` on a flat-bipyramid soup, with `marked`
//! marked, maxNormalDeviation = pi and minQuality = `minQuality`. Returns the
//! result and the global flip set.
template <class MeshT, class Exec>
static FlipResult diagonalRun( MPI_Comm comm, const TriangleSoup<double>& soup,
                               const std::set<EdgeKey>& marked,
                               FlipConflict mode, double minQuality,
                               std::set<std::pair<EdgeKey, EdgeKey>>& flips,
                               int& local )
{
    using mem = typename Exec::memory_space;
    MeshT mesh( comm );
    MeshHalo<mem> halo;
    buildFromTriangleSoup( mesh, soup );
    auto faceOwner = facePartitionByAxis( mesh );
    distribute( mesh, halo, faceOwner );

    long long V, E, F;
    counts( mesh, V, E, F );
    if ( V != 8 || E != 18 || F != 12 )
        ++local; // fixture drifted

    DefaultFlipPolicy policy;
    policy.maxNormalDeviation = std::acos( -1.0 );
    policy.minQuality = minQuality;
    policy.conflict = mode;
    const FlipResult res =
        flipEdges( mesh, halo, maskFromKeys( mesh, marked ), policy );

    local += checkAll( mesh, V, E, F );
    local += checkCountersPartition( res );
    local += TesseraTest::checkKeyTables( mesh ); // no two edges keyed (0,1)
    flips = globalFlipSet( comm, res );
    if ( static_cast<long long>( flips.size() ) != res.accepted )
        ++local; // the flip map is not complete/consistent
    if ( res.requested != static_cast<long long>( marked.size() ) ||
         res.rejectedBoundary != 0 || res.rejectedDuplicateEdge != 0 )
        ++local; // (0,1) is not a pre-flip edge
    return res;
}

//! Case 15: two flips that would create the same new diagonal are never both
//! accepted, under either relation. The higher-priority geometric survivor
//! wins and the other counts into rejectedConflict; a higher-priority claimant
//! that fails geometry does not veto the other.
template <class MeshT, class Exec>
static int caseSameDiagonal( int rank, const char* tag )
{
    using FlipSet = std::set<std::pair<EdgeKey, EdgeKey>>;
    int local = 0;
    auto key = []( int a, int b )
    {
        return makeEdgeKey( static_cast<GlobalId>( a ),
                            static_cast<GlobalId>( b ) );
    };
    const EdgeKey e23 = key( 2, 3 ), e56 = key( 5, 6 ), axis = key( 0, 1 );
    const std::set<EdgeKey> marked = { e23, e56 };
    EdgeKey unused;
    const TriangleSoup<double> flat = flatBipyramidSoup( unused );
    const TriangleSoup<double> pushed = pushedBipyramidSoup();

    // THE PREMISE, from the soup coordinates: each ring edge's squared length
    // (as the library computes its priority) and the worst radius ratio of its
    // new faces (lo, 1, 0) and (hi, 0, 1).
    auto len2Of = []( const TriangleSoup<double>& s, const EdgeKey& k )
    {
        const double* P = s.positions.data();
        return edgeLen2Canonical( k.id[0], P + 3 * k.id[0], k.id[1],
                                  P + 3 * k.id[1], 3 );
    };
    auto worstNew = []( const TriangleSoup<double>& s, const EdgeKey& k )
    {
        const double* P = s.positions.data();
        const double* lo = P + 3 * k.id[0];
        const double* hi = P + 3 * k.id[1];
        return std::min(
            Tessera::detail::flipRadiusRatio( lo, P + 3, P + 0 ),
            Tessera::detail::flipRadiusRatio( hi, P + 0, P + 3 ) );
    };
    const double flat23 = len2Of( flat, e23 ), flat56 = len2Of( flat, e56 );
    const double push23 = len2Of( pushed, e23 ), push56 = len2Of( pushed, e56 );
    const double q23 = worstNew( pushed, e23 ), q56 = worstNew( pushed, e56 );
    if ( !( push56 > push23 ) )
        ++local; // (5,6) must have the higher priority on the pushed soup
    if ( !( q56 < kDiagThreshold && q23 >= kDiagThreshold ) )
        ++local; // the threshold must separate the two claimants
    // The flat soup's winner, by (squared length DESCENDING, key ASCENDING).
    const EdgeKey flatWinner =
        ( flat56 > flat23 || ( flat56 == flat23 && e56 < e23 ) ) ? e56 : e23;

    const FlipConflict modes[2] = { FlipConflict::SharedFace,
                                    FlipConflict::EndpointInQuad };
    const char* names[2] = { "SharedFace", "EndpointInQuad" };
    std::string line[2];
    for ( int m = 0; m < 2; ++m )
    {
        // Success: the same-diagonal pair accepts exactly one flip, the one
        // the priority picks, at every rank count.
        FlipSet w, s;
        const FlipResult rw = diagonalRun<MeshT, Exec>(
            MPI_COMM_WORLD, flat, marked, modes[m], 0.0, w, local );
        const FlipResult rs = diagonalRun<MeshT, Exec>(
            MPI_COMM_SELF, flat, marked, modes[m], 0.0, s, local );
        if ( rw.accepted != 1 || rw.rejectedConflict != 1 ||
             rw.rejectedGeometric != 0 )
            ++local;
        if ( rs.accepted != rw.accepted ||
             rs.rejectedConflict != rw.rejectedConflict )
            ++local;
        if ( w != s )
            ++local; // the winner depends on the decomposition
        const FlipSet wantFlat = { { flatWinner, axis } };
        if ( w != wantFlat )
            ++local; // not the priority's pick, or not the axis

        // Failure direction: the higher-priority (5,6) fails geometry, so
        // (2,3) is accepted and (5,6) counts as geometric, not conflict.
        FlipSet p;
        const FlipResult rp = diagonalRun<MeshT, Exec>(
            MPI_COMM_WORLD, pushed, marked, modes[m], kDiagThreshold, p,
            local );
        if ( rp.accepted != 1 || rp.rejectedGeometric != 1 ||
             rp.rejectedConflict != 0 )
            ++local;
        const FlipSet wantPushed = { { e23, axis } };
        if ( p != wantPushed )
            ++local;

        char buf[512];
        std::snprintf( buf, sizeof( buf ),
                       "%s: same-diagonal pair accepted=%lld conflict=%lld "
                       "{%s->%s} (SELF set %s); pushed pair accepted=%lld "
                       "geom=%lld conflict=%lld {%s}",
                       names[m], rw.accepted, rw.rejectedConflict,
                       w.empty() ? "" : keyStr( w.begin()->first ).c_str(),
                       w.empty() ? "" : keyStr( w.begin()->second ).c_str(),
                       w == s ? "identical" : "DIFFERS", rp.accepted,
                       rp.rejectedGeometric, rp.rejectedConflict,
                       p.empty() ? "" : keyStr( p.begin()->first ).c_str() );
        line[m] = buf;
    }

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
    {
        std::printf( "  [%s] case15 premise: flat len2 (2,3)=%.17g "
                     "(5,6)=%.17g; pushed len2 (2,3)=%.17g (5,6)=%.17g, worst "
                     "new-face radius ratio (2,3)=%.17g (5,6)=%.17g, "
                     "minQuality=%.17g\n",
                     tag, flat23, flat56, push23, push56, q23, q56,
                     kDiagThreshold );
        for ( int m = 0; m < 2; ++m )
            std::printf( "  [%s] case15 %s\n", tag, line[m].c_str() );
        std::printf( "  [%s] case15 (same new diagonal) %s\n", tag,
                     glob == 0 ? "ok" : "FAIL" );
    }
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
    fails += caseQualityHook<MeshT, Exec>( rank, tag );
    fails += caseEndpointInQuad<MeshT, Exec>( rank, size, tag );
    fails += caseSameDiagonal<MeshT, Exec>( rank, tag );
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
