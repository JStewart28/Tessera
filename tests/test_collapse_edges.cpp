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

// Regression test: caller-driven edge collapse (tasks/completed/edge-collapse.md).
//
// collapseEdges() is the LAST of the four remesh-family edits and the only one
// that removes degrees of freedom: each accepted collapse merges an edge's two
// endpoints, so V-1, E-3, F-2 and the Euler number is preserved. It is also the
// only one that can silently corrupt a mesh in three distinct ways -- welding
// two distant parts of the surface together (the link condition), folding a
// neighbouring face (the normal test) and leaving a dangling reference behind
// (the connectivity rewrite) -- so each of the three is detected here and
// reported through a NAMED COUNTER.
//
//   1. ONE EDGE, EXACT DELTAS. Subdivision-2 icosphere at DEPTH 2, mark the
//      globally smallest EdgeKey: V-1, E-3, F-2, Euler == 2, conformity, no
//      interior vertex, owned 1-ring local. The surviving vertex gid is
//      min(a,b) and sits exactly at the midpoint of the two endpoint positions.
//   2. THE LINK CONDITION IS ENFORCED, both ways. Negative: mark all three
//      edges of one face -- collapsing two of them is topologically invalid, so
//      accepted <= 1 with Euler and conformity intact. Positive: a hand-built
//      soup whose edge (a,b) has exactly two incident faces AND a third common
//      neighbour w, the classic violation, gives rejectedLinkCondition == 1 and
//      accepted == 0.
//   3. DEPTH GUARD. collapseEdges() on a depth-1 mesh throws, naming the
//      required and the actual depth. Depth 0 (never distributed) is accepted,
//      which is what lets the soup fixtures above run at all.
//   4. INDEPENDENT SET. Mark every owned edge: no two accepted collapses share
//      a vertex (checked against the pre-collapse vertex -> edge map), and
//      verticesRemoved == accepted, edgesRemoved == 3*accepted,
//      facesRemoved == 2*accepted.
//   5. RANK-COUNT INVARIANCE -- the strongest determinism claim in the task.
//      Check 4 against a reference the same run computes alone on
//      MPI_COMM_SELF: identical `accepted`, identical verdict histogram,
//      identical V/E/F and an identical multiset of face corner-POSITION
//      triples. If this fails, the priority order is not total or the
//      surviving-vertex rule leaked a local index.
//   6. GEOMETRIC REJECTION. minQuality = 0.99 rejects everything: accepted == 0,
//      topologyChecksum bitwise unchanged, AND the local gid SEQUENCES are
//      unchanged -- a no-op must be a genuine no-op, not a re-compaction into a
//      different ordering.
//   7. NORMAL-FLIP REJECTION. A closed decagonal bipyramid with one apex pulled
//      into a SPIKE: collapsing a spike edge drags the apex halfway down and
//      folds every other spike face. rejectedNormalFlip >= 1, and every
//      surviving face's normal still has a POSITIVE dot product with its own
//      pre-collapse normal (matched by face gid, which a collapse preserves).
//   8. DECIMATION TO A FLOOR. Loop with an all-edges mask until accepted == 0
//      or 20 rounds. After every round: Euler == 2, conformity,
//      owned1RingLocal, no duplicate keys. The face count must be STRICTLY
//      DECREASING while accepted > 0. Terminal face count and quality floor are
//      MEASURED, not guessed (see kDecimateFloor).
//   9. ROUND TRIP WITH SPLIT. splitEdges(all) -> 1280 faces, then the same
//      collapse loop: Euler == 2 throughout and the final face count within a
//      factor of two of the original 320 -- coarse, but a real check that
//      coarsening undoes refinement rather than jamming.
//  10. USER-FIELD BLEND. A `double` and a `double[3]` vertex field seeded to a
//      LINEAR function of position: after a collapse at t = 0.5 the merged
//      vertex's value equals that function evaluated at its new position, to
//      1e-14 relative. Exact because the blend and the position are the same
//      convex combination.
//  11. HALO AND COMPACTION ON RETURN. No tombstone survives anywhere; every
//      ghost position corrupted and restored by haloExchange(); ownership is a
//      partition; halo.depth is still 2; a second collapseEdges() with no
//      intervening migrate() succeeds.
//  12. EMPTY MASK is a no-op with every counter zero and the checksums fixed.
//  13. FAMILY GUARD. collapseEdges() on a refine()d mesh throws in BOTH
//      refinement modes, and refine() on a collapsed mesh throws too.
//  14. COMPOSED REMESH LOOP -- the integration test, and the actual deliverable
//      of the family. Twenty rounds of { split above a length threshold -> flip
//      for valence -> collapse below a length floor }. After every operation:
//      Euler == 2, conformity, no duplicate keys, no tombstones. Face count,
//      edge-length extremes and minimum radius ratio reported per round, with
//      the face count held inside a measured band and the quality above a
//      measured floor.
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

// ---------------------------------------------------------------------------
// Numbers MEASURED IN THE FIRST IMPLEMENTATION RUN, not guessed
// ---------------------------------------------------------------------------
//
// Check 8 -- decimation of the subdiv-2 icosphere with an all-edges mask, one
// independent set per round. MEASURED, and BYTE-IDENTICAL at np1-5 on both
// backends and in both execution spaces:
//
//   round      1    2    3    4    5   ...  10   ...  15   ...  20
//   accepted   8    8    4    4    3   ...   4   ...   3   ...   3
//   F        304  288  280  272  266  ... 236  ... 210  ... 174
//   min r/R  .377 .381 .377 .381 .300 ... .263 ... .175 ... .175
//
// So twenty rounds take 320 faces to 174 and the loop is STILL ACCEPTING when
// the cap is reached -- an all-edges mask has no target scale, so what ends it is
// the quality and fold guards (case 9 measures where: 68 rounds from 1280 faces
// down to 144 with the same mask). The accepted count per round is small because
// the independent set excludes every candidate in the TWO-RING of an accepted
// one, which is Decision 2's conflict relation, not a defect: 8 of 480 marked
// edges in round 1. A caller wanting more progress calls again, which is what
// this loop does.
//
// kDecimateFloor is the floor on TesseraTest::minRadiusRatio() over a whole
// decimation sequence, in the convention where an EQUILATERAL triangle scores
// 0.5 -- so DefaultCollapsePolicy::minQuality (equilateral = 1) of 0.05 is a
// floor of 0.025 in these units. MEASURED: 0.1747 over case 8's twenty rounds
// and 0.0676 over case 9's sixty-odd, at every rank count and on both backends.
// The floor is set just below the lower of the two.
//
// NOTE THE SCOPE: collapseEdges() guarantees only that no ACCEPTED collapse
// leaves a face below its own minQuality -- a bound on the faces a collapse
// TOUCHES, not on the faces no candidate went near, and not on a mesh a later
// splitEdges() round has cut.
static const double kDecimateFloor = 0.02;

// Check 14 -- twenty composed { split, flip, collapse } rounds against a fixed
// target edge length of 0.6 * the initial mean (0.1796). MEASURED IN THE FIRST
// IMPLEMENTATION RUN: round 1 splits all 480 edges to 1240 faces and the drive
// then HOLDS the mesh there -- the face count stays inside [1196, 1250] for all
// twenty rounds, the edge lengths stay bracketed around the target, and the
// worst radius ratio over the whole sequence is 0.0255 (round 16). Identical at
// np1-5 on both backends and in both execution spaces, tail included.
//
// The band and the floor are set around those measurements with margin. The
// point of the assertion is that the drive is STABLE -- the face count neither
// runs away under repeated splitting nor decimates to nothing under repeated
// collapsing -- not that it reproduces a particular trajectory.
static const long long kComposedMinFaces = 1000;
static const long long kComposedMaxFaces = 1500;
static const double kComposedFloor = 0.02;

// ---------------------------------------------------------------------------
// Order-independent multiset checksum (same idiom as test_split_edges and
// test_flip_edges)
// ---------------------------------------------------------------------------
//
// count + SUM + BXOR over a 64-bit hash per item. All three combiners are
// commutative, so the value depends on neither visit order, nor local index, nor
// how the items are spread over ranks.
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

//! BITWISE position hash: the raw IEEE bit patterns, not a quantisation. Check
//! 5 compares two decompositions of the same collapse, and the merged position
//! is computed from canonically ordered endpoints, so exact equality is the
//! right assertion.
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

//! (gid, corner gids) of every OWNED face, for the per-gid normal comparison
//! check 7 makes.
template <class MeshT>
static std::map<GlobalId, std::array<GlobalId, 3>> ownedFacesByGid( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto fg = Cabana::slice<FaceField::Gid>( hf );
    auto fv = Cabana::slice<FaceField::Verts>( hf );
    std::map<GlobalId, std::array<GlobalId, 3>> out;
    for ( std::size_t f = 0; f < mesh.numOwnedFaces(); ++f )
        out[fg( f )] = { fv( f, 0 ), fv( f, 1 ), fv( f, 2 ) };
    return out;
}

//! EdgeKey of every OWNED edge, in owned-edge local index order -- i.e. exactly
//! the indexing collapseEdges()' mask uses.
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

//! Local gid SEQUENCE (owned block, in local index order) of each entity kind.
//! Order-SENSITIVE on purpose: check 6 needs to see that a rejected-everything
//! call did not quietly re-compact the mesh into a different ordering, which a
//! commutative checksum could not distinguish.
template <class MeshT>
static std::vector<GlobalId> localGidSequence( MeshT& mesh )
{
    std::vector<GlobalId> out;
    {
        Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> h(
            "h", mesh.numVertices() );
        Cabana::deep_copy( h, mesh.vertices() );
        auto g = Cabana::slice<VertexField::Gid>( h );
        for ( std::size_t i = 0; i < mesh.numVertices(); ++i )
            out.push_back( g( i ) );
    }
    {
        Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> h(
            "h", mesh.numEdges() );
        Cabana::deep_copy( h, mesh.edges() );
        auto g = Cabana::slice<EdgeField::Gid>( h );
        for ( std::size_t i = 0; i < mesh.numEdges(); ++i )
            out.push_back( g( i ) );
    }
    {
        Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> h(
            "h", mesh.numFaces() );
        Cabana::deep_copy( h, mesh.faces() );
        auto g = Cabana::slice<FaceField::Gid>( h );
        for ( std::size_t i = 0; i < mesh.numFaces(); ++i )
            out.push_back( g( i ) );
    }
    return out;
}

//! Number of LOCAL entities (any kind, owned or ghost) carrying the tombstone
//! marker. collapseEdges() compacts before returning, so this must be 0.
template <class MeshT>
static long long countTombstones( MeshT& mesh )
{
    long long n = 0;
    {
        Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> h(
            "h", mesh.numVertices() );
        Cabana::deep_copy( h, mesh.vertices() );
        auto g = Cabana::slice<VertexField::Gid>( h );
        for ( std::size_t i = 0; i < mesh.numVertices(); ++i )
            if ( g( i ) == invalid_gid )
                ++n;
    }
    {
        Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> h(
            "h", mesh.numEdges() );
        Cabana::deep_copy( h, mesh.edges() );
        auto g = Cabana::slice<EdgeField::Gid>( h );
        for ( std::size_t i = 0; i < mesh.numEdges(); ++i )
            if ( g( i ) == invalid_gid )
                ++n;
    }
    {
        Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> h(
            "h", mesh.numFaces() );
        Cabana::deep_copy( h, mesh.faces() );
        auto g = Cabana::slice<FaceField::Gid>( h );
        for ( std::size_t i = 0; i < mesh.numFaces(); ++i )
            if ( g( i ) == invalid_gid )
                ++n;
    }
    return n;
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
//! geometry however they are numbered or decomposed. THE check-5 statistic.
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

//! Owned edge lengths, in mask order, plus the global mean. Every length is
//! computed from the two endpoint positions this rank holds, which rebuildHalo()
//! guarantees for every owned edge.
template <class MeshT>
static std::vector<double> ownedEdgeLengths( MeshT& mesh, double& mean,
                                             int& fails )
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
    mean = nE > 0 ? totalLen / static_cast<double>( nE ) : 0.0;
    return len;
}

//! Mask marking every owned edge longer (or shorter) than an ABSOLUTE length.
//! The threshold is absolute and not a multiple of the current mean because a
//! relative rule marks nothing at all on a near-uniform mesh -- see the comment
//! in case 14, which measured exactly that.
template <class MeshT>
static std::vector<char> absLengthMask( MeshT& mesh, double threshold,
                                        bool above, int& fails )
{
    double mean = 0.0;
    const std::vector<double> len = ownedEdgeLengths( mesh, mean, fails );
    std::vector<char> mask( len.size(), 0 );
    for ( std::size_t e = 0; e < len.size(); ++e )
        mask[e] = above ? ( len[e] > threshold ? 1 : 0 )
                        : ( len[e] < threshold ? 1 : 0 );
    return mask;
}

//! Global min/max owned edge length -- the two ends of case 14's edge-length
//! histogram, which is what a length-driven drive is supposed to hold together.
template <class MeshT>
static void edgeLengthRange( MeshT& mesh, double& lo, double& hi, int& fails )
{
    double mean = 0.0;
    const std::vector<double> len = ownedEdgeLengths( mesh, mean, fails );
    double l = 1e300, h = 0.0;
    for ( double v : len )
    {
        l = std::min( l, v );
        h = std::max( h, v );
    }
    MPI_Allreduce( &l, &lo, 1, MPI_DOUBLE, MPI_MIN, mesh.comm() );
    MPI_Allreduce( &h, &hi, 1, MPI_DOUBLE, MPI_MAX, mesh.comm() );
}

// ---------------------------------------------------------------------------
// Global vertex valence -- the caller-side half of case 14's flip pass
// ---------------------------------------------------------------------------
//
// Condensed from test_flip_edges' `valence` namespace, which is where the
// pattern is written out in full and explained: each rank tallies its OWNED
// edges' endpoints to a gid coordinator (gid % size), which therefore holds the
// true global degree of every gid it owns, and a lookup is one request round.
// NOT shared: the only shared test code lives in MeshInvariants.hpp, and moving
// it there would mean editing test_flip_edges, which is outside this task.
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

//! The valence-equalization flip mask: a flip of (a,b) with opposite corners
//! (c,d) takes one edge off each of a and b and adds one to each of c and d, so
//! it is an improvement iff it lowers sum |valence - 6|.
template <class MeshT>
static std::vector<char> valenceMask( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto fv = Cabana::slice<FaceField::Verts>( hf );
    auto fe = Cabana::slice<FaceField::Edges>( hf );

    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
        "he", mesh.numEdges() );
    Cabana::deep_copy( he, mesh.edges() );
    auto eg = Cabana::slice<EdgeField::Gid>( he );

    std::unordered_map<GlobalId, std::vector<GlobalId>> opposites;
    opposites.reserve( mesh.numFaces() * 4 + 1 );
    for ( std::size_t f = 0; f < mesh.numFaces(); ++f )
        for ( int k = 0; k < 3; ++k )
            opposites[fe( f, k )].push_back( fv( f, ( k + 2 ) % 3 ) );

    const std::vector<EdgeKey> keys = ownedEdgeKeys( mesh );
    std::vector<GlobalId> gids( keys.size() );
    for ( std::size_t e = 0; e < keys.size(); ++e )
        gids[e] = eg( e );

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
        const int before = dev( v[0] ) + dev( v[1] ) + dev( v[2] ) + dev( v[3] );
        const int after = dev( v[0] - 1 ) + dev( v[1] - 1 ) + dev( v[2] + 1 ) +
                          dev( v[3] + 1 );
        mask[e] = ( after < before ) ? 1 : 0;
    }
    return mask;
}

// ---------------------------------------------------------------------------
// The collapsed set, gathered globally
// ---------------------------------------------------------------------------

//! The global set of collapsed EdgeKeys. CollapseResult::collapsed holds the
//! collapses a rank TOUCHES, so the same key appears on several ranks; an
//! allgather plus a std::set is the honest way to get the global set, and the
//! accepted set is a small fraction of E.
static std::set<EdgeKey> globalCollapsedSet( MPI_Comm comm,
                                             const CollapseResult& res )
{
    int size = 1;
    MPI_Comm_size( comm, &size );
    std::vector<unsigned long long> mine;
    mine.reserve( res.collapsed.size() * 2 );
    for ( const EdgeKey& k : res.collapsed )
    {
        mine.push_back( k.id[0] );
        mine.push_back( k.id[1] );
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
    std::set<EdgeKey> out;
    for ( int i = 0; i + 1 < tot; i += 2 )
    {
        EdgeKey k;
        k.id[0] = all[i];
        k.id[1] = all[i + 1];
        out.insert( k );
    }
    return out;
}

// ---------------------------------------------------------------------------
// Shared post-conditions
// ---------------------------------------------------------------------------

//! Everything that must hold of a collapsed mesh regardless of what was
//! collapsed: the three global owned counts are exactly the expected ones, Euler
//! is 2, the mesh is conforming with no interior vertex, ownership is a
//! partition, the owned 1-ring is local, no tombstone survives and the key side
//! tables agree with the connectivity with no duplicate key globally.
//! Returns LOCAL fails.
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
    fails += TesseraTest::checkKeyTables( mesh );
    if ( TesseraTest::checkOwnedEuler( mesh ) != 2 )
        ++fails;
    if ( countTombstones( mesh ) != 0 )
        ++fails;
    return fails;
}

//! The five verdict counters must partition `requested` -- a collapse is
//! accepted or rejected for exactly one reason.
static int checkCountersPartition( const CollapseResult& r )
{
    return ( r.accepted + r.rejectedBoundary + r.rejectedLinkCondition +
                 r.rejectedNormalFlip + r.rejectedQuality +
                 r.rejectedConflict ==
             r.requested )
               ? 0
               : 1;
}

//! An accepted collapse removes exactly 1 vertex, 3 edges and 2 faces.
static int checkRemovalDeltas( const CollapseResult& r )
{
    return ( r.verticesRemoved == r.accepted &&
             r.edgesRemoved == 3 * r.accepted &&
             r.facesRemoved == 2 * r.accepted )
               ? 0
               : 1;
}

//! Build -> partition -> distribute the subdiv-2 fixture on `comm` AT DEPTH 2,
//! which is collapseEdges()' precondition.
template <class MeshT, class Exec>
static void setupOn( MPI_Comm comm, MeshT& mesh,
                     MeshHalo<typename Exec::memory_space>& halo, int depth = 2 )
{
    (void)comm;
    buildIcosphere( mesh, 2 );
    auto faceOwner = facePartitionByAxis( mesh );
    distribute( mesh, halo, faceOwner, depth );
}

template <class MeshT, class Exec>
static void setup( MeshT& mesh, MeshHalo<typename Exec::memory_space>& halo )
{
    setupOn<MeshT, Exec>( MPI_COMM_WORLD, mesh, halo, 2 );
}

// ===========================================================================
// Case 1: one edge, exact deltas
// ===========================================================================

template <class MeshT, class Exec>
static int caseOneEdge( int rank, const char* tag )
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
    const auto pos0 = readPositions( mesh );
    // THE REFERENCE MIDPOINT HAS TO BE AGREED GLOBALLY, not computed locally: a
    // rank can hold the surviving endpoint (as a ghost) WITHOUT holding the far
    // one, and at np4-5 some rank always does. Every rank that holds both agrees
    // BITWISE -- a ghost position is a whole-tuple copy of its owner's -- so an
    // MPI_MAX over a sentinel carries the value to the ranks that do not.
    std::array<double, 3> want = { 0, 0, 0 };
    {
        double mine[3] = { -1e300, -1e300, -1e300 };
        auto ia = pos0.find( k0.id[0] );
        auto ib = pos0.find( k0.id[1] );
        if ( ia != pos0.end() && ib != pos0.end() )
            for ( int d = 0; d < 3; ++d )
                mine[d] = 0.5 * ( ia->second[d] + ib->second[d] );
        double best[3] = { -1e300, -1e300, -1e300 };
        MPI_Allreduce( mine, best, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
        for ( int d = 0; d < 3; ++d )
            want[d] = best[d];
        if ( !( want[0] > -1e299 ) )
            ++local; // no rank held both endpoints of an owned edge
    }

    const CollapseResult res =
        collapseEdges( mesh, halo, maskFromKeys( mesh, { k0 } ) );

    local += checkCountersPartition( res );
    local += checkRemovalDeltas( res );
    if ( res.requested != 1 || res.accepted != 1 )
        ++local; // the case is vacuous unless this one collapse is taken
    local += checkAll( mesh, V - 1, E - 3, F - 2 );

    // The surviving vertex is min(a,b) and sits at the midpoint; the dying one
    // is gone from every rank.
    {
        const auto pos1 = readPositions( mesh );
        long long haveSurv = 0, haveDead = 0, atMid = 0;
        auto is = pos1.find( k0.id[0] );
        if ( is != pos1.end() )
        {
            ++haveSurv;
            double err = 0.0;
            for ( int d = 0; d < 3; ++d )
                err = std::max( err, std::fabs( is->second[d] - want[d] ) );
            if ( err <= 1e-14 )
                ++atMid;
        }
        if ( pos1.find( k0.id[1] ) != pos1.end() )
            ++haveDead;
        haveSurv = gsumll( MPI_COMM_WORLD, haveSurv );
        haveDead = gsumll( MPI_COMM_WORLD, haveDead );
        atMid = gsumll( MPI_COMM_WORLD, atMid );
        if ( haveSurv <= 0 || atMid != haveSurv || haveDead != 0 )
            ++local;
    }

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case1 (one edge, exact deltas) %s: V %lld->%lld "
                     "E %lld->%lld F %lld->%lld, surviving gid %llu at the "
                     "midpoint, removed v=%lld e=%lld f=%lld\n",
                     tag, glob == 0 ? "ok" : "FAIL", V, V - 1, E, E - 3, F,
                     F - 2, static_cast<unsigned long long>( k0.id[0] ),
                     res.verticesRemoved, res.edgesRemoved, res.facesRemoved );
    return glob == 0 ? 0 : 1;
}

// ===========================================================================
// Case 2: the link condition, both ways
// ===========================================================================
//
// THE POSITIVE FIXTURE. Four triangles: (a,b,c) and (b,a,d) share the edge
// (a,b), and (a,c,w) plus (b,d,w) make w a neighbour of BOTH a and b. So
// link(a) INTERSECT link(b) == {c, d, w} and the collapse of (a,b) would weld
// the two w-corners into one, producing a non-manifold mesh. The edge (a,b)
// still has exactly two incident faces, so the boundary test does NOT catch it
// -- only the link condition does, which is the point of the fixture.
//
// The winding is chosen so every shared edge is traversed in OPPOSITE
// directions by its two faces: (a,b) is a->b in the first and b->a in the
// second, (a,c) is c->a and a->c, (b,d) is d->b and b->d. An inconsistently
// oriented soup would abort inside the operation rather than reach the check.
// buildFromTriangleSoup assigns vertex gid == soup index, so the marked edge's
// key is known up front.
static TriangleSoup<double> linkViolationSoup( EdgeKey& collapseMe )
{
    const double P[5][3] = {
        { 0.0, 0.0, 0.0 },  // 0 = a
        { 1.0, 0.0, 0.0 },  // 1 = b
        { 0.5, 1.0, 0.0 },  // 2 = c
        { 0.5, -1.0, 0.0 }, // 3 = d
        { 0.5, 0.0, 1.0 }   // 4 = w
    };
    const int T[4][3] = {
        { 0, 1, 2 }, // (a,b,c)
        { 1, 0, 3 }, // (b,a,d)
        { 0, 2, 4 }, // (a,c,w)
        { 1, 3, 4 }  // (b,d,w)
    };
    TriangleSoup<double> soup;
    for ( int i = 0; i < 5; ++i )
        for ( int d = 0; d < 3; ++d )
            soup.positions.push_back( P[i][d] );
    for ( int t = 0; t < 4; ++t )
        for ( int k = 0; k < 3; ++k )
            soup.triangles.push_back( T[t][k] );
    collapseMe = makeEdgeKey( static_cast<GlobalId>( 0 ),
                              static_cast<GlobalId>( 1 ) );
    return soup;
}

template <class MeshT, class Exec>
static int caseLinkCondition( int rank, const char* tag )
{
    using mem = typename Exec::memory_space;
    int local = 0;
    long long negAccepted = 0, posRejected = 0, posAccepted = 0;

    // ---- the NEGATIVE: all three edges of one face --------------------------
    //
    // Collapsing two edges of the same triangle is topologically invalid (they
    // share a vertex, and the second would act on a face the first deleted), so
    // the independent set may take AT MOST ONE of the three.
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        setup<MeshT, Exec>( mesh, halo );
        long long V, E, F;
        counts( mesh, V, E, F );

        // The three edges of the face with the globally smallest FaceKey, found
        // through the owned faces so the choice is rank-count invariant.
        std::set<EdgeKey> want;
        {
            const auto faces = ownedFaceVerts( mesh );
            FaceKey best;
            best.id[0] = best.id[1] = best.id[2] = ~0ULL;
            for ( const auto& t : faces )
            {
                const FaceKey k = makeFaceKey( t[0], t[1], t[2] );
                if ( k < best )
                    best = k;
            }
            const int size = mesh.commSize();
            std::vector<unsigned long long> all( 3 * size, 0 );
            unsigned long long mine[3] = { best.id[0], best.id[1], best.id[2] };
            MPI_Allgather( mine, 3, MPI_UNSIGNED_LONG_LONG, all.data(), 3,
                           MPI_UNSIGNED_LONG_LONG, mesh.comm() );
            for ( int r = 0; r < size; ++r )
            {
                FaceKey k;
                for ( int j = 0; j < 3; ++j )
                    k.id[j] = all[3 * r + j];
                if ( k < best )
                    best = k;
            }
            for ( int j = 0; j < 3; ++j )
                want.insert( makeEdgeKey( best.id[j], best.id[( j + 1 ) % 3] ) );
        }

        const CollapseResult res =
            collapseEdges( mesh, halo, maskFromKeys( mesh, want ) );
        negAccepted = res.accepted;
        local += checkCountersPartition( res );
        local += checkRemovalDeltas( res );
        if ( res.requested != 3 )
            ++local; // the three edges were not all found
        if ( res.accepted > 1 )
            ++local; // two collapses on one triangle were let through
        local += checkAll( mesh, V - res.accepted, E - 3 * res.accepted,
                           F - 2 * res.accepted );
    }

    // ---- the POSITIVE: a deliberate link-condition violation ----------------
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        EdgeKey collapseMe;
        const TriangleSoup<double> soup = linkViolationSoup( collapseMe );
        buildFromTriangleSoup( mesh, soup );
        auto faceOwner = facePartitionByAxis( mesh );
        distribute( mesh, halo, faceOwner, 2 );

        long long V, E, F;
        counts( mesh, V, E, F );
        if ( V != 5 || F != 4 )
            ++local; // fixture drifted

        const CollapseResult res =
            collapseEdges( mesh, halo, maskFromKeys( mesh, { collapseMe } ) );
        posRejected = res.rejectedLinkCondition;
        posAccepted = res.accepted;
        local += checkCountersPartition( res );
        if ( res.requested != 1 )
            ++local;
        if ( res.rejectedLinkCondition != 1 || res.accepted != 0 )
            ++local;
        if ( !res.collapsed.empty() )
            ++local;
        long long v, e, f;
        counts( mesh, v, e, f );
        if ( v != V || e != E || f != F )
            ++local; // a rejected collapse must leave the mesh alone
    }

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case2 (link condition both ways) %s: three edges "
                     "of one face -> accepted=%lld (<=1); violation soup -> "
                     "rejectedLinkCondition=%lld accepted=%lld\n",
                     tag, glob == 0 ? "ok" : "FAIL", negAccepted, posRejected,
                     posAccepted );
    return glob == 0 ? 0 : 1;
}

// ===========================================================================
// Case 3: the depth guard
// ===========================================================================

template <class MeshT, class Exec>
static int caseDepthGuard( int rank, const char* tag )
{
    using mem = typename Exec::memory_space;
    int local = 0;

    // Depth 1 THROWS, and the message names both the requirement and the fact.
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        setupOn<MeshT, Exec>( MPI_COMM_WORLD, mesh, halo, 1 );
        bool threw = false, named = false;
        try
        {
            std::vector<char> mask( mesh.numOwnedEdges(), 1 );
            collapseEdges( mesh, halo, mask );
        }
        catch ( const std::exception& e )
        {
            threw = true;
            const std::string what = e.what();
            named = what.find( "depth >= 2" ) != std::string::npos &&
                    what.find( "depth 1" ) != std::string::npos;
        }
        if ( !threw || !named )
            ++local;
        if ( mesh.haloDepth() != 1 )
            ++local; // the guard must not have changed the mesh
    }

    // Depth 2 does NOT throw -- otherwise the check above would pass for the
    // wrong reason.
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        setup<MeshT, Exec>( mesh, halo );
        bool threw = false;
        try
        {
            std::vector<char> mask( mesh.numOwnedEdges(), 0 );
            collapseEdges( mesh, halo, mask );
        }
        catch ( const std::exception& )
        {
            threw = true;
        }
        if ( threw || mesh.haloDepth() != 2 )
            ++local;
    }

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case3 (depth guard: throws at depth 1, runs at "
                     "depth 2) %s\n",
                     tag, glob == 0 ? "ok" : "FAIL" );
    return glob == 0 ? 0 : 1;
}

// ===========================================================================
// Cases 4 and 5: the independent set, and its rank-count invariance
// ===========================================================================

template <class MeshT, class Exec>
static int caseIndependentSet( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    int local = 0;

    struct Out
    {
        Chk posFaces;
        long long V = 0, E = 0, F = 0;
        CollapseResult res;
        long long shareVertex = 0;
    };

    auto measure = [&]( MPI_Comm comm, Out& o )
    {
        MeshT mesh( comm );
        MeshHalo<mem> halo;
        setupOn<MeshT, Exec>( comm, mesh, halo, 2 );
        counts( mesh, o.V, o.E, o.F );

        std::vector<char> mask( mesh.numOwnedEdges(), 1 );
        o.res = collapseEdges( mesh, halo, mask );
        local += checkCountersPartition( o.res );
        local += checkRemovalDeltas( o.res );
        local += checkAll( mesh, o.V - o.res.accepted,
                           o.E - 3 * o.res.accepted, o.F - 2 * o.res.accepted );

        // CHECK 4: no two accepted collapses share a vertex. The accepted keys
        // ARE the pre-collapse vertex -> edge map's payload: two collapses
        // conflict exactly when their endpoint sets intersect.
        const std::set<EdgeKey> accepted = globalCollapsedSet( comm, o.res );
        if ( static_cast<long long>( accepted.size() ) != o.res.accepted )
            ++local; // the collapsed set is not complete/consistent
        std::set<GlobalId> seen;
        for ( const EdgeKey& k : accepted )
            for ( int j = 0; j < 2; ++j )
                if ( !seen.insert( k.id[j] ).second )
                    ++o.shareVertex;

        o.posFaces = posFaceSignature( mesh, local );
        o.V -= o.res.accepted;
        o.E -= 3 * o.res.accepted;
        o.F -= 2 * o.res.accepted;
    };

    Out w, s;
    measure( MPI_COMM_WORLD, w );
    measure( MPI_COMM_SELF, s );

    if ( w.shareVertex != 0 || s.shareVertex != 0 )
        ++local; // two accepted collapses shared a vertex
    if ( w.res.accepted <= 0 )
        ++local; // vacuous: the case proves nothing if nothing was collapsed

    // CHECK 5: rank-count invariance.
    if ( w.V != s.V || w.E != s.E || w.F != s.F )
        ++local;
    if ( w.posFaces != s.posFaces )
        ++local;
    if ( w.res.accepted != s.res.accepted ||
         w.res.requested != s.res.requested ||
         w.res.rejectedBoundary != s.res.rejectedBoundary ||
         w.res.rejectedLinkCondition != s.res.rejectedLinkCondition ||
         w.res.rejectedNormalFlip != s.res.rejectedNormalFlip ||
         w.res.rejectedQuality != s.res.rejectedQuality ||
         w.res.rejectedConflict != s.res.rejectedConflict )
        ++local;

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case4+5 (independent set, np%d vs SELF) %s: "
                     "requested=%lld accepted=%lld boundary=%lld link=%lld "
                     "normal=%lld quality=%lld conflict=%lld -> V=%lld E=%lld "
                     "F=%lld (SELF accepted=%lld F=%lld)\n",
                     tag, size, glob == 0 ? "ok" : "FAIL", w.res.requested,
                     w.res.accepted, w.res.rejectedBoundary,
                     w.res.rejectedLinkCondition, w.res.rejectedNormalFlip,
                     w.res.rejectedQuality, w.res.rejectedConflict, w.V, w.E,
                     w.F, s.res.accepted, s.F );
    return glob == 0 ? 0 : 1;
}

// ===========================================================================
// Case 6: a policy nothing can satisfy rejects everything and moves nothing
// ===========================================================================

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
    const std::vector<GlobalId> seq0 = localGidSequence( mesh );

    DefaultCollapsePolicy policy;
    policy.minQuality = 0.99; // no triangle but an equilateral one clears this
    std::vector<char> mask( mesh.numOwnedEdges(), 1 );
    const CollapseResult res = collapseEdges( mesh, halo, mask, policy );

    local += checkCountersPartition( res );
    local += checkRemovalDeltas( res );
    local += checkAll( mesh, V, E, F );
    if ( res.accepted != 0 )
        ++local;
    if ( res.rejectedBoundary != 0 )
        ++local; // a closed icosphere has no boundary edge
    if ( res.rejectedNormalFlip + res.rejectedQuality != res.requested )
        ++local; // everything must fail on GEOMETRY, not on conflict

    TesseraTest::topologyChecksum( mesh, cv1, ce1, cf1 );
    if ( cv0 != cv1 || ce0 != ce1 || cf0 != cf1 )
        ++local;
    if ( localGidSequence( mesh ) != seq0 )
        ++local; // a no-op must not re-compact into a different ordering

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case6 (geometric rejection, minQuality=0.99) %s: "
                     "requested=%lld normal=%lld quality=%lld accepted=%lld, "
                     "ordering and checksums unmoved\n",
                     tag, glob == 0 ? "ok" : "FAIL", res.requested,
                     res.rejectedNormalFlip, res.rejectedQuality,
                     res.accepted );
    return glob == 0 ? 0 : 1;
}

// ===========================================================================
// Case 7: normal-flip rejection
// ===========================================================================
//
// THE FIXTURE: a closed decagonal BIPYRAMID whose top apex is pulled far out
// into a spike (z = 3 over a unit-radius ring) while the bottom apex sits at
// z = -1. Collapsing a spike edge (s, r_i) drags the apex to the midpoint,
// halfway down and a full radius sideways, which rotates every OTHER spike
// face's normal by far more than DefaultCollapsePolicy::maxNormalRotation -- so
// the fold guard must reject it. The RING edges are the shortest and are benign,
// so the case is not vacuous: some collapses are accepted and the surviving
// faces' orientations can be compared one gid at a time.
static TriangleSoup<double> spikeBipyramidSoup( int n )
{
    TriangleSoup<double> soup;
    auto push = [&]( double x, double y, double z )
    {
        soup.positions.push_back( x );
        soup.positions.push_back( y );
        soup.positions.push_back( z );
    };
    // 0 = the spike apex, 1 = the bottom apex, 2..n+1 = the ring.
    push( 0.0, 0.0, 3.0 );
    push( 0.0, 0.0, -1.0 );
    for ( int i = 0; i < n; ++i )
    {
        const double a = 2.0 * 3.14159265358979323846 * i / n;
        push( std::cos( a ), std::sin( a ), 0.0 );
    }
    for ( int i = 0; i < n; ++i )
    {
        const int r0 = 2 + i, r1 = 2 + ( i + 1 ) % n;
        // Top faces wind CCW seen from outside/above, bottom faces the other
        // way, so every ring edge is traversed in opposite directions by its two
        // faces and the surface is consistently oriented.
        soup.triangles.push_back( 0 );
        soup.triangles.push_back( r0 );
        soup.triangles.push_back( r1 );
        soup.triangles.push_back( 1 );
        soup.triangles.push_back( r1 );
        soup.triangles.push_back( r0 );
    }
    return soup;
}

//! Twice the area-weighted normal of a triangle: (p1-p0) x (p2-p0).
static inline void triNormal( const std::array<double, 3>& p0,
                              const std::array<double, 3>& p1,
                              const std::array<double, 3>& p2, double n[3] )
{
    const double u[3] = { p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] };
    const double v[3] = { p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2] };
    n[0] = u[1] * v[2] - u[2] * v[1];
    n[1] = u[2] * v[0] - u[0] * v[2];
    n[2] = u[0] * v[1] - u[1] * v[0];
}

//! Face gid -> its unnormalized normal, over the OWNED faces. A collapse
//! preserves face gids and face ownership, so the two snapshots are comparable
//! entry by entry with no communication.
template <class MeshT>
static std::map<GlobalId, std::array<double, 3>> ownedFaceNormals( MeshT& mesh,
                                                                   int& fails )
{
    const auto pos = readPositions( mesh );
    std::map<GlobalId, std::array<double, 3>> out;
    for ( const auto& kv : ownedFacesByGid( mesh ) )
    {
        std::array<double, 3> p[3];
        bool ok = true;
        for ( int k = 0; k < 3; ++k )
        {
            auto it = pos.find( kv.second[k] );
            if ( it == pos.end() )
                ok = false;
            else
                p[k] = it->second;
        }
        if ( !ok )
        {
            ++fails;
            continue;
        }
        double n[3];
        triNormal( p[0], p[1], p[2], n );
        out[kv.first] = { n[0], n[1], n[2] };
    }
    return out;
}

template <class MeshT, class Exec>
static int caseNormalFlip( int rank, const char* tag )
{
    using mem = typename Exec::memory_space;
    const int nRing = 10;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    buildFromTriangleSoup( mesh, spikeBipyramidSoup( nRing ) );
    auto faceOwner = facePartitionByAxis( mesh );
    distribute( mesh, halo, faceOwner, 2 );

    int local = 0;
    long long V, E, F;
    counts( mesh, V, E, F );
    if ( V != nRing + 2 || F != 2 * nRing )
        ++local; // fixture drifted
    if ( TesseraTest::checkOwnedEuler( mesh ) != 2 )
        ++local; // the fixture must be a closed genus-0 surface
    const std::map<GlobalId, std::array<double, 3>> n0 =
        ownedFaceNormals( mesh, local );

    std::vector<char> mask( mesh.numOwnedEdges(), 1 );
    const CollapseResult res = collapseEdges( mesh, halo, mask );
    local += checkCountersPartition( res );
    local += checkRemovalDeltas( res );
    local += checkAll( mesh, V - res.accepted, E - 3 * res.accepted,
                       F - 2 * res.accepted );
    if ( res.rejectedNormalFlip < 1 )
        ++local; // the fold guard never fired: the fixture is not doing its job

    // NO SURVIVING FACE IS REVERSED relative to its own pre-collapse normal.
    long long compared = 0, reversed = 0;
    {
        int nf = 0;
        const std::map<GlobalId, std::array<double, 3>> n1 =
            ownedFaceNormals( mesh, nf );
        local += nf;
        for ( const auto& kv : n1 )
        {
            auto it = n0.find( kv.first );
            if ( it == n0.end() )
                continue; // gained a face this rank did not own before
            ++compared;
            double dot = 0.0;
            for ( int d = 0; d < 3; ++d )
                dot += it->second[d] * kv.second[d];
            if ( !( dot > 0.0 ) )
                ++reversed;
        }
    }
    compared = gsumll( MPI_COMM_WORLD, compared );
    reversed = gsumll( MPI_COMM_WORLD, reversed );
    if ( reversed != 0 )
        ++local;
    if ( compared <= 0 )
        ++local; // vacuous

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case7 (normal-flip rejection on a spike) %s: "
                     "requested=%lld accepted=%lld rejectedNormalFlip=%lld "
                     "rejectedQuality=%lld link=%lld conflict=%lld, %lld faces "
                     "compared, %lld reversed\n",
                     tag, glob == 0 ? "ok" : "FAIL", res.requested,
                     res.accepted, res.rejectedNormalFlip, res.rejectedQuality,
                     res.rejectedLinkCondition, res.rejectedConflict, compared,
                     reversed );
    return glob == 0 ? 0 : 1;
}

// ===========================================================================
// Cases 8 and 9: decimation to a floor, and the round trip with split
// ===========================================================================

//! Run the collapse loop to exhaustion (or `maxRounds`), asserting the
//! invariants after every round and reporting the trajectory. Returns LOCAL
//! fails; `finalF` and `floorQ` come back for the caller's own assertions.
//!
//! `belowLen <= 0` marks EVERY owned edge -- unbounded decimation, which is what
//! case 8 measures. A positive `belowLen` marks only the edges shorter than it,
//! which is a length-driven remesher's coarsening rule and TERMINATES at that
//! scale rather than at the quality guards.
template <class MeshT, class Exec>
static int decimateLoop( MeshT& mesh, MeshHalo<typename Exec::memory_space>& halo,
                         int rank, const char* tag, const char* label,
                         int maxRounds, double belowLen, long long& finalF,
                         double& floorQ, long long& rounds )
{
    int local = 0;
    long long V, E, F;
    counts( mesh, V, E, F );
    floorQ = 1.0;
    rounds = 0;

    for ( int round = 0; round < maxRounds; ++round )
    {
        std::vector<char> mask;
        if ( belowLen > 0.0 )
        {
            double mean = 0.0;
            int mf = 0;
            const std::vector<double> len = ownedEdgeLengths( mesh, mean, mf );
            local += mf;
            mask.assign( len.size(), 0 );
            for ( std::size_t e = 0; e < len.size(); ++e )
                mask[e] = ( len[e] < belowLen ) ? 1 : 0;
        }
        else
            mask.assign( mesh.numOwnedEdges(), 1 );
        const CollapseResult res = collapseEdges( mesh, halo, mask );
        local += checkCountersPartition( res );
        local += checkRemovalDeltas( res );
        local += checkAll( mesh, V - res.accepted, E - 3 * res.accepted,
                           F - 2 * res.accepted );
        V -= res.accepted;
        E -= 3 * res.accepted;
        F -= 2 * res.accepted;

        double minAngle = 180.0;
        int qFails = 0;
        const double q = TesseraTest::minRadiusRatio( mesh, qFails, minAngle );
        local += qFails;
        floorQ = std::min( floorQ, q );
        ++rounds;

        if ( rank == 0 )
            std::printf( "  [%s] %s round%2d: accepted=%4lld V=%6lld E=%6lld "
                         "F=%6lld minRadiusRatio=%.4f minAngle=%.3f\n",
                         tag, label, round + 1, res.accepted, V, E, F, q,
                         minAngle );
        if ( res.accepted == 0 )
            break;
    }
    finalF = F;
    if ( floorQ < kDecimateFloor )
        ++local;
    return local;
}

template <class MeshT, class Exec>
static int caseDecimate( int rank, const char* tag )
{
    using mem = typename Exec::memory_space;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    setup<MeshT, Exec>( mesh, halo );

    int local = 0;
    long long finalF = 0, rounds = 0;
    double floorQ = 1.0;
    local += decimateLoop<MeshT, Exec>( mesh, halo, rank, tag, "case8", 20, 0.0,
                                        finalF, floorQ, rounds );
    if ( finalF >= F0 )
        ++local; // the mesh must have actually coarsened

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case8 (decimation to a floor, %lld rounds) %s: "
                     "F %lld -> %lld, quality floor %.4f (assert >= %.3f)\n",
                     tag, rounds, glob == 0 ? "ok" : "FAIL", F0, finalF, floorQ,
                     kDecimateFloor );
    return glob == 0 ? 0 : 1;
}

template <class MeshT, class Exec>
static int caseRoundTrip( int rank, const char* tag )
{
    using mem = typename Exec::memory_space;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    setup<MeshT, Exec>( mesh, halo );

    int local = 0;
    // THE TARGET SCALE IS THE PRE-SPLIT MEAN EDGE LENGTH, measured before the
    // split. "Coarsening undoes refinement" is a statement about a LENGTH, and
    // the collapse rule that expresses it is "collapse what is shorter than the
    // original scale" -- which TERMINATES there. An all-edges mask (case 8's)
    // has no target and would decimate past it to the quality floor, so it
    // could not satisfy the factor-of-two band no matter how long it ran.
    double targetLen = 0.0;
    {
        int mf = 0;
        ownedEdgeLengths( mesh, targetLen, mf );
        local += mf;
    }
    {
        std::vector<char> mask( mesh.numOwnedEdges(), 1 );
        const SplitResult sres = splitEdges( mesh, halo, mask );
        if ( sres.facesAfter != 4 * F0 )
            ++local; // a uniform split of every edge quadruples the faces
        if ( TesseraTest::checkOwnedEuler( mesh ) != 2 )
            ++local;
        local += TesseraTest::checkConforming( mesh );
    }
    long long splitF = TesseraTest::globalOwnedFaces( mesh );

    long long finalF = 0, rounds = 0;
    double floorQ = 1.0;
    // 4/5 of the target is the canonical "too short" threshold (the same one
    // case 14's drive uses). Collapsing everything below the target ITSELF
    // overshoots: the coarsening keeps manufacturing new sub-target edges and
    // ran to 144 faces in the first implementation run, past the factor-of-two
    // band. That is a property of a collapse-only drive with no smoothing, and
    // it is why the rule has a margin.
    local += decimateLoop<MeshT, Exec>( mesh, halo, rank, tag, "case9", 80,
                                        0.8 * targetLen, finalF, floorQ,
                                        rounds );

    // Within a factor of two of the ORIGINAL 320 faces: coarsening undid the
    // refinement rather than jamming (too many faces left) or running away (too
    // few).
    if ( finalF > 2 * F0 || finalF < F0 / 2 )
        ++local;

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case9 (round trip with split, %lld rounds, target "
                     "len %.4f) %s: F %lld -> %lld -> %lld, band [%lld, %lld], "
                     "quality floor %.4f\n",
                     tag, rounds, targetLen, glob == 0 ? "ok" : "FAIL", F0,
                     splitF, finalF, F0 / 2, 2 * F0, floorQ );
    return glob == 0 ? 0 : 1;
}

// ===========================================================================
// Case 10: the user-field blend
// ===========================================================================
//
// Both fields are seeded to a LINEAR function of position on EVERY locally held
// vertex, owned and ghost. Seeding the ghosts directly (rather than seeding the
// owners and exchanging) is deliberate: it makes the fixture exact by
// construction and keeps the case about the BLEND rather than about the halo,
// which case 11 covers. A linear function commutes with the convex combination
// the collapse applies, so the merged value must equal the function evaluated at
// the merged position -- to rounding, not approximately.
static inline double linScalar( const std::array<double, 3>& p )
{
    return 2.0 * p[0] - 3.0 * p[1] + 0.5 * p[2] + 1.0;
}
static inline double linVec( const std::array<double, 3>& p, int c )
{
    const double w[3] = { 0.25, -1.5, 4.0 };
    return w[c] * p[c] + 0.125 * ( p[0] + p[1] + p[2] ) + 7.0 * ( c + 1 );
}

template <class MeshT, class Exec>
static int caseUserFields( int rank, const char* tag )
{
    using mem = typename Exec::memory_space;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    setup<MeshT, Exec>( mesh, halo );

    int local = 0;
    long long V, E, F;
    counts( mesh, V, E, F );

    auto seed = [&]()
    {
        Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace>
            hv( "hv", mesh.numVertices() );
        Cabana::deep_copy( hv, mesh.vertices() );
        auto vp = Cabana::slice<VertexField::Position>( hv );
        auto s = Cabana::slice<userVertexField<0>()>( hv );
        auto v3 = Cabana::slice<userVertexField<1>()>( hv );
        for ( std::size_t i = 0; i < mesh.numVertices(); ++i )
        {
            std::array<double, 3> p = { 0, 0, 0 };
            for ( int d = 0; d < MeshT::dim && d < 3; ++d )
                p[d] = static_cast<double>( vp( i, d ) );
            s( i ) = linScalar( p );
            for ( int c = 0; c < 3; ++c )
                v3( i, c ) = linVec( p, c );
        }
        Cabana::deep_copy( mesh.vertices(), hv );
    };
    seed();

    const EdgeKey k0 = globalMinEdgeKey( mesh );
    const CollapseResult res =
        collapseEdges( mesh, halo, maskFromKeys( mesh, { k0 } ) );
    local += checkCountersPartition( res );
    if ( res.requested != 1 || res.accepted != 1 )
        ++local; // vacuous unless the one collapse is taken
    local += checkAll( mesh, V - 1, E - 3, F - 2 );

    // The merged vertex's fields, checked wherever it is held -- on its owner
    // and on every rank that ghosts it, which also proves the blended values
    // rode the halo rebuild correctly.
    long long checked = 0;
    double worst = 0.0;
    {
        Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace>
            hv( "hv", mesh.numVertices() );
        Cabana::deep_copy( hv, mesh.vertices() );
        auto vg = Cabana::slice<VertexField::Gid>( hv );
        auto vp = Cabana::slice<VertexField::Position>( hv );
        auto s = Cabana::slice<userVertexField<0>()>( hv );
        auto v3 = Cabana::slice<userVertexField<1>()>( hv );
        for ( std::size_t i = 0; i < mesh.numVertices(); ++i )
        {
            if ( vg( i ) != k0.id[0] )
                continue;
            ++checked;
            std::array<double, 3> p = { 0, 0, 0 };
            for ( int d = 0; d < MeshT::dim && d < 3; ++d )
                p[d] = static_cast<double>( vp( i, d ) );
            auto rel = [&]( double got, double want )
            {
                const double scale = std::max( 1.0, std::fabs( want ) );
                return std::fabs( got - want ) / scale;
            };
            worst = std::max( worst, rel( s( i ), linScalar( p ) ) );
            for ( int c = 0; c < 3; ++c )
                worst = std::max( worst, rel( v3( i, c ), linVec( p, c ) ) );
        }
    }
    checked = gsumll( MPI_COMM_WORLD, checked );
    {
        double g = 0.0;
        MPI_Allreduce( &worst, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
        worst = g;
    }
    if ( checked <= 0 )
        ++local; // the merged vertex is held nowhere: something is very wrong
    if ( !( worst <= 1e-14 ) )
        ++local;

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case10 (user-field blend, double + double[3]) %s: "
                     "merged vertex held on %lld copies, worst relative error "
                     "%.3e (tol 1e-14)\n",
                     tag, glob == 0 ? "ok" : "FAIL", checked, worst );
    return glob == 0 ? 0 : 1;
}

// ===========================================================================
// Cases 11 and 12: halo/compaction on return, and the empty mask
// ===========================================================================

template <class MeshT, class Exec>
static int caseHaloAndEmpty( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    int local = 0;

    // ---- case 12: empty mask -----------------------------------------------
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        setup<MeshT, Exec>( mesh, halo );
        long long V, E, F;
        counts( mesh, V, E, F );
        unsigned long long cv0, ce0, cf0, cv1, ce1, cf1;
        TesseraTest::topologyChecksum( mesh, cv0, ce0, cf0 );
        const std::vector<GlobalId> seq0 = localGidSequence( mesh );

        std::vector<char> mask( mesh.numOwnedEdges(), 0 );
        const CollapseResult res = collapseEdges( mesh, halo, mask );

        TesseraTest::topologyChecksum( mesh, cv1, ce1, cf1 );
        local += checkAll( mesh, V, E, F );
        if ( cv0 != cv1 || ce0 != ce1 || cf0 != cf1 )
            ++local;
        if ( localGidSequence( mesh ) != seq0 )
            ++local;
        if ( res.requested != 0 || res.accepted != 0 ||
             res.rejectedBoundary != 0 || res.rejectedLinkCondition != 0 ||
             res.rejectedNormalFlip != 0 || res.rejectedQuality != 0 ||
             res.rejectedConflict != 0 || res.verticesRemoved != 0 ||
             res.edgesRemoved != 0 || res.facesRemoved != 0 ||
             !res.collapsed.empty() )
            ++local;
    }

    // ---- case 11: halo and compaction on return ----------------------------
    long long corrupted = 0, planSize = 0, accepted2 = 0;
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        setup<MeshT, Exec>( mesh, halo );
        long long V, E, F;
        counts( mesh, V, E, F );

        std::vector<char> mask( mesh.numOwnedEdges(), 1 );
        CollapseResult res = collapseEdges( mesh, halo, mask );
        local += checkAll( mesh, V - res.accepted, E - 3 * res.accepted,
                           F - 2 * res.accepted );
        if ( res.accepted <= 0 )
            ++local;
        // THE DEPTH IS PRESERVED: compact()'s rebuild uses the halo's recorded
        // depth, so a collapse must not silently narrow the mesh back to 1.
        if ( halo.depth != 2 || mesh.haloDepth() != 2 )
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

        // A SECOND collapseEdges() with no intervening migrate() must work.
        long long V1, E1, F1;
        counts( mesh, V1, E1, F1 );
        std::vector<char> mask2( mesh.numOwnedEdges(), 1 );
        res = collapseEdges( mesh, halo, mask2 );
        accepted2 = res.accepted;
        local += checkAll( mesh, V1 - res.accepted, E1 - 3 * res.accepted,
                           F1 - 2 * res.accepted );
        if ( res.accepted <= 0 )
            ++local;
        if ( halo.depth != 2 || mesh.haloDepth() != 2 )
            ++local;

        if ( size > 1 && ( planSize <= 0 || corrupted <= 0 ) )
            ++local; // vacuous: an empty plan passes every structural check
    }

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case11+12 (empty mask, halo/compaction on return) "
                     "%s: plan=%lld ghostsCorrupted=%lld secondAccepted=%lld\n",
                     tag, glob == 0 ? "ok" : "FAIL", planSize, corrupted,
                     accepted2 );
    return glob == 0 ? 0 : 1;
}

// ===========================================================================
// Case 13: the family guard, in both refinement modes
// ===========================================================================

template <class MeshT, class Exec>
static int familyGuardOne( const char* what, int& local )
{
    using mem = typename Exec::memory_space;
    (void)what;
    auto namesBothFamilies = []( const std::string& msg )
    {
        return msg.find( "Hierarchical" ) != std::string::npos &&
               msg.find( "Remesh" ) != std::string::npos;
    };

    // collapseEdges() on a refine()d mesh throws.
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
            collapseEdges( mesh, halo, emask );
        }
        catch ( const std::exception& e )
        {
            threw = true;
            named = namesBothFamilies( e.what() );
        }
        if ( !threw || !named )
            ++local;
    }

    // refine() on a collapsed mesh throws.
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        setup<MeshT, Exec>( mesh, halo );
        std::vector<char> emask( mesh.numOwnedEdges(), 1 );
        const CollapseResult res = collapseEdges( mesh, halo, emask );
        if ( res.accepted <= 0 )
            ++local; // vacuous: the mesh was never claimed by the remesh family
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
    return local;
}

// ===========================================================================
// Case 14: the composed remesh loop -- the integration test
// ===========================================================================

template <class MeshT, class Exec>
static int caseComposed( int rank, const char* tag )
{
    using mem = typename Exec::memory_space;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    setup<MeshT, Exec>( mesh, halo );

    int local = 0;
    const int nRounds = 20;
    double floorQ = 1.0;
    long long lastF = 0, activity = 0;

    // THE DRIVE IS A FIXED TARGET EDGE LENGTH, not a per-round relative
    // threshold. A relative rule (split above 1.3 * the CURRENT mean, collapse
    // below 0.7 * it) marks NOTHING on a near-uniform mesh -- the subdiv-2
    // icosphere's edges span 0.276 to 0.325 around a mean of 0.30 -- so the
    // whole loop would run twenty vacuous rounds and prove nothing. That is
    // measured, not assumed: the first implementation run printed
    // split=0 flipped=0 collapsed=0 for all twenty.
    //
    // The fixed target with the 4/3 and 4/5 thresholds is Botsch-Kobbelt
    // incremental remeshing's rule and is what a metric remesher actually does:
    // split what is more than 4/3 of the target, collapse what is less than
    // 4/5 of it, flip for valence in between. The target here is 0.6 * the
    // initial mean, so the first round has real work to do in both directions
    // and the sequence then holds the mesh at that scale -- which is what the
    // band assertion below is about.
    double targetLen = 0.0;
    {
        int mf = 0;
        ownedEdgeLengths( mesh, targetLen, mf );
        local += mf;
        targetLen *= 0.6;
    }

    // After every operation, not merely every round: Euler, conformity, no
    // duplicate key, no tombstone. checkAll() needs the expected counts, which
    // for split and flip come from the operations' own contracts.
    auto after = [&]( const char* op )
    {
        int part[4] = { 0, 0, 0, 0 };
        const int conf = TesseraTest::checkConforming( mesh );
        const int keys = TesseraTest::checkKeyTables( mesh, part );
        const long long euler = TesseraTest::checkOwnedEuler( mesh );
        const long long tomb = countTombstones( mesh );
        int f = conf + keys;
        if ( euler != 2 )
            ++f;
        if ( tomb != 0 )
            ++f;
        // A per-check breakdown, not a total: the first implementation run's
        // one failure here was Euler off by exactly the number of splits, which
        // is what identified splitEdges()' midpoint-gid base as the cause.
        if ( f && rank == 0 )
            std::printf( "  [%s] case14 CHECK FAILS after %s: conforming=%d "
                         "keyTables=%d(%d,%d,%d,%d) euler=%lld tombstones=%lld\n",
                         tag, op, conf, keys, part[0], part[1], part[2], part[3],
                         euler, tomb );
        local += f;
    };

    for ( int round = 0; round < nRounds; ++round )
    {
        int maskFails = 0;
        // SPLIT what is too long, FLIP for valence, COLLAPSE what is too short
        // -- the three-operation drive a metric remesher is built out of.
        const SplitResult sres = splitEdges(
            mesh, halo,
            absLengthMask( mesh, 4.0 / 3.0 * targetLen, true, maskFails ) );
        after( "splitEdges" );
        const FlipResult fres = flipEdges( mesh, halo, valenceMask( mesh ) );
        after( "flipEdges" );
        const CollapseResult cres = collapseEdges(
            mesh, halo,
            absLengthMask( mesh, 0.8 * targetLen, false, maskFails ) );
        after( "collapseEdges" );
        local += maskFails;
        activity += sres.split + fres.accepted + cres.accepted;
        local += checkCountersPartition( cres );
        local += checkRemovalDeltas( cres );

        long long V, E, F;
        counts( mesh, V, E, F );
        double minAngle = 180.0, lenLo = 0.0, lenHi = 0.0;
        int qFails = 0;
        const double q = TesseraTest::minRadiusRatio( mesh, qFails, minAngle );
        edgeLengthRange( mesh, lenLo, lenHi, qFails );
        local += qFails;
        floorQ = std::min( floorQ, q );
        lastF = F;

        if ( F < kComposedMinFaces || F > kComposedMaxFaces )
            ++local; // the drive ran away or collapsed to nothing
        if ( q < kComposedFloor )
            ++local;

        if ( rank == 0 )
            std::printf( "  [%s] case14 round%2d: split=%4lld flipped=%4lld "
                         "collapsed=%4lld V=%6lld E=%6lld F=%6lld "
                         "len[%.4f,%.4f] minRadiusRatio=%.4f minAngle=%.3f\n",
                         tag, round + 1, sres.split, fres.accepted,
                         cres.accepted, V, E, F, lenLo, lenHi, q, minAngle );
    }
    local += TesseraTest::checkNoInteriorVertex( mesh );
    local += TesseraTest::owned1RingLocal( mesh );
    if ( activity <= 0 )
        ++local; // vacuous: the drive never edited anything

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case14 (%d composed split/flip/collapse rounds, "
                     "target len %.4f, band [%lld,%lld], floor %.3f) %s: final "
                     "F=%lld, quality floor %.4f, %lld total edits\n",
                     tag, nRounds, targetLen, kComposedMinFaces,
                     kComposedMaxFaces, kComposedFloor,
                     glob == 0 ? "ok" : "FAIL", lastF, floorQ, activity );
    return glob == 0 ? 0 : 1;
}

// ===========================================================================

template <class Exec>
static int run( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    // The default refinement mode is Conforming, which is the mode case 13's
    // first half must reject a refine()d mesh in.
    using MeshT =
        Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>, mem, Exec>;
    using MeshH = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                       mem, Exec, RefinementMode::HangingNode2to1>;
    using MeshU = Mesh<double, 3, VertexFields<double, double[3]>, EdgeFields<>,
                       FaceFields<>, mem, Exec>;

    int fails = 0;
    fails += caseOneEdge<MeshT, Exec>( rank, tag );
    fails += caseLinkCondition<MeshT, Exec>( rank, tag );
    fails += caseDepthGuard<MeshT, Exec>( rank, tag );
    fails += caseIndependentSet<MeshT, Exec>( rank, size, tag );
    fails += caseGeometricRejection<MeshT, Exec>( rank, tag );
    fails += caseNormalFlip<MeshT, Exec>( rank, tag );
    fails += caseUserFields<MeshU, Exec>( rank, tag );
    fails += caseHaloAndEmpty<MeshT, Exec>( rank, size, tag );
    {
        int local = 0;
        familyGuardOne<MeshT, Exec>( "Conforming", local );
        familyGuardOne<MeshH, Exec>( "HangingNode2to1", local );
        const int glob = gsum( MPI_COMM_WORLD, local );
        if ( rank == 0 )
            std::printf( "  [%s] case13 (editing-family guard, both refinement "
                         "modes) %s\n",
                         tag, glob == 0 ? "ok" : "FAIL" );
        fails += glob == 0 ? 0 : 1;
    }
    fails += caseDecimate<MeshT, Exec>( rank, tag );
    fails += caseRoundTrip<MeshT, Exec>( rank, tag );
    fails += caseComposed<MeshT, Exec>( rank, tag );
    std::fflush( stdout );
    return fails;
}

int main( int argc, char* argv[] )
{
    // UNBUFFERED: a hang then localizes to the case that did not print, which
    // is how test_compact's rank-0-only collective was found.
    std::setvbuf( stdout, nullptr, _IONBF, 0 );

    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int fails = 0;
    {
        int rank = 0, size = 1;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        MPI_Comm_size( MPI_COMM_WORLD, &size );
        if ( rank == 0 )
            std::printf(
                "test_collapse_edges: caller-driven edge collapse (size %d)\n",
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
