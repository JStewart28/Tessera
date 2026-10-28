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

// Regression test: DISTRIBUTED INITIAL MESH CONSTRUCTION
// (tasks/completed/distributed-coarse-build.md).
//
// The initial mesh used to be built in FULL ON EVERY RANK before it was cut:
// buildFromTriangleSoup() is host-side, serial, and takes a replicated soup;
// buildIcosphere() generates the whole soup on every rank; distribute() then
// computes ownership locally, which it can *because* the mesh is replicated. Peak
// memory per rank was therefore proportional to the GLOBAL mesh size, paid
// simultaneously everywhere, and the shape of any non-icosphere initial mesh was
// constrained the same way -- a caller supplying its own geometry had to
// materialize the whole thing on every rank first.
//
// buildFromTriangleSoupDistributed() (per-rank local patches + a canonical key
// per local vertex) and buildIcosphereDistributed() (subdivision-tree partition,
// no axis sort) remove that. This is their acceptance test.
//
// GROUND TRUTH is never Tessera's distributed path checking itself:
//   * the REPLICATED build (buildIcosphere + distribute) is the reference for
//     every geometric comparison, and it is partition-free by construction, so
//     agreeing with it at ranks 1-5 IS rank-count reproducibility (check 4);
//   * the comparison is by the GID-INDEPENDENT identity -- gid numbering
//     legitimately differs between two partitions -- namely the sorted vertex
//     POSITION multiset and the face CORNER-POSITION triple multiset, compared
//     BITWISE on the raw IEEE bit patterns;
//   * the 2-ring reference for check 5 is a BFS on the global vertex adjacency
//     reassembled from every rank's owned edges (an allgather), which is a
//     property of the mesh rather than of either builder;
//   * the count identities in check 6 are closed-form arithmetic (V'=V+E,
//     E'=2E+3F, F'=4F), not a Tessera-vs-Tessera comparison.
//
// CHECKS
//   1. COUNTS AND INVARIANTS. buildIcosphereDistributed at subdivision 2:
//      globalOwnedVertices == 162, Edges == 480, Faces == 320, owned Euler == 2,
//      checkOwnershipPartition / owned1RingLocal / checkConforming all pass, at
//      every rank count.
//   2. EQUIVALENCE WITH THE REPLICATED PATH -- the definitive check. The same
//      subdivision built twice, once each way; the vertex position multiset and
//      the face corner-position triple multiset must be BITWISE equal. Run at
//      subdivisions 1, 2 and 3. This is what makes the two builders
//      interchangeable rather than merely both plausible.
//   3. NOBODY HOLDS THE GLOBAL MESH. At ranks >= 2, subdivision 5 (10242
//      vertices, 20480 faces): mesh.numVertices() < globalOwnedVertices and
//      mesh.numFaces() < globalOwnedFaces on EVERY rank, and
//      numOwnedFaces() <= 2*globalOwnedFaces/size. This is the property the whole
//      task exists to establish -- without it the suite cannot tell the new path
//      from the old one -- so the per-rank local counts are printed too.
//   4. REPRODUCIBILITY ACROSS RANK COUNTS. Check 2's multiset checksum is printed
//      and asserted against the replicated reference, which is partition-free; so
//      the assertion holding at np1..np5 is bitwise agreement across rank counts,
//      and the printed checksums make it visible in the log rather than inferred.
//   5. HALO CORRECTNESS. haloExchange() leaves every ghost vertex position equal
//      to its owner's, bitwise; halo.depth is as requested; and at haloDepth == 2
//      every owned vertex's 2-ring from buildVertexStencil(mesh, 2) equals the
//      global-adjacency reference exactly -- proving the new builder feeds
//      rebuildHalo() correctly at depth > 1, not just at depth 1.
//   6. refine() WORKS ON THE RESULT. Uniform refine of a distributed-built mesh:
//      V'=V+E, E'=2E+3F, F'=4F, checkConforming, checkMidpointAgreement,
//      check21BalanceRed.
//   7. migrate() AND loadBalance() WORK ON THE RESULT. Identity migrate then a
//      real loadBalance; every distribution invariant holds after each.
//   8. I/O ROUND TRIP. writeMesh then readMesh; the position multiset is
//      unchanged.
//   9. DELIVERABLE A DIRECTLY, WITH A NON-ICOSPHERE. An octahedron hand-split into
//      two OVERLAPPING 4-face patches (both patches carry the whole equator ring,
//      and at ranks >= 3 several ranks claim the same patch), asserted equal to the
//      same octahedron via buildFromTriangleSoup + distribute by the
//      position-multiset criterion. Exercises overlap resolution, which the
//      icosphere path's disjoint index ranges never reach.
//  10. KEY COLLISION IS CAUGHT. Two genuinely different vertices given the same
//      canonical key must throw naming the key, on every rank, rather than
//      silently welding them.
//  11. DEGENERATE size > numFaces. A 2-face patch supplied identically by every
//      rank, so at ranks >= 2 some rank owns ZERO faces: the build completes, the
//      collectives do not deadlock, that rank holds zero owned entities, and
//      ownership is still a global partition.
//
// Runs on host (Serial) and device (default, HIP), ranks 1-5.

#include "MeshInvariants.hpp"

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace Tessera;

// ===========================================================================
// Small MPI helpers
// ===========================================================================

//! Concatenate every rank's `local` into one vector held identically by all.
template <class T>
std::vector<T> allGatherAll( MPI_Comm comm, const std::vector<T>& local )
{
    int size = 1;
    MPI_Comm_size( comm, &size );
    const int nbytes = static_cast<int>( local.size() * sizeof( T ) );
    std::vector<int> counts( size, 0 ), displs( size, 0 );
    MPI_Allgather( &nbytes, 1, MPI_INT, counts.data(), 1, MPI_INT, comm );
    int total = 0;
    for ( int i = 0; i < size; ++i )
    {
        displs[i] = total;
        total += counts[i];
    }
    std::vector<T> out( static_cast<std::size_t>( total ) / sizeof( T ) );
    MPI_Allgatherv( local.data(), nbytes, MPI_BYTE, out.data(), counts.data(),
                    displs.data(), MPI_BYTE, comm );
    return out;
}

inline int globalFails( MPI_Comm comm, int local )
{
    int g = 0;
    MPI_Allreduce( &local, &g, 1, MPI_INT, MPI_SUM, comm );
    return g;
}

// ===========================================================================
// Gid-independent geometric identity
// ===========================================================================
//
// Gid NUMBERING differs between two partitions of the same surface, so nothing
// gid-keyed is comparable between the replicated and the distributed builder.
// What IS comparable, and what actually pins the geometry, is the multiset of
// vertex positions and the multiset of face corner-position triples -- compared
// on the raw IEEE bit patterns, so "equal" means bitwise and no tolerance is
// smuggled in.

using Bits3 = std::array<unsigned long long, 3>;
using Tri9 = std::array<unsigned long long, 9>;

template <class Scalar>
Bits3 bitsOf( const Scalar* p )
{
    Bits3 b{};
    for ( int d = 0; d < 3; ++d )
    {
        double v = static_cast<double>( p[d] );
        unsigned long long u = 0;
        std::memcpy( &u, &v, sizeof( double ) );
        b[d] = u;
    }
    return b;
}

//! Every rank's OWNED vertex positions, gathered and sorted lexicographically by
//! bit pattern. Partition-independent by construction (owned entities partition
//! the global mesh), so it is directly comparable between two builders.
template <class MeshT>
std::vector<Bits3> vertexPositionMultiset( MeshT& mesh )
{
    using Scalar = typename MeshT::scalar_type;
    const std::size_t nov = mesh.numOwnedVertices();
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "pm_hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto pos = Cabana::slice<VertexField::Position>( hv );

    std::vector<Bits3> local;
    local.reserve( nov );
    for ( std::size_t i = 0; i < nov; ++i )
    {
        Scalar p[3] = { pos( i, 0 ), pos( i, 1 ), pos( i, 2 ) };
        local.push_back( bitsOf( p ) );
    }
    auto all = allGatherAll( mesh.comm(), local );
    std::sort( all.begin(), all.end() );
    return all;
}

//! Every rank's OWNED faces as corner-position triples, ROTATED so the
//! lexicographically smallest corner comes first. Rotation (not sorting) is the
//! canonicalization, because a rotation of a CCW triple is CCW -- so this still
//! detects a flipped winding, while tolerating the two builders happening to
//! start a triangle at a different corner.
//!
//! A corner of an owned face may be a ghost; after the build every vertex an
//! owned face references is held locally WITH its position (rebuildHalo round
//! G's postcondition), so a purely local gid->position map suffices.
template <class MeshT>
std::vector<Tri9> faceCornerMultiset( MeshT& mesh )
{
    using Scalar = typename MeshT::scalar_type;
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "fc_hv", mesh.numVertices() );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "fc_hf", mesh.numFaces() );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto vgid = Cabana::slice<VertexField::Gid>( hv );
    auto vpos = Cabana::slice<VertexField::Position>( hv );
    auto fverts = Cabana::slice<FaceField::Verts>( hf );

    std::map<GlobalId, Bits3> bitsOfGid;
    for ( std::size_t i = 0; i < mesh.numVertices(); ++i )
    {
        Scalar p[3] = { vpos( i, 0 ), vpos( i, 1 ), vpos( i, 2 ) };
        bitsOfGid[vgid( i )] = bitsOf( p );
    }

    std::vector<Tri9> local;
    local.reserve( mesh.numOwnedFaces() );
    for ( std::size_t f = 0; f < mesh.numOwnedFaces(); ++f )
    {
        Bits3 c[3];
        for ( int k = 0; k < 3; ++k )
            c[k] = bitsOfGid.at( fverts( f, k ) );
        int r = 0;
        for ( int k = 1; k < 3; ++k )
            if ( c[k] < c[r] )
                r = k;
        Tri9 t{};
        for ( int k = 0; k < 3; ++k )
            for ( int d = 0; d < 3; ++d )
                t[3 * k + d] = c[( r + k ) % 3][d];
        local.push_back( t );
    }
    auto all = allGatherAll( mesh.comm(), local );
    std::sort( all.begin(), all.end() );
    return all;
}

//! Order-independent checksum of a multiset, printed so the log carries direct
//! evidence that the value is identical at every rank count.
template <class T>
unsigned long long multisetChecksum( const std::vector<T>& v )
{
    unsigned long long h = 1469598103934665603ULL;
    for ( const auto& e : v )
        for ( unsigned long long x : e )
        {
            h ^= x;
            h *= 1099511628211ULL;
        }
    return h;
}

// ===========================================================================
// Global vertex adjacency reference (for the k-ring check)
// ===========================================================================
//
// Owned edges partition the global edge set, so an allgather of every rank's
// owned edge endpoint pairs reconstructs the WHOLE global vertex adjacency on
// every rank. The true k-ring of any gid is then a BFS on that -- exact,
// rank-count independent, and derived from neither builder's local view.

struct EdgePair
{
    GlobalId a, b;
};

template <class MeshT>
std::map<GlobalId, std::set<GlobalId>> globalVertexAdjacency( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
        "adj_he", mesh.numEdges() );
    Cabana::deep_copy( he, mesh.edges() );
    auto ev = Cabana::slice<EdgeField::Verts>( he );

    std::vector<EdgePair> local;
    local.reserve( mesh.numOwnedEdges() );
    for ( std::size_t e = 0; e < mesh.numOwnedEdges(); ++e )
        local.push_back( { ev( e, 0 ), ev( e, 1 ) } );
    auto all = allGatherAll( mesh.comm(), local );

    std::map<GlobalId, std::set<GlobalId>> adj;
    for ( const auto& p : all )
    {
        adj[p.a].insert( p.b );
        adj[p.b].insert( p.a );
    }
    return adj;
}

//! Exact k-ring of `v` on the global adjacency (self excluded).
inline std::set<GlobalId>
kRing( const std::map<GlobalId, std::set<GlobalId>>& adj, GlobalId v, int k )
{
    std::set<GlobalId> seen{ v }, frontier{ v };
    for ( int d = 0; d < k; ++d )
    {
        std::set<GlobalId> next;
        for ( GlobalId u : frontier )
        {
            auto it = adj.find( u );
            if ( it == adj.end() )
                continue;
            for ( GlobalId w : it->second )
                if ( seen.insert( w ).second )
                    next.insert( w );
        }
        frontier.swap( next );
    }
    seen.erase( v );
    return seen;
}

//! For every OWNED vertex, buildVertexStencil(mesh,k)'s row must equal the exact
//! k-ring on the global reference. Returns LOCAL fails.
template <class MeshT>
int checkKRingRows( MeshT& mesh, int k )
{
    const auto adj = globalVertexAdjacency( mesh );
    auto stencil = buildVertexStencil( mesh, k );
    const auto& csr = stencil.csr.get();

    auto h_off =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), csr.offsets );
    auto h_nbr = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                      csr.neighbors );
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "kr_hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto vgid = Cabana::slice<VertexField::Gid>( hv );

    int fails = 0;
    for ( std::size_t v = 0; v < mesh.numOwnedVertices(); ++v )
    {
        std::set<GlobalId> got;
        for ( int j = h_off( v ); j < h_off( v + 1 ); ++j )
            got.insert( vgid( h_nbr( j ) ) );
        if ( got != kRing( adj, vgid( v ), k ) )
            ++fails;
    }
    return fails;
}

//! Every GHOST vertex's position equals its owner's, bitwise. Ground truth is the
//! allgathered owned (gid, position) set, so this does not consult the halo plan
//! that is being validated. Returns LOCAL fails.
template <class MeshT>
int checkGhostPositionsAreOwners( MeshT& mesh )
{
    using Scalar = typename MeshT::scalar_type;
    struct VP
    {
        GlobalId gid;
        unsigned long long b[3];
    };
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "gp_hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto vgid = Cabana::slice<VertexField::Gid>( hv );
    auto vpos = Cabana::slice<VertexField::Position>( hv );

    std::vector<VP> local;
    for ( std::size_t i = 0; i < mesh.numOwnedVertices(); ++i )
    {
        Scalar p[3] = { vpos( i, 0 ), vpos( i, 1 ), vpos( i, 2 ) };
        Bits3 b = bitsOf( p );
        local.push_back( { vgid( i ), { b[0], b[1], b[2] } } );
    }
    auto all = allGatherAll( mesh.comm(), local );
    std::map<GlobalId, Bits3> owner;
    for ( const auto& r : all )
        owner[r.gid] = Bits3{ r.b[0], r.b[1], r.b[2] };

    int fails = 0;
    for ( std::size_t i = mesh.numOwnedVertices(); i < mesh.numVertices(); ++i )
    {
        Scalar p[3] = { vpos( i, 0 ), vpos( i, 1 ), vpos( i, 2 ) };
        auto it = owner.find( vgid( i ) );
        if ( it == owner.end() || it->second != bitsOf( p ) )
            ++fails;
    }
    return fails;
}

//! Overwrite every ghost position with a sentinel, so a following haloExchange()
//! has something real to restore (the structural checks all pass on an EMPTY
//! plan, which is the regression this guards).
template <class MeshT>
long long corruptGhostPositions( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "cg_hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto vpos = Cabana::slice<VertexField::Position>( hv );
    long long n = 0;
    for ( std::size_t i = mesh.numOwnedVertices(); i < mesh.numVertices(); ++i )
    {
        for ( int d = 0; d < 3; ++d )
            vpos( i, d ) = static_cast<typename MeshT::scalar_type>( -7.5 );
        ++n;
    }
    Cabana::deep_copy( mesh.vertices(), hv );
    return n;
}

// ===========================================================================
// Hand-built surfaces for Deliverable A
// ===========================================================================

//! The regular octahedron: 6 vertices, 12 edges, 8 faces, CCW seen from outside.
//! `patch` selects the upper fan (0), the lower fan (1) or both (-1). BOTH
//! patches carry the whole equator ring 0..3, so the two overlap on four
//! vertices -- which is exactly what the key dedup has to resolve.
template <class Scalar>
TriangleSoup<Scalar> octahedronPatch( int patch, std::vector<VertexKey>& keys )
{
    const double vpos[6][3] = { { 1, 0, 0 },  { -1, 0, 0 }, { 0, 1, 0 },
                                { 0, -1, 0 }, { 0, 0, 1 },  { 0, 0, -1 } };
    const int upper[12] = { 0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4 };
    const int lower[12] = { 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5 };

    std::vector<int> tris;
    if ( patch != 1 )
        tris.insert( tris.end(), upper, upper + 12 );
    if ( patch != 0 )
        tris.insert( tris.end(), lower, lower + 12 );

    // Keep only the vertices this patch touches, so the patch really is local.
    std::map<int, int> localOf;
    TriangleSoup<Scalar> soup;
    keys.clear();
    for ( int g : tris )
    {
        auto it = localOf.find( g );
        int li;
        if ( it == localOf.end() )
        {
            li = static_cast<int>( keys.size() );
            localOf.emplace( g, li );
            keys.push_back( makeVertexKey( static_cast<GlobalId>( g ) ) );
            for ( int d = 0; d < 3; ++d )
                soup.positions.push_back( static_cast<Scalar>( vpos[g][d] ) );
        }
        else
            li = it->second;
        soup.triangles.push_back( li );
    }
    return soup;
}

//! Two triangles sharing an edge: 4 vertices, 5 edges, 2 faces. Used for the
//! degenerate `size > numFaces` case (check 11), so it is deliberately an open
//! disc rather than a closed surface -- the point is that the build completes and
//! the collectives do not deadlock when a rank owns nothing.
template <class Scalar>
TriangleSoup<Scalar> twoFacePatch( std::vector<VertexKey>& keys )
{
    const double vpos[4][3] = {
        { 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 } };
    TriangleSoup<Scalar> soup;
    keys.clear();
    for ( int i = 0; i < 4; ++i )
    {
        keys.push_back( makeVertexKey( static_cast<GlobalId>( i ) ) );
        for ( int d = 0; d < 3; ++d )
            soup.positions.push_back( static_cast<Scalar>( vpos[i][d] ) );
    }
    soup.triangles = { 0, 1, 2, 0, 2, 3 };
    return soup;
}

// ===========================================================================
// The test body
// ===========================================================================

template <class ExecSpace>
int run( int rank, int size, const char* tag, const std::string& ioStem )
{
    using MemSpace = typename ExecSpace::memory_space;
    using MeshT = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                       MemSpace, ExecSpace>;
    MPI_Comm comm = MPI_COMM_WORLD;
    int fails = 0;

    // -- CHECK 1: counts and invariants at subdivision 2 --------------------
    {
        MeshT m( comm );
        MeshHalo<MemSpace> h;
        buildIcosphereDistributed( m, h, 2 );
        haloExchange( m, h );

        const long long V = TesseraTest::globalOwnedVertices( m );
        const long long E = TesseraTest::globalOwnedEdges( m );
        const long long F = TesseraTest::globalOwnedFaces( m );
        const long long X = TesseraTest::checkOwnedEuler( m );
        int c1 = 0;
        if ( V != 162 || E != 480 || F != 320 || X != 2 )
            ++c1;
        c1 += TesseraTest::checkOwnershipPartition( m, V, E, F );
        c1 += TesseraTest::owned1RingLocal( m );
        c1 += TesseraTest::checkConforming( m );
        const int g1 = globalFails( comm, c1 );
        fails += g1;
        if ( rank == 0 )
            printf( "[dbuild] %-7s np%d check1 subdiv2: V=%lld E=%lld F=%lld "
                    "X=%lld fails=%d\n",
                    tag, size, V, E, F, X, g1 );
    }

    // -- CHECK 2 + 4: bitwise equivalence with the replicated path ----------
    for ( int s = 1; s <= 3; ++s )
    {
        MeshT ref( comm );
        MeshHalo<MemSpace> refHalo;
        buildIcosphere( ref, s );
        auto faceOwner = facePartitionByAxis( ref, 2 );
        distribute( ref, refHalo, faceOwner, 1 );
        haloExchange( ref, refHalo );

        MeshT dis( comm );
        MeshHalo<MemSpace> disHalo;
        buildIcosphereDistributed( dis, disHalo, s );
        haloExchange( dis, disHalo );

        const auto refV = vertexPositionMultiset( ref );
        const auto disV = vertexPositionMultiset( dis );
        const auto refF = faceCornerMultiset( ref );
        const auto disF = faceCornerMultiset( dis );

        int c2 = 0;
        if ( refV != disV )
            ++c2;
        if ( refF != disF )
            ++c2;
        // Non-vacuity: the reference must actually be the whole mesh.
        const long long expV = ( s == 1 ) ? 42 : ( s == 2 ? 162 : 642 );
        const long long expF = 20LL << ( 2 * s );
        if ( static_cast<long long>( refV.size() ) != expV ||
             static_cast<long long>( refF.size() ) != expF )
            ++c2;
        fails += c2;
        if ( rank == 0 )
            printf( "[dbuild] %-7s np%d check2/4 subdiv%d: |V|=%zu |F|=%zu "
                    "vsum=%016llx fsum=%016llx refvsum=%016llx fails=%d\n",
                    tag, size, s, disV.size(), disF.size(),
                    multisetChecksum( disV ), multisetChecksum( disF ),
                    multisetChecksum( refV ), c2 );
    }

    // -- CHECK 3: nobody holds the global mesh ------------------------------
    {
        MeshT m( comm );
        MeshHalo<MemSpace> h;
        buildIcosphereDistributed( m, h, 5 );

        const long long V = TesseraTest::globalOwnedVertices( m );
        const long long F = TesseraTest::globalOwnedFaces( m );
        int c3 = 0;
        if ( V != 10242 || F != 20480 )
            ++c3;
        if ( size >= 2 )
        {
            if ( static_cast<long long>( m.numVertices() ) >= V )
                ++c3;
            if ( static_cast<long long>( m.numFaces() ) >= F )
                ++c3;
            if ( static_cast<long long>( m.numOwnedFaces() ) > 2 * F / size )
                ++c3;
        }
        c3 += TesseraTest::checkOwnershipPartition(
            m, V, TesseraTest::globalOwnedEdges( m ), F );
        const int g3 = globalFails( comm, c3 );
        fails += g3;
        // Printed from every rank: the measurement IS the deliverable.
        printf( "[dbuild] %-7s np%d rank%d check3 subdiv5: localV=%zu "
                "localF=%zu ownedV=%zu ownedF=%zu globalV=%lld globalF=%lld\n",
                tag, size, rank, m.numVertices(), m.numFaces(),
                m.numOwnedVertices(), m.numOwnedFaces(), V, F );
        fflush( stdout );
        if ( rank == 0 )
            printf( "[dbuild] %-7s np%d check3: fails=%d\n", tag, size, g3 );
    }

    // -- CHECK 5: halo correctness, depth 1 and depth 2 ---------------------
    for ( int depth = 1; depth <= 2; ++depth )
    {
        MeshT m( comm );
        MeshHalo<MemSpace> h;
        buildIcosphereDistributed( m, h, 2, depth );
        int c5 = 0;
        if ( h.depth != depth || m.haloDepth() != depth )
            ++c5;
        const long long corrupted = corruptGhostPositions( m );
        haloExchange( m, h );
        c5 += checkGhostPositionsAreOwners( m );
        // Non-vacuity: at ranks >= 2 there must have been ghosts to restore.
        if ( size > 1 && corrupted <= 0 )
            ++c5;
        for ( int k = 1; k <= depth; ++k )
            c5 += checkKRingRows( m, k );
        const int g5 = globalFails( comm, c5 );
        fails += g5;
        if ( rank == 0 )
            printf( "[dbuild] %-7s np%d check5 depth%d: corrupted=%lld "
                    "fails=%d\n",
                    tag, size, depth, corrupted, g5 );
    }

    // -- CHECK 6: refine() works on the result ------------------------------
    {
        MeshT m( comm );
        MeshHalo<MemSpace> h;
        buildIcosphereDistributed( m, h, 2 );
        const long long V = TesseraTest::globalOwnedVertices( m );
        const long long E = TesseraTest::globalOwnedEdges( m );
        const long long F = TesseraTest::globalOwnedFaces( m );

        std::vector<char> mask( m.numOwnedFaces(), 1 );
        RefineResult rr = refine( m, h, mask );
        haloExchange( m, h );

        const long long V2 = TesseraTest::globalOwnedVertices( m );
        const long long E2 = TesseraTest::globalOwnedEdges( m );
        const long long F2 = TesseraTest::globalOwnedFaces( m );
        int c6 = 0;
        if ( V2 != V + E || E2 != 2 * E + 3 * F || F2 != 4 * F )
            ++c6;
        c6 += TesseraTest::checkConforming( m );
        c6 += TesseraTest::checkMidpointAgreement( comm, size, rr.midpoints );
        c6 += TesseraTest::check21BalanceRed( m );
        c6 += TesseraTest::checkOwnershipPartition( m, V2, E2, F2 );
        c6 += TesseraTest::owned1RingLocal( m );
        const int g6 = globalFails( comm, c6 );
        fails += g6;
        if ( rank == 0 )
            printf(
                "[dbuild] %-7s np%d check6 refine: V %lld->%lld E %lld->%lld "
                "F %lld->%lld fails=%d\n",
                tag, size, V, V2, E, E2, F, F2, g6 );
    }

    // -- CHECK 7: migrate() and loadBalance() work on the result ------------
    {
        MeshT m( comm );
        MeshHalo<MemSpace> h;
        buildIcosphereDistributed( m, h, 3 );
        int c7 = 0;
        {
            std::vector<Rank> dest( m.numOwnedFaces(),
                                    static_cast<Rank>( rank ) );
            migrate( m, h, dest );
            haloExchange( m, h );
            c7 += TesseraTest::checkOwnershipPartition(
                m, TesseraTest::globalOwnedVertices( m ),
                TesseraTest::globalOwnedEdges( m ),
                TesseraTest::globalOwnedFaces( m ) );
            c7 += TesseraTest::owned1RingLocal( m );
        }
        loadBalance( m, h );
        haloExchange( m, h );
        const long long V = TesseraTest::globalOwnedVertices( m );
        const long long E = TesseraTest::globalOwnedEdges( m );
        const long long F = TesseraTest::globalOwnedFaces( m );
        if ( V != 642 || E != 1920 || F != 1280 )
            ++c7;
        c7 += TesseraTest::checkOwnershipPartition( m, V, E, F );
        c7 += TesseraTest::owned1RingLocal( m );
        c7 += TesseraTest::checkConforming( m );
        c7 += checkGhostPositionsAreOwners( m );
        const int g7 = globalFails( comm, c7 );
        fails += g7;
        if ( rank == 0 )
            printf( "[dbuild] %-7s np%d check7 migrate/loadBalance: V=%lld "
                    "E=%lld F=%lld fails=%d\n",
                    tag, size, V, E, F, g7 );
    }

    // -- CHECK 8: I/O round trip -------------------------------------------
    {
        MeshT m( comm );
        MeshHalo<MemSpace> h;
        buildIcosphereDistributed( m, h, 2 );
        haloExchange( m, h );
        const auto before = vertexPositionMultiset( m );

        writeMesh( m, ioStem );

        MeshT rd( comm );
        MeshHalo<MemSpace> rdHalo;
        readMesh( rd, rdHalo, ioStem );
        haloExchange( rd, rdHalo );
        const auto after = vertexPositionMultiset( rd );

        int c8 = ( before == after ) ? 0 : 1;
        c8 += TesseraTest::checkConforming( rd );
        MPI_Barrier( comm );
        if ( rank == 0 )
        {
            std::remove( ( ioStem + ".h5" ).c_str() );
            std::remove( ( ioStem + ".xmf" ).c_str() );
        }
        const int g8 = globalFails( comm, c8 );
        fails += g8;
        if ( rank == 0 )
            printf( "[dbuild] %-7s np%d check8 io: |V|=%zu sum=%016llx "
                    "fails=%d\n",
                    tag, size, after.size(), multisetChecksum( after ), g8 );
    }

    // -- CHECK 9: Deliverable A directly, overlapping octahedron patches -----
    {
        // Reference: the whole octahedron, replicated, then distributed.
        std::vector<VertexKey> allKeys;
        auto whole = octahedronPatch<double>( -1, allKeys );
        MeshT ref( comm );
        MeshHalo<MemSpace> refHalo;
        buildFromTriangleSoup( ref, whole );
        auto faceOwner = facePartitionByAxis( ref, 2 );
        distribute( ref, refHalo, faceOwner, 1 );
        haloExchange( ref, refHalo );

        // Distributed: alternating overlapping half-patches. At size 1 one rank
        // supplies both halves; at size >= 3 several ranks claim the SAME patch,
        // so the dedup resolves a full duplicate as well as a shared ring.
        std::vector<VertexKey> keys;
        auto patch = ( size == 1 ) ? octahedronPatch<double>( -1, keys )
                                   : octahedronPatch<double>( rank % 2, keys );
        MeshT dis( comm );
        MeshHalo<MemSpace> disHalo;
        buildFromTriangleSoupDistributed( dis, disHalo, patch, keys );
        haloExchange( dis, disHalo );

        const long long V = TesseraTest::globalOwnedVertices( dis );
        const long long E = TesseraTest::globalOwnedEdges( dis );
        const long long F = TesseraTest::globalOwnedFaces( dis );
        int c9 = 0;
        if ( V != 6 || E != 12 || F != 8 )
            ++c9;
        if ( vertexPositionMultiset( ref ) != vertexPositionMultiset( dis ) )
            ++c9;
        if ( faceCornerMultiset( ref ) != faceCornerMultiset( dis ) )
            ++c9;
        c9 += TesseraTest::checkOwnershipPartition( dis, V, E, F );
        c9 += TesseraTest::owned1RingLocal( dis );
        c9 += TesseraTest::checkConforming( dis );
        c9 += checkGhostPositionsAreOwners( dis );
        const int g9 = globalFails( comm, c9 );
        fails += g9;
        if ( rank == 0 )
            printf( "[dbuild] %-7s np%d check9 octahedron patches: V=%lld "
                    "E=%lld F=%lld fails=%d\n",
                    tag, size, V, E, F, g9 );
    }

    // -- CHECK 10: a canonical-key collision throws, naming the key ---------
    {
        // Two vertices given the SAME key at different positions. Every rank
        // supplies the collision, so the case is identical at every rank count
        // and the collective throw is exercised either way.
        TriangleSoup<double> soup;
        const double p[4][3] = { { 0, 0, 0 },
                                 { 1, 0, 0 },
                                 { 0, 1, 0 },
                                 { 0, 0, 1 } }; // p[3] shares p[0]'s key
        for ( int i = 0; i < 4; ++i )
            for ( int d = 0; d < 3; ++d )
                soup.positions.push_back( p[i][d] );
        soup.triangles = { 0, 1, 2, 3, 2, 1 };
        std::vector<VertexKey> keys = { makeVertexKey( 0 ), makeVertexKey( 1 ),
                                        makeVertexKey( 2 ),
                                        makeVertexKey( 0 ) };

        MeshT m( comm );
        MeshHalo<MemSpace> h;
        int threw = 0;
        std::string what;
        try
        {
            buildFromTriangleSoupDistributed( m, h, soup, keys );
        }
        catch ( const std::runtime_error& e )
        {
            threw = 1;
            what = e.what();
        }
        int c10 = threw ? 0 : 1;
        // The message must name the offending key, or it is not actionable.
        if ( threw && what.find( "KEY COLLISION" ) == std::string::npos )
            ++c10;
        if ( threw && what.find( "VertexKey {0," ) == std::string::npos )
            ++c10;
        const int g10 = globalFails( comm, c10 );
        fails += g10;
        if ( rank == 0 )
            printf( "[dbuild] %-7s np%d check10 key collision: threw=%d "
                    "fails=%d\n",
                    tag, size, threw, g10 );
    }

    // -- CHECK 11: degenerate size > numFaces, some rank owns nothing -------
    {
        std::vector<VertexKey> keys;
        auto patch = twoFacePatch<double>( keys );
        MeshT m( comm );
        MeshHalo<MemSpace> h;
        buildFromTriangleSoupDistributed( m, h, patch, keys );
        haloExchange( m, h );

        const long long V = TesseraTest::globalOwnedVertices( m );
        const long long E = TesseraTest::globalOwnedEdges( m );
        const long long F = TesseraTest::globalOwnedFaces( m );
        int c11 = 0;
        if ( V != 4 || E != 5 || F != 2 )
            ++c11;
        c11 += TesseraTest::checkOwnershipPartition( m, V, E, F );
        // Every rank supplies the whole patch, so the lowest claimant (rank 0)
        // owns all of it and every other rank owns nothing at all.
        if ( rank != 0 &&
             ( m.numOwnedFaces() != 0 || m.numOwnedVertices() != 0 ||
               m.numOwnedEdges() != 0 ) )
            ++c11;
        if ( rank == 0 &&
             ( m.numOwnedFaces() != 2 || m.numOwnedVertices() != 4 ||
               m.numOwnedEdges() != 5 ) )
            ++c11;
        const int g11 = globalFails( comm, c11 );
        fails += g11;
        if ( rank == 0 )
            printf(
                "[dbuild] %-7s np%d check11 zero-owner ranks: V=%lld E=%lld "
                "F=%lld fails=%d\n",
                tag, size, V, E, F, g11 );
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

        // Unique HDF5 stem per executable + rank count (the SERIAL and HIP
        // registrations are different executables, and ctest may run them
        // concurrently), matching test_io.cpp's idiom.
        std::string base = ( argc > 0 ) ? std::string( argv[0] )
                                        : std::string( "test_dbuild" );
        const std::size_t slash = base.find_last_of( '/' );
        if ( slash != std::string::npos )
            base = base.substr( slash + 1 );
        base += "_np" + std::to_string( size );
        if ( const char* tmpdir = std::getenv( "TESSERA_IO_TMPDIR" ) )
            base = std::string( tmpdir ) + "/" + base;

        fails += run<Kokkos::Serial>( rank, size, "Serial", base + "_serial" );
        if constexpr ( !std::is_same<Kokkos::DefaultExecutionSpace,
                                     Kokkos::Serial>::value )
            fails += run<Kokkos::DefaultExecutionSpace>( rank, size, "Default",
                                                         base + "_default" );

        if ( rank == 0 )
            printf( "[dbuild] TOTAL fails=%d\n", fails );
    }
    Kokkos::finalize();
    MPI_Finalize();
    return fails == 0 ? 0 : 1;
}
