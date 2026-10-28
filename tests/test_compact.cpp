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

// Regression test: mesh compaction (tasks/completed/mesh-compaction.md).
//
// compact() removes every tombstoned entity (`Gid == invalid_gid`) and every
// vertex/edge no surviving face references, restores the owned-first
// gid-ascending ordering, and rebuilds both CSRs, both key tables and the three
// halo plans -- PRESERVING GIDS. compactAndRenumberGids() additionally
// renumbers the gid space down to the live count, which is the mitigation for
// the dense-by-gid indexing in buildKindPlan()/make_g2l()/build_order().
//
// THE FIXTURE, and why it is exact. Nothing produces dead entities yet
// (collapse is a later task), so the tombstone set is made by CLOSED SUB-MESH
// REMOVAL: remove the 16 faces of one base-icosahedron patch, plus exactly the
// vertices and edges that no surviving face references. That set closes by
// construction and the result is a surface WITH BOUNDARY -- a legitimate
// exercise of the machinery even though it is not what collapse will produce.
//
// The ground truth is a REPLICATED REFERENCE MESH built on MPI_COMM_SELF, the
// idiom test_halo_depth uses: buildIcosphere() is deterministic and generates
// the identical soup on every rank, and distribute() carries the replicated
// index through as the gid, so face gid f of the distributed mesh is triangle f
// of the reference and every expected count, gid set and position is computed
// rather than hardcoded. The patch-as-a-gid-range rule is valid on THIS path
// only: face gid == soup triangle index (Tessera_MeshBuilder.hpp) and
// subdivision emits each face's four children consecutively
// (Tessera_Icosphere.hpp), so at subdivision 2 base face i is exactly the face
// gids [16i, 16i+16). buildFromTriangleSoupDistributed() numbers by partition
// and the rule does NOT hold there.
//
//  1. NO-OP. Nothing tombstoned: V/E/F, topologyChecksum and every gid
//     unchanged, CompactStats all zero, gid space unchanged.
//  2. IDEMPOTENCE. compact() twice; the second is a no-op by check 1's criteria.
//  3. CLOSED SUB-MESH REMOVAL. One base patch. Global owned counts drop by
//     exactly the removed counts; CompactStats matches; EVERY SURVIVING GID IS
//     UNCHANGED (checked against a pre-edit gid -> position snapshot, and
//     against the reference's surviving gid set); owned-first with gids
//     ascending in each block; owned1RingLocal; and the surviving faces' corner
//     POSITION triples are exactly the pre-edit set minus the removed ones.
//  4. HALO AFTER REMOVAL. Every ghost position corrupted and restored by
//     haloExchange(); ownership still a partition; no removed entity survives
//     anywhere, so none can appear in a plan.
//  5. EULER ON THE RESULT. V - E + F == 2 - (number of boundary components),
//     with the component count computed from the reference, not hardcoded.
//  6. DANGLING REFERENCE THROWS. One owned vertex tombstoned and nothing else:
//     every rank throws, the message names a live face and the dead vertex gid,
//     and the mesh is not mutated. Plus the editing-family guard both ways.
//  7. EVERYTHING DEAD ON THE LAST RANK. At ranks >= 2, remove every face rank
//     size-1 owns. That rank ends with zero owned AND zero local entities, the
//     call completes everywhere, Euler reflects the reduced mesh, haloExchange
//     succeeds, and the empty rank is no peer's send or recv target.
//  8. GENERATION GUARD. compact() strictly advances mesh.generation(), which is
//     what makes a handle taken beforehand abort on next use. The abort itself
//     is test_staleslice_guard.cpp's subject (it needs fork(), which is not
//     safe to do under MPI + HIP at ranks 1-5).
//  9. RENUMBERING. After check 3's removal: gids are [0, N) contiguous per kind
//     globally, gidSpaceAfter < gidSpaceBefore and equals the live count, every
//     connectivity field resolves, the face corner-position multiset is
//     UNCHANGED by the renumbering (positions are the gid-independent identity
//     of the mesh), and the halo is still correct.
// 10. RENUMBERING IS RANK-COUNT DETERMINISTIC. The global (gid -> position) map
//     equals the one the same run computes alone on MPI_COMM_SELF. This is what
//     the order-statistic definition of the new gid buys; a per-rank MPI_Exscan
//     block would fail it.
// 11. REPEATED ROUNDS DO NOT LEAK GID SPACE. Ten rounds of "remove a patch,
//     compact()": gidSpaceBefore is monotone and stays pinned near its initial
//     value while the live count falls, then one compactAndRenumberGids()
//     returns it to exactly the live count. Pins the hazard the task exists to
//     close; the measured numbers are printed.
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
#include <string>
#include <type_traits>
#include <vector>

using namespace Tessera;

// Coarse subdiv-2 icosphere entity counts (the Step-6b fixture).
static const long long V0 = 162, E0 = 480, F0 = 320;
//! Faces per base-icosahedron patch at subdivision 2 (4^2).
static const int kPatch = 16;

static int gsum( MPI_Comm comm, int v )
{
    int out = 0;
    MPI_Allreduce( &v, &out, 1, MPI_INT, MPI_SUM, comm );
    return out;
}

// ---------------------------------------------------------------------------
// Replicated ground truth
// ---------------------------------------------------------------------------

//! The whole subdiv-2 icosphere as plain host data: per face its 3 vertex gids
//! and 3 edge gids, and every vertex position. Built on MPI_COMM_SELF, so every
//! rank computes the identical table with no communication.
struct Reference
{
    std::vector<std::array<GlobalId, 3>> fv, fe;
    std::vector<std::array<double, 3>> pos;
};

template <class MeshT>
static Reference buildReference()
{
    MeshT ref( MPI_COMM_SELF );
    buildIcosphere( ref, 2 );

    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", ref.numFaces() );
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", ref.numVertices() );
    Cabana::deep_copy( hf, ref.faces() );
    Cabana::deep_copy( hv, ref.vertices() );
    auto fg = Cabana::slice<FaceField::Gid>( hf );
    auto fv = Cabana::slice<FaceField::Verts>( hf );
    auto fe = Cabana::slice<FaceField::Edges>( hf );
    auto vg = Cabana::slice<VertexField::Gid>( hv );
    auto vp = Cabana::slice<VertexField::Position>( hv );

    Reference R;
    R.fv.resize( ref.numFaces() );
    R.fe.resize( ref.numFaces() );
    R.pos.resize( ref.numVertices() );
    for ( std::size_t f = 0; f < ref.numFaces(); ++f )
        for ( int k = 0; k < 3; ++k )
        {
            R.fv[fg( f )][k] = fv( f, k );
            R.fe[fg( f )][k] = fe( f, k );
        }
    for ( std::size_t v = 0; v < ref.numVertices(); ++v )
        for ( int d = 0; d < 3; ++d )
            R.pos[vg( v )][d] = static_cast<double>( vp( v, d ) );
    return R;
}

//! The globally dead entity sets implied by a dead FACE set: the faces
//! themselves, plus exactly the vertices and edges no surviving face
//! references. Closed by construction.
struct DeadSet
{
    std::set<GlobalId> f, v, e;
};

static DeadSet closureOf( const Reference& R, const std::set<GlobalId>& deadF )
{
    std::set<GlobalId> liveV, liveE;
    for ( std::size_t f = 0; f < R.fv.size(); ++f )
        if ( !deadF.count( static_cast<GlobalId>( f ) ) )
            for ( int k = 0; k < 3; ++k )
            {
                liveV.insert( R.fv[f][k] );
                liveE.insert( R.fe[f][k] );
            }

    DeadSet d;
    d.f = deadF;
    for ( std::size_t f = 0; f < R.fv.size(); ++f )
        for ( int k = 0; k < 3; ++k )
        {
            if ( !liveV.count( R.fv[f][k] ) )
                d.v.insert( R.fv[f][k] );
            if ( !liveE.count( R.fe[f][k] ) )
                d.e.insert( R.fe[f][k] );
        }
    return d;
}

//! Face gids of base-icosahedron patches [first, last).
static std::set<GlobalId> patchFaces( int first, int last )
{
    std::set<GlobalId> s;
    for ( int p = first; p < last; ++p )
        for ( int i = 0; i < kPatch; ++i )
            s.insert( static_cast<GlobalId>( p * kPatch + i ) );
    return s;
}

//! Number of connected components of the boundary of the surviving surface: an
//! edge with exactly one live incident face is a boundary edge, and the
//! components are those of the graph its endpoints form.
static int boundaryComponents( const Reference& R,
                               const std::set<GlobalId>& deadF )
{
    std::map<GlobalId, int> inc; // edge gid -> live incidence count
    std::map<GlobalId, std::array<GlobalId, 2>> ends;
    for ( std::size_t f = 0; f < R.fv.size(); ++f )
    {
        if ( deadF.count( static_cast<GlobalId>( f ) ) )
            continue;
        for ( int k = 0; k < 3; ++k )
        {
            ++inc[R.fe[f][k]];
            ends[R.fe[f][k]] = { R.fv[f][k], R.fv[f][( k + 1 ) % 3] };
        }
    }
    std::map<GlobalId, std::set<GlobalId>> adj;
    for ( const auto& kv : inc )
        if ( kv.second == 1 )
        {
            const auto& e = ends.at( kv.first );
            adj[e[0]].insert( e[1] );
            adj[e[1]].insert( e[0] );
        }

    std::set<GlobalId> seen;
    int comps = 0;
    for ( const auto& kv : adj )
    {
        if ( seen.count( kv.first ) )
            continue;
        ++comps;
        std::vector<GlobalId> stack{ kv.first };
        seen.insert( kv.first );
        while ( !stack.empty() )
        {
            const GlobalId u = stack.back();
            stack.pop_back();
            for ( GlobalId w : adj[u] )
                if ( seen.insert( w ).second )
                    stack.push_back( w );
        }
    }
    return comps;
}

// ---------------------------------------------------------------------------
// Mesh readers
// ---------------------------------------------------------------------------

template <class MeshT>
static void setup( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo )
{
    buildIcosphere( mesh, 2 );
    auto faceOwner = facePartitionByAxis( mesh );
    distribute( mesh, halo, faceOwner );
}

//! Local gids of one kind, by local index.
template <class AoSoAType>
static std::vector<GlobalId> localGids( const AoSoAType& a, std::size_t n )
{
    Cabana::AoSoA<typename AoSoAType::member_types, Kokkos::HostSpace> h(
        "h", a.size() );
    Cabana::deep_copy( h, a );
    auto g = Cabana::slice<0>( h );
    std::vector<GlobalId> out( n );
    for ( std::size_t i = 0; i < n; ++i )
        out[i] = g( i );
    return out;
}

//! gid -> position of every LOCAL vertex.
template <class MeshT>
static std::map<GlobalId, std::array<double, 3>> localVertexPos( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto g = Cabana::slice<VertexField::Gid>( hv );
    auto p = Cabana::slice<VertexField::Position>( hv );
    std::map<GlobalId, std::array<double, 3>> out;
    for ( std::size_t i = 0; i < mesh.numVertices(); ++i )
        out[g( i )] = { static_cast<double>( p( i, 0 ) ),
                        static_cast<double>( p( i, 1 ) ),
                        static_cast<double>( p( i, 2 ) ) };
    return out;
}

//! Each OWNED face as its three corner POSITIONS, sorted within the face, so
//! the multiset is comparable across rank counts and gid numberings.
using Triple = std::array<double, 9>;

template <class MeshT>
static std::vector<Triple> ownedFaceTriples( MeshT& mesh, int& fails )
{
    const auto pos = localVertexPos( mesh );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto fv = Cabana::slice<FaceField::Verts>( hf );

    std::vector<Triple> out;
    out.reserve( mesh.numOwnedFaces() );
    for ( std::size_t f = 0; f < mesh.numOwnedFaces(); ++f )
    {
        std::array<std::array<double, 3>, 3> c;
        bool ok = true;
        for ( int k = 0; k < 3; ++k )
        {
            auto it = pos.find( fv( f, k ) );
            if ( it == pos.end() )
            {
                ok = false; // an owned face's corner must be held locally
                break;
            }
            c[k] = it->second;
        }
        if ( !ok )
        {
            ++fails;
            continue;
        }
        std::sort( c.begin(), c.end() );
        Triple t;
        for ( int k = 0; k < 3; ++k )
            for ( int d = 0; d < 3; ++d )
                t[3 * k + d] = c[k][d];
        out.push_back( t );
    }
    return out;
}

//! The reference's surviving face corner-position triples, same normalization.
static std::vector<Triple> referenceTriples( const Reference& R,
                                             const std::set<GlobalId>& deadF )
{
    std::vector<Triple> out;
    for ( std::size_t f = 0; f < R.fv.size(); ++f )
    {
        if ( deadF.count( static_cast<GlobalId>( f ) ) )
            continue;
        std::array<std::array<double, 3>, 3> c;
        for ( int k = 0; k < 3; ++k )
            c[k] = R.pos[R.fv[f][k]];
        std::sort( c.begin(), c.end() );
        Triple t;
        for ( int k = 0; k < 3; ++k )
            for ( int d = 0; d < 3; ++d )
                t[3 * k + d] = c[k][d];
        out.push_back( t );
    }
    return out;
}

//! Gather every rank's triples onto rank 0, sorted. Returns empty off root.
static std::vector<Triple> gatherTriples( MPI_Comm comm, int size,
                                          const std::vector<Triple>& mine )
{
    std::vector<std::vector<Triple>> send( size );
    send[0] = mine;
    auto got = allToAllV( comm, send );
    std::vector<Triple> all( got.data.begin(), got.data.end() );
    std::sort( all.begin(), all.end() );
    return all;
}

//! Gather (gid, value) pairs of one kind onto rank 0, sorted by gid.
struct GidPos
{
    GlobalId gid;
    double p[9];
};

static std::vector<GidPos> gatherGidPos( MPI_Comm comm, int size,
                                         const std::vector<GidPos>& mine )
{
    std::vector<std::vector<GidPos>> send( size );
    send[0] = mine;
    auto got = allToAllV( comm, send );
    std::vector<GidPos> all( got.data.begin(), got.data.end() );
    std::sort( all.begin(), all.end(),
               []( const GidPos& a, const GidPos& b ) { return a.gid < b.gid; } );
    return all;
}

//! Owned vertices as (gid, position) and owned faces as (gid, sorted corner
//! position triple) -- the gid-keyed identity of the mesh, for check 10.
template <class MeshT>
static void gidKeyedIdentity( MeshT& mesh, std::vector<GidPos>& verts,
                              std::vector<GidPos>& faces, int& fails )
{
    const auto pos = localVertexPos( mesh );
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", mesh.numVertices() );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto vg = Cabana::slice<VertexField::Gid>( hv );
    auto vp = Cabana::slice<VertexField::Position>( hv );
    auto fg = Cabana::slice<FaceField::Gid>( hf );
    auto fv = Cabana::slice<FaceField::Verts>( hf );

    verts.clear();
    faces.clear();
    for ( std::size_t v = 0; v < mesh.numOwnedVertices(); ++v )
    {
        GidPos m;
        std::memset( &m, 0, sizeof( m ) );
        m.gid = vg( v );
        for ( int d = 0; d < 3; ++d )
            m.p[d] = static_cast<double>( vp( v, d ) );
        verts.push_back( m );
    }
    for ( std::size_t f = 0; f < mesh.numOwnedFaces(); ++f )
    {
        std::array<std::array<double, 3>, 3> c;
        bool ok = true;
        for ( int k = 0; k < 3; ++k )
        {
            auto it = pos.find( fv( f, k ) );
            if ( it == pos.end() )
            {
                ok = false;
                break;
            }
            c[k] = it->second;
        }
        if ( !ok )
        {
            ++fails;
            continue;
        }
        std::sort( c.begin(), c.end() );
        GidPos m;
        std::memset( &m, 0, sizeof( m ) );
        m.gid = fg( f );
        for ( int k = 0; k < 3; ++k )
            for ( int d = 0; d < 3; ++d )
                m.p[3 * k + d] = c[k][d];
        faces.push_back( m );
    }
}

// ---------------------------------------------------------------------------
// Editing
// ---------------------------------------------------------------------------

//! Tombstone every LOCAL entity (owned or ghost) whose gid is in the dead set.
//! Marking the ghost copies too is harmless -- deadness is owner-scoped -- and
//! is what a real caller sweeping its local entities would do.
template <class MeshT>
static void applyTombstones( MeshT& mesh, const DeadSet& d )
{
    const auto vg = localGids( mesh.vertices(), mesh.numVertices() );
    const auto eg = localGids( mesh.edges(), mesh.numEdges() );
    const auto fg = localGids( mesh.faces(), mesh.numFaces() );
    for ( std::size_t i = 0; i < fg.size(); ++i )
        if ( d.f.count( fg[i] ) )
            tombstoneFace( mesh, static_cast<LocalIndex>( i ) );
    for ( std::size_t i = 0; i < eg.size(); ++i )
        if ( d.e.count( eg[i] ) )
            tombstoneEdge( mesh, static_cast<LocalIndex>( i ) );
    for ( std::size_t i = 0; i < vg.size(); ++i )
        if ( d.v.count( vg[i] ) )
            tombstoneVertex( mesh, static_cast<LocalIndex>( i ) );
}

//! Corrupt every GHOST vertex position, then haloExchange() and check every
//! local position is back to the reference value for its gid. Returns LOCAL
//! fails. Non-vacuous at ranks >= 2 (asserted by the caller through the plan).
template <class MeshT>
static int ghostRoundTrip( MeshT& mesh,
                           MeshHalo<typename MeshT::memory_space>& halo,
                           const Reference& R )
{
    const std::size_t nv = mesh.numVertices();
    const std::size_t nov = mesh.numOwnedVertices();
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", nv );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto p = Cabana::slice<VertexField::Position>( hv );
    for ( std::size_t i = nov; i < nv; ++i )
        for ( int d = 0; d < 3; ++d )
            p( i, d ) = -12345.0;
    Cabana::deep_copy( mesh.vertices(), hv );

    haloExchange( mesh, halo );

    Cabana::deep_copy( hv, mesh.vertices() );
    auto g = Cabana::slice<VertexField::Gid>( hv );
    auto q = Cabana::slice<VertexField::Position>( hv );
    int fails = 0;
    for ( std::size_t i = 0; i < nv; ++i )
    {
        if ( g( i ) == invalid_gid || g( i ) >= R.pos.size() )
        {
            ++fails;
            continue;
        }
        for ( int d = 0; d < 3; ++d )
            if ( static_cast<double>( q( i, d ) ) != R.pos[g( i )][d] )
                ++fails;
    }
    return fails;
}

//! Owned-first with gids strictly ascending in each block, and the owned block
//! really owned. Returns LOCAL fails.
template <class MeshT>
static int checkOrdering( MeshT& mesh )
{
    int fails = 0;
    auto one = [&]( const auto& aosoa, std::size_t nOwned, std::size_t n,
                    auto ownerSliceIndex )
    {
        (void)ownerSliceIndex;
        Cabana::AoSoA<typename std::decay<decltype( aosoa )>::type::member_types,
                      Kokkos::HostSpace>
            h( "h", n );
        Cabana::deep_copy( h, aosoa );
        auto g = Cabana::slice<0>( h );
        auto o = Cabana::slice<1>( h ); // Owner is member 1 for every kind
        for ( std::size_t i = 0; i < n; ++i )
        {
            if ( g( i ) == invalid_gid )
                ++fails; // no tombstone may survive a compaction
            if ( i > 0 && i != nOwned && !( g( i - 1 ) < g( i ) ) )
                ++fails; // not ascending within its block
        }
        for ( std::size_t i = 0; i < nOwned; ++i )
            if ( o( i ) != static_cast<Rank>( mesh.rank() ) )
                ++fails;
        for ( std::size_t i = nOwned; i < n; ++i )
            if ( o( i ) == static_cast<Rank>( mesh.rank() ) )
                ++fails;
    };
    one( mesh.vertices(), mesh.numOwnedVertices(), mesh.numVertices(), 0 );
    one( mesh.edges(), mesh.numOwnedEdges(), mesh.numEdges(), 0 );
    one( mesh.faces(), mesh.numOwnedFaces(), mesh.numFaces(), 0 );
    return fails;
}

//! No local entity of any kind carries a gid in the dead set. Since plans index
//! local slots, this is what "the removed entities appear in no plan" means.
template <class MeshT>
static int checkNoneSurvive( MeshT& mesh, const DeadSet& d )
{
    int fails = 0;
    for ( GlobalId g : localGids( mesh.vertices(), mesh.numVertices() ) )
        fails += d.v.count( g ) ? 1 : 0;
    for ( GlobalId g : localGids( mesh.edges(), mesh.numEdges() ) )
        fails += d.e.count( g ) ? 1 : 0;
    for ( GlobalId g : localGids( mesh.faces(), mesh.numFaces() ) )
        fails += d.f.count( g ) ? 1 : 0;
    return fails;
}

// ===========================================================================
// Cases
// ===========================================================================

//! Cases 1 and 2: the no-op and its idempotence.
template <class MeshT>
static int caseNoOp( int rank, const char* tag )
{
    using mem = typename MeshT::memory_space;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    setup( mesh, halo );

    unsigned long long c0[3], c1[3], c2[3];
    TesseraTest::topologyChecksum( mesh, c0[0], c0[1], c0[2] );
    const auto pos0 = localVertexPos( mesh );

    int local = 0;
    const CompactStats s1 = compact( mesh, halo );
    TesseraTest::topologyChecksum( mesh, c1[0], c1[1], c1[2] );
    const CompactStats s2 = compact( mesh, halo );
    TesseraTest::topologyChecksum( mesh, c2[0], c2[1], c2[2] );

    for ( int i = 0; i < 3; ++i )
        if ( c1[i] != c0[i] || c2[i] != c0[i] )
            ++local;
    if ( TesseraTest::globalOwnedVertices( mesh ) != V0 ||
         TesseraTest::globalOwnedEdges( mesh ) != E0 ||
         TesseraTest::globalOwnedFaces( mesh ) != F0 )
        ++local;
    for ( const CompactStats& s : { s1, s2 } )
    {
        if ( s.verticesRemoved || s.edgesRemoved || s.facesRemoved )
            ++local;
        if ( s.gidSpaceBefore != V0 + E0 + F0 ||
             s.gidSpaceAfter != V0 + E0 + F0 )
            ++local;
    }
    // Every gid still labels the same point, and none was dropped.
    const auto pos1 = localVertexPos( mesh );
    for ( const auto& kv : pos0 )
    {
        auto it = pos1.find( kv.first );
        if ( it == pos1.end() || it->second != kv.second )
            ++local;
    }
    local += TesseraTest::owned1RingLocal( mesh );
    local += TesseraTest::checkOwnershipPartition( mesh, V0, E0, F0 );
    local += checkOrdering( mesh );

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case1/2 (no-op, idempotence) %s\n", tag,
                     glob == 0 ? "ok" : "FAIL" );
    return glob == 0 ? 0 : 1;
}

//! Cases 3, 4, 5: closed sub-mesh removal, the halo after it, and Euler.
template <class MeshT>
static int caseRemovePatch( int rank, int size, const char* tag,
                            const Reference& R )
{
    using mem = typename MeshT::memory_space;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    setup( mesh, halo );

    const std::set<GlobalId> deadF = patchFaces( 0, 1 );
    const DeadSet dead = closureOf( R, deadF );

    int local = 0;
    const auto pos0 = localVertexPos( mesh );
    const auto tri0 = gatherTriples( MPI_COMM_WORLD, size,
                                     ownedFaceTriples( mesh, local ) );

    applyTombstones( mesh, dead );
    const CompactStats st = compact( mesh, halo );

    const long long nv = static_cast<long long>( dead.v.size() );
    const long long ne = static_cast<long long>( dead.e.size() );
    const long long nf = static_cast<long long>( dead.f.size() );
    if ( st.verticesRemoved != nv || st.edgesRemoved != ne ||
         st.facesRemoved != nf )
        ++local;
    const long long V = TesseraTest::globalOwnedVertices( mesh );
    const long long E = TesseraTest::globalOwnedEdges( mesh );
    const long long F = TesseraTest::globalOwnedFaces( mesh );
    if ( V != V0 - nv || E != E0 - ne || F != F0 - nf )
        ++local;

    // Gid preservation: every surviving gid still labels the point it did, and
    // the surviving gid set is exactly the reference's.
    const auto pos1 = localVertexPos( mesh );
    for ( const auto& kv : pos1 )
    {
        auto it = pos0.find( kv.first );
        if ( it == pos0.end() || it->second != kv.second )
            ++local;
    }
    // The gid SPACE is untouched too, since the patch does not contain the
    // largest gid of any kind -- computed from the reference, not assumed.
    {
        long long mx[3] = { -1, -1, -1 };
        for ( std::size_t f = 0; f < R.fv.size(); ++f )
        {
            if ( deadF.count( static_cast<GlobalId>( f ) ) )
                continue;
            mx[2] = std::max( mx[2], static_cast<long long>( f ) );
            for ( int k = 0; k < 3; ++k )
            {
                mx[0] = std::max( mx[0],
                                  static_cast<long long>( R.fv[f][k] ) );
                mx[1] = std::max( mx[1],
                                  static_cast<long long>( R.fe[f][k] ) );
            }
        }
        if ( st.gidSpaceBefore != V0 + E0 + F0 )
            ++local;
        if ( st.gidSpaceAfter != ( mx[0] + 1 ) + ( mx[1] + 1 ) + ( mx[2] + 1 ) )
            ++local;
    }

    local += checkOrdering( mesh );
    local += checkNoneSurvive( mesh, dead );
    local += TesseraTest::owned1RingLocal( mesh );
    local += TesseraTest::checkOwnershipPartition( mesh, V, E, F );

    // The surviving faces' corner-position triples are the pre-edit set minus
    // the removed ones, checked both against the reference and against the
    // gathered pre-edit set.
    const auto tri1 = gatherTriples( MPI_COMM_WORLD, size,
                                     ownedFaceTriples( mesh, local ) );
    if ( rank == 0 )
    {
        std::vector<Triple> want = referenceTriples( R, deadF );
        std::sort( want.begin(), want.end() );
        if ( tri1 != want )
            ++local;
        if ( tri0.size() != want.size() + deadF.size() )
            ++local;
        std::vector<Triple> removed;
        std::set_difference( tri0.begin(), tri0.end(), tri1.begin(), tri1.end(),
                             std::back_inserter( removed ) );
        if ( removed.size() != deadF.size() )
            ++local;
    }

    // Case 4: the halo after the removal.
    local += ghostRoundTrip( mesh, halo, R );
    if ( size >= 2 && halo.vplan.totalRecv() == 0 && mesh.numOwnedFaces() > 0 )
        ++local; // non-vacuity: a rank with faces must ghost something

    // Case 5: Euler. A sphere minus k open discs has V - E + F == 2 - k.
    // ownedEulerGlobal() is COLLECTIVE, so it is evaluated here and only its
    // value is printed -- calling it inside the rank-0 print guard below
    // deadlocks every other rank.
    const int comps = boundaryComponents( R, deadF );
    const long long chi = TesseraTest::ownedEulerGlobal( mesh );
    if ( chi != 2 - comps )
        ++local;

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case3/4/5 (patch removal V%lld E%lld F%lld -> "
                     "%lld/%lld/%lld, %d boundary component(s), chi=%lld) %s\n",
                     tag, V0, E0, F0, V, E, F, comps, chi,
                     glob == 0 ? "ok" : "FAIL" );
    return glob == 0 ? 0 : 1;
}

//! Case 6: a non-closing tombstone set throws, and the family guard.
template <class MeshT>
static int caseDangling( int rank, const char* tag )
{
    using mem = typename MeshT::memory_space;
    int local = 0;

    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        setup( mesh, halo );

        // Kill owned vertex gid 0 -- on its owner, and nothing else. Its
        // incident faces are all still live, on this rank and on any peer.
        const auto vg = localGids( mesh.vertices(), mesh.numOwnedVertices() );
        for ( std::size_t i = 0; i < vg.size(); ++i )
            if ( vg[i] == 0 )
                tombstoneVertex( mesh, static_cast<LocalIndex>( i ) );

        bool threw = false, named = false;
        try
        {
            compact( mesh, halo );
        }
        catch ( const std::exception& e )
        {
            threw = true;
            const std::string m = e.what();
            named = m.find( "does not close" ) != std::string::npos &&
                    ( m.find( "dead vertex gid 0" ) != std::string::npos ||
                      m.find( "collective half" ) != std::string::npos );
        }
        if ( !threw || !named )
            ++local;
        // Nothing was mutated: the counts still include the tombstoned vertex.
        if ( TesseraTest::globalOwnedFaces( mesh ) != F0 )
            ++local;
        // At least one rank must have named the offending face outright.
        int here = 0;
        try
        {
            compact( mesh, halo );
        }
        catch ( const std::exception& e )
        {
            const std::string m = e.what();
            here = ( m.find( "dead vertex gid 0" ) != std::string::npos ) ? 1
                                                                         : 0;
        }
        if ( gsum( MPI_COMM_WORLD, here ) < 1 )
            ++local;
    }

    // Family guard, both directions.
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        setup( mesh, halo );
        std::vector<char> fmask( mesh.numOwnedFaces(), 1 );
        refine( mesh, halo, fmask );
        bool threw = false;
        try
        {
            compact( mesh, halo );
        }
        catch ( const std::exception& e )
        {
            const std::string m = e.what();
            threw = m.find( "Hierarchical" ) != std::string::npos &&
                    m.find( "Remesh" ) != std::string::npos;
        }
        if ( !threw )
            ++local;
    }
    {
        MeshT mesh( MPI_COMM_WORLD );
        MeshHalo<mem> halo;
        setup( mesh, halo );
        compact( mesh, halo ); // tags the mesh Remesh
        if ( mesh.editFamily() != EditFamily::Remesh )
            ++local;
        bool threw = false;
        try
        {
            std::vector<char> fmask( mesh.numOwnedFaces(), 1 );
            refine( mesh, halo, fmask );
        }
        catch ( const std::exception& e )
        {
            const std::string m = e.what();
            threw = m.find( "Hierarchical" ) != std::string::npos &&
                    m.find( "Remesh" ) != std::string::npos;
        }
        if ( !threw )
            ++local;
    }

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case6 (dangling throws, family guard) %s\n", tag,
                     glob == 0 ? "ok" : "FAIL" );
    return glob == 0 ? 0 : 1;
}

//! Case 7: every face of the last rank removed. Skipped at one rank.
template <class MeshT>
static int caseEmptyRank( int rank, int size, const char* tag,
                          const Reference& R )
{
    if ( size < 2 )
        return 0;
    using mem = typename MeshT::memory_space;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    setup( mesh, halo );

    // The face gids the last rank owns, made global (small: 320 faces).
    std::vector<std::vector<GlobalId>> send( size );
    if ( rank == size - 1 )
        for ( GlobalId g : localGids( mesh.faces(), mesh.numOwnedFaces() ) )
            for ( int r = 0; r < size; ++r )
                send[r].push_back( g );
    auto got = allToAllV( MPI_COMM_WORLD, send );
    const std::set<GlobalId> deadF( got.data.begin(), got.data.end() );

    int local = 0;
    if ( deadF.empty() )
        ++local; // the fixture is vacuous if the last rank owns nothing

    const DeadSet dead = closureOf( R, deadF );
    applyTombstones( mesh, dead );
    const CompactStats st = compact( mesh, halo );

    if ( st.facesRemoved != static_cast<long long>( deadF.size() ) )
        ++local;
    if ( rank == size - 1 )
    {
        if ( mesh.numOwnedVertices() != 0 || mesh.numOwnedEdges() != 0 ||
             mesh.numOwnedFaces() != 0 )
            ++local;
        if ( mesh.numVertices() != 0 || mesh.numEdges() != 0 ||
             mesh.numFaces() != 0 )
            ++local;
        if ( halo.vplan.totalSend() != 0 || halo.vplan.totalRecv() != 0 ||
             halo.eplan.totalSend() != 0 || halo.eplan.totalRecv() != 0 ||
             halo.fplan.totalSend() != 0 || halo.fplan.totalRecv() != 0 )
            ++local;
    }
    else
    {
        // No peer names the empty rank in any plan.
        for ( const auto* plan :
              { &halo.vplan, &halo.eplan, &halo.fplan } )
        {
            for ( int p : plan->send_peers )
                if ( p == size - 1 )
                    ++local;
            for ( int p : plan->recv_peers )
                if ( p == size - 1 )
                    ++local;
        }
    }

    const long long V = TesseraTest::globalOwnedVertices( mesh );
    const long long E = TesseraTest::globalOwnedEdges( mesh );
    const long long F = TesseraTest::globalOwnedFaces( mesh );
    if ( F != F0 - static_cast<long long>( deadF.size() ) )
        ++local;
    if ( TesseraTest::ownedEulerGlobal( mesh ) !=
         2 - boundaryComponents( R, deadF ) )
        ++local;
    local += TesseraTest::checkOwnershipPartition( mesh, V, E, F );
    local += TesseraTest::owned1RingLocal( mesh );
    local += checkOrdering( mesh );
    local += ghostRoundTrip( mesh, halo, R );

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case7 (rank %d emptied, %zu faces removed) %s\n",
                     tag, size - 1, deadF.size(), glob == 0 ? "ok" : "FAIL" );
    return glob == 0 ? 0 : 1;
}

//! Case 8: compact() advances the generation counter.
template <class MeshT>
static int caseGeneration( int rank, const char* tag )
{
    using mem = typename MeshT::memory_space;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    setup( mesh, halo );

    int local = 0;
    const std::size_t g0 = mesh.generation();
    compact( mesh, halo );
    if ( !( mesh.generation() > g0 ) )
        ++local;

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case8 (generation bump) %s\n", tag,
                     glob == 0 ? "ok" : "FAIL" );
    return glob == 0 ? 0 : 1;
}

//! Cases 9 and 10: renumbering, and its rank-count determinism.
template <class MeshT>
static int caseRenumber( int rank, int size, const char* tag,
                         const Reference& R )
{
    using mem = typename MeshT::memory_space;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    setup( mesh, halo );

    const std::set<GlobalId> deadF = patchFaces( 0, 1 );
    const DeadSet dead = closureOf( R, deadF );

    int local = 0;
    applyTombstones( mesh, dead );
    const CompactStats pre = compact( mesh, halo );
    (void)pre;
    const auto triBefore = gatherTriples( MPI_COMM_WORLD, size,
                                          ownedFaceTriples( mesh, local ) );

    const CompactStats st = compactAndRenumberGids( mesh, halo );

    const long long V = TesseraTest::globalOwnedVertices( mesh );
    const long long E = TesseraTest::globalOwnedEdges( mesh );
    const long long F = TesseraTest::globalOwnedFaces( mesh );
    if ( st.gidSpaceAfter != V + E + F )
        ++local;
    if ( !( st.gidSpaceAfter < st.gidSpaceBefore ) )
        ++local;

    // gids are [0, N) contiguous per kind, globally.
    auto contiguous = [&]( const std::vector<GlobalId>& owned, long long N )
    {
        std::vector<std::vector<GlobalId>> send( size );
        send[0] = owned;
        auto got = allToAllV( MPI_COMM_WORLD, send );
        if ( rank != 0 )
            return 0;
        std::vector<GlobalId> all( got.data.begin(), got.data.end() );
        std::sort( all.begin(), all.end() );
        if ( static_cast<long long>( all.size() ) != N )
            return 1;
        for ( long long i = 0; i < N; ++i )
            if ( all[i] != static_cast<GlobalId>( i ) )
                return 1;
        return 0;
    };
    local += contiguous( localGids( mesh.vertices(), mesh.numOwnedVertices() ),
                         V );
    local += contiguous( localGids( mesh.edges(), mesh.numOwnedEdges() ), E );
    local += contiguous( localGids( mesh.faces(), mesh.numOwnedFaces() ), F );

    // Connectivity resolves and the geometry is untouched by the relabelling.
    local += checkOrdering( mesh );
    local += TesseraTest::owned1RingLocal( mesh );
    local += TesseraTest::checkOwnershipPartition( mesh, V, E, F );
    const auto triAfter = gatherTriples( MPI_COMM_WORLD, size,
                                         ownedFaceTriples( mesh, local ) );
    if ( rank == 0 && triAfter != triBefore )
        ++local;

    // The halo is correct in the new numbering: corrupt every ghost, exchange,
    // and require the owners' values back (positions, not gids, so the
    // reference is still the ground truth once mapped through the new gids).
    {
        std::vector<GidPos> vs, fs;
        gidKeyedIdentity( mesh, vs, fs, local );
        std::map<GlobalId, std::array<double, 3>> want;
        // Owner values, gathered so every rank can check its own ghosts.
        std::vector<std::vector<GidPos>> send( size, vs );
        auto got = allToAllV( MPI_COMM_WORLD, send );
        for ( const auto& m : got.data )
            want[m.gid] = { m.p[0], m.p[1], m.p[2] };

        const std::size_t nv = mesh.numVertices();
        Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace>
            hv( "hv", nv );
        Cabana::deep_copy( hv, mesh.vertices() );
        auto p = Cabana::slice<VertexField::Position>( hv );
        for ( std::size_t i = mesh.numOwnedVertices(); i < nv; ++i )
            for ( int d = 0; d < 3; ++d )
                p( i, d ) = -12345.0;
        Cabana::deep_copy( mesh.vertices(), hv );
        haloExchange( mesh, halo );
        Cabana::deep_copy( hv, mesh.vertices() );
        auto g = Cabana::slice<VertexField::Gid>( hv );
        auto q = Cabana::slice<VertexField::Position>( hv );
        for ( std::size_t i = 0; i < nv; ++i )
        {
            auto it = want.find( g( i ) );
            if ( it == want.end() )
            {
                ++local;
                continue;
            }
            for ( int d = 0; d < 3; ++d )
                if ( static_cast<double>( q( i, d ) ) != it->second[d] )
                    ++local;
        }
    }

    // Case 10: the same edit alone on MPI_COMM_SELF gives the same global
    // gid -> position map. Nothing here is comparable across rank counts unless
    // the renumbering is an order statistic rather than an exscan block.
    {
        std::vector<GidPos> vs, fs;
        gidKeyedIdentity( mesh, vs, fs, local );
        const auto vAll = gatherGidPos( MPI_COMM_WORLD, size, vs );
        const auto fAll = gatherGidPos( MPI_COMM_WORLD, size, fs );

        if ( rank == 0 )
        {
            MeshT solo( MPI_COMM_SELF );
            MeshHalo<mem> soloHalo;
            buildIcosphere( solo, 2 );
            auto owner = facePartitionByAxis( solo );
            distribute( solo, soloHalo, owner );
            applyTombstones( solo, dead );
            compact( solo, soloHalo );
            compactAndRenumberGids( solo, soloHalo );

            std::vector<GidPos> sv, sf;
            gidKeyedIdentity( solo, sv, sf, local );
            std::sort( sv.begin(), sv.end(), []( const GidPos& a,
                                                 const GidPos& b )
                       { return a.gid < b.gid; } );
            std::sort( sf.begin(), sf.end(), []( const GidPos& a,
                                                 const GidPos& b )
                       { return a.gid < b.gid; } );
            auto same = []( const std::vector<GidPos>& a,
                            const std::vector<GidPos>& b )
            {
                if ( a.size() != b.size() )
                    return false;
                for ( std::size_t i = 0; i < a.size(); ++i )
                {
                    if ( a[i].gid != b[i].gid )
                        return false;
                    for ( int d = 0; d < 9; ++d )
                        if ( a[i].p[d] != b[i].p[d] )
                            return false;
                }
                return true;
            };
            if ( !same( vAll, sv ) || !same( fAll, sf ) )
                ++local;
        }
    }

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case9/10 (renumber: gid space %lld -> %lld, live "
                     "%lld) %s\n",
                     tag, st.gidSpaceBefore, st.gidSpaceAfter, V + E + F,
                     glob == 0 ? "ok" : "FAIL" );
    return glob == 0 ? 0 : 1;
}

//! Case 11: ten rounds of remove-and-compact do not shrink the gid space, then
//! one renumbering returns it to the live count.
template <class MeshT>
static int caseGidSpaceLeak( int rank, const char* tag, const Reference& R )
{
    using mem = typename MeshT::memory_space;
    MeshT mesh( MPI_COMM_WORLD );
    MeshHalo<mem> halo;
    setup( mesh, halo );

    int local = 0;
    long long prevSpace = -1;
    long long space = 0, live = 0;
    for ( int round = 0; round < 10; ++round )
    {
        const std::set<GlobalId> deadF = patchFaces( 0, round + 1 );
        const DeadSet dead = closureOf( R, deadF );
        applyTombstones( mesh, dead );
        const CompactStats st = compact( mesh, halo );

        if ( prevSpace >= 0 && st.gidSpaceBefore > prevSpace )
            ++local; // gids are preserved, so the space can never grow here
        prevSpace = st.gidSpaceBefore;

        space = st.gidSpaceAfter;
        live = TesseraTest::globalOwnedVertices( mesh ) +
               TesseraTest::globalOwnedEdges( mesh ) +
               TesseraTest::globalOwnedFaces( mesh );
        local += TesseraTest::owned1RingLocal( mesh );
        local += checkOrdering( mesh );
        if ( rank == 0 )
            std::printf( "      round %2d: gid space %4lld, live %4lld, "
                         "wasted %4lld\n",
                         round + 1, space, live, space - live );
    }
    // The leak: the space is still pinned near its initial value while a third
    // of the mesh is gone. This is what grows without bound under a
    // split/collapse workload, and what the dense gid -> local vectors in
    // buildKindPlan()/make_g2l()/build_order() are sized by.
    if ( !( space > live ) )
        ++local;

    const CompactStats st = compactAndRenumberGids( mesh, halo );
    const long long liveAfter = TesseraTest::globalOwnedVertices( mesh ) +
                                TesseraTest::globalOwnedEdges( mesh ) +
                                TesseraTest::globalOwnedFaces( mesh );
    if ( st.gidSpaceAfter != liveAfter )
        ++local;
    if ( liveAfter != live )
        ++local; // the renumbering must not remove anything
    local += TesseraTest::owned1RingLocal( mesh );
    local += checkOrdering( mesh );

    const int glob = gsum( MPI_COMM_WORLD, local );
    if ( rank == 0 )
        std::printf( "  [%s] case11 (leak: %lld wasted over 10 rounds -> 0 "
                     "after renumber, gid space %lld == live %lld) %s\n",
                     tag, space - live, st.gidSpaceAfter, liveAfter,
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

    const Reference R = buildReference<MeshT>();

    int fails = 0;
    fails += caseNoOp<MeshT>( rank, tag );
    fails += caseRemovePatch<MeshT>( rank, size, tag, R );
    fails += caseDangling<MeshT>( rank, tag );
    fails += caseEmptyRank<MeshT>( rank, size, tag, R );
    fails += caseGeneration<MeshT>( rank, tag );
    fails += caseRenumber<MeshT>( rank, size, tag, R );
    fails += caseGidSpaceLeak<MeshT>( rank, tag, R );
    std::fflush( stdout );
    return fails;
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    // Unbuffered: every case prints as it completes, so a hang localizes to
    // the case that did NOT print rather than being swallowed by the buffer.
    std::setvbuf( stdout, nullptr, _IONBF, 0 );

    int fails = 0;
    {
        int rank = 0, size = 1;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        MPI_Comm_size( MPI_COMM_WORLD, &size );
        if ( rank == 0 )
            std::printf( "test_compact: tombstone removal and gid renumbering "
                         "(size %d)\n",
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
