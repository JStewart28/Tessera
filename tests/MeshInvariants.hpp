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

#ifndef TESSERA_TEST_MESH_INVARIANTS_HPP
#define TESSERA_TEST_MESH_INVARIANTS_HPP

// Shared distributed-mesh invariant checks, reused by the Step 5/6/7/8 regression
// tests. Every helper returns a LOCAL failure count whose SUM across ranks is the
// true global failure count (collective helpers place each check on exactly one
// rank), so a caller can MPI_Allreduce(SUM) the total and treat non-zero as fail.

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <map>
#include <set>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace TesseraTest
{

using Tessera::GlobalId;
using Tessera::LocalIndex;
using Tessera::Rank;

// Owned gids [begin,end) of an AoSoA (gid is member 0 for every entity kind).
template <class AoSoAType>
std::vector<GlobalId> ownedGids( const AoSoAType& a, std::size_t begin,
                                 std::size_t end )
{
    Cabana::AoSoA<typename AoSoAType::member_types, Kokkos::HostSpace> h(
        "h", a.size() );
    Cabana::deep_copy( h, a );
    auto g = Cabana::slice<0>( h );
    std::vector<GlobalId> out;
    out.reserve( end - begin );
    for ( std::size_t i = begin; i < end; ++i )
        out.push_back( g( i ) );
    return out;
}

// One entity kind's partition check: (1) total owned across ranks == Nglobal
// (checked on rank 0 only), (2) no gid owned by two ranks (each gid routed to a
// coordinator rank = gid % size, which flags duplicates). Returns LOCAL fails.
inline int checkKindPartition( MPI_Comm comm, int rank, int comm_size,
                               const std::vector<GlobalId>& owned,
                               long long Nglobal )
{
    long long local = static_cast<long long>( owned.size() );
    long long total = 0;
    MPI_Allreduce( &local, &total, 1, MPI_LONG_LONG, MPI_SUM, comm );

    int fails = 0;
    if ( rank == 0 && total != Nglobal )
        ++fails;

    std::vector<std::vector<GlobalId>> send( comm_size );
    for ( GlobalId g : owned )
        send[g % comm_size].push_back( g );
    auto res = Tessera::allToAllV( comm, send );
    std::set<GlobalId> seen;
    for ( GlobalId g : res.data )
        if ( !seen.insert( g ).second )
            ++fails; // this gid was owned by more than one rank
    return fails;
}

// Ownership is a partition of every entity kind. LOCAL fails (sum == global).
template <class MeshT>
int checkOwnershipPartition( MeshT& mesh, long long NvG, long long NeG,
                             long long NfG )
{
    MPI_Comm comm = mesh.comm();
    const int rank = mesh.rank();
    const int size = mesh.commSize();
    int fails = 0;
    fails += checkKindPartition(
        comm, rank, size,
        ownedGids( mesh.vertices(), 0, mesh.numOwnedVertices() ), NvG );
    fails += checkKindPartition(
        comm, rank, size, ownedGids( mesh.edges(), 0, mesh.numOwnedEdges() ),
        NeG );
    fails += checkKindPartition(
        comm, rank, size, ownedGids( mesh.faces(), 0, mesh.numOwnedFaces() ),
        NfG );
    return fails;
}

// Every owned vertex holds its full 1-ring locally: each incident face contains
// the vertex, and that face's three vertices and three edges are all present
// locally. LOCAL fails (no communication; sum across ranks == global).
template <class MeshT>
int owned1RingLocal( MeshT& mesh )
{
    const std::size_t nv = mesh.numVertices();
    const std::size_t nf = mesh.numFaces();

    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", nv );
    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
        "he", mesh.numEdges() );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", nf );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( he, mesh.edges() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto vgid = Cabana::slice<Tessera::VertexField::Gid>( hv );
    auto egid = Cabana::slice<Tessera::EdgeField::Gid>( he );
    auto fverts = Cabana::slice<Tessera::FaceField::Verts>( hf );
    auto fedges = Cabana::slice<Tessera::FaceField::Edges>( hf );

    std::unordered_set<GlobalId> vset, eset;
    for ( std::size_t i = 0; i < nv; ++i )
        vset.insert( vgid( i ) );
    for ( std::size_t i = 0; i < mesh.numEdges(); ++i )
        eset.insert( egid( i ) );

    auto off = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), mesh.vertexFaces().offsets );
    auto nbr = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), mesh.vertexFaces().neighbors );

    int fails = 0;
    for ( std::size_t v = 0; v < mesh.numOwnedVertices(); ++v )
    {
        const GlobalId vg = vgid( v );
        for ( int p = off( v ); p < off( v + 1 ); ++p )
        {
            const LocalIndex lf = nbr( p );
            bool has = false;
            for ( int k = 0; k < 3; ++k )
            {
                if ( fverts( lf, k ) == vg )
                    has = true;
                if ( vset.find( fverts( lf, k ) ) == vset.end() )
                    ++fails; // adjacent vertex not held locally
                if ( eset.find( fedges( lf, k ) ) == eset.end() )
                    ++fails; // incident edge not held locally
            }
            if ( !has )
                ++fails; // CSR listed a face not incident to this vertex
        }
    }
    return fails;
}

// Rank-count-independent topology checksum: XOR of owned gids per kind, reduced
// with MPI_BXOR so every rank gets the same value. Two distributions of the same
// global mesh (e.g. at different rank counts, or before/after I/O) agree iff their
// checksums agree. Used by Steps 6b/7/8.
template <class MeshT>
void topologyChecksum( MeshT& mesh, unsigned long long& cv,
                       unsigned long long& ce, unsigned long long& cf )
{
    auto xorKind = [&]( const std::vector<GlobalId>& owned )
    {
        unsigned long long local = 0;
        for ( GlobalId g : owned )
            local ^= static_cast<unsigned long long>( g );
        unsigned long long global = 0;
        MPI_Allreduce( &local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR,
                       mesh.comm() );
        return global;
    };
    cv = xorKind( ownedGids( mesh.vertices(), 0, mesh.numOwnedVertices() ) );
    ce = xorKind( ownedGids( mesh.edges(), 0, mesh.numOwnedEdges() ) );
    cf = xorKind( ownedGids( mesh.faces(), 0, mesh.numOwnedFaces() ) );
}

// ---------------------------------------------------------------------------
// Step 6b (distributed refinement) invariants
// ---------------------------------------------------------------------------

// The owned-count reductions are the library's (Tessera_Reduction.hpp) — these
// are name-preserving forwards so the existing call sites do not churn. Nothing
// here hand-rolls MPI_Allreduce for them any more.

// Global owned-only Euler number Σ_ranks( ownedV - ownedE + ownedF ). For a
// conforming closed genus-0 surface (e.g. after a UNIFORM refine) this is 2;
// adaptive refinement introduces bounded hanging nodes and does not preserve it.
template <class MeshT>
long long ownedEulerGlobal( MeshT& mesh )
{
    return Tessera::globalOwnedEuler( mesh );
}

// Sum of an owned count across ranks (owned entities partition the global mesh).
template <class MeshT>
long long globalOwnedVertices( MeshT& mesh )
{
    return Tessera::globalOwnedVertices( mesh );
}
template <class MeshT>
long long globalOwnedEdges( MeshT& mesh )
{
    return Tessera::globalOwnedEdges( mesh );
}
template <class MeshT>
long long globalOwnedFaces( MeshT& mesh )
{
    return Tessera::globalOwnedFaces( mesh );
}

// 2:1 balance over an arbitrary face layer, given each face's corner gids and
// level. Each face advertises (edge, level) to the edge's coordinator, which
// compares the two incidences. Edges with a single incidence are SKIPPED: on a
// hanging-node layer a bisected edge survives only on its kept side, which is
// precisely the bounded non-conformity the 2:1 rule allows.
// Returns LOCAL fails (sum across ranks == global).
inline int
check21BalanceOn( MPI_Comm comm, int size,
                  const std::vector<std::array<GlobalId, 3>>& faceVerts,
                  const std::vector<Tessera::Level>& faceLevel )
{
    struct LvMsg
    {
        Tessera::EdgeKey key;
        Tessera::Level level;
    };
    std::vector<std::vector<LvMsg>> send( size );
    for ( std::size_t f = 0; f < faceVerts.size(); ++f )
        for ( int k = 0; k < 3; ++k )
        {
            const Tessera::EdgeKey key = Tessera::makeEdgeKey(
                faceVerts[f][k], faceVerts[f][( k + 1 ) % 3] );
            send[Tessera::detail::edgeCoordRank( key, size )].push_back(
                { key, faceLevel[f] } );
        }
    auto got = Tessera::allToAllV( comm, send );

    std::map<Tessera::EdgeKey, std::vector<Tessera::Level>> byEdge;
    for ( const auto& m : got.data )
        byEdge[m.key].push_back( m.level );

    int fails = 0;
    for ( const auto& kv : byEdge )
        if ( kv.second.size() == 2 )
        {
            const int d = static_cast<int>( kv.second[0] ) -
                          static_cast<int>( kv.second[1] );
            if ( d > 1 || d < -1 )
                ++fails;
        }
    return fails;
}

// Owned corner gids + level of every OWNED face of the VISIBLE layer.
template <class MeshT>
void ownedVisibleFaceLevels( MeshT& mesh,
                             std::vector<std::array<GlobalId, 3>>& verts,
                             std::vector<Tessera::Level>& level )
{
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto fv = Cabana::slice<Tessera::FaceField::Verts>( hf );
    auto fl = Cabana::slice<Tessera::FaceField::Level>( hf );
    const std::size_t nof = mesh.numOwnedFaces();
    verts.resize( nof );
    level.resize( nof );
    for ( std::size_t f = 0; f < nof; ++f )
    {
        for ( int k = 0; k < 3; ++k )
            verts[f][k] = fv( f, k );
        level[f] = fl( f );
    }
}

// 2:1 balance of the VISIBLE face layer. LOCAL fails (sum == global).
template <class MeshT>
int check21Balance( MeshT& mesh )
{
    std::vector<std::array<GlobalId, 3>> verts;
    std::vector<Tessera::Level> level;
    ownedVisibleFaceLevels( mesh, verts, level );
    return check21BalanceOn( mesh.comm(), mesh.commSize(), verts, level );
}

// Cross-rank midpoint-gid agreement (the key Step-6b guarantee): every rank that
// creates a midpoint for a shared edge must use the same gid. Each (edge, gid)
// pair is routed to the edge's coordinator, which flags any edge seen with two
// distinct gids. Returns LOCAL fails (sum == global).
inline int checkMidpointAgreement(
    MPI_Comm comm, int size,
    const std::vector<std::pair<Tessera::EdgeKey, Tessera::GlobalId>>& mids )
{
    struct KG
    {
        Tessera::EdgeKey key;
        Tessera::GlobalId gid;
    };
    std::vector<std::vector<KG>> send( size );
    for ( const auto& kg : mids )
        send[Tessera::detail::edgeCoordRank( kg.first, size )].push_back(
            { kg.first, kg.second } );
    auto got = Tessera::allToAllV( comm, send );

    std::map<Tessera::EdgeKey, Tessera::GlobalId> seen;
    int fails = 0;
    for ( const auto& m : got.data )
    {
        auto it = seen.find( m.key );
        if ( it == seen.end() )
            seen.emplace( m.key, m.gid );
        else if ( it->second != m.gid )
            ++fails; // same edge, different midpoint gid across ranks
    }
    return fails;
}

// The three EdgeKeys of every face in `verts`, deduplicated.
inline std::set<Tessera::EdgeKey>
edgeSetOf( const std::vector<std::array<GlobalId, 3>>& verts )
{
    std::set<Tessera::EdgeKey> out;
    for ( const auto& t : verts )
        for ( int k = 0; k < 3; ++k )
            out.insert( Tessera::makeEdgeKey( t[k], t[( k + 1 ) % 3] ) );
    return out;
}

// Globally-decided ground truth for a split-edge map (RefineResult::midpoints or
// SplitResult::midpoints), needing no replica of the refinement algorithm. Every
// reported key is routed to its edge coordinator, which therefore knows the true
// global split-edge set; each rank then asks the coordinator about every edge of
// its pre-edit owned faces and checks presence-in-my-map == is-globally-split.
// Two failure modes are caught:
//   COMPLETENESS  an edge of one of my faces is split somewhere but absent here
//                 (the pre-Task-3 refine() bug: the kept side never heard);
//   SOUNDNESS     I report a key that is not an edge of any face I own.
// Returns LOCAL fails (sum across ranks == global).
inline int checkSplitEdgeCoverage(
    MPI_Comm comm, int size,
    const std::vector<std::array<GlobalId, 3>>& preOwnedFaceVerts,
    const std::vector<std::pair<Tessera::EdgeKey, GlobalId>>& mids )
{
    struct KeyMsg
    {
        Tessera::EdgeKey key;
    };
    struct SplitMsg
    {
        Tessera::EdgeKey key;
        unsigned char split;
    };

    std::set<Tessera::EdgeKey> mine;
    for ( const auto& kv : mids )
        mine.insert( kv.first );
    const std::set<Tessera::EdgeKey> myEdges = edgeSetOf( preOwnedFaceVerts );

    int fails = 0;
    for ( const Tessera::EdgeKey& k : mine )
        if ( myEdges.find( k ) == myEdges.end() )
            ++fails; // reported an edge this rank does not touch

    // Advertise every reported key -> coordinator learns the global split set.
    std::vector<std::vector<KeyMsg>> adv( size );
    for ( const Tessera::EdgeKey& k : mine )
        adv[Tessera::detail::edgeCoordRank( k, size )].push_back( { k } );
    auto advGot = Tessera::allToAllV( comm, adv );
    std::set<Tessera::EdgeKey> globalSplit;
    for ( const auto& m : advGot.data )
        globalSplit.insert( m.key );

    // Ask the coordinator about every edge of my pre-edit owned faces.
    std::vector<std::vector<KeyMsg>> req( size );
    for ( const Tessera::EdgeKey& k : myEdges )
        req[Tessera::detail::edgeCoordRank( k, size )].push_back( { k } );
    auto reqGot = Tessera::allToAllV( comm, req );

    std::vector<std::vector<SplitMsg>> reply( size );
    for ( int s = 0; s < size; ++s )
    {
        const KeyMsg* p = reqGot.from( s );
        const int cnt = reqGot.count( s );
        for ( int i = 0; i < cnt; ++i )
            reply[s].push_back(
                { p[i].key, static_cast<unsigned char>(
                                globalSplit.count( p[i].key ) ? 1 : 0 ) } );
    }
    auto replyGot = Tessera::allToAllV( comm, reply );

    for ( const auto& m : replyGot.data )
    {
        const bool have = mine.find( m.key ) != mine.end();
        if ( have != ( m.split != 0 ) )
            ++fails; // incomplete (split but missing) or spurious
    }
    return fails;
}

//! The canonical-key side tables agree with the AoSoA connectivity entry for
//! entry over every LOCAL entity, and no two OWNED edges (or faces) share a key
//! globally. The duplicate test is routed through the same coordinators
//! flipEdges() uses, so it sees the whole global key set.
//!
//! SHARED, not duplicated, for the same reason edgeSetOf() and
//! checkSplitEdgeCoverage() are: test_distribute asserts it on the mesh
//! distribute() just produced and test_flip_edges asserts it as its check 8
//! after a flip pass, and the two only mean the same thing if they are the same
//! function.
//!
//! `breakdown`, if given, receives the four partial counts in order: local edge
//! table, local face table, duplicate owned edge key, duplicate owned face key.
//! Returns LOCAL fails (sum across ranks == global).
template <class MeshT>
int checkKeyTables( MeshT& mesh, int* breakdown = nullptr )
{
    int fails = 0;
    int part[4] = { 0, 0, 0, 0 };
    const int size = mesh.commSize();

    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
        "he", mesh.numEdges() );
    Cabana::deep_copy( he, mesh.edges() );
    auto ev = Cabana::slice<Tessera::EdgeField::Verts>( he );
    auto ek = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                   mesh.edgeKeys() );
    if ( ek.extent( 0 ) != mesh.numEdges() )
        ++part[0];
    else
        for ( std::size_t e = 0; e < mesh.numEdges(); ++e )
            if ( !( ek( e ) ==
                    Tessera::makeEdgeKey( ev( e, 0 ), ev( e, 1 ) ) ) )
                ++part[0];

    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto fv = Cabana::slice<Tessera::FaceField::Verts>( hf );
    auto fk = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                   mesh.faceKeys() );
    if ( fk.extent( 0 ) != mesh.numFaces() )
        ++part[1];
    else
        for ( std::size_t f = 0; f < mesh.numFaces(); ++f )
            if ( !( fk( f ) == Tessera::makeFaceKey( fv( f, 0 ), fv( f, 1 ),
                                                     fv( f, 2 ) ) ) )
                ++part[1];

    // Global duplicate test over the OWNED edge keys.
    struct KeyMsg
    {
        Tessera::EdgeKey key;
    };
    std::vector<std::vector<KeyMsg>> adv( size );
    for ( std::size_t e = 0; e < mesh.numOwnedEdges(); ++e )
    {
        const Tessera::EdgeKey k =
            Tessera::makeEdgeKey( ev( e, 0 ), ev( e, 1 ) );
        adv[Tessera::detail::edgeCoordRank( k, size )].push_back( { k } );
    }
    auto got = Tessera::allToAllV( mesh.comm(), adv );
    std::map<Tessera::EdgeKey, int> seen;
    for ( const auto& m : got.data )
        ++seen[m.key];
    for ( const auto& kv : seen )
        if ( kv.second > 1 )
            ++part[2]; // two owned edges with the same endpoints

    // ... and over the OWNED face keys, routed on the face key's first id.
    struct FKeyMsg
    {
        Tessera::FaceKey key;
    };
    std::vector<std::vector<FKeyMsg>> fadv( size );
    for ( std::size_t f = 0; f < mesh.numOwnedFaces(); ++f )
    {
        const Tessera::FaceKey k =
            Tessera::makeFaceKey( fv( f, 0 ), fv( f, 1 ), fv( f, 2 ) );
        fadv[k.id[0] % static_cast<GlobalId>( size )].push_back( { k } );
    }
    auto fgot = Tessera::allToAllV( mesh.comm(), fadv );
    std::map<Tessera::FaceKey, int> fseen;
    for ( const auto& m : fgot.data )
        ++fseen[m.key];
    for ( const auto& kv : fseen )
        if ( kv.second > 1 )
            ++part[3];

    for ( int i = 0; i < 4; ++i )
    {
        fails += part[i];
        if ( breakdown )
            breakdown[i] = part[i];
    }
    return fails;
}

// ---------------------------------------------------------------------------
// Triangle shape statistics
// ---------------------------------------------------------------------------

//! Global minimum inradius/circumradius over the owned faces. 0.5 exactly for an
//! equilateral triangle, 0 for a degenerate one. Reduced over mesh.comm().
//! `minAngleDeg`, also reduced, is the global smallest triangle angle -- the
//! second statistic because r/R and the min angle degrade for different reasons
//! and a needle triangle can be caught by one before the other.
//!
//! SHARED, not duplicated: test_split_edges asserts a measured floor on it over
//! a length-driven split sequence and test_flip_edges reports it before and
//! after a flip pass, and the two numbers are only comparable if they are the
//! same function. Lives here for the same reason edgeSetOf() and
//! checkSplitEdgeCoverage() do.
//!
//! `fails` is incremented once per owned face whose corners are not all held
//! locally, or which is degenerate; such a face contributes nothing.
template <class MeshT>
double minRadiusRatio( MeshT& mesh, int& fails, double& minAngleDeg )
{
    // gid -> position over every locally held vertex, and the owned faces'
    // corner gids: both read straight from the AoSoAs so the helper depends on
    // nothing but the mesh.
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto vg = Cabana::slice<Tessera::VertexField::Gid>( hv );
    auto vp = Cabana::slice<Tessera::VertexField::Position>( hv );
    std::map<GlobalId, std::array<double, 3>> pos;
    for ( std::size_t i = 0; i < mesh.numVertices(); ++i )
    {
        std::array<double, 3> p = { 0, 0, 0 };
        for ( int d = 0; d < MeshT::dim && d < 3; ++d )
            p[d] = static_cast<double>( vp( i, d ) );
        pos[vg( i )] = p;
    }

    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto fv = Cabana::slice<Tessera::FaceField::Verts>( hf );

    double worst = 1.0, worstAngle = 180.0;
    for ( std::size_t f = 0; f < mesh.numOwnedFaces(); ++f )
    {
        std::array<double, 3> p[3];
        bool ok = true;
        for ( int k = 0; k < 3; ++k )
        {
            auto it = pos.find( fv( f, k ) );
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
        double side[3] = { 0, 0, 0 };
        for ( int k = 0; k < 3; ++k )
        {
            double s = 0.0;
            for ( int d = 0; d < 3; ++d )
            {
                const double dd = p[( k + 1 ) % 3][d] - p[k][d];
                s += dd * dd;
            }
            side[k] = std::sqrt( s );
        }
        // Area from the cross product of two edge vectors.
        double u[3], v[3];
        for ( int d = 0; d < 3; ++d )
        {
            u[d] = p[1][d] - p[0][d];
            v[d] = p[2][d] - p[0][d];
        }
        const double cx = u[1] * v[2] - u[2] * v[1];
        const double cy = u[2] * v[0] - u[0] * v[2];
        const double cz = u[0] * v[1] - u[1] * v[0];
        const double area = 0.5 * std::sqrt( cx * cx + cy * cy + cz * cz );
        const double abc = side[0] * side[1] * side[2];
        if ( abc <= 0.0 )
        {
            ++fails;
            continue;
        }
        const double s = 0.5 * ( side[0] + side[1] + side[2] );
        // r/R = (area/s) / (abc/(4 area)) = 4 area^2 / (s abc)
        worst = std::min( worst, 4.0 * area * area / ( s * abc ) );

        // Law of cosines. side[k] runs corner k -> k+1, so the angle at corner
        // k+1 is between side[k] and side[k+1], opposite side[k+2].
        for ( int k = 0; k < 3; ++k )
        {
            const double a = side[k], b = side[( k + 1 ) % 3],
                         c = side[( k + 2 ) % 3];
            double cosA = ( a * a + b * b - c * c ) / ( 2.0 * a * b );
            cosA = std::max( -1.0, std::min( 1.0, cosA ) );
            worstAngle = std::min(
                worstAngle, std::acos( cosA ) * 180.0 / 3.14159265358979323846 );
        }
    }
    double global = worst;
    MPI_Allreduce( &worst, &global, 1, MPI_DOUBLE, MPI_MIN, mesh.comm() );
    minAngleDeg = worstAngle;
    MPI_Allreduce( &worstAngle, &minAngleDeg, 1, MPI_DOUBLE, MPI_MIN,
                   mesh.comm() );
    return global;
}

// ---------------------------------------------------------------------------
// Conforming-refinement invariants (tasks/completed/conforming-refinement.md, Task 4)
// ---------------------------------------------------------------------------
//
// These are the acceptance criteria for RefinementMode::Conforming. Three of
// them (checkConforming, checkOwnedEuler, checkNoInteriorVertex) are properties
// of ANY conforming surface and are deliberately mode-agnostic, so a test can
// run them on a HangingNode2to1 result too and assert that they FAIL there --
// otherwise a mask too weak to create a hanging node would prove nothing.

// The VISIBLE owned faces of a Conforming mesh as plain VisibleFace structs
// (corners, gid, level, and the ClosureParent / ClosureParentVerts bookkeeping).
template <class MeshT>
std::vector<Tessera::VisibleFace> ownedVisibleFaces( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    return Tessera::readVisibleFaces<MeshT>( hf, mesh.numOwnedFaces() );
}

// THE conformity criterion: every edge of the GLOBAL mesh has exactly two
// incident faces. A hanging node on edge (a,b) means the refined side carries
// (a,m) and (m,b) instead, leaving (a,b) with a single incidence -- so this
// single check subsumes the topological half of "no T-junctions".
//
// Verified through the edge coordinator (edgeCoordRank + allToAllV) rather than
// the halo, so the verdict is rank-count independent and needs no ghost layer
// (refine() leaves an owned-only mesh). Every rank advertises each edge of each
// of its OWNED faces exactly once; a face is owned by exactly one rank, so the
// coordinator's per-edge count is the true global incidence count.
// Returns LOCAL fails (sum across ranks == global).
template <class MeshT>
int checkConforming( MeshT& mesh )
{
    MPI_Comm comm = mesh.comm();
    const int size = mesh.commSize();

    std::vector<std::array<GlobalId, 3>> verts;
    std::vector<Tessera::Level> level;
    ownedVisibleFaceLevels( mesh, verts, level );

    struct IncMsg
    {
        Tessera::EdgeKey key;
    };
    std::vector<std::vector<IncMsg>> send( size );
    for ( const auto& t : verts )
        for ( int k = 0; k < 3; ++k )
        {
            const Tessera::EdgeKey key =
                Tessera::makeEdgeKey( t[k], t[( k + 1 ) % 3] );
            send[Tessera::detail::edgeCoordRank( key, size )].push_back(
                { key } );
        }
    auto got = Tessera::allToAllV( comm, send );

    std::map<Tessera::EdgeKey, int> incidence;
    for ( const auto& m : got.data )
        ++incidence[m.key];

    int fails = 0;
    for ( const auto& kv : incidence )
        if ( kv.second != 2 )
            ++fails;
    return fails;
}

// Owned-only Euler number V - E + F, summed over ranks. THE headline acceptance
// criterion for conforming refinement: it is 2 for a closed genus-0 surface
// under an ARBITRARY (adaptive) mask, which is exactly what the hanging-node
// mode fails -- there it holds only for a uniform refine. Named separately from
// ownedEulerGlobal() because it is the criterion, not just a statistic; the
// value is identical.
template <class MeshT>
long long checkOwnedEuler( MeshT& mesh )
{
    return ownedEulerGlobal( mesh );
}

// GEOMETRIC "no T-junction": no vertex lies strictly inside any edge. The purely
// topological reading -- "some m has edges (u,m) and (m,w)" -- is true of every
// ordinary triangle, so the test must check that m is COLLINEAR with (u,w) AND
// strictly between its endpoints.
//
// Unlike the other checks this one cannot be routed through an edge coordinator:
// it needs vertex POSITIONS, and after refine() a face may name a vertex no rank
// but its owner holds (always true across a partition boundary, and in
// Conforming mode also true of a closure child's midpoint corner). So the global
// owned vertices and owned faces are replicated on rank 0 and the exact test is
// run there. That is affordable and, more importantly, rank-count independent by
// construction -- the same style of partition-free reference the rest of this
// suite uses. Returns LOCAL fails (0 off root; sum across ranks == global).
template <class MeshT>
int checkNoInteriorVertex( MeshT& mesh )
{
    MPI_Comm comm = mesh.comm();
    const int rank = mesh.rank();
    const int size = mesh.commSize();

    struct VtxMsg
    {
        GlobalId gid;
        double p[3];
    };
    struct FaceMsg
    {
        GlobalId v[3];
    };

    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto vg = Cabana::slice<Tessera::VertexField::Gid>( hv );
    auto vp = Cabana::slice<Tessera::VertexField::Position>( hv );

    std::vector<std::vector<VtxMsg>> vsend( size );
    for ( std::size_t i = 0; i < mesh.numOwnedVertices(); ++i )
    {
        VtxMsg m;
        m.gid = vg( i );
        for ( int d = 0; d < 3; ++d )
            m.p[d] = ( d < MeshT::dim ) ? static_cast<double>( vp( i, d ) ) : 0;
        vsend[0].push_back( m );
    }
    auto vgot = Tessera::allToAllV( comm, vsend );

    std::vector<std::array<GlobalId, 3>> verts;
    std::vector<Tessera::Level> level;
    ownedVisibleFaceLevels( mesh, verts, level );
    std::vector<std::vector<FaceMsg>> fsend( size );
    for ( const auto& t : verts )
        fsend[0].push_back( { { t[0], t[1], t[2] } } );
    auto fgot = Tessera::allToAllV( comm, fsend );

    if ( rank != 0 )
        return 0;

    std::map<GlobalId, std::array<double, 3>> pos;
    for ( const auto& m : vgot.data )
        pos[m.gid] = { m.p[0], m.p[1], m.p[2] };

    int fails = 0;
    std::set<Tessera::EdgeKey> edges;
    for ( const auto& f : fgot.data )
        for ( int k = 0; k < 3; ++k )
        {
            edges.insert( Tessera::makeEdgeKey( f.v[k], f.v[( k + 1 ) % 3] ) );
            if ( pos.find( f.v[k] ) == pos.end() )
                ++fails; // a face names a vertex NO rank owns
        }
    if ( fails )
        return fails;

    std::map<GlobalId, std::vector<GlobalId>> nbr;
    for ( const auto& k : edges )
    {
        nbr[k.id[0]].push_back( k.id[1] );
        nbr[k.id[1]].push_back( k.id[0] );
    }
    for ( auto& kv : nbr )
        std::sort( kv.second.begin(), kv.second.end() );

    for ( const auto& k : edges )
    {
        const std::array<double, 3>& pu = pos.at( k.id[0] );
        const std::array<double, 3>& pw = pos.at( k.id[1] );
        double d[3], len2 = 0.0;
        for ( int c = 0; c < 3; ++c )
        {
            d[c] = pw[c] - pu[c];
            len2 += d[c] * d[c];
        }
        std::vector<GlobalId> common;
        const auto& nu = nbr[k.id[0]];
        const auto& nw = nbr[k.id[1]];
        std::set_intersection( nu.begin(), nu.end(), nw.begin(), nw.end(),
                               std::back_inserter( common ) );
        for ( GlobalId m : common )
        {
            const std::array<double, 3>& pm = pos.at( m );
            double e[3], proj = 0.0;
            for ( int c = 0; c < 3; ++c )
            {
                e[c] = pm[c] - pu[c];
                proj += e[c] * d[c];
            }
            const double cx = e[1] * d[2] - e[2] * d[1];
            const double cy = e[2] * d[0] - e[0] * d[2];
            const double cz = e[0] * d[1] - e[1] * d[0];
            if ( cx * cx + cy * cy + cz * cz <= 1e-20 * len2 * len2 &&
                 proj > 1e-12 * len2 && proj < ( 1.0 - 1e-12 ) * len2 )
                ++fails; // m is collinear with and strictly inside (u,w)
        }
    }
    return fails;
}

// 2:1 balance of the RED layer. In Conforming mode the visible layer is the
// closure, whose children carry their parent's level, so the balance invariant
// the fixpoint actually maintains is a property of the RED faces -- recovered
// here by un-closing. Identical to check21Balance() in HangingNode2to1 mode
// (there the red layer IS the visible layer).
// Returns LOCAL fails (sum across ranks == global).
template <class MeshT>
int check21BalanceRed( MeshT& mesh )
{
    if constexpr ( MeshT::refinement_mode !=
                   Tessera::RefinementMode::Conforming )
    {
        return check21Balance( mesh );
    }
    else
    {
        const Tessera::UncloseResult un =
            Tessera::unclose( ownedVisibleFaces( mesh ) );
        std::vector<std::array<GlobalId, 3>> verts( un.red.size() );
        std::vector<Tessera::Level> level( un.red.size() );
        for ( std::size_t r = 0; r < un.red.size(); ++r )
        {
            for ( int k = 0; k < 3; ++k )
                verts[r][k] = un.red[r].v[k];
            level[r] = un.red[r].level;
        }
        return check21BalanceOn( mesh.comm(), mesh.commSize(), verts, level );
    }
}

// unclose o close == identity, checked against the mesh as it actually stands.
// Two independent properties, both purely local (a closure child names its
// parent outright, so nothing here communicates):
//
//   (1) FIDELITY. Re-closing the un-closed red layer with the SAME split-edge
//       map refine() used reproduces the mesh's visible layer exactly -- as a
//       multiset of (sorted corner triple, level, parent gid, parent corners).
//       Child gids are excluded from the comparison only because their base is
//       an exscan result, not because they are unconstrained; (3) pins them.
//       A closure that fired where it should not have, or a pattern applied to
//       the wrong rotation, fails here.
//   (2) INVERSE. Un-closing that re-closure returns the red layer bit-for-bit
//       on gid, corner gids, and level.
//   (3) GID SANITY. Visible face gids are locally distinct, and a passed-through
//       red face's gid equals its red gid (`parent == invalid_gid`).
//
// `midpoints` is RefineResult::midpoints from the refine() call that produced
// this mesh -- the very map the closure consumed. A no-op (0) in
// HangingNode2to1 mode, where there is no closure layer.
// Returns LOCAL fails (sum across ranks == global).
template <class MeshT>
int checkClosureInverse(
    MeshT& mesh,
    const std::vector<std::pair<Tessera::EdgeKey, GlobalId>>& midpoints )
{
    if constexpr ( MeshT::refinement_mode !=
                   Tessera::RefinementMode::Conforming )
    {
        (void)mesh;
        (void)midpoints;
        return 0;
    }
    else
    {
        using Tessera::GlobalId;
        const std::vector<Tessera::VisibleFace> visible =
            ownedVisibleFaces( mesh );
        const Tessera::UncloseResult un = Tessera::unclose( visible );

        std::map<Tessera::EdgeKey, GlobalId> midpointOf;
        for ( const auto& kv : midpoints )
            midpointOf.emplace( kv.first, kv.second );

        // The blue diagonal is chosen from the two split edges' squared lengths,
        // so a faithful RE-closure has to reproduce them. It can: the split
        // edges' endpoints are corners of this rank's own faces and refine()
        // carries existing vertex positions through verbatim, so recomputing
        // from the mesh as it now stands through the same edgeLen2Canonical()
        // gives bit-identical values to the ones the closure consumed. A
        // missing one is not silently defaulted -- closeFaces() aborts naming
        // the edge.
        std::map<Tessera::EdgeKey, double> len2Of;
        {
            const std::size_t nv = mesh.numVertices();
            Cabana::AoSoA<typename MeshT::vertex_member_types,
                          Kokkos::HostSpace>
                hv( "hv", nv );
            Cabana::deep_copy( hv, mesh.vertices() );
            auto vg = Cabana::slice<Tessera::VertexField::Gid>( hv );
            auto vp = Cabana::slice<Tessera::VertexField::Position>( hv );
            std::map<GlobalId, std::size_t> lv;
            for ( std::size_t i = 0; i < nv; ++i )
                lv.emplace( vg( i ), i );
            Tessera::addSplitEdgeLengths(
                midpointOf, MeshT::dim,
                [&]( GlobalId g, double* p )
                {
                    auto it = lv.find( g );
                    if ( it == lv.end() )
                        return false;
                    for ( int d = 0; d < MeshT::dim; ++d )
                        p[d] = static_cast<double>( vp( it->second, d ) );
                    return true;
                },
                len2Of );
        }

        // Hand the re-closure a fresh gid block above everything live, exactly
        // as refine() does, so unclose()'s duplicate-gid guard stays meaningful.
        GlobalId base = 0;
        for ( const auto& f : visible )
            base = std::max( base, f.gid );
        const Tessera::CloseResult cl = Tessera::closeFaces(
            un.red, midpointOf, base + 1, std::vector<char>(),
            Tessera::invalid_gid, len2Of );

        int fails = 0;

        // (1) fidelity, as a multiset keyed on everything but the child gid.
        using Sig = std::tuple<GlobalId, GlobalId, GlobalId, long long,
                               GlobalId, GlobalId, GlobalId, GlobalId>;
        auto sigOf = []( const Tessera::VisibleFace& f )
        {
            std::array<GlobalId, 3> v = { f.v[0], f.v[1], f.v[2] };
            std::sort( v.begin(), v.end() );
            return Sig{ v[0],
                        v[1],
                        v[2],
                        static_cast<long long>( f.level ),
                        f.parent,
                        f.parentVerts[0],
                        f.parentVerts[1],
                        f.parentVerts[2] };
        };
        std::multiset<Sig> got, want;
        for ( const auto& f : visible )
            got.insert( sigOf( f ) );
        for ( const auto& f : cl.visible )
            want.insert( sigOf( f ) );
        if ( got != want )
            ++fails;

        // (2) inverse: unclose o close reproduces the red layer bit-for-bit.
        const Tessera::UncloseResult un2 = Tessera::unclose( cl.visible );
        if ( un2.red.size() != un.red.size() )
            ++fails;
        else
        {
            std::map<GlobalId, const Tessera::RedFace*> byGid;
            for ( const auto& r : un.red )
                byGid[r.gid] = &r;
            for ( const auto& r : un2.red )
            {
                auto it = byGid.find( r.gid );
                if ( it == byGid.end() || it->second->level != r.level )
                {
                    ++fails;
                    continue;
                }
                for ( int k = 0; k < 3; ++k )
                    if ( it->second->v[k] != r.v[k] )
                        ++fails;
            }
        }

        // (3) gid sanity on the visible layer: locally distinct gids, and a
        //     passed-through red face carries NO closure bookkeeping (a
        //     zero-filled ClosureParentVerts would read as vertex gid 0 here).
        std::set<GlobalId> seen;
        for ( const auto& f : visible )
        {
            if ( !seen.insert( f.gid ).second )
                ++fails;
            if ( f.parent == Tessera::invalid_gid )
                for ( int k = 0; k < 3; ++k )
                    if ( f.parentVerts[k] != Tessera::invalid_gid )
                        ++fails;
        }

        return fails;
    }
}

// Closure siblings are CO-RESIDENT: no retired red parent has children on two
// ranks. unclose() is local per child, so a split sibling group would have each
// holding rank restore the SAME red parent -- a duplicated face, a broken
// ownership partition, and a global face count that grows on every refine.
// Co-residency holds by construction after refine() (closeFaces() runs on one
// rank's red layer) and is maintained by migrate()'s sibling-cohesion fixup;
// this is the check that the fixup actually covered every case.
//
// Verified through the same gid coordinator (gid % size) the other partition
// checks use, so it is rank-count independent and needs no ghost layer. A
// no-op (0) in HangingNode2to1 mode. Returns LOCAL fails (sum == global).
template <class MeshT>
int checkSiblingCoresidency( MeshT& mesh )
{
    if constexpr ( MeshT::refinement_mode !=
                   Tessera::RefinementMode::Conforming )
    {
        (void)mesh;
        return 0;
    }
    else
    {
        MPI_Comm comm = mesh.comm();
        const int rank = mesh.rank();
        const int size = mesh.commSize();

        struct ClaimMsg
        {
            GlobalId parent;
            Rank owner;
        };
        std::set<GlobalId> mine;
        for ( const auto& f : ownedVisibleFaces( mesh ) )
            if ( f.parent != Tessera::invalid_gid )
                mine.insert( f.parent );

        std::vector<std::vector<ClaimMsg>> send( size );
        for ( GlobalId p : mine )
            send[p % size].push_back( { p, static_cast<Rank>( rank ) } );
        auto got = Tessera::allToAllV( comm, send );

        std::map<GlobalId, Rank> claimed;
        int fails = 0;
        for ( const auto& m : got.data )
        {
            auto it = claimed.find( m.parent );
            if ( it == claimed.end() )
                claimed.emplace( m.parent, m.owner );
            else if ( it->second != m.owner )
                ++fails; // siblings of this parent live on two ranks
        }
        return fails;
    }
}

// Number of distinct retired red parents this rank holds children of, i.e. the
// local closure-sibling group count. 0 in HangingNode2to1 mode. Reported by the
// conforming migrate test so the Task-8 run yields the group/fixup figures.
template <class MeshT>
long long closureSiblingGroups( MeshT& mesh )
{
    if constexpr ( MeshT::refinement_mode !=
                   Tessera::RefinementMode::Conforming )
    {
        (void)mesh;
        return 0;
    }
    else
    {
        std::set<GlobalId> p;
        for ( const auto& f : ownedVisibleFaces( mesh ) )
            if ( f.parent != Tessera::invalid_gid )
                p.insert( f.parent );
        return static_cast<long long>( p.size() );
    }
}

} // namespace TesseraTest

#endif // TESSERA_TEST_MESH_INVARIANTS_HPP
