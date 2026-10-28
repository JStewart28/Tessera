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

// Regression test: CONFIGURABLE HALO DEPTH (tasks/completed/halo-depth.md).
//
// The ghost layer used to be 1-deep and not configurable, which covers a
// one-ring stencil exactly and nothing wider. buildVertexStencil( mesh, 2 ) on a
// distributed mesh was therefore SILENTLY SHORT within one hop of a partition
// boundary: an owned vertex whose 2-ring reached past the ghost layer simply got
// a shorter CSR row, which looks exactly like a correct one. An operator built on
// it produces a plausible field with a small error localized on partition
// boundaries -- an error that MOVES when the rank count changes and that no
// existing invariant check detects.
//
// This test is the acceptance test for the fix: `depth` on distribute() and
// rebuildHalo(), preserved by refine()/migrate(), and policed by
// buildVertexStencil().
//
// GROUND TRUTH. Exact and rank-count independent: the owned entities partition
// the global mesh, so an MPI_Allgatherv of every rank's owned edge endpoints
// reconstructs the WHOLE global vertex adjacency on every rank, and the true
// k-ring of any gid is a BFS on that. Unlike a replicated reference icosphere it
// keeps working after refine()/migrate(), which is what checks 6 and 7 need. The
// meshes here are small (a subdivision-2 icosphere and four adaptive rounds on
// it), so the allgather is cheap.
//
// CHECKS
//   1. DEPTH-2 RING COMPLETENESS (the definitive one). distribute(..., depth=2),
//      haloExchange. For every OWNED vertex the 2-ring gid set from
//      buildVertexStencil( mesh, 2 ) equals the 2-ring on the global reference:
//      zero missing, zero extra, at every rank count. This is the check the
//      README entry said could not pass. Check 1b additionally cross-checks the
//      two independent implementations of that closure -- distribute()'s local
//      marking loop over the replicated mesh and rebuildHalo()'s ring of
//      coordinator queries -- against each other, since a short closure only
//      shows up as a short k-ring for the owned vertices next to the gap.
//   2. DEPTH 1 IS UNCHANGED. The edit is in detail::finishHaloAndAssemble(),
//      which migrate() and refine() also use, so the depth-1 path must be
//      untouched. distribute()'s depth-1 closure is an INDEPENDENT
//      implementation (purely local, marked once over the replicated mesh);
//      rebuildHalo( mesh, halo, 1 ) straight after it must reproduce its local
//      counts, its owned counts, its topologyChecksum and its plan sizes
//      exactly. The rest of the guard is the 150-test gate, every member of
//      which runs the depth-1 path.
//   3. DEPTH IS MONOTONE AND EFFECTIVE. At ranks >= 2 the global ghost vertex
//      count at depth 2 is strictly greater than at depth 1, and the 1-ring rows
//      are still exactly correct at depth 2 -- a wider closure must not perturb
//      the inner ring.
//   4. OWNERSHIP IS DEPTH-INVARIANT. checkOwnershipPartition, the three
//      globalOwned* counts, ownedEulerGlobal and topologyChecksum are identical
//      at depth 1 and depth 2. This pins the argument that round B of the
//      rebuild runs ONCE, outside the ring loop.
//   5. GHOST VALUES ARE THE OWNERS' VALUES. After haloExchange at depth 2 every
//      ghost vertex position equals the owner's position for that gid, bitwise.
//   6. refine() PRESERVES DEPTH. Build at depth 2, then FOUR refine() rounds
//      back-to-back with nothing in between -- the sequence test_refine_rehalo
//      already exercises at depth 1, so this adds depth rather than re-proving
//      the re-halo. After each round: halo.depth == 2, mesh.haloDepth() == 2,
//      check 1 still passes, and the conforming invariants hold. Includes the
//      non-vacuity guards test_refine_rehalo uses -- plans non-empty at ranks
//      >= 2, and a deliberately corrupted ghost resynced by haloExchange --
//      because the structural checks all pass on an EMPTY plan.
//   7. migrate() PRESERVES DEPTH. Identity migrate and a real loadBalance() at
//      depth 2; halo.depth == 2 and check 1 still passes after each.
//   8. IDEMPOTENCE. rebuildHalo() twice yields identical local counts and
//      identical plans (totalSend/totalRecv and the flattened index arrays).
//   9. STENCIL GUARD. buildVertexStencil( mesh, 2 ) throws std::invalid_argument
//      on a depth-1 mesh and does not on a depth-2 mesh.
//  10. SINGLE RANK. Depth 1 and 2 give an empty plan and identical meshes, and
//      haloExchange is a no-op. The whole mesh is locally resident after ring 0,
//      so the collective early exit in the ring loop is what stops ring 1 from
//      costing anything -- observable here as depth 2 producing exactly the
//      depth-1 mesh.
//
// Runs on host (Serial) and device (default, HIP), ranks 1-5.

#include "MeshInvariants.hpp"

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <vector>

using namespace Tessera;

// ===========================================================================
// Small MPI helpers
// ===========================================================================

//! Concatenate every rank's `local` into one vector held identically by all.
//! Byte-wise, so it works for any trivially-copyable record.
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
    const T dummy{};
    MPI_Allgatherv( local.empty() ? &dummy : local.data(), nbytes, MPI_BYTE,
                    out.empty() ? nullptr : out.data(), counts.data(),
                    displs.data(), MPI_BYTE, comm );
    return out;
}

inline int globalFails( MPI_Comm comm, int local )
{
    int g = 0;
    MPI_Allreduce( &local, &g, 1, MPI_INT, MPI_SUM, comm );
    return g;
}

inline long long commSum( MPI_Comm comm, long long local )
{
    long long g = 0;
    MPI_Allreduce( &local, &g, 1, MPI_LONG_LONG, MPI_SUM, comm );
    return g;
}

// ===========================================================================
// Global reference: the whole mesh's vertex adjacency, on every rank
// ===========================================================================

struct EdgePair
{
    GlobalId a = 0;
    GlobalId b = 0;
};

//! Vertex adjacency of the GLOBAL mesh, assembled from every rank's OWNED edges.
//! Owned entities partition the global mesh, so this is exact and identical on
//! every rank -- and independent of how the mesh happens to be distributed,
//! which is exactly the property a halo-depth check needs.
template <class MeshT>
std::map<GlobalId, std::set<GlobalId>> globalVertexAdjacency( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
        "ref_he", mesh.numEdges() );
    Cabana::deep_copy( he, mesh.edges() );
    auto e_v = Cabana::slice<EdgeField::Verts>( he );

    std::vector<EdgePair> mine;
    mine.reserve( mesh.numOwnedEdges() );
    for ( std::size_t e = 0; e < mesh.numOwnedEdges(); ++e )
        mine.push_back( { e_v( e, 0 ), e_v( e, 1 ) } );

    std::map<GlobalId, std::set<GlobalId>> adj;
    for ( const EdgePair& p : allGatherAll( mesh.comm(), mine ) )
    {
        adj[p.a].insert( p.b );
        adj[p.b].insert( p.a );
    }
    return adj;
}

//! True k-ring of `g` on the global reference (self excluded).
inline std::set<GlobalId>
referenceKRing( const std::map<GlobalId, std::set<GlobalId>>& adj, GlobalId g,
                int k )
{
    std::set<GlobalId> ring;
    std::vector<GlobalId> frontier{ g }, next;
    for ( int d = 0; d < k; ++d )
    {
        next.clear();
        for ( GlobalId u : frontier )
        {
            auto it = adj.find( u );
            if ( it == adj.end() )
                continue;
            for ( GlobalId w : it->second )
                if ( w != g && ring.insert( w ).second )
                    next.push_back( w );
        }
        frontier = next;
    }
    return ring;
}

//! Owned + ghost vertex gids in local-index order.
template <class MeshT>
std::vector<GlobalId> localVertexGids( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "gid_hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto g = Cabana::slice<VertexField::Gid>( hv );
    std::vector<GlobalId> out( mesh.numVertices() );
    for ( std::size_t i = 0; i < out.size(); ++i )
        out[i] = g( i );
    return out;
}

//! CHECKS 1 and 3's engine: every OWNED vertex's k-ring row from the stencil,
//! translated to gids, must equal the reference k-ring. Returns LOCAL fails, and
//! reports the first mismatch so a failure is diagnosable rather than a count.
template <class MeshT>
int checkKRingRows( MeshT& mesh, int k, const char* tag )
{
    auto stencil = buildVertexStencil( mesh, k );
    const auto& csr = stencil.csr.get();
    auto off =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), csr.offsets );
    auto nbr = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                    csr.neighbors );
    const std::vector<GlobalId> gids = localVertexGids( mesh );
    const std::map<GlobalId, std::set<GlobalId>> adj =
        globalVertexAdjacency( mesh );

    int fails = 0;
    for ( std::size_t v = 0; v < mesh.numOwnedVertices(); ++v )
    {
        std::set<GlobalId> got;
        for ( int p = off( v ); p < off( v + 1 ); ++p )
            got.insert( gids[static_cast<std::size_t>( nbr( p ) )] );
        const std::set<GlobalId> want = referenceKRing( adj, gids[v], k );
        if ( got != want )
        {
            if ( fails == 0 )
            {
                int missing = 0, extra = 0;
                for ( GlobalId w : want )
                    if ( !got.count( w ) )
                        ++missing;
                for ( GlobalId w : got )
                    if ( !want.count( w ) )
                        ++extra;
                printf(
                    "[halo_depth] %s k=%d rank-local vertex %zu (gid %llu): "
                    "%d missing, %d extra (row %zu, ref %zu)\n",
                    tag, k, v, (unsigned long long)gids[v], missing, extra,
                    got.size(), want.size() );
            }
            ++fails;
        }
    }
    return fails;
}

// ===========================================================================
// Ghost / plan probes
// ===========================================================================

template <class MeshT>
long long globalGhostVertices( MeshT& mesh )
{
    return commSum( mesh.comm(),
                    static_cast<long long>( mesh.numVertices() ) -
                        static_cast<long long>( mesh.numOwnedVertices() ) );
}

template <class MeshT>
long long globalHaloPlanSize( MeshT& mesh,
                              MeshHalo<typename MeshT::memory_space>& halo )
{
    long long local = 0;
    for ( const auto* p : { &halo.vplan, &halo.eplan, &halo.fplan } )
        local += static_cast<long long>( p->totalSend() ) +
                 static_cast<long long>( p->totalRecv() );
    return commSum( mesh.comm(), local );
}

//! Flattened (totalSend, totalRecv, send_idx..., recv_idx...) of the three
//! plans -- the full plan content, for the idempotence comparison.
template <class MemorySpace>
std::vector<long long> planFingerprint( MeshHalo<MemorySpace>& halo )
{
    std::vector<long long> out;
    for ( const auto* p : { &halo.vplan, &halo.eplan, &halo.fplan } )
    {
        out.push_back( static_cast<long long>( p->totalSend() ) );
        out.push_back( static_cast<long long>( p->totalRecv() ) );
        for ( int r : p->send_peers )
            out.push_back( r );
        for ( int o : p->send_off )
            out.push_back( o );
        for ( int r : p->recv_peers )
            out.push_back( r );
        for ( int o : p->recv_off )
            out.push_back( o );
        for ( const auto* idx : { &p->send_idx, &p->recv_idx } )
        {
            auto h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          *idx );
            for ( std::size_t i = 0; i < h.extent( 0 ); ++i )
                out.push_back( h( i ) );
        }
    }
    return out;
}

//! CHECK 5. Every ghost vertex position must equal its OWNER's position for the
//! same gid, bitwise. The owners' (gid, position) pairs are gathered globally,
//! so this compares against the authoritative value and not against a neighbour.
template <class MeshT>
int checkGhostPositionsAreOwners( MeshT& mesh )
{
    using Scalar = typename MeshT::scalar_type;
    constexpr int Dim = MeshT::dim;
    struct GidPos
    {
        GlobalId gid = 0;
        double x[3] = { 0, 0, 0 };
    };

    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "pos_hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto gid = Cabana::slice<VertexField::Gid>( hv );
    auto pos = Cabana::slice<VertexField::Position>( hv );

    std::vector<GidPos> mine;
    mine.reserve( mesh.numOwnedVertices() );
    for ( std::size_t v = 0; v < mesh.numOwnedVertices(); ++v )
    {
        GidPos r;
        r.gid = gid( v );
        for ( int d = 0; d < Dim; ++d )
            r.x[d] = static_cast<double>( pos( v, d ) );
        mine.push_back( r );
    }
    std::map<GlobalId, GidPos> owner;
    for ( const GidPos& r : allGatherAll( mesh.comm(), mine ) )
        owner[r.gid] = r;

    int fails = 0;
    for ( std::size_t v = mesh.numOwnedVertices(); v < mesh.numVertices(); ++v )
    {
        auto it = owner.find( gid( v ) );
        if ( it == owner.end() )
        {
            ++fails; // a ghost nobody owns
            continue;
        }
        for ( int d = 0; d < Dim; ++d )
            if ( static_cast<double>( pos( v, d ) ) !=
                 static_cast<double>( static_cast<Scalar>( it->second.x[d] ) ) )
            {
                ++fails;
                break;
            }
    }
    return fails;
}

//! Overwrite every GHOST vertex position with a value no owner would produce.
template <class MeshT>
void corruptGhostPositions( MeshT& mesh )
{
    using Scalar = typename MeshT::scalar_type;
    constexpr int Dim = MeshT::dim;
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "corrupt_hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto pos = Cabana::slice<VertexField::Position>( hv );
    for ( std::size_t i = mesh.numOwnedVertices(); i < mesh.numVertices(); ++i )
        for ( int d = 0; d < Dim; ++d )
            pos( i, d ) = Scalar( -1234.5 );
    Cabana::deep_copy( mesh.vertices(), hv );
}

template <class MeshT>
long long countCorruptGhosts( MeshT& mesh )
{
    using Scalar = typename MeshT::scalar_type;
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "count_hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto pos = Cabana::slice<VertexField::Position>( hv );
    long long bad = 0;
    for ( std::size_t i = mesh.numOwnedVertices(); i < mesh.numVertices(); ++i )
        if ( pos( i, 0 ) == Scalar( -1234.5 ) )
            ++bad;
    return commSum( mesh.comm(), bad );
}

//! Rank-count-independent summary of a distribution: the three global owned
//! counts, the owned Euler number and the three topology checksums. Depth must
//! not change any of them (check 4).
template <class MeshT>
std::vector<long long> ownershipSignature( MeshT& mesh )
{
    unsigned long long cv = 0, ce = 0, cf = 0;
    TesseraTest::topologyChecksum( mesh, cv, ce, cf );
    return { TesseraTest::globalOwnedVertices( mesh ),
             TesseraTest::globalOwnedEdges( mesh ),
             TesseraTest::globalOwnedFaces( mesh ),
             TesseraTest::ownedEulerGlobal( mesh ),
             static_cast<long long>( cv ),
             static_cast<long long>( ce ),
             static_cast<long long>( cf ) };
}

//! Local (per-rank) shape of a distribution: local and owned counts per kind.
template <class MeshT>
std::vector<long long> localShape( MeshT& mesh )
{
    return { static_cast<long long>( mesh.numVertices() ),
             static_cast<long long>( mesh.numEdges() ),
             static_cast<long long>( mesh.numFaces() ),
             static_cast<long long>( mesh.numOwnedVertices() ),
             static_cast<long long>( mesh.numOwnedEdges() ),
             static_cast<long long>( mesh.numOwnedFaces() ) };
}

// ===========================================================================
// The conforming invariants that must survive a wider halo (check 6)
// ===========================================================================
template <class MeshT>
int conformingChecks( MeshT& mesh, const RefineResult& res )
{
    int fails = 0;
    const long long NvG = TesseraTest::globalOwnedVertices( mesh );
    const long long NeG = TesseraTest::globalOwnedEdges( mesh );
    const long long NfG = TesseraTest::globalOwnedFaces( mesh );
    fails += TesseraTest::checkOwnershipPartition( mesh, NvG, NeG, NfG );
    fails += TesseraTest::owned1RingLocal( mesh );
    fails += TesseraTest::checkMidpointAgreement( mesh.comm(), mesh.commSize(),
                                                  res.midpoints );
    fails += TesseraTest::checkConforming( mesh );
    fails += TesseraTest::checkNoInteriorVertex( mesh );
    fails += TesseraTest::check21BalanceRed( mesh );
    if ( TesseraTest::ownedEulerGlobal( mesh ) != 2 )
        ++fails;
    return fails;
}

//! Adaptive mask: refine face gids divisible by `m`. Gid-derived, so it is the
//! same global face set at any rank count.
template <class MeshT>
std::vector<char> gidMask( MeshT& mesh, int m )
{
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "mask_hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto g = Cabana::slice<FaceField::Gid>( hf );
    std::vector<char> mask( mesh.numOwnedFaces(), 0 );
    for ( std::size_t f = 0; f < mask.size(); ++f )
        mask[f] = ( g( f ) % static_cast<GlobalId>( m ) == 0 ) ? 1 : 0;
    return mask;
}

// ===========================================================================
// The run
// ===========================================================================
template <class ExecSpace>
int run( int rank, int size, const char* tag )
{
    using mem = typename ExecSpace::memory_space;
    using MeshT =
        Mesh<double, 3, Cabana::MemberTypes<>, Cabana::MemberTypes<>,
             Cabana::MemberTypes<>, mem, ExecSpace, RefinementMode::Conforming>;
    MPI_Comm comm = MPI_COMM_WORLD;
    int fails = 0;

    auto build = []( MeshT& m, MeshHalo<mem>& h, int depth )
    {
        buildIcosphere( m, 2 );
        auto faceOwner = facePartitionByAxis( m );
        distribute( m, h, faceOwner, depth );
        haloExchange( m, h );
    };

    // -- depth 1 reference distribution -------------------------------------
    MeshT m1( comm );
    MeshHalo<mem> h1;
    build( m1, h1, 1 );
    if ( h1.depth != 1 || m1.haloDepth() != 1 )
        ++fails;

    const std::vector<long long> shape1 = localShape( m1 );
    const std::vector<long long> sig1 = ownershipSignature( m1 );
    const long long ghosts1 = globalGhostVertices( m1 );
    const long long plan1 = globalHaloPlanSize( m1, h1 );
    fails += globalFails( comm, TesseraTest::checkOwnershipPartition(
                                    m1, sig1[0], sig1[1], sig1[2] ) );
    // The 1-ring is complete at depth 1 -- the property that always held.
    fails += globalFails( comm, checkKRingRows( m1, 1, "depth1" ) );

    // -- CHECK 2: the shared rebuild's depth-1 output == distribute()'s -------
    // distribute() computes the depth-1 closure locally on the replicated mesh;
    // rebuildHalo() computes it by communication through the code the depth loop
    // edits. They must agree exactly, per rank, including the plans.
    {
        MeshT m( comm );
        MeshHalo<mem> h;
        build( m, h, 1 );
        rebuildHalo( m, h, 1 );
        haloExchange( m, h );
        if ( localShape( m ) != shape1 )
            ++fails;
        if ( ownershipSignature( m ) != sig1 )
            ++fails;
        if ( globalHaloPlanSize( m, h ) != plan1 )
            ++fails;
        if ( globalGhostVertices( m ) != ghosts1 )
            ++fails;
        if ( h.depth != 1 || m.haloDepth() != 1 )
            ++fails;

        // -- CHECK 8: idempotence of the rebuild -----------------------------
        const std::vector<long long> shapeA = localShape( m );
        const std::vector<long long> planA = planFingerprint( h );
        rebuildHalo( m, h, 1 );
        if ( localShape( m ) != shapeA || planFingerprint( h ) != planA )
            ++fails;
    }

    // -- depth 2 distribution ------------------------------------------------
    MeshT m2( comm );
    MeshHalo<mem> h2;
    build( m2, h2, 2 );
    if ( h2.depth != 2 || m2.haloDepth() != 2 )
        ++fails;

    // -- CHECK 1: the 2-ring of every owned vertex is complete ---------------
    fails += globalFails( comm, checkKRingRows( m2, 2, "depth2" ) );

    // -- CHECK 1b: the two depth-2 closures agree ----------------------------
    // distribute()'s closure is a local marking loop over the replicated mesh;
    // rebuildHalo()'s is a ring of coordinator queries. They are independent
    // implementations of the same definition, so at depth 2 (as at depth 1, in
    // check 2) they must produce the identical local set. Cross-checking them
    // catches an under-expansion that a k-ring check alone can miss, because a
    // short closure only shows up as a short k-ring for the owned vertices that
    // happen to sit next to the missing ring.
    {
        MeshT mr( comm );
        MeshHalo<mem> hr;
        build( mr, hr, 1 );
        rebuildHalo( mr, hr, 2 );
        haloExchange( mr, hr );
        if ( localShape( mr ) != localShape( m2 ) )
            ++fails;
        if ( ownershipSignature( mr ) != sig1 )
            ++fails;
        if ( hr.depth != 2 || mr.haloDepth() != 2 )
            ++fails;
        fails += globalFails( comm, checkKRingRows( mr, 2, "rebuild2" ) );
    }

    // -- CHECK 3: monotone and effective, inner ring undisturbed -------------
    const long long ghosts2 = globalGhostVertices( m2 );
    if ( size > 1 && ghosts2 <= ghosts1 )
        ++fails;
    fails += globalFails( comm, checkKRingRows( m2, 1, "depth2-inner" ) );

    // -- CHECK 4: ownership is depth-invariant -------------------------------
    if ( ownershipSignature( m2 ) != sig1 )
        ++fails;
    fails += globalFails( comm, TesseraTest::checkOwnershipPartition(
                                    m2, sig1[0], sig1[1], sig1[2] ) );

    // -- CHECK 5: ghost values are the owners' values ------------------------
    fails += globalFails( comm, checkGhostPositionsAreOwners( m2 ) );

    // -- CHECK 9: the stencil guard ------------------------------------------
    {
        bool threw = false;
        try
        {
            auto s = buildVertexStencil( m1, 2 );
            (void)s;
        }
        catch ( const std::invalid_argument& )
        {
            threw = true;
        }
        if ( !threw )
            ++fails; // k=2 on a depth-1 mesh must be LOUD, never quietly short
        try
        {
            auto s = buildVertexStencil( m2, 2 );
            (void)s;
        }
        catch ( ... )
        {
            ++fails; // k=2 on a depth-2 mesh must be allowed
        }
    }

    // -- CHECK 10: single rank ------------------------------------------------
    // The whole mesh is locally resident after ring 0, so ring 1 acquires
    // nothing and the collective early exit ends the loop: depth 2 must be
    // exactly the depth-1 mesh, with an empty plan and a no-op haloExchange.
    if ( size == 1 )
    {
        if ( localShape( m2 ) != shape1 )
            ++fails;
        if ( plan1 != 0 || globalHaloPlanSize( m2, h2 ) != 0 )
            ++fails;
        if ( globalGhostVertices( m2 ) != 0 || ghosts2 != 0 )
            ++fails;
        unsigned long long a0, b0, c0, a1, b1, c1;
        TesseraTest::topologyChecksum( m2, a0, b0, c0 );
        haloExchange( m2, h2 );
        TesseraTest::topologyChecksum( m2, a1, b1, c1 );
        if ( a0 != a1 || b0 != b1 || c0 != c1 )
            ++fails;
    }

    if ( rank == 0 )
        printf( "[halo_depth] %-7s np%d setup: ghosts d1=%lld d2=%lld "
                "plan d1=%lld V=%lld E=%lld F=%lld fails=%d\n",
                tag, size, ghosts1, ghosts2, plan1, sig1[0], sig1[1], sig1[2],
                fails );

    // -- CHECK 6: refine() preserves depth over four rounds ------------------
    {
        MeshT m( comm );
        MeshHalo<mem> h;
        build( m, h, 2 );
        const int masks[4] = { 7, 5, 3, 4 };
        for ( int r = 0; r < 4; ++r )
        {
            auto res = refine( m, h, gidMask( m, masks[r] ) );
            if ( h.depth != 2 || m.haloDepth() != 2 )
                ++fails;
            fails += globalFails( comm, conformingChecks( m, res ) );
            fails += globalFails( comm, checkKRingRows( m, 2, "refine" ) );

            // Non-vacuity: every structural check above passes on an EMPTY
            // plan, so without these two a regression to a silent no-op halo
            // would not be detected.
            const long long planSize = globalHaloPlanSize( m, h );
            if ( size > 1 && planSize <= 0 )
                ++fails;
            corruptGhostPositions( m );
            const long long corrupted = countCorruptGhosts( m );
            haloExchange( m, h );
            if ( countCorruptGhosts( m ) != 0 )
                ++fails;
            if ( size > 1 && corrupted <= 0 )
                ++fails;
            fails += globalFails( comm, checkGhostPositionsAreOwners( m ) );

            const long long gV = TesseraTest::globalOwnedVertices( m );
            const long long gF = TesseraTest::globalOwnedFaces( m );
            if ( rank == 0 )
                printf( "[halo_depth] %-7s np%d refine round %d: depth=%d "
                        "V=%lld F=%lld plan=%lld ghostsCorrupted=%lld "
                        "fails=%d\n",
                        tag, size, r + 1, h.depth, gV, gF, planSize, corrupted,
                        fails );
        }

        // -- CHECK 7: migrate() preserves depth ------------------------------
        // Identity first (nothing moves, so only the halo half runs), then a
        // real Zoltan2 load balance on the refined conforming mesh.
        {
            std::vector<Rank> dest( m.numOwnedFaces(),
                                    static_cast<Rank>( rank ) );
            migrate( m, h, dest );
            if ( h.depth != 2 || m.haloDepth() != 2 )
                ++fails;
            haloExchange( m, h );
            fails += globalFails( comm, checkKRingRows( m, 2, "migrate-id" ) );
        }
        {
            loadBalance( m, h );
            if ( h.depth != 2 || m.haloDepth() != 2 )
                ++fails;
            haloExchange( m, h );
            fails += globalFails( comm, checkKRingRows( m, 2, "loadbalance" ) );
            fails += globalFails( comm, checkGhostPositionsAreOwners( m ) );
            const long long NvG = TesseraTest::globalOwnedVertices( m );
            const long long NeG = TesseraTest::globalOwnedEdges( m );
            const long long NfG = TesseraTest::globalOwnedFaces( m );
            fails += globalFails( comm, TesseraTest::checkOwnershipPartition(
                                            m, NvG, NeG, NfG ) );
            if ( rank == 0 )
                printf( "[halo_depth] %-7s np%d migrate/loadBalance: depth=%d "
                        "V=%lld F=%lld fails=%d\n",
                        tag, size, h.depth, NvG, NfG, fails );
        }
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

        fails += run<Kokkos::Serial>( rank, size, "Serial" );
        if constexpr ( !std::is_same<Kokkos::DefaultExecutionSpace,
                                     Kokkos::Serial>::value )
            fails +=
                run<Kokkos::DefaultExecutionSpace>( rank, size, "Default" );

        if ( rank == 0 )
            printf( "[halo_depth] TOTAL fails=%d\n", fails );
    }
    Kokkos::finalize();
    MPI_Finalize();
    return fails == 0 ? 0 : 1;
}
