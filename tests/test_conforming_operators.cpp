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

// Regression test: SURFACE OPERATORS on a conforming mesh (Task 7 of
// tasks/completed/conforming-refinement.md). This is the PAYOFF test -- the reason the
// consuming codebase needs conforming refinement at all.
//
// A hanging node is a vertex `m` that lies on an edge (a,b) of a kept face
// (a,b,c) without being one of its corners. Nothing in the topology says so:
// m's incident faces, its edge 1-ring, and its vertex->faces CSR row are all
// perfectly self-consistent -- they just describe a HALF-DISC instead of a
// disc. Every operator assembled over that neighbourhood (a k=1 stencil, a
// face->vertex reduction) is therefore silently wrong at m: it never sees the
// kept face that m geometrically touches. The conforming closure retriangulates
// that kept face so m becomes a real corner of it, and this test measures that
// the resulting neighbourhoods are consistent.
//
// Pipeline: buildIcosphere -> distribute -> two adaptive conforming refine()
// rounds -> migrate -> haloExchange. Then, over the CLOSURE VERTICES (the
// former hanging nodes) specifically:
//
//   (1) IDENTIFICATION. A closure vertex is a corner of some closure child that
//       is NOT a corner of that child's red parent -- i.e. a midpoint of one of
//       the parent's edges, created by a refining neighbour. This reads the
//       ClosureParent / ClosureParentVerts bookkeeping directly, over ALL
//       locally held faces (owned and ghost) so an owned vertex's full incident
//       set is covered. NON-VACUITY: the global count of OWNED closure vertices
//       must be positive at every rank count, or the whole test proves nothing
//       and fails loudly.
//
//   (2) CLOSED 1-RING FAN. Every edge incident to an owned vertex is shared by
//       exactly two of that vertex's incident faces. This is the local form of
//       "no hanging nodes": at a hanging node the fan is an open half-disc and
//       the two boundary edges have a single incident face each. Checked for
//       every owned vertex, and reported separately for the closure ones.
//
//   (3) STENCIL TOPOLOGY. buildVertexStencil(mesh, 1) derives the 1-ring from
//       the vertex->edges CSR; the reference here is derived INDEPENDENTLY from
//       the face table (the other two corners of every incident face). On a
//       conforming mesh the two must coincide, and the vertex->faces CSR row
//       must be exactly the set of local faces containing the vertex.
//
//   (4) applyStencil WITH THE ANALYTIC FIELD f(p) = p_x and uniform weights,
//       against a reference computed from the haloed positions. The max error is
//       reported SEPARATELY for closure and interior vertices, so a
//       closure-specific regression cannot hide inside an aggregate figure.
//
//   (5) reduceVertexFromFaces WITH THE ONE-THIRD-AREA OP. Globally, the owned
//       vertex-area sum equals the owned faceArea sum (a partition-independent
//       identity). Per closure vertex, the accumulated area equals the true
//       one-third sum over its FULL incident-face set, computed host-side from
//       positions -- independent of the CSR the reduction walked.
//
// Prints the closure-vertex count per rank count and both max-error figures for
// the Task-8 measurement table.
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
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <type_traits>
#include <unordered_map>
#include <vector>

using namespace Tessera;

using Scalar = double;

static inline long long globalSum( long long v )
{
    long long g = 0;
    MPI_Allreduce( &v, &g, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD );
    return g;
}
static inline double globalSumD( double v )
{
    double g = 0;
    MPI_Allreduce( &v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    return g;
}
static inline double globalMaxD( double v )
{
    double g = 0;
    MPI_Allreduce( &v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
    return g;
}

//! Gid of every OWNED face (host snapshot).
template <class MeshT>
static std::vector<GlobalId> ownedFaceGids( MeshT& mesh )
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

//! Adaptive mask: refine face gids divisible by `m`. Global-gid keyed, so the
//! same faces are marked however the mesh is partitioned.
template <class MeshT>
static std::vector<char> gidMask( MeshT& mesh, int m )
{
    const std::vector<GlobalId> g = ownedFaceGids( mesh );
    std::vector<char> mask( g.size(), 0 );
    for ( std::size_t f = 0; f < g.size(); ++f )
        mask[f] = ( g[f] % m == 0 ) ? 1 : 0;
    return mask;
}

//! Caller-owned device functor: vertexArea(v) += 1/3 * faceArea(f). The same op
//! the reduce_faces unit test uses, so the identity checked here is the same
//! one -- what is new is that it must hold at former hanging nodes too.
template <class MeshT>
struct ThirdAreaOp
{
    template <class FaceSlice, class VertSlice>
    KOKKOS_INLINE_FUNCTION void operator()( int v, int f,
                                            const MeshGeometry<MeshT>& g,
                                            FaceSlice, VertSlice vs ) const
    {
        vs( v ) += faceArea( g, f ) / Scalar( 3 );
    }
};

//! Everything the checks below need, read once from the mesh host-side.
struct LocalView
{
    std::size_t nv = 0, nf = 0, nOwnedV = 0;
    std::vector<GlobalId> vgid;
    std::vector<std::array<double, 3>> vpos;
    std::unordered_map<GlobalId, int> gid2local;
    //! Corner LOCAL indices of every local face (-1 if a corner is not held).
    std::vector<std::array<int, 3>> fverts;
    //! Local face indices incident to each local vertex, ascending.
    std::vector<std::vector<int>> faceInc;
    //! Face-derived 1-ring (local indices, ascending) of each local vertex.
    std::vector<std::vector<int>> faceRing;
    //! Per local vertex: is it a closure vertex (a former hanging node)?
    std::vector<char> isClosure;
};

template <class MeshT>
static LocalView readLocal( MeshT& mesh )
{
    LocalView L;
    L.nv = mesh.numVertices();
    L.nf = mesh.numFaces();
    L.nOwnedV = mesh.numOwnedVertices();

    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", L.nv );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", L.nf );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto vg = Cabana::slice<VertexField::Gid>( hv );
    auto vp = Cabana::slice<VertexField::Position>( hv );

    L.vgid.resize( L.nv );
    L.vpos.resize( L.nv );
    for ( std::size_t i = 0; i < L.nv; ++i )
    {
        L.vgid[i] = vg( i );
        for ( int d = 0; d < 3; ++d )
            L.vpos[i][d] = static_cast<double>( vp( i, d ) );
        L.gid2local[vg( i )] = static_cast<int>( i );
    }

    // The visible faces INCLUDING ghosts: an owned vertex's incident faces may
    // be owned by a neighbour, and the closure bookkeeping travels with them.
    const std::vector<VisibleFace> vis =
        readVisibleFaces<MeshT>( hf, mesh.numFaces() );

    L.fverts.assign( L.nf, { -1, -1, -1 } );
    L.faceInc.assign( L.nv, {} );
    L.faceRing.assign( L.nv, {} );
    L.isClosure.assign( L.nv, 0 );

    for ( std::size_t f = 0; f < L.nf; ++f )
    {
        for ( int k = 0; k < 3; ++k )
        {
            auto it = L.gid2local.find( vis[f].v[k] );
            L.fverts[f][k] = ( it == L.gid2local.end() ) ? -1 : it->second;
        }
        for ( int k = 0; k < 3; ++k )
        {
            const int lv = L.fverts[f][k];
            if ( lv < 0 )
                continue;
            L.faceInc[lv].push_back( static_cast<int>( f ) );
            for ( int j = 1; j <= 2; ++j )
                if ( L.fverts[f][( k + j ) % 3] >= 0 )
                    L.faceRing[lv].push_back( L.fverts[f][( k + j ) % 3] );
        }

        // (1) closure vertices: a corner of a closure child that is not a
        //     corner of its retired red parent is a midpoint of a parent edge,
        //     i.e. the hanging node the closure absorbed.
        if ( vis[f].parent == invalid_gid )
            continue;
        for ( int k = 0; k < 3; ++k )
        {
            const GlobalId g = vis[f].v[k];
            if ( g == vis[f].parentVerts[0] || g == vis[f].parentVerts[1] ||
                 g == vis[f].parentVerts[2] )
                continue;
            auto it = L.gid2local.find( g );
            if ( it != L.gid2local.end() )
                L.isClosure[it->second] = 1;
        }
    }

    for ( std::size_t v = 0; v < L.nv; ++v )
    {
        std::sort( L.faceRing[v].begin(), L.faceRing[v].end() );
        L.faceRing[v].erase(
            std::unique( L.faceRing[v].begin(), L.faceRing[v].end() ),
            L.faceRing[v].end() );
    }
    return L;
}

//! (2) Every edge incident to an owned vertex is shared by exactly two of that
//! vertex's incident faces -- the local form of "the 1-ring is a closed fan".
//! At a hanging node the fan is an open half-disc and this fails.
static void checkClosedFan( const LocalView& L, int& failsClosure,
                            int& failsInterior )
{
    failsClosure = 0;
    failsInterior = 0;
    for ( std::size_t v = 0; v < L.nOwnedV; ++v )
    {
        std::map<int, int> spokeCount; // opposite-corner local index -> count
        for ( int f : L.faceInc[v] )
            for ( int k = 0; k < 3; ++k )
            {
                if ( L.fverts[f][k] != static_cast<int>( v ) )
                    continue;
                ++spokeCount[L.fverts[f][( k + 1 ) % 3]];
                ++spokeCount[L.fverts[f][( k + 2 ) % 3]];
            }
        int bad = 0;
        for ( const auto& kv : spokeCount )
            if ( kv.first >= 0 && kv.second != 2 )
                ++bad;
        if ( L.isClosure[v] )
            failsClosure += bad;
        else
            failsInterior += bad;
    }
}

//! Area of local face `f` from host positions (independent of MeshGeometry).
static double hostFaceArea( const LocalView& L, int f )
{
    const std::array<double, 3>& a = L.vpos[L.fverts[f][0]];
    const std::array<double, 3>& b = L.vpos[L.fverts[f][1]];
    const std::array<double, 3>& c = L.vpos[L.fverts[f][2]];
    const double u[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
    const double w[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
    const double n[3] = { u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2],
                          u[0] * w[1] - u[1] * w[0] };
    return 0.5 * std::sqrt( n[0] * n[0] + n[1] * n[1] + n[2] * n[2] );
}

template <class Exec, class MeshT>
static int case_operators( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    MPI_Comm comm = MPI_COMM_WORLD;
    constexpr std::size_t IN = userVertexField<0>();
    constexpr std::size_t OUT = userVertexField<1>();
    constexpr std::size_t VAREA = userVertexField<2>();
    int fails = 0;

    // ---- pipeline ---------------------------------------------------------
    MeshT mesh( comm );
    buildIcosphere( mesh, 2 );
    MeshHalo<mem> halo;
    {
        auto faceOwner = facePartitionByAxis( mesh );
        distribute( mesh, halo, faceOwner );
    }
    // refine() rebuilds the 1-deep halo itself, so the neighbourhood checks
    // below see complete owned 1-rings with nothing done in between. This used
    // to need an identity migrate() after each round: refine()'s Phase 3a
    // interpolates each midpoint it owns from both endpoint positions, and
    // across a partition boundary one of those endpoints is a ghost.
    refine( mesh, halo, gidMask( mesh, 7 ) );
    refine( mesh, halo, gidMask( mesh, 5 ) );

    // The neighbourhood checks below are only meaningful on a mesh whose owned
    // 1-rings are complete and whose global topology is conforming.
    {
        int local = TesseraTest::owned1RingLocal( mesh );
        local += TesseraTest::checkConforming( mesh );
        local += TesseraTest::checkSiblingCoresidency( mesh );
        local += TesseraTest::checkOwnershipPartition(
            mesh, TesseraTest::globalOwnedVertices( mesh ),
            TesseraTest::globalOwnedEdges( mesh ),
            TesseraTest::globalOwnedFaces( mesh ) );
        int g = 0;
        MPI_Allreduce( &local, &g, 1, MPI_INT, MPI_SUM, comm );
        if ( g != 0 )
            ++fails;
        if ( TesseraTest::checkOwnedEuler( mesh ) != 2 )
            ++fails;
    }

    const LocalView L = readLocal( mesh );

    // ---- (1) closure vertices, and the non-vacuity guard ------------------
    long long nClosureOwned = 0;
    for ( std::size_t v = 0; v < L.nOwnedV; ++v )
        nClosureOwned += L.isClosure[v] ? 1 : 0;
    const long long gClosure = globalSum( nClosureOwned );
    if ( gClosure <= 0 )
        ++fails; // vacuous: no former hanging node exists to test

    // ---- (2) closed 1-ring fan -------------------------------------------
    int fanClosure = 0, fanInterior = 0;
    checkClosedFan( L, fanClosure, fanInterior );
    if ( globalSum( fanClosure + fanInterior ) != 0 )
        ++fails;

    // ---- (3) stencil topology --------------------------------------------
    auto stencil = buildVertexStencil( mesh, 1 );
    const auto& csr = stencil.csr.get();
    auto soff =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), csr.offsets );
    auto snbr = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                     csr.neighbors );
    auto voff = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), mesh.vertexFaces().offsets );
    auto vnbr = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), mesh.vertexFaces().neighbors );

    long long ringBadClosure = 0, ringBadInterior = 0;
    for ( std::size_t v = 0; v < L.nOwnedV; ++v )
    {
        std::vector<int> row;
        for ( int p = soff( v ); p < soff( v + 1 ); ++p )
            row.push_back( static_cast<int>( snbr( p ) ) );
        std::sort( row.begin(), row.end() );

        std::vector<int> inc;
        for ( int p = voff( v ); p < voff( v + 1 ); ++p )
            inc.push_back( static_cast<int>( vnbr( p ) ) );
        std::sort( inc.begin(), inc.end() );

        const bool bad = ( row != L.faceRing[v] ) || ( inc != L.faceInc[v] );
        if ( bad )
        {
            if ( L.isClosure[v] )
                ++ringBadClosure;
            else
                ++ringBadInterior;
        }
    }
    if ( globalSum( ringBadClosure + ringBadInterior ) != 0 )
        ++fails;

    // ---- (4) applyStencil -------------------------------------------------
    const int nOwnedV = static_cast<int>( mesh.numOwnedVertices() );
    Kokkos::View<Scalar*, mem> w( "w", csr.numEntries() );
    Kokkos::deep_copy( w, Scalar( 1 ) );
    {
        auto pos = mesh.template vertexSlice<VertexField::Position>();
        auto in = mesh.template vertexSlice<IN>();
        Kokkos::parallel_for(
            "set_in", Kokkos::RangePolicy<Exec>( 0, nOwnedV ),
            KOKKOS_LAMBDA( const int i ) { in( i ) = pos( i, 0 ); } );
        Kokkos::fence();
    }
    haloExchange( mesh, halo ); // fill ghost `in`
    {
        auto in = mesh.template vertexSlice<IN>();
        auto out = mesh.template vertexSlice<OUT>();
        applyStencil( mesh, stencil, w, in, out );
    }

    // ---- (5) reduceVertexFromFaces ---------------------------------------
    auto geom = buildMeshGeometry( mesh );
    {
        auto va = mesh.template vertexSlice<VAREA>();
        Kokkos::parallel_for(
            "zero_varea",
            Kokkos::RangePolicy<Exec>( 0,
                                       static_cast<int>( mesh.numVertices() ) ),
            KOKKOS_LAMBDA( const int i ) { va( i ) = Scalar( 0 ); } );
        Kokkos::fence();
    }
    reduceVertexFromFaces(
        mesh, geom, mesh.template faceSlice<FaceField::Gid>(),
        mesh.template vertexSlice<VAREA>(), ThirdAreaOp<MeshT>{} );

    // Read both results back host-side and compare against references derived
    // from the haloed positions (never from the field halo path under test).
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto outSlice = Cabana::slice<OUT>( hv );
    auto areaSlice = Cabana::slice<VAREA>( hv );

    const double tol = 1e-9;
    double errClosure = 0.0, errInterior = 0.0;
    double areaErrClosure = 0.0, areaErrInterior = 0.0;
    double localVArea = 0.0;
    for ( std::size_t v = 0; v < L.nOwnedV; ++v )
    {
        double ref = 0.0;
        for ( int p = soff( v ); p < soff( v + 1 ); ++p )
            ref += L.vpos[snbr( p )][0];
        const double e = std::abs( static_cast<double>( outSlice( v ) ) - ref );

        double aref = 0.0;
        for ( int f : L.faceInc[v] )
            aref += hostFaceArea( L, f ) / 3.0;
        const double ae =
            std::abs( static_cast<double>( areaSlice( v ) ) - aref );
        localVArea += static_cast<double>( areaSlice( v ) );

        if ( L.isClosure[v] )
        {
            errClosure = std::max( errClosure, e );
            areaErrClosure = std::max( areaErrClosure, ae );
        }
        else
        {
            errInterior = std::max( errInterior, e );
            areaErrInterior = std::max( areaErrInterior, ae );
        }
    }
    errClosure = globalMaxD( errClosure );
    errInterior = globalMaxD( errInterior );
    areaErrClosure = globalMaxD( areaErrClosure );
    areaErrInterior = globalMaxD( areaErrInterior );

    // The two max errors are asserted SEPARATELY at the same tolerance: an
    // aggregate figure would let a closure-specific regression hide behind the
    // (far more numerous) interior vertices.
    if ( errClosure > tol || errInterior > tol )
        ++fails;
    if ( areaErrClosure > tol || areaErrInterior > tol )
        ++fails;

    // Global identity: the owned vertex-area sum is the total surface area.
    double localFArea = 0.0;
    for ( std::size_t f = 0; f < mesh.numOwnedFaces(); ++f )
        localFArea += hostFaceArea( L, static_cast<int>( f ) );
    const double gVArea = globalSumD( localVArea );
    const double gFArea = globalSumD( localFArea );
    if ( std::abs( gVArea - gFArea ) > 1e-9 * gFArea )
        ++fails;

    // Every figure printed below is reduced HERE, on all ranks -- a collective
    // inside the rank-0 printf would deadlock at ranks >= 2.
    const long long gV = TesseraTest::globalOwnedVertices( mesh );
    const long long gF = TesseraTest::globalOwnedFaces( mesh );
    const long long gFanC = globalSum( fanClosure );
    const long long gFanI = globalSum( fanInterior );
    const long long gRingC = globalSum( ringBadClosure );
    const long long gRingI = globalSum( ringBadInterior );

    (void)size;
    if ( rank == 0 )
        std::printf( "  [%s] operators %s (V=%lld F=%lld closureVerts=%lld "
                     "fanBad=%lld/%lld ringBad=%lld/%lld "
                     "stencilMaxErr closure=%.3e interior=%.3e "
                     "areaMaxErr closure=%.3e interior=%.3e "
                     "vArea=%.9f fArea=%.9f)\n",
                     tag, fails == 0 ? "ok" : "FAIL", gV, gF, gClosure, gFanC,
                     gFanI, gRingC, gRingI, errClosure, errInterior,
                     areaErrClosure, areaErrInterior, gVArea, gFArea );
    return fails;
}

template <class Exec>
static int run( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    // three user vertex fields: [0] = in, [1] = out, [2] = accumulated area.
    using MeshT =
        Mesh<Scalar, 3, VertexFields<Scalar, Scalar, Scalar>, EdgeFields<>,
             FaceFields<>, mem, Exec, RefinementMode::Conforming>;
    return case_operators<Exec, MeshT>( rank, size, tag );
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
            std::printf( "test_conforming_operators: stencil / reduction "
                         "consistency at former hanging nodes (size %d)\n",
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
