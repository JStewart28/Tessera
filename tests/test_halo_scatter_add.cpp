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

// Regression test: GHOST SCATTER-ADD, the reverse halo
// (tasks/completed/halo-scatter-add.md).
//
// haloExchange() is a pure gather (owner -> ghost, overwrite). haloScatterAdd()
// is its missing reverse (ghost -> owner, +=), which is the other half of the
// standard distributed-assembly pattern: a per-vertex quantity assembled by
// iterating OWNED faces leaves the owner of a boundary vertex with a partial sum
// and every ghost copy with a different partial sum, and only a reverse
// accumulate makes the owned value the true global sum.
//
// GROUND TRUTH IS THE REPLICATED REFERENCE MESH, NEVER TESSERA'S OWN DISTRIBUTED
// STATE. Before distribute() the mesh is replicated with gid == index, so this
// test snapshots its connectivity/positions and then recomputes, on every rank,
//   * ownership   (face -> faceOwner; vertex/edge -> min faceOwner over the
//                  incident faces), and
//   * the local set of EVERY rank  (the `depth`-deep closure of that rank's owned
//                  vertices, exactly as documented for distribute()),
// from that snapshot plus the faceOwner array that is distribute()'s own input.
// The per-gid GHOST MULTIPLICITY -- how many ranks hold a copy of an entity -- and
// the true global incident-face count of every vertex then follow with no
// reference to what Tessera actually built, so a self-consistent wrong answer
// cannot pass. As a guard on the replication itself, the reference's predicted
// per-rank local counts are asserted against the real mesh's.
//
// CHECKS
//   1. GHOST-MULTIPLICITY COUNT (exact integer ground truth). Set the scalar
//      vertex user field to 1.0 on EVERY local vertex, owned and ghost, then
//      haloScatterAdd. Each owned vertex must hold exactly
//      1 + (number of other ranks holding a ghost copy) == its multiplicity.
//      Small integers in floating point, so this is an equality assert.
//   2. FACE-LOOP ASSEMBLY (the real use case). Zero the field, loop OWNED faces
//      adding 1.0 to each corner's local slot (ghost slots included -- a corner
//      of an owned face need not be owned), haloScatterAdd. Every owned vertex
//      must hold its true GLOBAL incident-face count: 6 on a subdivision-2
//      icosphere except the 12 original icosahedron vertices, which have 5.
//      This is the check that fails today with a plain haloExchange.
//
//      NOTE, deliberate deviation from the task text, which said to loop LOCAL
//      faces: iterating local (owned + ghost) faces double-counts, because a
//      ghost face is an owned face somewhere else and would contribute twice.
//      The assembly pattern this operation exists to serve -- vertex areas,
//      vertex normals, a per-element residual -- always iterates OWNED faces,
//      which is what makes each face contribute exactly once globally. The
//      expected value stated in the task (the true global incident-face count)
//      is the one this loop produces.
//   3. VECTOR FIELD, COMPONENTWISE. Repeat check 2 with the double[3] field,
//      adding each owned face's centroid to its three corners, compared against
//      the replicated-reference assembly to 1e-14 relative. Catches a component
//      stride bug that check 2 cannot see.
//   4. GHOSTS ARE UNTOUCHED. Every ghost slot is bitwise unchanged across the
//      check-2 scatter (stronger than "<= the owner's"), and at ranks >= 2 at
//      least one boundary vertex's ghost partial is strictly less than the
//      owner's total, so the check is not vacuous. A following haloExchange then
//      makes every ghost equal its owner.
//   5. DOUBLE-COUNTING IS REAL. A second haloScatterAdd with no intervening
//      haloExchange yields the pinned double-counted value
//      2*total - (the owner's own partial), computed from the reference. Pins the
//      non-idempotence as a tested contract rather than a surprise.
//   6. DETERMINISM WITHIN A RUN. Re-zero, re-assemble and re-scatter the VECTOR
//      field (the case with real floating-point summation) and assert BITWISE
//      equality of the owned values with the first pass.
//   7. ROUND-TRIP IDENTITY. A field uniform across all ranks (set on owned only,
//      then haloExchange) scatter-adds to value * multiplicity -- a cheap
//      cross-check between the two directions' index lists that catches a
//      swapped send_idx/recv_idx.
//   8. SINGLE RANK. Empty plan, so the field is bitwise unchanged.
//   9. EDGES AND FACES via the kind-named wrappers. Check 1 on an edge user
//      field and a face user field, against the reference edge/face
//      multiplicities. A face is never shared as an owned duplicate and is
//      ghosted by fewer ranks than a vertex, so its expected multiplicity is
//      a different number -- derived from the reference, not assumed.
//  10. DEPTH 2. Check 1 at distribute(..., depth=2): the multiplicities grow and
//      the reference ground truth still matches.
//
// Runs on host (Serial) and device (default, HIP), ranks 1-5.

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <type_traits>
#include <vector>

using namespace Tessera;

namespace
{

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
// The replicated reference: the whole mesh, as it is BEFORE distribute()
// ===========================================================================
//
// gid == index here, so every array below is indexed by gid.

struct Reference
{
    int Nv = 0, Ne = 0, Nf = 0;
    std::vector<int> fv;           // 3*Nf face corner gids
    std::vector<int> fe;           // 3*Nf face edge gids
    std::vector<int> ef;           // 2*Ne incident face gids (-1 if none)
    std::vector<double> pos;       // 3*Nv positions
    std::vector<int> vfOff, vfNbr; // vertex -> incident faces CSR
};

template <class MeshT>
Reference extractReference( MeshT& mesh )
{
    Reference r;
    r.Nv = static_cast<int>( mesh.numVertices() );
    r.Ne = static_cast<int>( mesh.numEdges() );
    r.Nf = static_cast<int>( mesh.numFaces() );

    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "ref_hv", r.Nv );
    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
        "ref_he", r.Ne );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "ref_hf", r.Nf );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( he, mesh.edges() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto p = Cabana::slice<VertexField::Position>( hv );
    auto e_f = Cabana::slice<EdgeField::Faces>( he );
    auto f_v = Cabana::slice<FaceField::Verts>( hf );
    auto f_e = Cabana::slice<FaceField::Edges>( hf );

    r.pos.resize( 3 * static_cast<std::size_t>( r.Nv ) );
    for ( int v = 0; v < r.Nv; ++v )
        for ( int d = 0; d < 3; ++d )
            r.pos[3 * static_cast<std::size_t>( v ) + d] =
                static_cast<double>( p( v, d ) );

    r.ef.resize( 2 * static_cast<std::size_t>( r.Ne ) );
    for ( int e = 0; e < r.Ne; ++e )
        for ( int k = 0; k < 2; ++k )
            r.ef[2 * static_cast<std::size_t>( e ) + k] =
                ( e_f( e, k ) == invalid_gid )
                    ? -1
                    : static_cast<int>( e_f( e, k ) );

    r.fv.resize( 3 * static_cast<std::size_t>( r.Nf ) );
    r.fe.resize( 3 * static_cast<std::size_t>( r.Nf ) );
    for ( int f = 0; f < r.Nf; ++f )
        for ( int k = 0; k < 3; ++k )
        {
            r.fv[3 * static_cast<std::size_t>( f ) + k] =
                static_cast<int>( f_v( f, k ) );
            r.fe[3 * static_cast<std::size_t>( f ) + k] =
                static_cast<int>( f_e( f, k ) );
        }

    auto off = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), mesh.vertexFaces().offsets );
    auto nbr = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), mesh.vertexFaces().neighbors );
    r.vfOff.resize( off.extent( 0 ) );
    for ( std::size_t i = 0; i < off.extent( 0 ); ++i )
        r.vfOff[i] = static_cast<int>( off( i ) );
    r.vfNbr.resize( nbr.extent( 0 ) );
    for ( std::size_t i = 0; i < nbr.extent( 0 ); ++i )
        r.vfNbr[i] = static_cast<int>( nbr( i ) );
    return r;
}

// ===========================================================================
// Reference ownership + every rank's local set, and the multiplicities
// ===========================================================================
//
// This is an INDEPENDENT re-derivation of distribute()'s documented rules
// (Tessera_Distribute.hpp): ownership is the lowest incident face owner, and the
// local set is the `depth`-deep closure of the owned vertices, one iteration
// being { local faces += faces incident to a local vertex; local verts/edges +=
// the vertices/edges of the local faces }, seeded from the owned faces and the
// owned vertices.

struct RefSets
{
    int size = 1;
    std::vector<Rank> vOwner, eOwner;     // by gid
    std::vector<char> inV, inE, inF;      // [r * N + gid]
    std::vector<int> multV, multE, multF; // ranks holding a copy, by gid
};

RefSets referenceSets( const Reference& r, const std::vector<Rank>& faceOwner,
                       int size, int depth )
{
    RefSets s;
    s.size = size;
    const std::size_t Nv = static_cast<std::size_t>( r.Nv );
    const std::size_t Ne = static_cast<std::size_t>( r.Ne );
    const std::size_t Nf = static_cast<std::size_t>( r.Nf );

    s.vOwner.assign( Nv, 0 );
    for ( int v = 0; v < r.Nv; ++v )
    {
        Rank m = static_cast<Rank>( size );
        for ( int p = r.vfOff[v]; p < r.vfOff[v + 1]; ++p )
            m = std::min( m,
                          faceOwner[static_cast<std::size_t>( r.vfNbr[p] )] );
        s.vOwner[v] = ( m == static_cast<Rank>( size ) ) ? 0 : m;
    }
    s.eOwner.assign( Ne, 0 );
    for ( int e = 0; e < r.Ne; ++e )
    {
        Rank m = faceOwner[static_cast<std::size_t>( r.ef[2 * e] )];
        if ( r.ef[2 * e + 1] >= 0 )
            m = std::min(
                m, faceOwner[static_cast<std::size_t>( r.ef[2 * e + 1] )] );
        s.eOwner[e] = m;
    }

    s.inV.assign( Nv * static_cast<std::size_t>( size ), 0 );
    s.inE.assign( Ne * static_cast<std::size_t>( size ), 0 );
    s.inF.assign( Nf * static_cast<std::size_t>( size ), 0 );
    const int nRings = ( depth > 0 ) ? depth : 1;
    for ( int R = 0; R < size; ++R )
    {
        char* inV = &s.inV[static_cast<std::size_t>( R ) * Nv];
        char* inE = &s.inE[static_cast<std::size_t>( R ) * Ne];
        char* inF = &s.inF[static_cast<std::size_t>( R ) * Nf];
        for ( int f = 0; f < r.Nf; ++f )
            if ( faceOwner[static_cast<std::size_t>( f )] ==
                 static_cast<Rank>( R ) )
                inF[f] = 1;
        for ( int v = 0; v < r.Nv; ++v )
            if ( s.vOwner[static_cast<std::size_t>( v )] ==
                 static_cast<Rank>( R ) )
                inV[v] = 1;
        for ( int d = 0; d < nRings; ++d )
        {
            bool grew = false;
            for ( int v = 0; v < r.Nv; ++v )
                if ( inV[v] )
                    for ( int p = r.vfOff[v]; p < r.vfOff[v + 1]; ++p )
                        if ( !inF[r.vfNbr[p]] )
                        {
                            inF[r.vfNbr[p]] = 1;
                            grew = true;
                        }
            for ( int f = 0; f < r.Nf; ++f )
                if ( inF[f] )
                    for ( int k = 0; k < 3; ++k )
                    {
                        const int vg =
                            r.fv[3 * static_cast<std::size_t>( f ) + k];
                        if ( !inV[vg] )
                        {
                            inV[vg] = 1;
                            grew = true;
                        }
                        inE[r.fe[3 * static_cast<std::size_t>( f ) + k]] = 1;
                    }
            if ( !grew )
                break;
        }
    }

    auto mult = []( const std::vector<char>& in, std::size_t N, int nr )
    {
        std::vector<int> m( N, 0 );
        for ( int R = 0; R < nr; ++R )
            for ( std::size_t i = 0; i < N; ++i )
                if ( in[static_cast<std::size_t>( R ) * N + i] )
                    ++m[i];
        return m;
    };
    s.multV = mult( s.inV, Nv, size );
    s.multE = mult( s.inE, Ne, size );
    s.multF = mult( s.inF, Nf, size );
    return s;
}

//! True GLOBAL incident-face count of vertex gid `v` on the reference.
inline int refIncidentFaces( const Reference& r, int v )
{
    return r.vfOff[v + 1] - r.vfOff[v];
}

//! Faces incident on vertex `v` that rank `o` OWNS -- the owner's own partial in
//! the check-2 assembly, needed for check 5's pinned double-counted value.
inline int refOwnedIncidentFaces( const Reference& r,
                                  const std::vector<Rank>& faceOwner, int v,
                                  Rank o )
{
    int n = 0;
    for ( int p = r.vfOff[v]; p < r.vfOff[v + 1]; ++p )
        if ( faceOwner[static_cast<std::size_t>( r.vfNbr[p] )] == o )
            ++n;
    return n;
}

//! The reference face-centroid assembly: sum of the centroids of ALL faces
//! incident on each vertex, on the whole global mesh.
std::vector<double> refCentroidAssembly( const Reference& r )
{
    std::vector<double> out( 3 * static_cast<std::size_t>( r.Nv ), 0.0 );
    for ( int f = 0; f < r.Nf; ++f )
    {
        double c[3] = { 0, 0, 0 };
        for ( int k = 0; k < 3; ++k )
        {
            const int v = r.fv[3 * static_cast<std::size_t>( f ) + k];
            for ( int d = 0; d < 3; ++d )
                c[d] += r.pos[3 * static_cast<std::size_t>( v ) + d];
        }
        for ( int d = 0; d < 3; ++d )
            c[d] /= 3.0;
        for ( int k = 0; k < 3; ++k )
        {
            const int v = r.fv[3 * static_cast<std::size_t>( f ) + k];
            for ( int d = 0; d < 3; ++d )
                out[3 * static_cast<std::size_t>( v ) + d] += c[d];
        }
    }
    return out;
}

// ===========================================================================
// Host <-> device field helpers (Gid is member 0 for all three entity kinds)
// ===========================================================================

template <class AoSoAType>
std::vector<GlobalId> gidsOf( const AoSoAType& a )
{
    Cabana::AoSoA<typename AoSoAType::member_types, Kokkos::HostSpace> h(
        "gids_h", a.size() );
    Cabana::deep_copy( h, a );
    auto g = Cabana::slice<0>( h );
    std::vector<GlobalId> out( a.size() );
    for ( std::size_t i = 0; i < out.size(); ++i )
        out[i] = g( i );
    return out;
}

//! Read one scalar member of every local entity.
template <std::size_t F, class AoSoAType>
std::vector<double> readScalar( const AoSoAType& a )
{
    Cabana::AoSoA<typename AoSoAType::member_types, Kokkos::HostSpace> h(
        "rd_h", a.size() );
    Cabana::deep_copy( h, a );
    auto s = Cabana::slice<F>( h );
    std::vector<double> out( a.size() );
    for ( std::size_t i = 0; i < out.size(); ++i )
        out[i] = static_cast<double>( s( i ) );
    return out;
}

//! Write one scalar member of every local entity (other members untouched).
template <std::size_t F, class AoSoAType>
void writeScalar( AoSoAType& a, const std::vector<double>& vals )
{
    Cabana::AoSoA<typename AoSoAType::member_types, Kokkos::HostSpace> h(
        "wr_h", a.size() );
    Cabana::deep_copy( h, a );
    auto s = Cabana::slice<F>( h );
    for ( std::size_t i = 0; i < vals.size(); ++i )
        s( i ) = vals[i];
    Cabana::deep_copy( a, h );
}

//! Read a rank-1 member of extent 3, flattened as 3*i + c.
template <std::size_t F, class AoSoAType>
std::vector<double> readVec3( const AoSoAType& a )
{
    Cabana::AoSoA<typename AoSoAType::member_types, Kokkos::HostSpace> h(
        "rdv_h", a.size() );
    Cabana::deep_copy( h, a );
    auto s = Cabana::slice<F>( h );
    std::vector<double> out( 3 * a.size() );
    for ( std::size_t i = 0; i < a.size(); ++i )
        for ( int c = 0; c < 3; ++c )
            out[3 * i + c] = static_cast<double>( s( i, c ) );
    return out;
}

template <std::size_t F, class AoSoAType>
void writeVec3( AoSoAType& a, const std::vector<double>& vals )
{
    Cabana::AoSoA<typename AoSoAType::member_types, Kokkos::HostSpace> h(
        "wrv_h", a.size() );
    Cabana::deep_copy( h, a );
    auto s = Cabana::slice<F>( h );
    for ( std::size_t i = 0; i < a.size(); ++i )
        for ( int c = 0; c < 3; ++c )
            s( i, c ) = vals[3 * i + c];
    Cabana::deep_copy( a, h );
}

// ===========================================================================
// The run
// ===========================================================================

template <class ExecSpace>
int run( int rank, int size, const char* tag )
{
    using mem = typename ExecSpace::memory_space;
    // The mesh carries user fields so both a scalar and a vector member are
    // exercised, plus one scalar on edges and faces for check 9.
    using MeshT =
        Mesh<double, 3, VertexFields<double, double[3]>, EdgeFields<double>,
             FaceFields<double>, mem, ExecSpace, RefinementMode::Conforming>;
    constexpr std::size_t VU0 = userVertexField<0>(); // double
    constexpr std::size_t VU1 = userVertexField<1>(); // double[3]
    constexpr std::size_t EU0 = userEdgeField<0>();
    constexpr std::size_t FU0 = userFaceField<0>();

    MPI_Comm comm = MPI_COMM_WORLD;
    int fails = 0;

    // ---- build, snapshot the reference, distribute at depth 1 --------------
    MeshT mesh( comm );
    buildIcosphere( mesh, 2 );
    const Reference ref = extractReference( mesh );
    const std::vector<Rank> faceOwner = facePartitionByAxis( mesh );
    const RefSets sets = referenceSets( ref, faceOwner, size, 1 );

    MeshHalo<mem> halo;
    distribute( mesh, halo, faceOwner, 1 );
    haloExchange( mesh, halo );

    const std::vector<GlobalId> vgid = gidsOf( mesh.vertices() );
    const std::vector<GlobalId> egid = gidsOf( mesh.edges() );
    const std::vector<GlobalId> fgid = gidsOf( mesh.faces() );
    const std::size_t nV = mesh.numVertices(), nOV = mesh.numOwnedVertices();
    const std::size_t nF = mesh.numFaces(), nOF = mesh.numOwnedFaces();

    // GUARD ON THE REFERENCE ITSELF. If the re-derived local-set rule disagrees
    // with distribute()'s, every "ground truth" below is fiction -- so assert the
    // predicted local/owned counts per rank before using them.
    {
        long long pv = 0, pe = 0, pf = 0, ov = 0, oe = 0, of = 0;
        for ( int i = 0; i < ref.Nv; ++i )
        {
            if ( sets.inV[static_cast<std::size_t>( rank ) * ref.Nv + i] )
                ++pv;
            if ( sets.vOwner[i] == static_cast<Rank>( rank ) )
                ++ov;
        }
        for ( int i = 0; i < ref.Ne; ++i )
        {
            if ( sets.inE[static_cast<std::size_t>( rank ) * ref.Ne + i] )
                ++pe;
            if ( sets.eOwner[i] == static_cast<Rank>( rank ) )
                ++oe;
        }
        for ( int i = 0; i < ref.Nf; ++i )
        {
            if ( sets.inF[static_cast<std::size_t>( rank ) * ref.Nf + i] )
                ++pf;
            if ( faceOwner[static_cast<std::size_t>( i )] ==
                 static_cast<Rank>( rank ) )
                ++of;
        }
        if ( pv != (long long)mesh.numVertices() ||
             pe != (long long)mesh.numEdges() ||
             pf != (long long)mesh.numFaces() ||
             ov != (long long)mesh.numOwnedVertices() ||
             oe != (long long)mesh.numOwnedEdges() ||
             of != (long long)mesh.numOwnedFaces() )
        {
            printf( "[scatter_add] %s np%d rank %d REFERENCE MISMATCH: "
                    "pred V/E/F %lld/%lld/%lld owned %lld/%lld/%lld vs "
                    "mesh %zu/%zu/%zu owned %zu/%zu/%zu\n",
                    tag, size, rank, pv, pe, pf, ov, oe, of, mesh.numVertices(),
                    mesh.numEdges(), mesh.numFaces(), mesh.numOwnedVertices(),
                    mesh.numOwnedEdges(), mesh.numOwnedFaces() );
            ++fails;
        }
    }

    // ---- CHECK 1: ghost multiplicity ---------------------------------------
    {
        writeScalar<VU0>( mesh.vertices(), std::vector<double>( nV, 1.0 ) );
        haloScatterAddVertices<VU0>( mesh, halo );
        const std::vector<double> got = readScalar<VU0>( mesh.vertices() );
        int bad = 0;
        for ( std::size_t v = 0; v < nOV; ++v )
        {
            const double want = static_cast<double>(
                sets.multV[static_cast<std::size_t>( vgid[v] )] );
            if ( got[v] != want )
            {
                if ( bad == 0 )
                    printf( "[scatter_add] %s np%d check1 vertex gid %llu: "
                            "got %.17g want %.17g\n",
                            tag, size, (unsigned long long)vgid[v], got[v],
                            want );
                ++bad;
            }
        }
        fails += globalFails( comm, bad );
    }

    // ---- CHECK 2: face-loop assembly (+ 4, 5) ------------------------------
    // Loop OWNED faces, add 1.0 to each corner's LOCAL slot (a corner of an
    // owned face may well be a ghost), then push the ghost partials home.
    std::vector<double> assembled; // owned values after the first scatter
    {
        std::map<GlobalId, std::size_t> vlocal;
        for ( std::size_t v = 0; v < nV; ++v )
            vlocal[vgid[v]] = v;
        std::vector<double> u( nV, 0.0 );
        for ( std::size_t f = 0; f < nOF; ++f )
        {
            const int g = static_cast<int>( fgid[f] );
            for ( int k = 0; k < 3; ++k )
                u[vlocal.at( static_cast<GlobalId>(
                    ref.fv[3 * static_cast<std::size_t>( g ) + k] ) )] += 1.0;
        }
        writeScalar<VU0>( mesh.vertices(), u );
        const std::vector<double> before = readScalar<VU0>( mesh.vertices() );
        haloScatterAddVertices<VU0>( mesh, halo );
        const std::vector<double> got = readScalar<VU0>( mesh.vertices() );
        assembled.assign( got.begin(), got.begin() + nOV );

        int bad = 0;
        for ( std::size_t v = 0; v < nOV; ++v )
        {
            const double want = static_cast<double>(
                refIncidentFaces( ref, static_cast<int>( vgid[v] ) ) );
            if ( got[v] != want )
            {
                if ( bad == 0 )
                    printf( "[scatter_add] %s np%d check2 vertex gid %llu: "
                            "got %.17g want %.17g\n",
                            tag, size, (unsigned long long)vgid[v], got[v],
                            want );
                ++bad;
            }
        }
        fails += globalFails( comm, bad );

        // -- CHECK 4a: every ghost slot is bitwise unchanged -----------------
        int ghostChanged = 0;
        long long strictlyLess = 0;
        for ( std::size_t v = nOV; v < nV; ++v )
        {
            if ( got[v] != before[v] )
                ++ghostChanged;
            const double total = static_cast<double>(
                refIncidentFaces( ref, static_cast<int>( vgid[v] ) ) );
            if ( got[v] > total )
                ++ghostChanged; // a partial can never exceed the global total
            if ( got[v] < total )
                ++strictlyLess;
        }
        fails += globalFails( comm, ghostChanged );
        // Non-vacuity: at ranks >= 2 some boundary vertex must hold a strictly
        // partial ghost value, or check 4a proves nothing.
        if ( size > 1 && commSum( comm, strictlyLess ) <= 0 )
            ++fails;

        // -- CHECK 5: a second scatter double-counts, by a pinned amount -----
        haloScatterAddVertices<VU0>( mesh, halo );
        const std::vector<double> twice = readScalar<VU0>( mesh.vertices() );
        int bad5 = 0;
        for ( std::size_t v = 0; v < nOV; ++v )
        {
            const int g = static_cast<int>( vgid[v] );
            const double total =
                static_cast<double>( refIncidentFaces( ref, g ) );
            const double mine = static_cast<double>( refOwnedIncidentFaces(
                ref, faceOwner, g, static_cast<Rank>( rank ) ) );
            if ( twice[v] != 2.0 * total - mine )
                ++bad5;
        }
        fails += globalFails( comm, bad5 );

        // -- CHECK 4b: haloExchange makes every ghost equal its owner --------
        writeScalar<VU0>( mesh.vertices(), got ); // back to the check-2 state
        haloExchange( mesh, halo );
        const std::vector<double> synced = readScalar<VU0>( mesh.vertices() );
        int bad4 = 0;
        for ( std::size_t v = 0; v < nV; ++v )
        {
            const double want = static_cast<double>(
                refIncidentFaces( ref, static_cast<int>( vgid[v] ) ) );
            if ( synced[v] != want )
                ++bad4;
        }
        fails += globalFails( comm, bad4 );

        // -- CHECK 8: single rank -- empty plan, field bitwise unchanged -----
        if ( size == 1 )
        {
            if ( halo.vplan.totalSend() != 0 || halo.vplan.totalRecv() != 0 ||
                 halo.eplan.totalSend() != 0 || halo.fplan.totalSend() != 0 )
                ++fails;
            const std::vector<double> pre = readScalar<VU0>( mesh.vertices() );
            haloScatterAddVertices<VU0>( mesh, halo );
            const std::vector<double> post = readScalar<VU0>( mesh.vertices() );
            if ( std::memcmp( pre.data(), post.data(),
                              pre.size() * sizeof( double ) ) != 0 )
                ++fails;
        }
    }

    // ---- CHECK 3 + 6: vector field, componentwise, twice -------------------
    {
        std::map<GlobalId, std::size_t> vlocal;
        for ( std::size_t v = 0; v < nV; ++v )
            vlocal[vgid[v]] = v;
        const std::vector<double> want = refCentroidAssembly( ref );

        std::vector<double> pass[2];
        for ( int it = 0; it < 2; ++it )
        {
            std::vector<double> u( 3 * nV, 0.0 );
            for ( std::size_t f = 0; f < nOF; ++f )
            {
                const int g = static_cast<int>( fgid[f] );
                double c[3] = { 0, 0, 0 };
                for ( int k = 0; k < 3; ++k )
                {
                    const int v = ref.fv[3 * static_cast<std::size_t>( g ) + k];
                    for ( int d = 0; d < 3; ++d )
                        c[d] += ref.pos[3 * static_cast<std::size_t>( v ) + d];
                }
                for ( int d = 0; d < 3; ++d )
                    c[d] /= 3.0;
                for ( int k = 0; k < 3; ++k )
                {
                    const std::size_t l = vlocal.at( static_cast<GlobalId>(
                        ref.fv[3 * static_cast<std::size_t>( g ) + k] ) );
                    for ( int d = 0; d < 3; ++d )
                        u[3 * l + d] += c[d];
                }
            }
            writeVec3<VU1>( mesh.vertices(), u );
            haloScatterAddVertices<VU1>( mesh, halo );
            const std::vector<double> got = readVec3<VU1>( mesh.vertices() );
            pass[it].assign( got.begin(), got.begin() + 3 * nOV );
        }

        int bad = 0;
        for ( std::size_t v = 0; v < nOV; ++v )
            for ( int d = 0; d < 3; ++d )
            {
                const double w =
                    want[3 * static_cast<std::size_t>( vgid[v] ) + d];
                const double g = pass[0][3 * v + d];
                const double scale = std::max( 1.0, std::abs( w ) );
                if ( std::abs( g - w ) > 1e-14 * scale )
                {
                    if ( bad == 0 )
                        printf( "[scatter_add] %s np%d check3 vertex gid %llu "
                                "comp %d: got %.17g want %.17g\n",
                                tag, size, (unsigned long long)vgid[v], d, g,
                                w );
                    ++bad;
                }
            }
        fails += globalFails( comm, bad );

        // CHECK 6: bitwise determinism of the two identical passes.
        if ( pass[0].size() != pass[1].size() ||
             std::memcmp( pass[0].data(), pass[1].data(),
                          pass[0].size() * sizeof( double ) ) != 0 )
            ++fails;
    }

    // ---- CHECK 7: round-trip identity (haloExchange then scatter-add) ------
    {
        constexpr double kVal = 3.5; // exact in binary, so the product is too
        std::vector<double> u = readScalar<VU0>( mesh.vertices() );
        for ( std::size_t v = 0; v < nOV; ++v )
            u[v] = kVal;
        for ( std::size_t v = nOV; v < nV; ++v )
            u[v] = -7.0; // must be overwritten by the gather, not summed
        writeScalar<VU0>( mesh.vertices(), u );
        haloExchange( mesh, halo );
        haloScatterAddVertices<VU0>( mesh, halo );
        const std::vector<double> got = readScalar<VU0>( mesh.vertices() );
        int bad = 0;
        for ( std::size_t v = 0; v < nOV; ++v )
        {
            const double want =
                kVal * static_cast<double>(
                           sets.multV[static_cast<std::size_t>( vgid[v] )] );
            if ( got[v] != want )
                ++bad;
        }
        fails += globalFails( comm, bad );
    }

    // ---- CHECK 9: edges and faces via the kind-named wrappers --------------
    {
        writeScalar<EU0>( mesh.edges(),
                          std::vector<double>( mesh.numEdges(), 1.0 ) );
        haloScatterAddEdges<EU0>( mesh, halo );
        const std::vector<double> ge = readScalar<EU0>( mesh.edges() );
        int bad = 0;
        for ( std::size_t e = 0; e < mesh.numOwnedEdges(); ++e )
            if ( ge[e] !=
                 static_cast<double>(
                     sets.multE[static_cast<std::size_t>( egid[e] )] ) )
                ++bad;
        fails += globalFails( comm, bad );

        writeScalar<FU0>( mesh.faces(), std::vector<double>( nF, 1.0 ) );
        haloScatterAddFaces<FU0>( mesh, halo );
        const std::vector<double> gf = readScalar<FU0>( mesh.faces() );
        int badf = 0;
        for ( std::size_t f = 0; f < nOF; ++f )
            if ( gf[f] !=
                 static_cast<double>(
                     sets.multF[static_cast<std::size_t>( fgid[f] )] ) )
                ++badf;
        fails += globalFails( comm, badf );

        // A face is never an owned duplicate and is ghosted by fewer ranks than
        // a vertex, so its multiplicity distribution genuinely differs -- record
        // it rather than assume it.
        long long sumV = 0, sumE = 0, sumF = 0;
        for ( int i = 0; i < ref.Nv; ++i )
            sumV += sets.multV[i];
        for ( int i = 0; i < ref.Ne; ++i )
            sumE += sets.multE[i];
        for ( int i = 0; i < ref.Nf; ++i )
            sumF += sets.multF[i];
        if ( rank == 0 )
            printf( "[scatter_add] %-7s np%d multiplicity sums: V=%lld E=%lld "
                    "F=%lld (N %d/%d/%d)\n",
                    tag, size, sumV, sumE, sumF, ref.Nv, ref.Ne, ref.Nf );
    }

    // ---- CHECK 10: depth 2 -------------------------------------------------
    {
        MeshT m2( comm );
        buildIcosphere( m2, 2 );
        const Reference r2 = extractReference( m2 );
        const std::vector<Rank> fo2 = facePartitionByAxis( m2 );
        const RefSets s2 = referenceSets( r2, fo2, size, 2 );
        MeshHalo<mem> h2;
        distribute( m2, h2, fo2, 2 );
        haloExchange( m2, h2 );
        if ( h2.depth != 2 || m2.haloDepth() != 2 )
            ++fails;

        const std::vector<GlobalId> g2 = gidsOf( m2.vertices() );
        writeScalar<VU0>( m2.vertices(),
                          std::vector<double>( m2.numVertices(), 1.0 ) );
        haloScatterAddVertices<VU0>( m2, h2 );
        const std::vector<double> got = readScalar<VU0>( m2.vertices() );
        int bad = 0;
        for ( std::size_t v = 0; v < m2.numOwnedVertices(); ++v )
            if ( got[v] != static_cast<double>(
                               s2.multV[static_cast<std::size_t>( g2[v] )] ) )
                ++bad;
        fails += globalFails( comm, bad );

        // The wider closure must genuinely raise the multiplicities at ranks > 1
        // (otherwise depth 2 is silently depth 1 and check 10 is vacuous).
        long long sum1 = 0, sum2 = 0;
        for ( int i = 0; i < ref.Nv; ++i )
            sum1 += sets.multV[i];
        for ( int i = 0; i < r2.Nv; ++i )
            sum2 += s2.multV[i];
        if ( size > 1 && sum2 <= sum1 )
            ++fails;
        if ( rank == 0 )
            printf( "[scatter_add] %-7s np%d depth2: multV sum %lld -> %lld "
                    "fails=%d\n",
                    tag, size, sum1, sum2, fails );
    }

    if ( rank == 0 )
        printf( "[scatter_add] %-7s np%d fails=%d\n", tag, size, fails );
    return fails;
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

        fails += run<Kokkos::Serial>( rank, size, "Serial" );
        if constexpr ( !std::is_same<Kokkos::DefaultExecutionSpace,
                                     Kokkos::Serial>::value )
            fails +=
                run<Kokkos::DefaultExecutionSpace>( rank, size, "Default" );

        if ( rank == 0 )
            printf( "[scatter_add] TOTAL fails=%d\n", fails );
    }
    Kokkos::finalize();
    MPI_Finalize();
    return fails == 0 ? 0 : 1;
}
