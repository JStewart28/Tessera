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

// Regression test: parallel HDF5 + XDMF writer/reader round-trip (Step 8).
// Builds a distributed icosphere with a non-trivial user field per vertex and
// per face, writes it, reads it back into a fresh mesh, and checks:
//   - ownership partition, owned-1-ring-local, owned Euler == 2 (uniform
//     coarse icosphere) on the round-tripped mesh,
//   - the rank-count-independent topology checksum is unchanged,
//   - the user-field checksums (vertex + face) are unchanged,
//   - the on-disk /*/gid BXOR (read serially by rank 0) equals the in-memory
//     checksum, demonstrating rank-count independence (the gate running
//     np1-5 confirms all five agree),
//   - a corrupt -> haloExchange -> restored-ghost check on the round-tripped
//     mesh (same as the distribute/migrate regression tests).
//
// A second case (Task 6 of tasks/completed/conforming-refinement.md) round-trips a
// RefinementMode::Conforming mesh after two adaptive refine rounds and checks
// that un-closing the read-back mesh recovers the RED layer bit-for-bit -- the
// closure bookkeeping is what makes a conforming file refinable again, and a
// file that dropped it would still pass every check above. See runConforming().
// Runs on host (Serial) and device (default, HIP), np1-5.

#include "MeshInvariants.hpp"

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <hdf5.h>
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

using namespace Tessera;

// Reinterpret a Scalar's raw bytes as an (up to 8-byte) integer for BXOR
// checksums -- exact for the small deterministic values this test writes.
template <class Scalar>
static inline unsigned long long bitsOf( Scalar v )
{
    unsigned long long u = 0;
    std::memcpy( &u, &v, sizeof( Scalar ) );
    return u;
}

template <class MeshT>
static unsigned long long vertexFieldChecksum( MeshT& mesh )
{
    using Scalar = typename MeshT::scalar_type;
    const std::size_t no = mesh.numOwnedVertices();
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> h(
        "h", mesh.numVertices() );
    Cabana::deep_copy( h, mesh.vertices() );
    auto f = Cabana::slice<Tessera::userVertexField<0>()>( h );
    unsigned long long local = 0;
    for ( std::size_t i = 0; i < no; ++i )
        local ^= bitsOf<Scalar>( f( i ) );
    unsigned long long global = 0;
    MPI_Allreduce( &local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR,
                   mesh.comm() );
    return global;
}

template <class MeshT>
static unsigned long long faceFieldChecksum( MeshT& mesh )
{
    using Scalar = typename MeshT::scalar_type;
    const std::size_t no = mesh.numOwnedFaces();
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> h(
        "h", mesh.numFaces() );
    Cabana::deep_copy( h, mesh.faces() );
    auto f = Cabana::slice<Tessera::userFaceField<0>()>( h );
    unsigned long long local = 0;
    for ( std::size_t i = 0; i < no; ++i )
        local ^= bitsOf<Scalar>( f( i ) );
    unsigned long long global = 0;
    MPI_Allreduce( &local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR,
                   mesh.comm() );
    return global;
}

// Corrupt every ghost tuple, halo-sync, verify restoration (as in
// test_distribute.cpp / test_migrate_mesh.cpp).
template <class AoSoAType, class Mem>
static int corrupt_sync_verify( MPI_Comm comm, int rank, AoSoAType& a,
                                std::size_t n_owned,
                                HaloExchangePlan<Mem>& plan )
{
    using exec = typename AoSoAType::execution_space;
    const int n = static_cast<int>( a.size() );
    const int no = static_cast<int>( n_owned );

    Cabana::AoSoA<typename AoSoAType::member_types, Kokkos::HostSpace> h0( "h0",
                                                                           n );
    Cabana::deep_copy( h0, a );
    auto g0 = Cabana::slice<0>( h0 );
    auto o0 = Cabana::slice<1>( h0 );
    std::vector<GlobalId> expg;
    std::vector<Rank> expo;
    for ( int i = no; i < n; ++i )
    {
        expg.push_back( g0( i ) );
        expo.push_back( o0( i ) );
    }

    auto g = Cabana::slice<0>( a );
    auto o = Cabana::slice<1>( a );
    Kokkos::parallel_for(
        "corrupt", Kokkos::RangePolicy<exec>( no, n ),
        KOKKOS_LAMBDA( const int i ) {
            g( i ) = ~static_cast<GlobalId>( 0 );
            o( i ) = -1;
        } );
    Kokkos::fence();

    haloExchange( comm, a, plan );

    Cabana::AoSoA<typename AoSoAType::member_types, Kokkos::HostSpace> h1( "h1",
                                                                           n );
    Cabana::deep_copy( h1, a );
    auto g1 = Cabana::slice<0>( h1 );
    auto o1 = Cabana::slice<1>( h1 );
    int fails = 0;
    for ( int i = 0; i < no; ++i )
        if ( o1( i ) != rank )
            ++fails;
    for ( int i = no; i < n; ++i )
    {
        if ( g1( i ) != expg[i - no] || o1( i ) != expo[i - no] )
            ++fails;
        if ( o1( i ) == -1 || g1( i ) == ~static_cast<GlobalId>( 0 ) )
            ++fails;
    }
    return fails;
}

// Serial (rank 0 only) BXOR of a whole /group/gid dataset, for the on-disk
// rank-count-independence check.
static unsigned long long readWholeGidChecksum( const std::string& file,
                                                const char* dsetPath )
{
    hid_t f = H5Fopen( file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
    hid_t dset = H5Dopen2( f, dsetPath, H5P_DEFAULT );
    hid_t space = H5Dget_space( dset );
    hsize_t dims[1] = { 0 };
    H5Sget_simple_extent_dims( space, dims, nullptr );
    std::vector<std::uint64_t> buf( dims[0] );
    if ( dims[0] > 0 )
        H5Dread( dset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 buf.data() );
    H5Sclose( space );
    H5Dclose( dset );
    H5Fclose( f );
    unsigned long long c = 0;
    for ( auto v : buf )
        c ^= v;
    return c;
}

template <class Exec>
int run( int rank, int size, const char* tag, const std::string& stem )
{
    using mem = typename Exec::memory_space;
    using Scalar = double;
    // Pinned to the hanging-node mode: this case covers the format-version-2
    // file WITHOUT the closure datasets, which is no longer the Mesh default.
    // runConforming() below is its conforming counterpart.
    using MeshT =
        Mesh<Scalar, 3, VertexFields<Scalar>, EdgeFields<>, FaceFields<Scalar>,
             mem, Exec, RefinementMode::HangingNode2to1>;

    MeshT mesh( MPI_COMM_WORLD );
    buildIcosphere( mesh, 3 );
    const long long NvG = static_cast<long long>( mesh.numVertices() );
    const long long NeG = static_cast<long long>( mesh.numEdges() );
    const long long NfG = static_cast<long long>( mesh.numFaces() );

    // Deterministic user fields (function of gid) set on the replicated mesh
    // so they carry through distribute() automatically.
    {
        auto gid = mesh.template vertexSlice<VertexField::Gid>();
        auto vf = mesh.template vertexSlice<userVertexField<0>()>();
        const int nv = static_cast<int>( mesh.numVertices() );
        Kokkos::parallel_for(
            "set_vfield", Kokkos::RangePolicy<Exec>( 0, nv ),
            KOKKOS_LAMBDA( const int i ) {
                vf( i ) = static_cast<Scalar>( ( gid( i ) * 2654435761ULL ) %
                                               1009ULL );
            } );
        auto fgid = mesh.template faceSlice<FaceField::Gid>();
        auto ff = mesh.template faceSlice<userFaceField<0>()>();
        const int nf = static_cast<int>( mesh.numFaces() );
        Kokkos::parallel_for(
            "set_ffield", Kokkos::RangePolicy<Exec>( 0, nf ),
            KOKKOS_LAMBDA( const int i ) {
                ff( i ) = static_cast<Scalar>( ( fgid( i ) * 40503ULL + 7 ) %
                                               997ULL );
            } );
        Kokkos::fence();
    }

    auto faceOwner = facePartitionByAxis( mesh );
    MeshHalo<mem> halo;
    distribute( mesh, halo, faceOwner );

    unsigned long long cv, ce, cf;
    TesseraTest::topologyChecksum( mesh, cv, ce, cf );
    const unsigned long long fcv = vertexFieldChecksum( mesh );
    const unsigned long long fcf = faceFieldChecksum( mesh );

    writeMesh( mesh, stem );
    MPI_Barrier( MPI_COMM_WORLD );

    MeshT mesh2( MPI_COMM_WORLD );
    MeshHalo<mem> halo2;
    readMesh( mesh2, halo2, stem );

    int fails = 0;
    {
        int f2 = TesseraTest::checkOwnershipPartition( mesh2, NvG, NeG, NfG );
        f2 += TesseraTest::owned1RingLocal( mesh2 );
        int g2 = 0;
        MPI_Allreduce( &f2, &g2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
        if ( g2 != 0 )
            ++fails;

        if ( TesseraTest::ownedEulerGlobal( mesh2 ) != 2 )
            ++fails;

        unsigned long long cv2, ce2, cf2;
        TesseraTest::topologyChecksum( mesh2, cv2, ce2, cf2 );
        if ( cv2 != cv || ce2 != ce || cf2 != cf )
            ++fails;

        if ( vertexFieldChecksum( mesh2 ) != fcv ||
             faceFieldChecksum( mesh2 ) != fcf )
            ++fails;
    }

    // On-disk rank-count independence: rank 0 reads the whole /*/gid
    // datasets serially and BXORs them; must equal the in-memory checksum
    // computed identically at every np.
    int diskFails = 0;
    if ( rank == 0 )
    {
        const std::string h5file = stem + ".h5";
        if ( readWholeGidChecksum( h5file, "/vertices/gid" ) != cv )
            ++diskFails;
        if ( readWholeGidChecksum( h5file, "/edges/gid" ) != ce )
            ++diskFails;
        if ( readWholeGidChecksum( h5file, "/faces/gid" ) != cf )
            ++diskFails;
    }
    int diskGlobal = 0;
    MPI_Allreduce( &diskFails, &diskGlobal, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    fails += diskGlobal;

    int haloFails = 0;
    haloFails += corrupt_sync_verify( mesh2.comm(), rank, mesh2.vertices(),
                                      mesh2.numOwnedVertices(), halo2.vplan );
    haloFails += corrupt_sync_verify( mesh2.comm(), rank, mesh2.edges(),
                                      mesh2.numOwnedEdges(), halo2.eplan );
    haloFails += corrupt_sync_verify( mesh2.comm(), rank, mesh2.faces(),
                                      mesh2.numOwnedFaces(), halo2.fplan );
    int haloGlobal = 0;
    MPI_Allreduce( &haloFails, &haloGlobal, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    fails += haloGlobal;

    if ( rank == 0 )
        std::printf( "  [%s] io round-trip %s (diskFails=%d haloFails=%d)\n",
                     tag, fails == 0 ? "ok" : "FAIL", diskGlobal, haloGlobal );

    // Clean up the written files.
    MPI_Barrier( MPI_COMM_WORLD );
    if ( rank == 0 )
    {
        std::remove( ( stem + ".h5" ).c_str() );
        std::remove( ( stem + ".xmf" ).c_str() );
    }

    int global = 0;
    MPI_Allreduce( &fails, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    return global;
}

// ===========================================================================
// Conforming case (Task 6): write -> read -> un-close recovers the RED layer
// ===========================================================================
//
// A conforming mesh's visible faces are the transient closure layer; the
// persistent thing is the RED layer un-close() derives from the two closure
// bookkeeping members. If those do not round-trip, a read-back mesh still looks
// fine -- same faces, same gids, Euler 2 -- and then silently un-closes to
// garbage on the next refine(). So this case checks the red layer directly, and
// then actually refines the read-back mesh.
//
// It also exercises the one thing the hanging-node path cannot hit: the
// reader's fresh dense-index block partition splits closure sibling groups
// across ranks, which migrate()'s local round S cannot see. readMesh() calls
// repairClosureCohesion() for exactly that, and checkSiblingCoresidency() below
// is what pins it.

//! Sum a LOCAL fail count into a global one.
static inline int globalFailsIo( int local )
{
    int g = 0;
    MPI_Allreduce( &local, &g, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    return g;
}

static inline unsigned long long mix64( unsigned long long x )
{
    x += 0x9E3779B97F4A7C15ULL;
    x = ( x ^ ( x >> 30 ) ) * 0xBF58476D1CE4E5B9ULL;
    x = ( x ^ ( x >> 27 ) ) * 0x94D049BB133111EBULL;
    return x ^ ( x >> 31 );
}

// Rank-count-independent identity of the whole global red layer: count, plus
// BXOR and SUM of a per-red-face hash over (gid, the three corner gids IN
// ORDER, level). Order is kept rather than sorted so a winding change fails
// too; XOR and SUM together catch both a duplicate and a swap.
template <class MeshT>
static void redLayerChecksum( MeshT& mesh, long long& n, unsigned long long& x,
                              unsigned long long& s )
{
    const std::vector<VisibleFace> visible =
        TesseraTest::ownedVisibleFaces( mesh );
    const UncloseResult un = unclose( visible );
    long long ln = static_cast<long long>( un.red.size() );
    unsigned long long lx = 0, ls = 0;
    for ( const RedFace& r : un.red )
    {
        unsigned long long h = mix64( r.gid );
        for ( int k = 0; k < 3; ++k )
            h = mix64( h ^ mix64( r.v[k] ) );
        h = mix64( h ^ static_cast<unsigned long long>( r.level ) );
        lx ^= h;
        ls += h;
    }
    MPI_Allreduce( &ln, &n, 1, MPI_LONG_LONG, MPI_SUM, mesh.comm() );
    MPI_Allreduce( &lx, &x, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, mesh.comm() );
    MPI_Allreduce( &ls, &s, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, mesh.comm() );
}

// Deterministic adaptive mask: owned faces whose gid is divisible by m. Same
// mask predicate the conforming refine tests use -- rank-count independent.
template <class MeshT>
static std::vector<char> gidMask( MeshT& mesh, int m )
{
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto g = Cabana::slice<FaceField::Gid>( hf );
    std::vector<char> mask( mesh.numOwnedFaces(), 0 );
    for ( std::size_t f = 0; f < mask.size(); ++f )
        mask[f] = ( g( f ) % m == 0 ) ? 1 : 0;
    return mask;
}

static long long fileSizeBytes( const std::string& path )
{
    std::FILE* fp = std::fopen( path.c_str(), "rb" );
    if ( !fp )
        return -1;
    std::fseek( fp, 0, SEEK_END );
    const long long n = static_cast<long long>( std::ftell( fp ) );
    std::fclose( fp );
    return n;
}

template <class Exec>
int runConforming( int rank, int size, const char* tag,
                   const std::string& stem )
{
    using mem = typename Exec::memory_space;
    using Scalar = double;
    using MeshT =
        Mesh<Scalar, 3, VertexFields<Scalar>, EdgeFields<>, FaceFields<Scalar>,
             mem, Exec, RefinementMode::Conforming>;

    int fails = 0;

    MeshT mesh( MPI_COMM_WORLD );
    buildIcosphere( mesh, 2 );
    {
        auto gid = mesh.template vertexSlice<VertexField::Gid>();
        auto vf = mesh.template vertexSlice<userVertexField<0>()>();
        const int nv = static_cast<int>( mesh.numVertices() );
        Kokkos::parallel_for(
            "set_vfield", Kokkos::RangePolicy<Exec>( 0, nv ),
            KOKKOS_LAMBDA( const int i ) {
                vf( i ) = static_cast<Scalar>( ( gid( i ) * 2654435761ULL ) %
                                               1009ULL );
            } );
        auto fgid = mesh.template faceSlice<FaceField::Gid>();
        auto ff = mesh.template faceSlice<userFaceField<0>()>();
        const int nf = static_cast<int>( mesh.numFaces() );
        Kokkos::parallel_for(
            "set_ffield", Kokkos::RangePolicy<Exec>( 0, nf ),
            KOKKOS_LAMBDA( const int i ) {
                ff( i ) = static_cast<Scalar>( ( fgid( i ) * 40503ULL + 7 ) %
                                               997ULL );
            } );
        Kokkos::fence();
    }

    auto faceOwner = facePartitionByAxis( mesh );
    MeshHalo<mem> halo;
    distribute( mesh, halo, faceOwner );

    // Two adaptive rounds, so the written file really carries a closure layer
    // (a uniform or empty mask would emit no closure children at all and the
    // round-trip would prove nothing).
    long long closureChildren = 0;
    for ( int round = 0; round < 2; ++round )
    {
        auto res = refine( mesh, halo, gidMask( mesh, round == 0 ? 7 : 5 ) );
        long long local = res.closure.nClosureChildren, g = 0;
        MPI_Allreduce( &local, &g, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD );
        closureChildren += g;
        std::vector<Rank> dest( mesh.numOwnedFaces(),
                                static_cast<Rank>( rank ) );
        migrate( mesh, halo, dest );
        haloExchange( mesh, halo );
    }
    if ( closureChildren <= 0 )
        ++fails; // vacuous: nothing was closed, so nothing closure-shaped ships

    const long long NvG = TesseraTest::globalOwnedVertices( mesh );
    const long long NeG = TesseraTest::globalOwnedEdges( mesh );
    const long long NfG = TesseraTest::globalOwnedFaces( mesh );
    unsigned long long cv, ce, cf;
    TesseraTest::topologyChecksum( mesh, cv, ce, cf );
    const unsigned long long fcv = vertexFieldChecksum( mesh );
    const unsigned long long fcf = faceFieldChecksum( mesh );
    long long redN;
    unsigned long long redX, redS;
    redLayerChecksum( mesh, redN, redX, redS );

    writeMesh( mesh, stem );
    MPI_Barrier( MPI_COMM_WORLD );
    const long long bytes = ( rank == 0 ) ? fileSizeBytes( stem + ".h5" ) : 0;

    MeshT mesh2( MPI_COMM_WORLD );
    MeshHalo<mem> halo2;
    readMesh( mesh2, halo2, stem );

    int local = TesseraTest::checkOwnershipPartition( mesh2, NvG, NeG, NfG );
    local += TesseraTest::owned1RingLocal( mesh2 );
    local += TesseraTest::checkConforming( mesh2 );
    local += TesseraTest::checkSiblingCoresidency( mesh2 );
    local += TesseraTest::check21BalanceRed( mesh2 );
    if ( globalFailsIo( local ) != 0 )
        ++fails;
    if ( TesseraTest::checkOwnedEuler( mesh2 ) != 2 )
        ++fails;

    unsigned long long cv2, ce2, cf2;
    TesseraTest::topologyChecksum( mesh2, cv2, ce2, cf2 );
    if ( cv2 != cv || ce2 != ce || cf2 != cf )
        ++fails;
    if ( vertexFieldChecksum( mesh2 ) != fcv ||
         faceFieldChecksum( mesh2 ) != fcf )
        ++fails; // also the UserBegin-shift canary: userFaceField<0>() must not
                 // have moved under the two appended closure members

    // THE criterion for this case: the red layer survives the round trip.
    long long redN2;
    unsigned long long redX2, redS2;
    redLayerChecksum( mesh2, redN2, redX2, redS2 );
    if ( redN2 != redN || redX2 != redX || redS2 != redS )
        ++fails;

    // ... and is usable: one more conforming refine on the read-back mesh.
    {
        auto res = refine( mesh2, halo2, gidMask( mesh2, 5 ) );
        int l = TesseraTest::checkConforming( mesh2 );
        l += TesseraTest::checkClosureInverse( mesh2, res.midpoints );
        l += TesseraTest::check21BalanceRed( mesh2 );
        if ( globalFailsIo( l ) != 0 )
            ++fails;
        if ( TesseraTest::checkOwnedEuler( mesh2 ) != 2 )
            ++fails;
    }

    if ( rank == 0 )
    {
        // Computed on-disk delta of the two closure datasets: 1 + 3 uint64 per
        // OWNED face. Task 8 reports the measured whole-file figures.
        const long long delta =
            NfG * static_cast<long long>( 4 * sizeof( GlobalId ) );
        std::printf( "  [%s] io conforming round-trip %s (redFaces=%lld "
                     "visibleF=%lld closureChildren=%lld h5=%lld B, closure "
                     "payload=%lld B = %.1f%%)\n",
                     tag, fails == 0 ? "ok" : "FAIL", redN, NfG,
                     closureChildren, bytes, delta,
                     bytes > 0 ? 100.0 * static_cast<double>( delta ) /
                                     static_cast<double>( bytes )
                               : 0.0 );
    }

    MPI_Barrier( MPI_COMM_WORLD );
    if ( rank == 0 )
    {
        std::remove( ( stem + ".h5" ).c_str() );
        std::remove( ( stem + ".xmf" ).c_str() );
    }

    int global = 0;
    MPI_Allreduce( &fails, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    (void)size;
    return global;
}

static std::string basenameOf( const std::string& path )
{
    const auto pos = path.find_last_of( "/\\" );
    return pos == std::string::npos ? path : path.substr( pos + 1 );
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
            std::printf( "test_io: parallel HDF5 + XDMF round-trip (size %d)\n",
                         size );

        // Unique stem per executable+rankcount (SERIAL/HIP exes differ,
        // np differs => unique), avoiding concurrent-run collisions.
        const std::string exeName =
            argc > 0 ? basenameOf( argv[0] ) : "test_io";
        std::string stem = exeName + "_np" + std::to_string( size );
        if ( const char* tmpdir = std::getenv( "TESSERA_IO_TMPDIR" ) )
            stem = std::string( tmpdir ) + "/" + stem;

        fails += run<Kokkos::Serial>( rank, size, "Serial", stem + "_serial" );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            fails += run<Kokkos::DefaultExecutionSpace>( rank, size, "Default",
                                                         stem + "_default" );

        fails += runConforming<Kokkos::Serial>( rank, size, "Serial",
                                                stem + "_conf_serial" );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            fails += runConforming<Kokkos::DefaultExecutionSpace>(
                rank, size, "Default", stem + "_conf_default" );
    }

    Kokkos::finalize();
    MPI_Finalize();
    return fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
