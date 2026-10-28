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

// Unit test: MeshSeries writes one master .xmf that Paraview opens as a single
// timestepped dataset (T2 of tasks/fix-file-grouping-io.md).
//
// Builds a distributed icosphere carrying one vertex and one face user field,
// writes three frames through ONE MeshSeries at times 0.0, 0.5, 1.25, and
// asserts on rank 0 that the master is the XDMF temporal collection the design
// specifies:
//   - <master>.xmf exists and the <master>.xmf.tmp it was renamed from does
//     not (the rename-over is what keeps a Paraview reload from ever seeing a
//     half-written master),
//   - exactly one CollectionType="Temporal", and exactly three <Time Value=,
//     <Topology and <Geometry -- three FULL child grids, since an adaptive
//     series genuinely changes Nv/Nf frame to frame,
//   - the three <Time Value= parse to 0.0, 0.5, 1.25 in that order (%.17g in
//     the writer, so a double round-trips exactly through the text),
//   - each child grid references exactly its own frame's .h5 BASENAME via
//     Format="HDF" and no other .h5, and each of those files exists,
//   - the <Attribute Name= list is identical, in the same order, across all
//     three child grids. This is risk R1's stated diagnostic (Paraview flags an
//     attribute missing at some steps as "partial", and the reported failure
//     mode garbles the FIRST scalar in the file), and it is why the mesh here
//     carries user fields: no other committed test leaves a user-field .xmf on
//     disk, so the attribute loop had no in-repo coverage before this file,
//   - <master>.xmfindex holds exactly the three frames' "<stem> <time>" lines,
//     although a stale index was planted before the first write(): a new
//     series replaces the index rather than appending to an earlier run's,
//   - python3's xml.etree parses the master, i.e. it is well-formed XML.
//     Paraview/pvpython are not installed on this system, so the reader-level
//     confirmation is V1's, done by hand elsewhere.
//
// Failure direction: a fourth write() at the same time 1.25 (non-increasing)
// and a write() whose frame stem sits in a subdirectory of the master's
// directory (risk R5 -- an .xmf references .h5 by basename, so a master and
// its frames in different directories yield unresolvable references that
// Paraview reports as an empty dataset, not as a path error). Both must throw
// std::runtime_error and leave the already-written master BYTE-unchanged: the
// checks run before any I/O, on every rank, so the throw is symmetric.
//
// SERIAL, np1-3: the master is rank-0 text with no device involvement.

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Tessera;

// ---------------------------------------------------------------------------
// Small text helpers (the assertions are on the emitted XML text)
// ---------------------------------------------------------------------------

static std::string basenameOf( const std::string& path )
{
    const auto pos = path.find_last_of( "/\\" );
    return pos == std::string::npos ? path : path.substr( pos + 1 );
}

static bool fileExists( const std::string& path )
{
    std::FILE* fp = std::fopen( path.c_str(), "rb" );
    if ( !fp )
        return false;
    std::fclose( fp );
    return true;
}

//! Whole file as text, or "" if it cannot be read.
static std::string readWholeFile( const std::string& path )
{
    std::FILE* fp = std::fopen( path.c_str(), "rb" );
    if ( !fp )
        return std::string();
    std::string out;
    char buf[4096];
    std::size_t n;
    while ( ( n = std::fread( buf, 1, sizeof( buf ), fp ) ) > 0 )
        out.append( buf, n );
    std::fclose( fp );
    return out;
}

static std::size_t countOccurrences( const std::string& hay,
                                     const std::string& needle )
{
    std::size_t n = 0, pos = 0;
    while ( ( pos = hay.find( needle, pos ) ) != std::string::npos )
    {
        ++n;
        pos += needle.size();
    }
    return n;
}

//! Every value of an attribute spelled `<key>VALUE"` , in file order.
static std::vector<std::string> extractValues( const std::string& text,
                                               const std::string& key )
{
    std::vector<std::string> out;
    std::size_t pos = 0;
    while ( ( pos = text.find( key, pos ) ) != std::string::npos )
    {
        const std::size_t b = pos + key.size();
        const std::size_t e = text.find( '"', b );
        if ( e == std::string::npos )
            break;
        out.push_back( text.substr( b, e - b ) );
        pos = e + 1;
    }
    return out;
}

//! Split the master into one substring per child `<Grid ... GridType="Uniform">`.
static std::vector<std::string> childGrids( const std::string& master )
{
    const std::string open = "<Grid Name=\"Tessera\" GridType=\"Uniform\">";
    std::vector<std::size_t> starts;
    std::size_t pos = 0;
    while ( ( pos = master.find( open, pos ) ) != std::string::npos )
    {
        starts.push_back( pos );
        pos += open.size();
    }
    std::vector<std::string> out;
    for ( std::size_t i = 0; i < starts.size(); ++i )
    {
        const std::size_t end =
            ( i + 1 < starts.size() ) ? starts[i + 1] : master.size();
        out.push_back( master.substr( starts[i], end - starts[i] ) );
    }
    return out;
}

//! Every `<something>.h5` referenced by a `NAME.h5:` DataItem token in `text`.
static std::vector<std::string> h5References( const std::string& text )
{
    std::vector<std::string> out;
    const std::string tok = ".h5:";
    std::size_t pos = 0;
    while ( ( pos = text.find( tok, pos ) ) != std::string::npos )
    {
        // Walk back to the start of the file name (it follows whitespace).
        std::size_t b = pos;
        while ( b > 0 && !std::isspace( static_cast<unsigned char>(
                             text[b - 1] ) ) )
            --b;
        out.push_back( text.substr( b, pos + 3 - b ) );
        pos += tok.size();
    }
    return out;
}

//! A python3 that runs, or "" if none does. Reported as a failure rather than
//! skipped: a well-formedness check that cannot run must not look like a pass.
static std::string findPython3()
{
    const char* cands[] = { "python3", "/usr/tce/bin/python3" };
    for ( const char* c : cands )
    {
        const std::string probe =
            std::string( c ) + " -c \"pass\" >/dev/null 2>&1";
        if ( std::system( probe.c_str() ) == 0 )
            return std::string( c );
    }
    return std::string();
}

// ---------------------------------------------------------------------------

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
            std::printf( "test_xdmf_series: MeshSeries master .xmf temporal "
                         "collection (size %d)\n",
                         size );

        // Unique stem per executable+rankcount, in the cwd (which must be on a
        // SHARED filesystem -- /tmp is node-local on Tuolumne), overridable
        // with TESSERA_IO_TMPDIR. Same convention as test_io.cpp.
        const std::string exeName =
            argc > 0 ? basenameOf( argv[0] ) : "test_xdmf_series";
        std::string masterStem = exeName + "_np" + std::to_string( size );
        if ( const char* tmpdir = std::getenv( "TESSERA_IO_TMPDIR" ) )
            masterStem = std::string( tmpdir ) + "/" + masterStem;

        using Exec = Kokkos::Serial;
        using mem = Exec::memory_space;
        using Scalar = double;
        // One vertex and one face user field, so the master exercises the XDMF
        // user-field attribute loop (see the header comment / R1).
        using MeshT = Mesh<Scalar, 3, VertexFields<Scalar>, EdgeFields<>,
                           FaceFields<Scalar>, mem, Exec,
                           RefinementMode::HangingNode2to1>;

        MeshT mesh( MPI_COMM_WORLD );
        buildIcosphere( mesh, 2 );
        {
            auto gid = mesh.vertexSlice<VertexField::Gid>();
            auto vf = mesh.vertexSlice<userVertexField<0>()>();
            const int nv = static_cast<int>( mesh.numVertices() );
            Kokkos::parallel_for(
                "set_vfield", Kokkos::RangePolicy<Exec>( 0, nv ),
                KOKKOS_LAMBDA( const int i ) {
                    vf( i ) = static_cast<Scalar>(
                        ( gid( i ) * 2654435761ULL ) % 1009ULL );
                } );
            auto fgid = mesh.faceSlice<FaceField::Gid>();
            auto ff = mesh.faceSlice<userFaceField<0>()>();
            const int nf = static_cast<int>( mesh.numFaces() );
            Kokkos::parallel_for(
                "set_ffield", Kokkos::RangePolicy<Exec>( 0, nf ),
                KOKKOS_LAMBDA( const int i ) {
                    ff( i ) = static_cast<Scalar>(
                        ( fgid( i ) * 40503ULL + 7 ) % 997ULL );
                } );
            Kokkos::fence();
        }

        auto faceOwner = facePartitionByAxis( mesh );
        MeshHalo<mem> halo;
        distribute( mesh, halo, faceOwner );

        // ---- Three frames through ONE series ------------------------------
        const double times[3] = { 0.0, 0.5, 1.25 };
        std::vector<std::string> frameStems;
        for ( int i = 0; i < 3; ++i )
            frameStems.push_back( masterStem + "_frame" +
                                  std::to_string( i ) );

        // A stale index from an "earlier run" into the same directory. A new
        // MeshSeries must replace it on its first write(), so the restart
        // record never carries the earlier run's frames.
        const std::string indexPath = masterStem + ".xmfindex";
        if ( rank == 0 )
        {
            std::FILE* fp = std::fopen( indexPath.c_str(), "w" );
            if ( fp )
            {
                std::fprintf( fp, "%s_stale0 0\n%s_stale1 7\n",
                              masterStem.c_str(), masterStem.c_str() );
                std::fclose( fp );
            }
        }
        MPI_Barrier( MPI_COMM_WORLD );

        MeshSeries series( masterStem );
        for ( int i = 0; i < 3; ++i )
            series.write( mesh, frameStems[static_cast<std::size_t>( i )],
                          times[i] );

        if ( series.numFrames() != 3 )
            ++fails; // rank-uniform accumulator: true on EVERY rank
        if ( series.masterStem() != masterStem )
            ++fails;

        MPI_Barrier( MPI_COMM_WORLD );

        const std::string masterXmf = masterStem + ".xmf";
        std::string masterText;
        int r0fails = 0;
        if ( rank == 0 )
        {
            if ( !fileExists( masterXmf ) )
            {
                ++r0fails;
                std::printf( "  FAIL: master %s does not exist\n",
                             masterXmf.c_str() );
            }
            if ( fileExists( masterXmf + ".tmp" ) )
            {
                ++r0fails;
                std::printf( "  FAIL: %s.tmp was left behind\n",
                             masterXmf.c_str() );
            }

            masterText = readWholeFile( masterXmf );

            const std::size_t nColl =
                countOccurrences( masterText, "CollectionType=\"Temporal\"" );
            const std::size_t nTopo =
                countOccurrences( masterText, "<Topology" );
            const std::size_t nGeom =
                countOccurrences( masterText, "<Geometry" );
            const auto timeVals = extractValues( masterText, "<Time Value=\"" );
            if ( nColl != 1 )
            {
                ++r0fails;
                std::printf( "  FAIL: %zu CollectionType=\"Temporal\", "
                             "expected 1\n",
                             nColl );
            }
            if ( timeVals.size() != 3 )
            {
                ++r0fails;
                std::printf( "  FAIL: %zu <Time Value=, expected 3\n",
                             timeVals.size() );
            }
            if ( nTopo != 3 || nGeom != 3 )
            {
                ++r0fails;
                std::printf( "  FAIL: %zu <Topology / %zu <Geometry, expected "
                             "3 / 3\n",
                             nTopo, nGeom );
            }
            for ( std::size_t i = 0; i < timeVals.size() && i < 3; ++i )
            {
                const double v = std::strtod( timeVals[i].c_str(), nullptr );
                if ( v != times[i] )
                {
                    ++r0fails;
                    std::printf( "  FAIL: <Time Value=\"%s\"> at step %zu "
                                 "parses to %.17g, expected %.17g\n",
                                 timeVals[i].c_str(), i, v, times[i] );
                }
            }

            // Per child grid: its own .h5 basename and nobody else's, and the
            // identical attribute list in the identical order.
            const auto grids = childGrids( masterText );
            if ( grids.size() != 3 )
            {
                ++r0fails;
                std::printf( "  FAIL: %zu child grids, expected 3\n",
                             grids.size() );
            }
            std::vector<std::string> attrs0;
            for ( std::size_t i = 0;
                  i < grids.size() && i < frameStems.size(); ++i )
            {
                const std::string expect =
                    basenameOf( frameStems[i] ) + ".h5";
                const auto refs = h5References( grids[i] );
                if ( countOccurrences( grids[i], "Format=\"HDF\"" ) == 0 ||
                     refs.empty() )
                {
                    ++r0fails;
                    std::printf( "  FAIL: child grid %zu has no Format=\"HDF\" "
                                 "reference\n",
                                 i );
                }
                for ( const auto& r : refs )
                    if ( r != expect )
                    {
                        ++r0fails;
                        std::printf( "  FAIL: child grid %zu references '%s', "
                                     "expected '%s'\n",
                                     i, r.c_str(), expect.c_str() );
                        break;
                    }
                if ( !fileExists( frameStems[i] + ".h5" ) )
                {
                    ++r0fails;
                    std::printf( "  FAIL: referenced %s.h5 does not exist\n",
                                 frameStems[i].c_str() );
                }

                const auto attrs =
                    extractValues( grids[i], "<Attribute Name=\"" );
                if ( i == 0 )
                    attrs0 = attrs;
                else if ( attrs != attrs0 )
                {
                    // R1's diagnostic: a drifting attribute set is what
                    // Paraview reports as "partial" arrays / garbled values.
                    ++r0fails;
                    std::printf( "  FAIL: child grid %zu attribute set differs "
                                 "from grid 0 (%zu vs %zu names)\n",
                                 i, attrs.size(), attrs0.size() );
                }
            }
            if ( attrs0.empty() )
            {
                ++r0fails;
                std::printf( "  FAIL: no <Attribute Name= in the master\n" );
            }

            // The index holds exactly this series' three frames, in order --
            // the stale lines planted before the first write() are gone.
            {
                std::string expect;
                char line[64];
                for ( int i = 0; i < 3; ++i )
                {
                    std::snprintf( line, sizeof( line ), " %.17g\n", times[i] );
                    expect += frameStems[static_cast<std::size_t>( i )] + line;
                }
                if ( readWholeFile( indexPath ) != expect )
                {
                    ++r0fails;
                    std::printf( "  FAIL: %s does not hold exactly this "
                                 "series' 3 frames (stale lines kept?)\n",
                                 indexPath.c_str() );
                }
            }

            // Well-formed XML, checked by an independent parser.
            const std::string py = findPython3();
            if ( py.empty() )
            {
                ++r0fails;
                std::printf( "  FAIL: no working python3 found, so the XML "
                             "well-formedness check could not run\n" );
            }
            else
            {
                const std::string cmd =
                    py + " -c \"import xml.etree.ElementTree as E; "
                         "E.parse('" +
                    masterXmf + "')\" >/dev/null 2>&1";
                if ( std::system( cmd.c_str() ) != 0 )
                {
                    ++r0fails;
                    std::printf( "  FAIL: %s is not well-formed XML\n",
                                 masterXmf.c_str() );
                }
            }
        }

        // ---- Failure direction --------------------------------------------
        // Both are rejected by validation that runs on EVERY rank before any
        // I/O, so every rank throws and the master on disk is untouched.
        MPI_Barrier( MPI_COMM_WORLD );
        {
            bool threw = false;
            try
            {
                series.write( mesh, masterStem + "_frame3", 1.25 );
            }
            catch ( const std::runtime_error& )
            {
                threw = true;
            }
            if ( !threw )
                ++fails;
            if ( series.numFrames() != 3 )
                ++fails;
            if ( rank == 0 )
            {
                if ( readWholeFile( masterXmf ) != masterText )
                {
                    ++r0fails;
                    std::printf( "  FAIL: rejected non-increasing time still "
                                 "changed the master\n" );
                }
                if ( fileExists( masterStem + "_frame3.h5" ) )
                {
                    ++r0fails;
                    std::printf( "  FAIL: rejected frame still wrote an .h5\n" );
                }
            }
        }
        MPI_Barrier( MPI_COMM_WORLD );
        {
            // Same directory rule as R5: a subdirectory is a different
            // directory, and the master could not reference it by basename.
            const std::string badStem =
                masterStem.substr( 0, masterStem.find_last_of( "/\\" ) + 1 ) +
                "sub/" + basenameOf( masterStem ) + "_frame3";
            bool threw = false;
            try
            {
                series.write( mesh, badStem, 2.0 );
            }
            catch ( const std::runtime_error& )
            {
                threw = true;
            }
            if ( !threw )
                ++fails;
            if ( series.numFrames() != 3 )
                ++fails;
            if ( rank == 0 && readWholeFile( masterXmf ) != masterText )
            {
                ++r0fails;
                std::printf( "  FAIL: rejected out-of-directory frame still "
                             "changed the master\n" );
            }
        }

        // r0fails is nonzero only on rank 0; the single Allreduce below folds
        // it in along with every rank's own count.
        fails += r0fails;

        int global = 0;
        MPI_Allreduce( &fails, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
        fails = global;

        if ( rank == 0 )
            std::printf( "  [Serial] xdmf series master %s (fails=%d)\n",
                         fails == 0 ? "ok" : "FAIL", fails );

        // Clean up: master, index, and every frame's .h5 + sidecar.
        MPI_Barrier( MPI_COMM_WORLD );
        if ( rank == 0 )
        {
            std::remove( masterXmf.c_str() );
            std::remove( indexPath.c_str() );
            for ( const auto& s : frameStems )
            {
                std::remove( ( s + ".h5" ).c_str() );
                std::remove( ( s + ".xmf" ).c_str() );
            }
        }
    }

    Kokkos::finalize();
    MPI_Finalize();
    return fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
