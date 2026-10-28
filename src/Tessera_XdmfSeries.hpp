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

#ifndef TESSERA_XDMF_SERIES_HPP
#define TESSERA_XDMF_SERIES_HPP

// MeshSeries -- the caller-facing handle that turns N writeMesh() frames into
// ONE Paraview dataset with N timesteps.
//
// writeMesh() cannot know a series exists: it is handed a stem and a time, no
// frame index and no state between calls, and the caller invents the stems. So
// the only place in the system that knows which files form a sequence, and in
// what time order, is the caller -- which is why the grouping fix is a
// caller-held object rather than a change inside writeXdmf().
//
// MeshSeries accumulates one XdmfTimeStep per frame and REWRITES the master
// <masterStem>.xmf after every frame, so the master on disk always describes
// the frames that actually exist: a run killed at frame 40 leaves a master with
// 40 timesteps, not a missing or truncated file. The alternatives were both
// worse -- leaving the file unterminated until finalize leaves invalid XML that
// Paraview refuses outright, and writing the master only in a finalize call
// yields no master at all for a killed run, which is routine on long HPC runs.
// The cost is O(frames) rank-0 text per frame, immaterial against a collective
// HDF5 write.
//
// Per-frame .xmf sidecars are still written, so any single frame stays
// individually inspectable; the output directory holds N+1 openable .xmf files
// and the master is the one to open (it is the only one with no frame index in
// its name). The per-frame .h5 layout is unchanged, so readMesh() still reads
// any individual frame.

#include "Tessera_HDF5Writer.hpp"
#include "Tessera_Xdmf.hpp"

#include <hdf5.h>
#include <mpi.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Tessera
{

namespace detail
{

//! The directory component of `path` -- "" when there is none. Compared between
//! a frame stem and the master stem, because an .xmf can only reference .h5
//! files in its own directory.
inline std::string xdmfDirname( const std::string& path )
{
    const auto pos = path.find_last_of( "/\\" );
    return pos == std::string::npos ? std::string() : path.substr( 0, pos );
}

//! A `<frameStem>.h5` opened SERIALLY (H5P_DEFAULT, not MPI-IO) to read back
//! the root attributes a series reopen needs. Unlike readIntAttr/readU64Attr
//! in Tessera_IoCommon.hpp, which return 0 for a missing attribute, every
//! failure here throws, naming the file and the attribute: a frame rebuilt
//! from a zero Nv would describe an empty grid without complaint.
class SeriesH5Reader
{
  public:
    explicit SeriesH5Reader( std::string path )
        : _path( std::move( path ) )
    {
        _file = H5Fopen( _path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
        if ( _file < 0 )
            throw std::runtime_error( "cannot open frame file '" + _path +
                                      "'" );
    }
    ~SeriesH5Reader() { H5Fclose( _file ); }
    SeriesH5Reader( const SeriesH5Reader& ) = delete;
    SeriesH5Reader& operator=( const SeriesH5Reader& ) = delete;

    int readInt( const std::string& name ) const
    {
        int value = 0;
        read( name, H5T_NATIVE_INT, &value );
        return value;
    }
    std::uint64_t readU64( const std::string& name ) const
    {
        std::uint64_t value = 0;
        read( name, H5T_NATIVE_UINT64, &value );
        return value;
    }

  private:
    void read( const std::string& name, hid_t memType, void* out ) const
    {
        const hid_t attr = H5Aopen( _file, name.c_str(), H5P_DEFAULT );
        if ( attr < 0 )
            throw std::runtime_error( "cannot open root attribute '" + name +
                                      "' in frame file '" + _path + "'" );
        const herr_t rc = H5Aread( attr, memType, out );
        H5Aclose( attr );
        if ( rc < 0 )
            throw std::runtime_error( "cannot read root attribute '" + name +
                                      "' in frame file '" + _path + "'" );
    }

    std::string _path;
    hid_t _file = -1;
};

//! Rebuild the XdmfFrame writeMeshH5() returned for `<frameStem>.h5`, from the
//! root attributes it wrote. Field datasets and display names follow the
//! writer: dataset "u<j>", display name "v"/"f" + "u<j>".
inline XdmfFrame readSeriesFrame( const std::string& frameStem )
{
    const SeriesH5Reader h5( frameStem + ".h5" );
    XdmfFrame f;
    f.h5name = xdmfBasename( frameStem ) + ".h5";
    f.dim = h5.readInt( "dim" );
    f.scalarBytes = h5.readInt( "scalar_bytes" );
    f.Nv = static_cast<unsigned long long>( h5.readU64( "Nv" ) );
    f.Nf = static_cast<unsigned long long>( h5.readU64( "Nf" ) );
    const int nUV = h5.readInt( "n_user_v_fields" );
    for ( int j = 0; j < nUV; ++j )
    {
        const std::string name = "u" + std::to_string( j );
        f.vFields.push_back(
            { name, "v" + name, h5.readInt( "uv_ext_" + std::to_string( j ) ) } );
    }
    const int nUF = h5.readInt( "n_user_f_fields" );
    for ( int j = 0; j < nUF; ++j )
    {
        const std::string name = "u" + std::to_string( j );
        f.fFields.push_back(
            { name, "f" + name, h5.readInt( "uf_ext_" + std::to_string( j ) ) } );
    }
    return f;
}

//! Parse `<masterStem>.xmfindex` and rebuild every step it lists. Serial; the
//! reopen constructor calls it on one rank. Each line is
//! "<frameStem> %.17g\n" with the stem written verbatim, so it is split at
//! its LAST space -- the stem may itself contain spaces. Throws on a missing
//! or empty index, a line that does not parse (including a final line with no
//! newline, i.e. a torn append), a non-increasing time, or a frame stem whose
//! directory differs from the master's. Messages are relative to the index;
//! the caller prefixes its path.
inline std::vector<XdmfTimeStep> readSeriesIndex( const std::string& masterStem )
{
    const std::string idxname = masterStem + ".xmfindex";
    std::FILE* fp = std::fopen( idxname.c_str(), "rb" );
    if ( !fp )
        throw std::runtime_error( std::string( "cannot open it for reading (" ) +
                                  std::strerror( errno ) + ")" );
    std::string text;
    char buf[4096];
    std::size_t n;
    while ( ( n = std::fread( buf, 1, sizeof( buf ), fp ) ) > 0 )
        text.append( buf, n );
    const bool readErr = std::ferror( fp ) != 0;
    std::fclose( fp );
    if ( readErr )
        throw std::runtime_error( "error reading it" );
    if ( text.empty() )
        throw std::runtime_error( "it is empty, so there is no series to "
                                  "continue" );
    if ( text.back() != '\n' )
        throw std::runtime_error( "its last line has no terminating newline "
                                  "(a torn append?)" );

    const std::string masterDir = xdmfDirname( masterStem );
    std::vector<XdmfTimeStep> steps;
    std::size_t lineNo = 0;
    for ( std::size_t b = 0; b < text.size(); )
    {
        const std::size_t e = text.find( '\n', b );
        const std::string line = text.substr( b, e - b );
        b = e + 1;
        ++lineNo;
        const std::string where = "line " + std::to_string( lineNo ) + " '" +
                                  line + "'";

        const std::size_t sp = line.find_last_of( ' ' );
        if ( sp == std::string::npos || sp == 0 || sp + 1 == line.size() )
            throw std::runtime_error( where +
                                      " is not \"<frameStem> <time>\"" );
        const std::string frameStem = line.substr( 0, sp );
        const std::string timeText = line.substr( sp + 1 );
        char* end = nullptr;
        const double time = std::strtod( timeText.c_str(), &end );
        if ( end != timeText.c_str() + timeText.size() )
            throw std::runtime_error( where + ": time '" + timeText +
                                      "' is not a number" );

        if ( !steps.empty() && !( time > steps.back().time ) )
            throw std::runtime_error(
                where + ": time " + timeText +
                " is not strictly greater than the previous line's" );

        const std::string frameDir = xdmfDirname( frameStem );
        if ( frameDir != masterDir )
            throw std::runtime_error(
                where + ": frame stem '" + frameStem + "' is in directory '" +
                frameDir + "' but the master '" + masterStem + "' is in '" +
                masterDir + "'; they must match" );

        steps.push_back( XdmfTimeStep{ readSeriesFrame( frameStem ), time } );
    }
    return steps;
}

// Byte packing for broadcasting rebuilt steps from the reading rank. Every
// rank is the same build on the same machine, so raw object bytes suffice.
template <class T>
inline void packPod( std::string& buf, const T& v )
{
    buf.append( reinterpret_cast<const char*>( &v ), sizeof( T ) );
}
template <class T>
inline T unpackPod( const std::string& buf, std::size_t& pos )
{
    T v;
    std::memcpy( &v, buf.data() + pos, sizeof( T ) );
    pos += sizeof( T );
    return v;
}
inline void packString( std::string& buf, const std::string& s )
{
    packPod( buf, static_cast<std::uint64_t>( s.size() ) );
    buf += s;
}
inline std::string unpackString( const std::string& buf, std::size_t& pos )
{
    const auto n = static_cast<std::size_t>( unpackPod<std::uint64_t>( buf, pos ) );
    std::string s = buf.substr( pos, n );
    pos += n;
    return s;
}

inline void packSteps( std::string& buf, const std::vector<XdmfTimeStep>& steps )
{
    auto packFields = [&]( const std::vector<XdmfField>& fields )
    {
        packPod( buf, static_cast<std::uint64_t>( fields.size() ) );
        for ( const auto& fld : fields )
        {
            packString( buf, fld.dataset );
            packString( buf, fld.name );
            packPod( buf, fld.extent );
        }
    };
    packPod( buf, static_cast<std::uint64_t>( steps.size() ) );
    for ( const auto& s : steps )
    {
        packPod( buf, s.time );
        packString( buf, s.frame.h5name );
        packPod( buf, s.frame.dim );
        packPod( buf, s.frame.scalarBytes );
        packPod( buf, s.frame.Nv );
        packPod( buf, s.frame.Nf );
        packFields( s.frame.vFields );
        packFields( s.frame.fFields );
    }
}

inline std::vector<XdmfTimeStep> unpackSteps( const std::string& buf,
                                              std::size_t pos )
{
    auto unpackFields = [&]()
    {
        std::vector<XdmfField> fields(
            static_cast<std::size_t>( unpackPod<std::uint64_t>( buf, pos ) ) );
        for ( auto& fld : fields )
        {
            fld.dataset = unpackString( buf, pos );
            fld.name = unpackString( buf, pos );
            fld.extent = unpackPod<int>( buf, pos );
        }
        return fields;
    };
    std::vector<XdmfTimeStep> steps(
        static_cast<std::size_t>( unpackPod<std::uint64_t>( buf, pos ) ) );
    for ( auto& s : steps )
    {
        s.time = unpackPod<double>( buf, pos );
        s.frame.h5name = unpackString( buf, pos );
        s.frame.dim = unpackPod<int>( buf, pos );
        s.frame.scalarBytes = unpackPod<int>( buf, pos );
        s.frame.Nv = unpackPod<unsigned long long>( buf, pos );
        s.frame.Nf = unpackPod<unsigned long long>( buf, pos );
        s.frame.vFields = unpackFields();
        s.frame.fFields = unpackFields();
    }
    return steps;
}

} // namespace detail

//! Tag selecting MeshSeries' reopen constructor.
struct ReopenSeries
{
};

//! A time series of mesh frames sharing one master `.xmf`.
//!
//! Frame stems stay CALLER-OWNED: the master lists each frame's file
//! explicitly, so frame names need not be numeric or even ordered, and a
//! downstream caller keeps whatever naming it already uses. The one real
//! constraint that follows is enforced loudly rather than worked around: an
//! .xmf references its .h5 by BASENAME, so every frame must sit in the same
//! directory as the master, and write() throws otherwise.
//!
//! Usage:
//! \code
//!   Tessera::MeshSeries series( "out/bubble" );   // -> out/bubble.xmf
//!   for ( int step = 0; step < nsteps; ++step )
//!       series.write( mesh, "out/bubble_" + pad( step ), t );  // t increasing
//!   // open out/bubble.xmf in Paraview: one dataset, nsteps timesteps
//!
//!   // After a restart: continue the same series, same master.
//!   Tessera::MeshSeries resumed( "out/bubble", Tessera::ReopenSeries{},
//!                                mesh.comm() );
//!   resumed.write( mesh, "out/bubble_" + pad( nsteps ), t );  // t > last
//! \endcode
class MeshSeries
{
  public:
    //! `masterStem` names the master file `<masterStem>.xmf` and, with it, the
    //! directory every frame of this series must be written to. Nothing is
    //! written until the first write().
    //!
    //! This constructor starts a NEW series: its first write() truncates any
    //! existing `<masterStem>.xmfindex`, so a rerun into the same directory
    //! never inherits the previous run's frames in its restart record. (The
    //! master needs no such step -- it is always rewritten from this object's
    //! own frames.) Frames of the earlier run that are still on disk are left
    //! alone and are simply no longer part of the series.
    explicit MeshSeries( std::string masterStem )
        : _masterStem( std::move( masterStem ) )
    {
    }

    //! Reopen the series recorded in `<masterStem>.xmfindex`, so that the next
    //! write() continues it: the reopened steps are kept, the index is
    //! appended to rather than replaced, and the next `time` must exceed the
    //! last recorded one.
    //!
    //! **Collective on `comm`** -- every rank must call it with the same
    //! `masterStem`, and `comm` should be the communicator of the meshes later
    //! passed to write(). Rank 0 reads the index and, serially, each listed
    //! `<frameStem>.h5`'s root attributes to rebuild that frame's XDMF
    //! metadata, then rewrites `<masterStem>.xmf` from them before returning,
    //! and broadcasts the steps, so numFrames() is the same on every rank.
    //!
    //! Throws std::runtime_error ON EVERY RANK, with the same message naming
    //! the index, if the index is missing or empty, a line does not parse as
    //! "<frameStem> <time>", its times are not strictly increasing, a frame
    //! stem's directory differs from the master's, a listed `.h5` or one of
    //! its root attributes cannot be read, or the master cannot be rewritten.
    MeshSeries( std::string masterStem, ReopenSeries, MPI_Comm comm )
        : _masterStem( std::move( masterStem ) )
    {
        int rank = 0;
        MPI_Comm_rank( comm, &rank );

        // Rank 0 does every read. Its outcome travels as one payload: 'S'
        // plus the packed steps, or 'E' plus the error text, so a failure
        // throws on every rank and no rank goes on to a collective write()
        // the others never reach.
        std::string payload;
        if ( rank == 0 )
        {
            try
            {
                std::vector<XdmfTimeStep> steps =
                    detail::readSeriesIndex( _masterStem );
                writeXdmfSeries( _masterStem, steps );
                payload = "S";
                detail::packSteps( payload, steps );
            }
            catch ( const std::exception& e )
            {
                payload = std::string( "ETessera::MeshSeries: cannot reopen "
                                       "the series in '" ) +
                          _masterStem + ".xmfindex': " + e.what();
            }
        }
        unsigned long long n = payload.size();
        MPI_Bcast( &n, 1, MPI_UNSIGNED_LONG_LONG, 0, comm );
        payload.resize( static_cast<std::size_t>( n ) );
        MPI_Bcast( payload.data(), static_cast<int>( n ), MPI_CHAR, 0, comm );

        if ( payload[0] != 'S' )
            throw std::runtime_error( payload.substr( 1 ) );
        _steps = detail::unpackSteps( payload, 1 );
    }

    //! Write one frame of the series: `<frameStem>.h5` + its own
    //! `<frameStem>.xmf` sidecar, then the rewritten master.
    //!
    //! **Collective on `mesh.comm()`** -- every rank must call it, in the same
    //! order, with the same `frameStem` and `time`. `time` is the caller's own
    //! quantity (physical time or a step index) in the caller's own units, and
    //! must be STRICTLY GREATER than the previous frame's.
    //!
    //! Throws std::runtime_error, before performing any I/O, on a non-
    //! increasing `time` or on a `frameStem` whose directory differs from the
    //! master's. Both checks run on every rank, so the throw is symmetric and
    //! cannot deadlock, and a rejected frame leaves the master on disk
    //! unchanged.
    template <class MeshT>
    void write( const MeshT& mesh, const std::string& frameStem, double time )
    {
        // ---- Validation, on EVERY rank, before any I/O ---------------------
        if ( !_steps.empty() && !( time > _steps.back().time ) )
            throw std::runtime_error(
                "Tessera::MeshSeries::write: time must be strictly increasing "
                "within a series, but frame '" +
                frameStem + "' has time " + std::to_string( time ) +
                " and the previous frame had " +
                std::to_string( _steps.back().time ) );

        const std::string frameDir = detail::xdmfDirname( frameStem );
        const std::string masterDir = detail::xdmfDirname( _masterStem );
        if ( frameDir != masterDir )
            throw std::runtime_error(
                "Tessera::MeshSeries::write: frame stem '" + frameStem +
                "' is in directory '" + frameDir + "' but the master '" +
                _masterStem + "' is in '" + masterDir +
                "'; they must match, because an .xmf references its .h5 by "
                "basename and so can only name files in its own directory" );

        // ---- The frame itself (collective; also writes its own sidecar) ----
        // Goes through the public writeMesh() rather than
        // detail::writeMeshH5(), so TIMER_WRITE_MESH still spans the frame.
        XdmfFrame frame = writeMesh( mesh, frameStem, time );

        // The first frame of this object starts the index afresh.
        const bool firstFrame = _steps.empty();

        // Rank-uniform accumulator: numFrames() and the monotonic-time check
        // then mean the same thing on every rank. The cost is a handful of
        // strings and integers per frame replicated everywhere.
        _steps.push_back( XdmfTimeStep{ std::move( frame ), time } );

        if ( mesh.rank() == 0 )
        {
            appendIndexLine( frameStem, time, firstFrame );
            writeXdmfSeries( _masterStem, _steps );
        }
    }

    //! Number of frames written so far. Rank-uniform.
    std::size_t numFrames() const { return _steps.size(); }

    //! The stem the master `<masterStem>.xmf` is written to.
    const std::string& masterStem() const { return _masterStem; }

  private:
    //! Append "<frameStem> <time>\n" to <masterStem>.xmfindex, opened and
    //! closed per frame so the line is on disk before write() returns. This is
    //! the restart record the ReopenSeries constructor consumes. %.17g so the
    //! double round-trips exactly. `truncate` (the first frame of a new series)
    //! replaces any existing index instead of appending to it.
    void appendIndexLine( const std::string& frameStem, double time,
                          bool truncate ) const
    {
        const std::string idxname = _masterStem + ".xmfindex";
        std::FILE* fp = std::fopen( idxname.c_str(), truncate ? "w" : "a" );
        if ( !fp )
            throw std::runtime_error(
                "Tessera::MeshSeries::write: cannot open '" + idxname +
                ( truncate ? "' for writing" : "' for appending" ) );
        std::fprintf( fp, "%s %.17g\n", frameStem.c_str(), time );
        if ( std::fclose( fp ) != 0 )
            throw std::runtime_error(
                "Tessera::MeshSeries::write: error closing '" + idxname + "'" );
    }

    std::string _masterStem;
    //! Every frame's XDMF metadata, in write order. The previous frame's time
    //! is _steps.back().time -- there is deliberately no separate last-time
    //! member to drift from it.
    std::vector<XdmfTimeStep> _steps;
};

} // namespace Tessera

#endif // TESSERA_XDMF_SERIES_HPP
