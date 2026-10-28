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

#include <cstddef>
#include <cstdio>
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

} // namespace detail

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
    //! the restart record a series reopen consumes; it is written now so a run
    //! predating that feature is still restartable. %.17g so the double
    //! round-trips exactly. `truncate` (the first frame of a new series)
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
