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

// End-to-end mesh pipeline example: build an icosphere, partition + distribute
// it, then repeatedly mark a random fraction of the owned faces for
// refinement and refine/re-halo/write a new output frame. Running the
// resulting sequence of <out>_frameN.h5/.xmf files through Paraview visualizes
// how the mesh evolves as it adaptively refines.
//
// Usage:
//   Tuolumne: source scripts/lib/tessera_env.sh
//             flux run --ntasks <N> --nodes=1 --exclusive --cores-per-task=1 \
//                 $(tessera_exe examples/02_mesh_pipeline/mesh_pipeline) [args]
//   Local:    mpirun -n <N> ./mesh_pipeline [args]
//
// Arguments (all optional):
//   --subdiv N   Initial icosphere subdivision level (default 2).
//   --axis N     facePartitionByAxis() axis, 0/1/2 (default 2).
//   --balance    Run the internal Zoltan2 loadBalance() after every refine
//                step instead of a plain (no-op-destination) halo rebuild.
//   --iters N    Number of adaptive random-refinement iterations after the
//                initial build (default 3); one output frame is written per
//                iteration, plus a frame 0 baseline.
//   --frac F     Fraction (0,1] of owned faces marked for refinement each
//                iteration (default 0.1).
//   --seed N     RNG seed for the per-rank random marking (default 42);
//                reproducible across reruns.
//   --refine-mode {hanging,conforming}
//                Which RefinementMode the mesh type is instantiated with
//                (default conforming). `hanging` keeps the 2:1-bounded
//                hanging-node behavior: a partial mask leaves T-junctions, so
//                the frames show level jumps across edges. `conforming` adds
//                the transient red-green-blue closure pass, so every frame is
//                a conforming triangulation with no hanging nodes -- visibly
//                different in Paraview at every refinement front, and the
//                owned-face counts are higher by the closure children.
//   --out STEM   Output file stem (default "mesh_pipeline"); the exec-space
//                tag, rank count, and frame index are appended automatically.

#include <Tessera.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using namespace Tessera;

// ----------------------------------------------------------------------------
// Command-line options
// ----------------------------------------------------------------------------
struct Options
{
    int subdiv = 2;
    int axis = 2;
    bool balance = false;
    int iterations = 3;
    double refineFraction = 0.1;
    unsigned seed = 42;
    bool conforming = true;
    std::string outStem = "mesh_pipeline";
};

Options parseOptions( int argc, char* argv[] )
{
    Options opt;
    for ( int i = 1; i < argc; ++i )
    {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string
        { return ( i + 1 < argc ) ? argv[++i] : std::string(); };
        if ( arg == "--subdiv" )
            opt.subdiv = std::atoi( next().c_str() );
        else if ( arg == "--axis" )
            opt.axis = std::atoi( next().c_str() );
        else if ( arg == "--balance" )
            opt.balance = true;
        else if ( arg == "--iters" )
            opt.iterations = std::atoi( next().c_str() );
        else if ( arg == "--frac" )
            opt.refineFraction = std::atof( next().c_str() );
        else if ( arg == "--seed" )
            opt.seed = static_cast<unsigned>( std::atoi( next().c_str() ) );
        else if ( arg == "--refine-mode" )
            opt.conforming = ( next() != "hanging" );
        else if ( arg == "--out" )
            opt.outStem = next();
    }
    return opt;
}

// Master stem of one run()'s series for a given execution-space tag and rank
// count: "<out>_<tag>_np<size>" -> MeshSeries writes <stem>.xmf, the file to
// open in Paraview, plus <stem>.xmfindex.
std::string seriesStem( const Options& opt, const std::string& tag, int size )
{
    return opt.outStem + "_" + tag + "_np" + std::to_string( size );
}

// Output file stem for frame `i` of that series: "<seriesStem>_frame<i>" ->
// .h5 plus its own .xmf sidecar. Built from seriesStem() so every frame sits
// in the master's directory, as MeshSeries requires.
std::string frameStem( const Options& opt, const std::string& tag, int size,
                       int i )
{
    return seriesStem( opt, tag, size ) + "_frame" + std::to_string( i );
}

// ----------------------------------------------------------------------------
// Pipeline, run once per execution space so the example also exercises
// whichever Kokkos backend Exec resolves to (Serial always; the platform
// default and, on Tuolumne, OpenMP as well -- see main()).
// ----------------------------------------------------------------------------
// `Mode` is the Mesh template's RefinementMode -- a compile-time parameter, so
// the two --refine-mode choices are two distinct mesh types and main() below
// instantiates run() twice. Nothing else in the pipeline changes: refine()
// dispatches internally, and the closure is invisible to every accessor.
template <class Exec, RefinementMode Mode>
void run( int rank, int size, const std::string& tag, const Options& opt )
{
    using mem = typename Exec::memory_space;
    using MeshT = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                       mem, Exec, Mode>;

    // Step 1: construct an empty distributed-mesh container bound to a
    // communicator. No entities exist yet -- the builder below populates it.
    MeshT mesh( MPI_COMM_WORLD );

    // Step 2: generate a closed triangulated sphere at the requested
    // subdivision depth. This single-rank algorithm runs redundantly and
    // identically on every rank, producing the replicated starting geometry
    // that the partition step below cuts up.
    buildIcosphere( mesh, opt.subdiv );

    // Step 3: compute a deterministic geometric partition of the replicated
    // faces along `axis` (identical on every rank, no communication needed
    // since the input is still replicated), then cut the mesh down to this
    // rank's owned entities plus a 1-deep ghost layer per that partition,
    // building the three halo exchange plans held in `halo`.
    auto faceOwner = facePartitionByAxis( mesh, opt.axis );
    MeshHalo<mem> halo;
    distribute( mesh, halo, faceOwner );

    // Step 4: sync the ghost layer and write the "as distributed" baseline
    // frame, before any adaptive refinement, as frame 0 of this run()'s
    // series. One MeshSeries per run() call, so each mesh type and execution
    // space gets its own master and frames of different mesh types never
    // share one. The example has no physical time, so each frame's time is
    // its frame index as a double: 0.0 here, double( it ) in the loop below.
    MeshSeries series( seriesStem( opt, tag, size ) );
    haloExchange( mesh, halo );
    series.write( mesh, frameStem( opt, tag, size, 0 ), 0.0 );
    if ( rank == 0 )
        std::printf( "  [%s] frame 0: ownedF=%zu ownedE=%zu ownedV=%zu -> %s\n",
                     tag.c_str(), mesh.numOwnedFaces(), mesh.numOwnedEdges(),
                     mesh.numOwnedVertices(),
                     frameStem( opt, tag, size, 0 ).c_str() );

    // Step 5: adaptive loop. Each iteration marks a random subset of this
    // rank's owned faces, refines them, rebuilds the halo (refine() clears
    // it as a side effect -- Tessera has no built-in geometric quality
    // criterion for *this* mask; markByQuality()/EdgeLengthCriterion /
    // CurvatureCriterion are the geometry-driven alternative, see README),
    // then re-syncs and writes the next frame.
    std::mt19937 rng( opt.seed + static_cast<unsigned>( rank ) );
    std::uniform_real_distribution<double> unif( 0.0, 1.0 );
    int totalRefineIters = 0;

    // Profiling demo: Tessera exposes only the reset/print mechanism; the
    // adaptive loop below stands in for a downstream simulation's timestep loop
    // and drives the cadence. Each iteration is one reporting "window" -- print
    // its per-region aggregate then reset -- while the lifetime registry keeps
    // the whole-run total for the summary after the loop. Every call is a no-op
    // unless the library was built with -DTessera_PROFILING_LEVEL>=1.
    TESSERA_RESET_TIMERS();
    for ( int it = 1; it <= opt.iterations; ++it )
    {
        // 5a. Mark: refine()'s mask is a pure caller decision -- here a
        // uniform random draw of `refineFraction` of this rank's owned
        // faces, standing in for a real AMR error indicator.
        const int nOwnedF = static_cast<int>( mesh.numOwnedFaces() );
        std::vector<char> mask( nOwnedF, 0 );
        for ( int f = 0; f < nOwnedF; ++f )
            mask[f] = ( unif( rng ) < opt.refineFraction ) ? 1 : 0;

        // 5b. Refine: 2:1-balanced 1->4 split of the marked faces (plus
        // whatever the cross-rank 2:1 fixpoint pulls in), followed -- in
        // RefinementMode::Conforming only -- by the transient closure pass that
        // retriangulates the kept neighbours so no hanging node survives. This
        // clears `halo` as a documented side effect -- no haloExchange() may
        // run until it is rebuilt.
        RefineResult rr = refine( mesh, halo, mask );
        totalRefineIters += rr.iterations;

        // 5c. Rebuild the halo (mandatory post-refine step), optionally
        // rebalancing the partition at the same time since refinement can
        // skew owned-face counts across ranks.
        if ( opt.balance )
        {
            // loadBalance() computes a fresh Zoltan2 geometric assignment and
            // migrates to it, which also rebuilds the halo.
            loadBalance( mesh, halo );
        }
        else
        {
            // A self-destination migrate() (every owned face "moves" to the
            // rank that already owns it) is a legitimate no-op move whose
            // side effect is exactly what's needed here: recompute ownership
            // and rebuild the ghost layer + halo plans without changing the
            // partition.
            std::vector<Rank> dest( mesh.numOwnedFaces(),
                                    static_cast<Rank>( rank ) );
            migrate( mesh, halo, dest );
        }

        // 5d. Sync the rebuilt halo and write this iteration's frame. Time is
        // the frame index, double( it ), as for frame 0 above.
        haloExchange( mesh, halo );
        series.write( mesh, frameStem( opt, tag, size, it ),
                      static_cast<double>( it ) );

        if ( rank == 0 )
            std::printf( "  [%s] frame %d: ownedF=%zu ownedE=%zu ownedV=%zu "
                         "(refine rounds=%d, cumulative=%d) -> %s\n",
                         tag.c_str(), it, mesh.numOwnedFaces(),
                         mesh.numOwnedEdges(), mesh.numOwnedVertices(),
                         rr.iterations, totalRefineIters,
                         frameStem( opt, tag, size, it ).c_str() );

        // End of this "timestep": report the window aggregate, then reset so
        // the next iteration measures only its own work.
        TESSERA_PRINT_TIMERS( MPI_COMM_WORLD );
        TESSERA_RESET_TIMERS();
    }

    // Whole-run summary: per-region totals accumulated across every iteration
    // of this run() (the lifetime registry is never cleared by the resets).
    TESSERA_PRINT_TIMERS_TOTAL( MPI_COMM_WORLD );

    if ( rank == 0 )
        std::printf( "  [%s] series: %zu frames -> open %s.xmf in Paraview\n",
                     tag.c_str(), series.numFrames(),
                     series.masterStem().c_str() );
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    int rank = 0, size = 1;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    Kokkos::initialize( argc, argv );
    {
        const Options opt = parseOptions( argc, argv );
        const char* modeTag = opt.conforming ? "conforming" : "hanging";
        if ( rank == 0 )
            std::printf(
                "=== Tessera mesh_pipeline: subdiv=%d axis=%d balance=%s "
                "iters=%d frac=%g seed=%u refine-mode=%s out=%s ===\n",
                opt.subdiv, opt.axis, opt.balance ? "on" : "off",
                opt.iterations, opt.refineFraction, opt.seed, modeTag,
                opt.outStem.c_str() );

        // The mode is a compile-time mesh parameter, so it selects between two
        // instantiations of the same pipeline rather than a runtime branch
        // inside it. The mode tag goes in the frame stem so a conforming and a
        // hanging-node run of the same --out can be compared side by side.
        auto dispatch = [&]( auto execTag, const char* spaceTag )
        {
            using Exec = decltype( execTag );
            const std::string tag = std::string( spaceTag ) + "_" + modeTag;
            if ( opt.conforming )
                run<Exec, RefinementMode::Conforming>( rank, size, tag, opt );
            else
                run<Exec, RefinementMode::HangingNode2to1>( rank, size, tag,
                                                            opt );
        };

        dispatch( Kokkos::Serial{}, "Serial" );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            dispatch( Kokkos::DefaultExecutionSpace{}, "Default" );
#ifdef KOKKOS_ENABLE_OPENMP
        if ( !std::is_same<Kokkos::OpenMP, Kokkos::Serial>::value &&
             !std::is_same<Kokkos::OpenMP,
                           Kokkos::DefaultExecutionSpace>::value )
            dispatch( Kokkos::OpenMP{}, "OpenMP" );
#endif
    }
    Kokkos::finalize();

    MPI_Finalize();
    return 0;
}
