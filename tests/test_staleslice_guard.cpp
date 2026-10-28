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

// Unit test: the reallocate-on-topology-change ownership guard.
//
// Case 1: a slice taken from the mesh, held across a call that bumps
// Mesh::generation() (standing in for a real distribute()/migrate()/refine()),
// then copied (as it would be when captured into a KOKKOS_LAMBDA) must abort
// with a diagnostic. Since std::abort() would take down the test binary, the
// copy is exercised in a forked child process and the parent checks that the
// child died abnormally.
//
// Case 2: haloExchange() is topology-preserving and must NOT bump
// Mesh::generation() -- a slice taken before it must survive being copied
// again afterward with no abort.
//
// Case 3: the CONFORMING refine path (tasks/completed/conforming-refinement.md, Task 7).
// It is a distinct hazard from the hanging-node path, not the same one twice:
// conforming refine() un-closes the transient closure layer and re-closes it,
// so it changes the local face count -- and reallocates the face AoSoA -- even
// when nothing at all is refined. A face slice, a MeshGeometry, or a
// VertexStencil held across it is stale and every one of them must abort when
// copied. All three are checked, since each reaches the guard by a different
// route (a bare slice; the position GenerationHandle inside the geometry
// accessor; the CSR GenerationHandle inside the stencil).
//
// Host-only (Kokkos::Serial / Kokkos::HostSpace) regardless of the build's
// default execution space: the bug under test is host-side generation
// bookkeeping, not device kernels, so this test is registered SERIAL-only.

#include <Tessera.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <sys/wait.h>
#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace Tessera;

using MeshT =
    Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
         Kokkos::HostSpace, Kokkos::Serial, RefinementMode::HangingNode2to1>;

//! Case 3's mesh: same in every respect but the refinement mode.
using ConfMeshT =
    Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
         Kokkos::HostSpace, Kokkos::Serial, RefinementMode::Conforming>;

// Runs `f` in a forked child process; returns true iff the child exited
// cleanly (code 0), false if it was signaled or exited nonzero.
template <class F>
bool runInChild( F&& f )
{
    std::fflush( nullptr );
    pid_t pid = fork();
    if ( pid == 0 )
    {
        f();
        std::_Exit( EXIT_SUCCESS );
    }
    int status = 0;
    waitpid( pid, &status, 0 );
    return WIFEXITED( status ) && WEXITSTATUS( status ) == EXIT_SUCCESS;
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int fails = 0;
    {
        MeshT mesh( MPI_COMM_WORLD );
        mesh.resizeVertices( 8 );

        // ---- Case 1: stale slice trips the guard --------------------------
        auto stale = mesh.template vertexSlice<VertexField::Position>();
        const std::size_t gen_before = mesh.generation();
        mesh.resizeVertices( 8 ); // same size, but still a count-changing op
        if ( mesh.generation() == gen_before )
        {
            std::fprintf(
                stderr, "FAIL: resizeVertices() did not bump generation()\n" );
            ++fails;
        }

        const bool clean_exit = runInChild(
            [&]()
            {
                auto copy = stale; // copy ctor validates -> should abort
                (void)copy;
            } );
        if ( clean_exit )
        {
            std::fprintf(
                stderr,
                "FAIL: copying a stale slice did not trip the guard\n" );
            ++fails;
        }

        // ---- Case 2: haloExchange() does not trip the guard ---------------
        auto fresh = mesh.template vertexSlice<VertexField::Position>();
        const std::size_t gen_before_halo = mesh.generation();

        MeshHalo<Kokkos::HostSpace> halo; // default-constructed: empty plans
        haloExchange( mesh, halo );       // no-op (empty plan), must not bump

        if ( mesh.generation() != gen_before_halo )
        {
            std::fprintf( stderr,
                          "FAIL: haloExchange() bumped generation()\n" );
            ++fails;
        }

        // If the guard were (incorrectly) wired to trip here, this copy would
        // abort and take the whole test process down with it -- a crash here
        // is itself a (very loud) test failure.
        auto still_fresh = fresh;
        (void)still_fresh;

        // ---- Case 3: conforming refine() invalidates every handle ---------
        ConfMeshT cmesh( MPI_COMM_WORLD );
        buildIcosphere( cmesh, 1 );
        MeshHalo<Kokkos::HostSpace> chalo;
        {
            auto faceOwner = facePartitionByAxis( cmesh );
            distribute( cmesh, chalo, faceOwner );
        }
        haloExchange( cmesh, chalo );

        auto stale_face = cmesh.template faceSlice<FaceField::Gid>();
        auto stale_geom = buildMeshGeometry( cmesh );
        auto stale_stencil = buildVertexStencil( cmesh, 1 );
        const std::size_t cgen_before = cmesh.generation();

        // A PARTIAL mask, so the closure really runs: the kept faces adjacent
        // to a refined one are retriangulated and the visible face count is not
        // simply 4x the refined count.
        {
            auto fg = cmesh.template faceSlice<FaceField::Gid>();
            std::vector<char> mask( cmesh.numOwnedFaces(), 0 );
            for ( std::size_t f = 0; f < mask.size(); ++f )
                mask[f] = ( fg( f ) % 3 == 0 ) ? 1 : 0;
            refine( cmesh, chalo, mask );
        }

        if ( cmesh.generation() == cgen_before )
        {
            std::fprintf(
                stderr,
                "FAIL: conforming refine() did not bump generation()\n" );
            ++fails;
        }

        if ( runInChild(
                 [&]()
                 {
                     auto c = stale_face;
                     (void)c;
                 } ) )
        {
            std::fprintf( stderr, "FAIL: a face slice held across a conforming "
                                  "refine() did not trip the guard\n" );
            ++fails;
        }
        if ( runInChild(
                 [&]()
                 {
                     auto c = stale_geom;
                     (void)c;
                 } ) )
        {
            std::fprintf( stderr, "FAIL: a MeshGeometry held across a "
                                  "conforming refine() did not trip the "
                                  "guard\n" );
            ++fails;
        }
        if ( runInChild(
                 [&]()
                 {
                     auto c = stale_stencil.csr;
                     (void)c;
                 } ) )
        {
            std::fprintf( stderr, "FAIL: a VertexStencil held across a "
                                  "conforming refine() did not trip the "
                                  "guard\n" );
            ++fails;
        }

        int rank = 0;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        std::printf( "  [rank %d] %s\n", rank, fails == 0 ? "ok" : "FAIL" );
    }

    Kokkos::finalize();

    int global_fails = 0;
    MPI_Allreduce( &fails, &global_fails, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    MPI_Finalize();
    return global_fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
