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

// Regression test: markByQuality() drives CONFORMING refinement (Task 6 of
// tasks/completed/conforming-refinement.md). The conforming counterpart of
// markquality_edge / markquality_curv, which pin the criteria themselves
// against a partition-free reference; this file pins the criteria *composed
// with* the closure.
//
// Two things are under test, and neither is a property of the criteria:
//
//   1. MASK TRANSLATION. A criterion returns a mask over VISIBLE owned faces --
//      which in conforming mode are closure children, not red faces -- and
//      refine() must translate it to the red layer (a red parent is marked iff
//      ANY of its children was). Nothing in the criterion knows this. If the
//      translation were wrong the refine would still "work", just on the wrong
//      faces, so the check is that every conforming post-condition holds after
//      a criterion-driven round: conformity, owned Euler == 2, no interior
//      vertex, red-layer 2:1 balance, midpoint agreement, closure inverse.
//
//   2. CurvatureCriterion's COORDINATOR ASSUMPTION BECOMES TRUE. Its edge
//      coordinator computes a dihedral only for edges with exactly two incident
//      faces (`inc.size() != 2 -> continue`) and silently skips the rest. On a
//      hanging-node mesh those skipped edges are precisely the T-junctions, so
//      a fold running through a refinement front is under-marked. On a
//      conforming mesh there are none. checkConforming() counts exactly the
//      edges that guard would skip -- via the same edgeCoordRank routing -- so
//      "0 on the conforming mesh, > 0 on the hanging-node control" is a direct
//      measurement of that improvement, not a proxy for it.
//
// FIXTURE. A subdiv-2 icosphere with vertex gid 0 pushed radially out by 3x:
// the ring of faces around it is both steeply folded and long-edged, so ONE
// fixture drives both criteria and both mark a nontrivial PROPER subset -- the
// partial mask a hanging-node mesh cannot close. EdgeLengthCriterion's
// threshold sits in the wide gap between the spike edges and the rest;
// CurvatureCriterion's is derived from the reference mesh's own dihedral
// distribution (midpoint of the largest gap), the same rule markquality_curv
// uses, so no edge sits near the threshold and the marked set is stable across
// host/device rounding.
//
// NON-VACUITY, enforced as hard failures: the mask must be a proper non-empty
// subset every round, closure children must actually be emitted, and the
// identical criterion run on a HangingNode2to1 control mesh must leave
// T-junctions behind. Any of these being trivially satisfied means the case
// proved nothing.
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
#include <cstdio>
#include <cstdlib>
#include <map>
#include <type_traits>
#include <vector>

using namespace Tessera;

//! Sum a LOCAL fail count into a global one.
static inline int globalFails( int local )
{
    int g = 0;
    MPI_Allreduce( &local, &g, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    return g;
}

static inline long long globalSum( long long local )
{
    long long g = 0;
    MPI_Allreduce( &local, &g, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD );
    return g;
}

// The shared fixture's spike: scale the position of the vertex with gid
// `target` by `factor`. Applied on the replicated mesh right after
// buildIcosphere (gid == index), so the reference and every distributed copy
// see exactly the same geometry.
template <class MeshT>
static void displaceVertex( MeshT& mesh, GlobalId target,
                            typename MeshT::scalar_type factor )
{
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto gid = Cabana::slice<VertexField::Gid>( hv );
    auto pos = Cabana::slice<VertexField::Position>( hv );
    for ( std::size_t i = 0; i < mesh.numVertices(); ++i )
        if ( gid( i ) == target )
            for ( int d = 0; d < 3; ++d )
                pos( i, d ) *= factor;
    Cabana::deep_copy( mesh.vertices(), hv );
}

// Dihedral threshold for CurvatureCriterion, derived from a replicated
// (un-partitioned) copy of the fixture: the midpoint of the largest gap in the
// sorted per-edge dihedral distribution. With a 3x displacement that gap
// separates the few steep fold edges from the many shallow ones by a wide
// margin. Identical on every rank (no MPI, same input), so it introduces no
// partition dependence of its own.
template <class MeshT>
static typename MeshT::scalar_type deriveDihedralThreshold( int subdiv,
                                                            GlobalId spikeVert )
{
    using Scalar = typename MeshT::scalar_type;

    MeshT ref( MPI_COMM_WORLD );
    buildIcosphere( ref, subdiv );
    displaceVertex( ref, spikeVert, Scalar( 3 ) );

    const std::size_t nf = ref.numFaces();
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", ref.numVertices() );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", nf );
    Cabana::deep_copy( hv, ref.vertices() );
    Cabana::deep_copy( hf, ref.faces() );
    auto pos = Cabana::slice<VertexField::Position>( hv );
    auto fv = Cabana::slice<FaceField::Verts>( hf );

    std::vector<std::array<double, 3>> nrm( nf );
    for ( std::size_t f = 0; f < nf; ++f )
    {
        double p[3][3];
        for ( int k = 0; k < 3; ++k )
            for ( int d = 0; d < 3; ++d )
                p[k][d] = static_cast<double>(
                    pos( static_cast<int>( fv( f, k ) ), d ) );
        const double e1[3] = { p[1][0] - p[0][0], p[1][1] - p[0][1],
                               p[1][2] - p[0][2] };
        const double e2[3] = { p[2][0] - p[0][0], p[2][1] - p[0][1],
                               p[2][2] - p[0][2] };
        double n[3] = { e1[1] * e2[2] - e1[2] * e2[1],
                        e1[2] * e2[0] - e1[0] * e2[2],
                        e1[0] * e2[1] - e1[1] * e2[0] };
        const double len = std::sqrt( n[0] * n[0] + n[1] * n[1] + n[2] * n[2] );
        const double inv = len > 0.0 ? 1.0 / len : 0.0;
        for ( int d = 0; d < 3; ++d )
            nrm[f][d] = n[d] * inv;
    }

    std::map<EdgeKey, std::vector<int>> byEdge;
    for ( std::size_t f = 0; f < nf; ++f )
        for ( int k = 0; k < 3; ++k )
            byEdge[makeEdgeKey( fv( f, k ), fv( f, ( k + 1 ) % 3 ) )].push_back(
                static_cast<int>( f ) );

    std::vector<double> ang;
    for ( const auto& kv : byEdge )
    {
        if ( kv.second.size() != 2 )
            continue;
        double dot = 0.0;
        for ( int d = 0; d < 3; ++d )
            dot += nrm[kv.second[0]][d] * nrm[kv.second[1]][d];
        dot = std::max( -1.0, std::min( 1.0, dot ) );
        ang.push_back( std::acos( dot ) );
    }
    std::sort( ang.begin(), ang.end() );

    double bestGap = -1.0, theta = 0.0;
    for ( std::size_t i = 0; i + 1 < ang.size(); ++i )
        if ( ang[i + 1] - ang[i] > bestGap )
        {
            bestGap = ang[i + 1] - ang[i];
            theta = 0.5 * ( ang[i] + ang[i + 1] );
        }
    return static_cast<Scalar>( theta );
}

//! What one criterion-driven run produced, for the printout and the
//! non-vacuity guards.
struct Figures
{
    long long marked = 0;          //!< global marked owned faces, all rounds
    long long visible = 0;         //!< global owned faces after the last round
    long long closureChildren = 0; //!< global closure children, all rounds
    long long badIncidence = 0;    //!< edges without exactly two incident faces
    long long euler = 0;
    long long interiorVerts = 0;
    bool properSubsetEveryRound = true;
};

// Drive `rounds` iterations of {markByQuality -> refine -> migrate -> halo} and
// measure. Templated on the mesh type so the SAME code runs the conforming mesh
// and its hanging-node control: the criteria, the mask, and the loop are
// mode-agnostic, which is the point -- the only difference is what refine()
// does with the mask internally.
//
// markByQuality() must see a mesh whose face-corner vertices are all locally
// held (both criteria map corner gids to local indices), so it is called on the
// haloed mesh at the top of each round, never on refine()'s owned-only output.
template <class MeshT, class Crit>
static Figures driveCriterion( MeshT& mesh,
                               MeshHalo<typename MeshT::memory_space>& halo,
                               const Crit& crit, int rounds, int rank,
                               int& fails, bool conforming )
{
    Figures fig;

    for ( int round = 0; round < rounds; ++round )
    {
        std::vector<char> mask = markByQuality( mesh, crit );
        if ( mask.size() != mesh.numOwnedFaces() )
            ++fails;

        long long localMarked = 0;
        for ( char m : mask )
            localMarked += ( m ? 1 : 0 );
        const long long gMarked = globalSum( localMarked );
        const long long gFaces =
            globalSum( static_cast<long long>( mesh.numOwnedFaces() ) );
        // A criterion that marks nothing (or everything) makes the closure
        // trivially inert -- there would be no kept face with a split edge.
        if ( gMarked <= 0 || gMarked >= gFaces )
            fig.properSubsetEveryRound = false;
        fig.marked += gMarked;

        auto res = refine( mesh, halo, mask );
        fig.closureChildren +=
            globalSum( static_cast<long long>( res.closure.nClosureChildren ) );

        int local = TesseraTest::checkMidpointAgreement(
            MPI_COMM_WORLD, mesh.commSize(), res.midpoints );
        if ( conforming )
        {
            local += TesseraTest::checkConforming( mesh );
            local += TesseraTest::checkNoInteriorVertex( mesh );
            local += TesseraTest::check21BalanceRed( mesh );
            local += TesseraTest::checkClosureInverse( mesh, res.midpoints );
            if ( globalFails( local ) != 0 )
                ++fails;
            // THE criterion: a criterion-driven adaptive mask still closes.
            if ( TesseraTest::checkOwnedEuler( mesh ) != 2 )
                ++fails;
        }
        else
        {
            // The control's own contracts still hold; its conformity is
            // measured, not asserted (it is expected to be broken).
            local += TesseraTest::check21Balance( mesh );
            if ( globalFails( local ) != 0 )
                ++fails;
        }

        // refine() rebuilds the 1-deep halo itself, so the 1-ring and ownership
        // checks below need nothing done in between. This used to be an identity
        // migrate() + haloExchange().
        if ( conforming )
        {
            int l = TesseraTest::checkSiblingCoresidency( mesh );
            l += TesseraTest::owned1RingLocal( mesh );
            l += TesseraTest::checkOwnershipPartition(
                mesh, TesseraTest::globalOwnedVertices( mesh ),
                TesseraTest::globalOwnedEdges( mesh ),
                TesseraTest::globalOwnedFaces( mesh ) );
            if ( globalFails( l ) != 0 )
                ++fails;
        }
    }

    fig.visible = TesseraTest::globalOwnedFaces( mesh );
    fig.badIncidence = globalSum(
        static_cast<long long>( TesseraTest::checkConforming( mesh ) ) );
    fig.interiorVerts = globalSum(
        static_cast<long long>( TesseraTest::checkNoInteriorVertex( mesh ) ) );
    fig.euler = TesseraTest::checkOwnedEuler( mesh );
    return fig;
}

// One criterion, run on a conforming mesh and on a hanging-node control from
// the same fixture with the same criterion object.
template <class Exec, class ConfMesh, class HangMesh, class Crit>
static int case_criterion( int rank, const char* tag, const char* critName,
                           const Crit& crit, int subdiv, GlobalId spikeVert,
                           typename ConfMesh::scalar_type factor )
{
    using mem = typename Exec::memory_space;
    int fails = 0;
    const int rounds = 2;

    ConfMesh mesh( MPI_COMM_WORLD );
    buildIcosphere( mesh, subdiv );
    displaceVertex( mesh, spikeVert, factor );
    MeshHalo<mem> halo;
    {
        auto faceOwner = facePartitionByAxis( mesh ); // axis=z: fold straddles
        distribute( mesh, halo, faceOwner );
    }

    HangMesh ctrl( MPI_COMM_WORLD );
    buildIcosphere( ctrl, subdiv );
    displaceVertex( ctrl, spikeVert, factor );
    MeshHalo<mem> ctrlHalo;
    {
        auto faceOwner = facePartitionByAxis( ctrl );
        distribute( ctrl, ctrlHalo, faceOwner );
    }

    const Figures conf =
        driveCriterion( mesh, halo, crit, rounds, rank, fails, true );
    const Figures hang =
        driveCriterion( ctrl, ctrlHalo, crit, rounds, rank, fails, false );

    if ( !conf.properSubsetEveryRound || !hang.properSubsetEveryRound )
        ++fails; // vacuous: the criterion did not produce a partial mask
    if ( conf.closureChildren <= 0 )
        ++fails; // vacuous: the closure never fired
    if ( hang.badIncidence + hang.interiorVerts <= 0 )
        ++fails; // vacuous: this mask leaves no T-junction to begin with, so
                 // the conforming result above would prove nothing
    if ( conf.badIncidence != 0 || conf.interiorVerts != 0 || conf.euler != 2 )
        ++fails;

    if ( rank == 0 )
        std::printf( "  [%s] markquality/%s %s: conforming marked=%lld "
                     "F=%lld closure=%lld euler=%lld badInc=%lld tjunc=%lld | "
                     "control marked=%lld F=%lld euler=%lld badInc=%lld "
                     "tjunc=%lld\n",
                     tag, critName, fails == 0 ? "ok" : "FAIL", conf.marked,
                     conf.visible, conf.closureChildren, conf.euler,
                     conf.badIncidence, conf.interiorVerts, hang.marked,
                     hang.visible, hang.euler, hang.badIncidence,
                     hang.interiorVerts );

    return fails;
}

template <class Scalar, class Exec>
static int run( int rank, const char* tag )
{
    using mem = typename Exec::memory_space;
    using ConfMesh = Mesh<Scalar, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                          mem, Exec, RefinementMode::Conforming>;
    using HangMesh = Mesh<Scalar, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                          mem, Exec, RefinementMode::HangingNode2to1>;

    const int subdiv = 2;
    const GlobalId spikeVert = 0;      // an original icosahedron corner
    const Scalar factor = Scalar( 3 ); // push it well outward

    int fails = 0;

    // 10a -- EdgeLengthCriterion. The spike's edges are ~2x the sphere radius
    // while every other edge of a subdiv-2 unit icosphere is well under 0.5, so
    // maxLen = 1 sits in a wide empty gap: the marked set is the spike ring and
    // nothing else, at any rank count and on host or device.
    fails += case_criterion<Exec, ConfMesh, HangMesh>(
        rank, tag, "edge", EdgeLengthCriterion<Scalar>{ Scalar( 1 ) }, subdiv,
        spikeVert, factor );

    // 10b -- CurvatureCriterion at the reference-derived dihedral threshold.
    const Scalar theta = deriveDihedralThreshold<ConfMesh>( subdiv, spikeVert );
    fails += case_criterion<Exec, ConfMesh, HangMesh>(
        rank, tag, "curv", CurvatureCriterion<Scalar>{ theta }, subdiv,
        spikeVert, factor );

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
        if ( rank == 0 )
            std::printf( "test_markquality_conforming: markByQuality driving "
                         "conforming refinement (size %d)\n",
                         size );

        fails += run<double, Kokkos::Serial>( rank, "Serial" );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            fails +=
                run<double, Kokkos::DefaultExecutionSpace>( rank, "Default" );
    }

    Kokkos::finalize();
    MPI_Finalize();
    return fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
