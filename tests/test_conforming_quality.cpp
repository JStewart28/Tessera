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

// Test: TRIANGLE SHAPE QUALITY OVER REFINEMENT DEPTH (Task 7 of
// tasks/completed/conforming-refinement.md) -- the guarantee that buys the closure its
// "transient" design.
//
// A closure that PERSISTED would be a disaster for shape: bisecting an already
// bisected green triangle, round after round, drives its min angle to zero with
// no lower bound. Tessera's closure is instead recomputed from scratch on every
// refine call: un-close discards the whole closure layer, the red engine sees
// only red faces, and the closure is rebuilt. Consequently EVERY visible
// triangle is either a red triangle or one of three fixed retriangulations of
// one, so the number of triangle similarity classes is BOUNDED and the worst
// shape is bounded with it -- independent of how many rounds have been run.
//
// This test measures that. It refines a shrinking geodesic cap for 16 rounds and
// tracks, per round: the global minimum triangle angle, the global maximum
// radius ratio, and the closure-face fraction -- each of those split by which of
// the four closure patterns produced the face, plus the same for the red layer
// that feeds them. It then asserts each stays inside a FIXED bound -- fixed being
// the whole point: a bound that had to grow with the round count would mean the
// closure is not transient after all.
//
//   MIN ANGLE. On an exactly equilateral parent the four patterns give: |S|=0
//   60 deg, |S|=1 (green, a median cut) 30 deg, |S|=2 (blue) 30 deg (the
//   30-30-120 sliver off the midline is the worst of the three children), |S|=3
//   (red-closure) 60 deg. Real red faces are icosphere triangles, which are
//   near- but not exactly equilateral, so the realised worst case sits somewhat
//   below 30.
//
//   RADIUS RATIO Q = abc(a+b+c) / (16 A^2), which is 1 for an equilateral
//   triangle and grows as the triangle degenerates (it is R/2r, circumradius
//   over inradius, normalised). The ideal-parent worst cases are Q = 1.37 for
//   the 30-60-90 green child and Q = 2.16 for the 30-30-120 blue child.
//
//   CLOSURE FRACTION. The closure covers the level-jump BOUNDARY, an O(perimeter)
//   set, while the mesh grows by area, so the fraction must stay bounded well
//   below 1 rather than tracking the mesh size.
//
// THE BOUNDS BELOW ARE MEASURED (Task 8 sub-task D7). Task 7 registered this
// test with bounds derived from the ideal-parent geometry alone -- 20 deg, Q <=
// 4.0, closure fraction <= 0.50 -- because nothing had been executed yet. D7 ran
// it to 16 rounds and calibrated them against the result. The 16-round run is
// what settled the open question, so its shape is recorded here:
//
//   |S|=0 (RED, the closure's INPUT)  maxQ 1.0278, minAngle 54.397 deg
//                                     -- IDENTICAL in all 16 rounds.
//   |S|=1 (green)                     maxQ 1.5672 from round 1, never moves.
//   |S|=2 (blue)                      maxQ 1.7759 (r6) -> 2.2344 (r8), then
//                                     FLAT through r16.
//   |S|=3 (red-closure)               never realised on this workload.
//   closure fraction                  peaks 0.1864 at r6, declines to 0.0623.
//   worst amplification Q/Q(parent)   2.2310, flat from r8.
//
// RE-MEASURED after the blue tie-break became GEOMETRIC (Decision 15). The rule
// takes the SHORTER diagonal, so blue's worst shape improved and it now saturates
// three steps earlier: D7 measured a third step to 2.5254 at r11 with
// amplification 2.4906; both are gone. Everything else -- red, green, the closure
// fraction, min angle, and every F and |S| count -- is UNCHANGED, which is the
// expected signature: the red layer is not a function of the diagonal. The bounds
// below are deliberately NOT tightened to the new worst; they still hold with
// more margin, and re-tightening a gate bound to a just-measured number buys
// nothing but a future false failure.
//
// The blue family is the only one that moves, and it moves in DISCRETE STEPS
// separated by several flat rounds, then saturates: rounds 8-16 are identical
// while F grows 1672 -> 24608 and the marked set grows 134 -> 2388. That is the
// signature of a maximum over a FINITE set being progressively discovered, not
// of unbounded growth -- the closure can only emit (red similarity class) x
// (which edges are split) x (which blue diagonal), and each new combination the
// growing cap reaches can raise the maximum once. Eight rounds could not tell
// the two apart (the last round measured was itself a step); sixteen can.
//
// So the bounds are FIXED in the round count, which is exactly the claim that
// buys the closure its transient design. The test now asserts that directly
// rather than only printing it: the red layer's own shape is bounded (the
// closure's input never degrades), the per-round worst is bounded, AND the worst
// must stop growing over the final quarter of the rounds. Bounds keep ~10%
// margin over the measured worst -- affordable because every printed quantity is
// byte-identical at every rank count on both backends: D7's 320 round lines over
// np1-5 x {SERIAL, HIP} x {Serial, Default} reduce to exactly 16 distinct lines,
// so there is no run-to-run spread to absorb.
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
#include <type_traits>
#include <unordered_map>
#include <vector>

using namespace Tessera;

// ---- MEASURED quality bounds (D7; see the header note for the 16-round data) --
//
// Each was D7's 16-round measured worst plus ~10%. The measured worst is in the
// comment, so a future change that moves one of these is visible as a diff
// against a number, not against a guess. Decision 15's geometric blue tie-break
// IMPROVED two of them (D7's value in parentheses); the bounds are deliberately
// not re-tightened onto the new worst.
static constexpr double kMinAngleDeg = 24.0;     //!< measured 25.987
static constexpr double kMaxRadiusRatio = 2.8;   //!< 2.2344 (D7 2.5254)
static constexpr double kMaxClosureFrac = 0.25;  //!< measured 0.1864 (peak, r6)
static constexpr double kMaxAmplification = 2.8; //!< 2.2310 (D7 2.4906)
//
// The RED layer is the closure's input. It is a pure 4-way subdivision of
// icosphere triangles, so its shape is round-independent by construction and
// measured dead flat; bounding it separately is what distinguishes "the closure
// degrades shape" from "the red engine does", which the aggregate cannot.
//! Measured 1.0278 in every one of the 16 rounds.
static constexpr double kRedMaxRadiusRatio = 1.10;
//! Measured 54.397 deg in every one of the 16 rounds.
static constexpr double kRedMinAngleDeg = 50.0;
//
// Rounds over which the worst radius ratio must have STOPPED growing. Measured:
// the last step is at round 8, so the final 8 of 16 rounds are flat. (D7 measured
// the last step at round 11, which is why the round count is 16 and not 12; the
// geometric tie-break removed that step but the depth is kept -- a round count
// chosen to be longer than the observed saturation point is the point.)
static constexpr int kSaturationRounds = 4;

static constexpr int kRounds = 16;
static constexpr double kPi = 3.14159265358979323846;

//! Rounds to run. `kRounds` by default; `TESSERA_QUALITY_ROUNDS` overrides it so
//! the round-independence claim can be re-measured to any depth without an edit.
//! Deeper than ~16 is not useful: the cap's faces reach ~1e-5 of a unit sphere
//! and the mesh grows ~1.35x per round.
static int roundCount()
{
    const char* e = std::getenv( "TESSERA_QUALITY_ROUNDS" );
    if ( e == nullptr )
        return kRounds;
    const int n = std::atoi( e );
    return n > 0 ? n : kRounds;
}

static inline long long globalSum( long long v )
{
    long long g = 0;
    MPI_Allreduce( &v, &g, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD );
    return g;
}
static inline double globalMinD( double v )
{
    double g = 0;
    MPI_Allreduce( &v, &g, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
    return g;
}
static inline double globalMaxD( double v )
{
    double g = 0;
    MPI_Allreduce( &v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
    return g;
}

//! Vertex gid -> position over every locally held vertex.
template <class MeshT>
static std::unordered_map<GlobalId, std::array<double, 3>>
readPositions( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", mesh.numVertices() );
    Cabana::deep_copy( hv, mesh.vertices() );
    auto vg = Cabana::slice<VertexField::Gid>( hv );
    auto vp = Cabana::slice<VertexField::Position>( hv );
    std::unordered_map<GlobalId, std::array<double, 3>> out;
    out.reserve( mesh.numVertices() * 2 );
    for ( std::size_t i = 0; i < mesh.numVertices(); ++i )
    {
        std::array<double, 3> p = { 0, 0, 0 };
        for ( int d = 0; d < MeshT::dim && d < 3; ++d )
            p[d] = static_cast<double>( vp( i, d ) );
        out[vg( i )] = p;
    }
    return out;
}

template <class MeshT>
static std::vector<VisibleFace> ownedVisible( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    return readVisibleFaces<MeshT>( hf, mesh.numOwnedFaces() );
}

//! Min angle (deg) and radius ratio Q of one triangle. False if degenerate.
static bool triMetrics( const std::array<double, 3> p[3], double& minAngleDeg,
                        double& Q )
{
    // Side lengths: s[k] is the side OPPOSITE corner k.
    double s[3];
    for ( int k = 0; k < 3; ++k )
    {
        const std::array<double, 3>& a = p[( k + 1 ) % 3];
        const std::array<double, 3>& b = p[( k + 2 ) % 3];
        double d2 = 0.0;
        for ( int c = 0; c < 3; ++c )
            d2 += ( b[c] - a[c] ) * ( b[c] - a[c] );
        s[k] = std::sqrt( d2 );
    }
    if ( s[0] <= 0.0 || s[1] <= 0.0 || s[2] <= 0.0 )
        return false;

    // Angles by the law of cosines; area from the cross product.
    minAngleDeg = 180.0;
    for ( int k = 0; k < 3; ++k )
    {
        const double num = s[( k + 1 ) % 3] * s[( k + 1 ) % 3] +
                           s[( k + 2 ) % 3] * s[( k + 2 ) % 3] - s[k] * s[k];
        const double den = 2.0 * s[( k + 1 ) % 3] * s[( k + 2 ) % 3];
        double cosA = num / den;
        cosA = std::max( -1.0, std::min( 1.0, cosA ) );
        minAngleDeg = std::min( minAngleDeg, std::acos( cosA ) * 180.0 / kPi );
    }

    const double u[3] = { p[1][0] - p[0][0], p[1][1] - p[0][1],
                          p[1][2] - p[0][2] };
    const double w[3] = { p[2][0] - p[0][0], p[2][1] - p[0][1],
                          p[2][2] - p[0][2] };
    const double n[3] = { u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2],
                          u[0] * w[1] - u[1] * w[0] };
    const double area =
        0.5 * std::sqrt( n[0] * n[0] + n[1] * n[1] + n[2] * n[2] );
    if ( area <= 0.0 )
        return false;
    Q = s[0] * s[1] * s[2] * ( s[0] + s[1] + s[2] ) / ( 16.0 * area * area );
    return true;
}

//! One round's measurements, already reduced across ranks.
struct Quality
{
    double minAngleDeg = 180.0;
    double maxRadiusRatio = 0.0;
    double closureFrac = 0.0;
    long long faces = 0;
    long long closureChildren = 0;
    long long marked = 0;
    //! Per closure PATTERN, indexed by the parent's |S|: 0 = a red face passed
    //! through unchanged, 1 = green, 2 = blue, 3 = red-closure. This is what
    //! makes the bound legible: the closure can only ever emit one of these
    //! four families, so if each family's worst shape is flat in the round
    //! count then the overall bound is too, and if the overall maximum moves it
    //! says WHICH family moved it.
    double maxQByPattern[4] = { 0.0, 0.0, 0.0, 0.0 };
    double minAngleByPattern[4] = { 180.0, 180.0, 180.0, 180.0 };
    //! Worst shape of a RED PARENT triangle (the input the closure retriangulates).
    double maxParentQ = 0.0;
    double minParentAngle = 180.0;
    //! Worst amplification Q(child) / Q(its red parent) -- how much shape the
    //! closure itself costs, with the parent's own distortion divided out. Not
    //! quite a property of the patterns alone: the split points are the red
    //! engine's midpoints, which are re-projected onto the sphere rather than
    //! being exact affine midpoints.
    double maxAmp = 0.0;
};

//! Shape of every OWNED visible face, from the haloed positions.
template <class MeshT>
static Quality measureQuality( MeshT& mesh, int& fails )
{
    const auto pos = readPositions( mesh );
    const std::vector<VisibleFace> vis = ownedVisible( mesh );

    // |S| of a closed red parent is (number of its children) - 1, and siblings
    // are co-resident (checkSiblingCoresidency), so this count is complete.
    std::unordered_map<GlobalId, int> childCount;
    childCount.reserve( vis.size() * 2 );
    for ( const auto& f : vis )
        if ( f.parent != invalid_gid )
            ++childCount[f.parent];

    double minAngle = 180.0, maxQ = 0.0;
    double maxQPat[4] = { 0.0, 0.0, 0.0, 0.0 };
    double minAngPat[4] = { 180.0, 180.0, 180.0, 180.0 };
    double maxParentQ = 0.0, minParentAngle = 180.0, maxAmp = 0.0;
    long long nClosure = 0;
    for ( const auto& f : vis )
    {
        int pattern = 0; // |S| = 0: a red face, passed through
        if ( f.parent != invalid_gid )
        {
            ++nClosure;
            pattern = childCount[f.parent] - 1;
            if ( pattern < 1 || pattern > 3 )
            {
                ++fails;  // a closure family of 2..4 children is the only legal
                continue; // shape; anything else is a bookkeeping defect
            }
        }

        std::array<double, 3> p[3];
        bool have = true;
        for ( int k = 0; k < 3; ++k )
        {
            auto it = pos.find( f.v[k] );
            if ( it == pos.end() )
            {
                have = false;
                break;
            }
            p[k] = it->second;
        }
        if ( !have )
        {
            ++fails; // every corner of an owned face must be held after halo
            continue;
        }

        double ang = 180.0, Q = 0.0;
        if ( !triMetrics( p, ang, Q ) )
        {
            ++fails; // a degenerate triangle is a hard failure, not a datum
            continue;
        }
        minAngle = std::min( minAngle, ang );
        maxQ = std::max( maxQ, Q );
        minAngPat[pattern] = std::min( minAngPat[pattern], ang );
        maxQPat[pattern] = std::max( maxQPat[pattern], Q );

        // The parent this child retriangulates. Its corners are corners of the
        // sibling group, which is co-resident, so they are held after the halo.
        if ( pattern == 0 )
        {
            maxParentQ = std::max( maxParentQ, Q );
            minParentAngle = std::min( minParentAngle, ang );
            continue;
        }
        std::array<double, 3> pp[3];
        for ( int k = 0; k < 3; ++k )
        {
            auto it = pos.find( f.parentVerts[k] );
            if ( it == pos.end() )
            {
                have = false;
                break;
            }
            pp[k] = it->second;
        }
        if ( !have )
        {
            ++fails; // a closure child's parent corners must be held too
            continue;
        }
        double pang = 180.0, pQ = 0.0;
        if ( !triMetrics( pp, pang, pQ ) )
        {
            ++fails;
            continue;
        }
        maxParentQ = std::max( maxParentQ, pQ );
        minParentAngle = std::min( minParentAngle, pang );
        maxAmp = std::max( maxAmp, Q / pQ );
    }

    Quality q;
    q.minAngleDeg = globalMinD( minAngle );
    q.maxRadiusRatio = globalMaxD( maxQ );
    for ( int s = 0; s < 4; ++s )
    {
        q.maxQByPattern[s] = globalMaxD( maxQPat[s] );
        q.minAngleByPattern[s] = globalMinD( minAngPat[s] );
    }
    q.maxParentQ = globalMaxD( maxParentQ );
    q.minParentAngle = globalMinD( minParentAngle );
    q.maxAmp = globalMaxD( maxAmp );
    q.faces = globalSum( static_cast<long long>( vis.size() ) );
    q.closureChildren = globalSum( nClosure );
    q.closureFrac = q.faces > 0 ? static_cast<double>( q.closureChildren ) /
                                      static_cast<double>( q.faces )
                                : 0.0;
    return q;
}

//! Geodesic-cap mask: mark a face iff the direction of its centroid is within
//! `halfAngle` of +z. Purely geometric, so it is the same face set at any rank
//! count; the cap shrinks by 0.6 per round while the faces inside it shrink by
//! 0.5, which keeps the marked set a small, slowly growing patch instead of
//! either dying out or engulfing the sphere.
template <class MeshT>
static std::vector<char>
capMask( MeshT& mesh,
         const std::unordered_map<GlobalId, std::array<double, 3>>& pos,
         double halfAngle, int& fails )
{
    const std::vector<VisibleFace> vis = ownedVisible( mesh );
    const double cosLim = std::cos( halfAngle );
    std::vector<char> mask( vis.size(), 0 );
    for ( std::size_t f = 0; f < vis.size(); ++f )
    {
        double c[3] = { 0, 0, 0 };
        bool have = true;
        for ( int k = 0; k < 3; ++k )
        {
            auto it = pos.find( vis[f].v[k] );
            if ( it == pos.end() )
            {
                have = false;
                break;
            }
            for ( int d = 0; d < 3; ++d )
                c[d] += it->second[d] / 3.0;
        }
        if ( !have )
        {
            ++fails;
            continue;
        }
        const double len = std::sqrt( c[0] * c[0] + c[1] * c[1] + c[2] * c[2] );
        mask[f] = ( len > 0.0 && c[2] / len > cosLim ) ? 1 : 0;
    }
    return mask;
}

template <class Exec, class MeshT>
static int case_quality( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    int fails = 0;

    MeshT mesh( MPI_COMM_WORLD );
    buildIcosphere( mesh, 2 );
    MeshHalo<mem> halo;
    {
        auto faceOwner = facePartitionByAxis( mesh );
        distribute( mesh, halo, faceOwner );
    }

    double worstAngle = 180.0, worstQ = 0.0, worstFrac = 0.0, worstAmp = 0.0;
    long long totalClosure = 0;
    bool properSubsetEveryRound = true;
    const int rounds = roundCount();
    std::vector<double> maxQPerRound;
    maxQPerRound.reserve( static_cast<std::size_t>( rounds ) );

    for ( int round = 0; round < rounds; ++round )
    {
        const double halfAngle = 0.35 * std::pow( 0.6, round );
        const auto pos = readPositions( mesh );
        std::vector<char> mask = capMask( mesh, pos, halfAngle, fails );

        long long localMarked = 0;
        for ( char m : mask )
            localMarked += m ? 1 : 0;
        const long long gMarked = globalSum( localMarked );
        const long long gFaces =
            globalSum( static_cast<long long>( mesh.numOwnedFaces() ) );
        if ( gMarked <= 0 || gMarked >= gFaces )
            properSubsetEveryRound = false;

        refine( mesh, halo, mask );

        // Nothing in between: refine() rebuilds the 1-deep halo itself, so the
        // quality measurement and the next round's geometric mask find corner
        // positions locally. This used to need an identity migrate().

        const Quality q = measureQuality( mesh, fails );
        if ( q.faces <= 0 )
            ++fails;
        totalClosure += q.closureChildren;
        worstAngle = std::min( worstAngle, q.minAngleDeg );
        worstQ = std::max( worstQ, q.maxRadiusRatio );
        worstFrac = std::max( worstFrac, q.closureFrac );
        worstAmp = std::max( worstAmp, q.maxAmp );

        // The bound must hold EVERY round, not just on average: a quality that
        // degrades with depth would show up as a late round crossing it.
        if ( q.minAngleDeg < kMinAngleDeg )
            ++fails;
        if ( q.maxRadiusRatio > kMaxRadiusRatio )
            ++fails;
        if ( q.closureFrac > kMaxClosureFrac )
            ++fails;
        if ( q.maxAmp > kMaxAmplification )
            ++fails;

        // The closure's INPUT must not degrade. The red layer is a pure 4-way
        // subdivision, so this is round-independent by construction -- if it
        // ever moves, the defect is in the red engine and the closure bounds
        // above are measuring someone else's damage.
        if ( q.maxQByPattern[0] > kRedMaxRadiusRatio ||
             q.minAngleByPattern[0] < kRedMinAngleDeg )
            ++fails;
        maxQPerRound.push_back( q.maxRadiusRatio );

        // Conformity must survive all eight rounds too -- a quality bound on a
        // mesh that stopped being conforming would prove nothing.
        int local = TesseraTest::checkConforming( mesh );
        local += TesseraTest::check21BalanceRed( mesh );
        local += TesseraTest::checkSiblingCoresidency( mesh );
        int g = 0;
        MPI_Allreduce( &local, &g, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
        if ( g != 0 )
            ++fails;
        if ( TesseraTest::checkOwnedEuler( mesh ) != 2 )
            ++fails;

        if ( rank == 0 )
        {
            std::printf( "  [%s] quality round%d marked=%lld F=%lld "
                         "minAngle=%.3f deg maxQ=%.4f closure=%lld (%.4f) "
                         "inv=%d | parent minAng=%.3f maxQ=%.4f amp=%.4f | "
                         "maxQ by |S| = [%.4f %.4f %.4f %.4f] minAng by |S| = "
                         "[%.3f %.3f %.3f %.3f]\n",
                         tag, round + 1, gMarked, q.faces, q.minAngleDeg,
                         q.maxRadiusRatio, q.closureChildren, q.closureFrac, g,
                         q.minParentAngle, q.maxParentQ, q.maxAmp,
                         q.maxQByPattern[0], q.maxQByPattern[1],
                         q.maxQByPattern[2], q.maxQByPattern[3],
                         q.minAngleByPattern[0], q.minAngleByPattern[1],
                         q.minAngleByPattern[2], q.minAngleByPattern[3] );
            std::fflush( stdout );
        }
    }

    if ( totalClosure <= 0 )
        ++fails; // vacuous: no closure face was ever produced
    if ( !properSubsetEveryRound )
        ++fails; // vacuous: some round refined nothing, or everything

    // SATURATION. A fixed bound that merely happens to hold for the rounds run
    // is not the claim; the claim is that the worst shape stops growing. The
    // blue family discovers new worst cases in discrete steps for a while (last
    // step measured at round 11 of 16), so require the final kSaturationRounds
    // to introduce no new worst case. This is the assertion 8 rounds could not
    // support and is why the round count is 16.
    double tailWorst = 0.0, headWorst = 0.0;
    const int nR = static_cast<int>( maxQPerRound.size() );
    const bool saturationTestable = nR >= kSaturationRounds + 8;
    if ( saturationTestable )
    {
        for ( int r = 0; r < nR - kSaturationRounds; ++r )
            headWorst = std::max( headWorst, maxQPerRound[r] );
        for ( int r = nR - kSaturationRounds; r < nR; ++r )
            tailWorst = std::max( tailWorst, maxQPerRound[r] );
        // Relative tolerance only: the tail's worst face is a different face
        // from the head's, so exact equality is not the right statement.
        if ( tailWorst > headWorst * ( 1.0 + 1e-6 ) )
            ++fails;
    }

    (void)size;
    if ( rank == 0 )
    {
        std::printf(
            "  [%s] quality over %d rounds %s (worst minAngle=%.3f "
            "deg [bound %.1f], worst Q=%.4f [bound %.2f], worst "
            "closureFrac=%.4f [bound %.2f], worst amp=%.4f [bound "
            "%.2f]) | saturation: last %d rounds worst Q=%.4f vs "
            "first %d rounds %.4f%s\n",
            tag, rounds, fails == 0 ? "ok" : "FAIL", worstAngle, kMinAngleDeg,
            worstQ, kMaxRadiusRatio, worstFrac, kMaxClosureFrac, worstAmp,
            kMaxAmplification, kSaturationRounds, tailWorst,
            nR - kSaturationRounds, headWorst,
            saturationTestable ? "" : " (not testable, too few rounds)" );
        std::fflush( stdout );
    }
    return fails;
}

template <class Exec>
static int run( int rank, int size, const char* tag )
{
    using mem = typename Exec::memory_space;
    using MeshT = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                       mem, Exec, RefinementMode::Conforming>;
    return case_quality<Exec, MeshT>( rank, size, tag );
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
            std::printf( "test_conforming_quality: triangle shape over %d "
                         "adaptive rounds (size %d)\n",
                         roundCount(), size );

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
