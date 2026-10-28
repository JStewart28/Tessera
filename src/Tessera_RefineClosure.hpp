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

#ifndef TESSERA_REFINE_CLOSURE_HPP
#define TESSERA_REFINE_CLOSURE_HPP

#include "Tessera_Fields.hpp"
#include "Tessera_RefinementMode.hpp"
#include "Tessera_Types.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <map>
#include <set>
#include <vector>

namespace Tessera
{

// ============================================================================
// Transient red-green-blue closure (RefinementMode::Conforming)
// ============================================================================
//
// A conforming mesh is maintained as TWO layers:
//
//   RED layer      persistent, authoritative. Faces produced by red 1->4 splits
//                  only, 2:1 level-balanced. Exactly what the hanging-node
//                  refine() produces. Each red face carries its red Level.
//
//   CLOSURE layer  transient, derived. A green / blue / red retriangulation of
//                  every red face that has a hanging node on one of its edges,
//                  so the VISIBLE mesh (what mesh.faces(), the slices, I/O, and
//                  the geometry/stencil API see) has no T-junctions.
//
// The closure is recomputed from scratch on every refine call. It MUST be
// transient: repeatedly bisecting an already-bisected closure triangle degrades
// its shape without bound, whereas a from-scratch closure means every visible
// triangle is a red triangle or one of three fixed retriangulations of one --
// a bounded number of similarity classes regardless of refinement depth.
//
// THE KEY SIMPLIFICATION: closure creates no new vertices. A hanging node on a
// kept face's edge is a midpoint the NEIGHBOUR's red split already created, so
// closure only reconnects existing vertices. Hence no new vertex gids, no
// position/user-field interpolation, no RefinePolicy involvement, and un-closing
// discards no vertex state. Closure is also purely local to one red face, hence
// to one rank (a face is owned by exactly one rank) -- the functions below need
// no communication.
//
// ----------------------------------------------------------------------------
// Closure patterns
// ----------------------------------------------------------------------------
//
// For a red face with corners (a,b,c) and edges e[k] = (v[k], v[(k+1)%3]), let S
// be the subset of its edges that are bisected in the red layer, with midpoints
// m_ab, m_bc, m_ca. The 2:1 balance bounds the level jump to one, so a split edge
// contributes exactly ONE midpoint and |S| in {0,1,2,3}:
//
//   |S|  pattern       children                                            count
//   ---  ------------  --------------------------------------------------  -----
//    0   none          the face is emitted unchanged                         1
//    1   green         (a, m_ab, c), (m_ab, b, c)                            2
//    2   blue          (a, m_ab, c), (m_ab, b, m_bc), (m_ab, m_bc, c)        3
//    3   red-closure   (a,m_ab,m_ca) (b,m_bc,m_ab) (c,m_ca,m_bc) (mid,mid,mid) 4
//
// so a face with |S| split edges is replaced by |S| + 1 visible faces.
//
// Rules:
//   * Every closure child inherits the parent's Level and the parent's face user
//     fields. The closure never modifies the red layer -- in particular the
//     |S| = 3 red-closure does NOT increment Level and does NOT promote the face
//     into the red layer. That deliberately avoids a promotion cascade.
//   * Winding is preserved: the |S| = 1 and |S| = 2 patterns are written for a
//     rotated corner triple (A,B,C) = (v[k], v[k+1], v[k+2]), and a rotation of a
//     CCW triple is CCW, so every child is emitted in the parent's orientation
//     and faceNormalRaw / CurvatureCriterion keep a consistent outward normal.
//   * The blue pattern has TWO valid diagonals across the quad (A, q0, q1, C).
//     The tie-break is GEOMETRIC: take the SHORTER diagonal. That reduces to a
//     rule needing only the two split edges' lengths, never the diagonals' --
//     with q0 = (A+B)/2 and q1 = (B+C)/2,
//
//         |q0-C|^2 - |A-q1|^2 = (3/4) ( |C-B|^2 - |B-A|^2 ),
//
//     so "shorter diagonal" is exactly "connect the midpoint of the LONGER split
//     edge to its opposite corner". It is the standard blue rule and it also
//     gives the better-shaped children (test_conforming_quality measures it).
//
//     WHY LENGTHS ARE PASSED IN RATHER THAN COMPUTED HERE, and it is the whole
//     reason this rule took two attempts (Task 8 D6, Decision 11 in
//     tasks/completed/conforming-refinement.md): the closure runs on the UN-CLOSED red
//     layer, whose corners come from closure children's ClosureParentVerts, and a
//     child may name a vertex gid its rank does not hold -- the documented risk
//     point 4. Those are PARENT corners, not neighbours, so a 1-deep halo does
//     not reach them either (instrumented at np2: one closeFaces() call asked for
//     12 positions the rank did not have, including original icosphere
//     vertices). Evaluating the rule from corner positions therefore aborts at
//     np >= 2. The length must be attached to the EDGE, not derived from the
//     corners: refine()'s Phase 2 already delivers each split edge's midpoint gid
//     from the midpoint owner -- which by construction holds both endpoints of
//     the edge it is bisecting -- to every co-sharer, so it carries the squared
//     length along in the same message (detail::KeyGid). Persistently split edges
//     never appear in that round trip and get their length locally, next to
//     recoverSplitEdges(), from the closure children's own corners.
//
//     Lengths must be BIT-IDENTICAL wherever computed or two ranks can pick
//     different diagonals and crack the mesh, so every producer goes through
//     edgeLen2Canonical() below.
//
//     WHAT THIS BUYS: the diagonal is a function of the mesh GEOMETRY alone, so
//     the visible layer is invariant under both a change of partition and a
//     change of RANK COUNT. The previous rule -- connect the lower-GID midpoint
//     -- was partition-independent but not rank-count independent, because
//     midpoint gids come from an MPI_Exscan and "agreed across the ranks of one
//     run" is not "the same value at a different rank count" (measured: the same
//     red mesh closed with a different blue diagonal on 4 of 20 blue parents at
//     np5, with np1-4 agreeing). test_conforming_determinism case A asserts the
//     invariance, and test_refine_closure drives both diagonals from geometry and
//     pins that RELABELLING the midpoint gids does not move one.
//
//     EXACT TIES are not hypothetical -- the icosphere is highly symmetric and
//     round 1 is the undisturbed icosphere -- so |B-A|^2 == |C-B|^2 falls back to
//     the old lower-midpoint-gid rule, which is deterministic and
//     partition-independent though still rank-count dependent. ClosureStats::
//     nBlueDiagTie counts how often that happens, so the residual dependence is
//     measured rather than assumed away.
//   * Face e[3] follows the existing convention e[k] = edge(v[k], v[(k+1)%3])
//     and is re-derived with the rest of the edge table by the caller.
//
// ----------------------------------------------------------------------------
// Bookkeeping: every child stores its parent outright
// ----------------------------------------------------------------------------
//
// Un-closing must reconstruct the red parent from its children, so each closure
// child carries `ClosureParent` (the parent's gid) and `ClosureParentVerts` (the
// parent's three corner gids, in the parent's winding order) -- see
// ClosureFaceMembers in Tessera_Fields.hpp. Any SINGLE child therefore determines
// its parent with no sibling lookup, which is what lets migrate() move siblings
// without a correctness hazard (only a duplication one -- Task 5).
//
// A red face's gid is RETIRED while it is closed: it lives only in its children's
// ClosureParent field, and un-close reuses it. Closure children take fresh gids
// above the global max face gid, so a retired gid can never collide with a later
// allocation.

//! A face of the RED layer: the persistent, authoritative refinement state.
struct RedFace
{
    //! Corner vertex gids in CCW (outward-seen-from-outside) order.
    GlobalId v[3] = { invalid_gid, invalid_gid, invalid_gid };
    GlobalId gid = invalid_gid;
    //! Red refinement level. A closure child of this face carries this value.
    Level level = 0;
};

//! A face of the VISIBLE layer: either a red face passed through unchanged
//! (`parent == invalid_gid`, `gid` is its own red gid) or one closure child of a
//! red face (`parent` / `parentVerts` name the retired red parent).
struct VisibleFace
{
    GlobalId v[3] = { invalid_gid, invalid_gid, invalid_gid };
    GlobalId gid = invalid_gid;
    Level level = 0;
    GlobalId parent = invalid_gid;
    GlobalId parentVerts[3] = { invalid_gid, invalid_gid, invalid_gid };
};

//! Per-call closure diagnostics. Printed by the conforming tests so the Task-8
//! run yields the closure-face fraction and |S| histogram without a re-run.
struct ClosureStats
{
    //! Red faces whose split-edge count |S| was 0, 1, 2, 3 respectively.
    int patternCount[4] = { 0, 0, 0, 0 };
    //! Visible faces emitted in total.
    int nVisible = 0;
    //! Of those, how many are closure children (parent != invalid_gid).
    int nClosureChildren = 0;
    //! Blue patterns resolved to each diagonal. Both should be non-zero on a
    //! real mesh; an all-or-nothing split hints the tie-break is not doing work.
    int nBlueDiagQ0C = 0; //!< diagonal q0 <-> C, i.e. (A,B) was the longer edge
    int nBlueDiagQ1A = 0; //!< diagonal q1 <-> A, i.e. (B,C) was the longer edge
    //! Blue patterns whose two split edges had EXACTLY equal squared length, so
    //! the geometric rule could not choose and the lower-midpoint-gid fallback
    //! decided. Those are the only blue diagonals that remain rank-count
    //! dependent; a zero here means the visible layer is fully rank-count
    //! invariant on this workload. Counted, not silently tolerated.
    int nBlueDiagTie = 0;
};

//! Squared length of the edge (ga, gb) with endpoint positions pa, pb.
//!
//! THE ONE CANONICAL PRODUCER of a blue tie-break length. The rule compares two
//! such values, so they must be bit-identical no matter which rank evaluates
//! them, in which order that rank holds the two endpoints, or whether the value
//! travelled through Phase 2's coordinator reply or was recovered locally --
//! otherwise two ranks can pick different diagonals for the same quad and crack
//! the mesh. Endpoints are ordered by gid before subtracting so that
//! independence is manifest rather than argued: IEEE negation is exact, so
//! (pa-pb)^2 and (pb-pa)^2 already agree, but the ordering keeps the guarantee
//! if the expression ever grows a term where it would not.
template <class Scalar>
inline double edgeLen2Canonical( GlobalId ga, const Scalar* pa, GlobalId gb,
                                 const Scalar* pb, int dim )
{
    const Scalar* lo = pa;
    const Scalar* hi = pb;
    if ( gb < ga )
    {
        lo = pb;
        hi = pa;
    }
    double s = 0.0;
    for ( int d = 0; d < dim; ++d )
    {
        const double dd =
            static_cast<double>( hi[d] ) - static_cast<double>( lo[d] );
        s += dd * dd;
    }
    return s;
}

//! Add a `len2Of` entry for every edge of `splitEdges` whose two endpoints this
//! rank holds, computed through edgeLen2Canonical() so it is bit-identical to
//! the value any other rank would produce for the same edge.
//!
//! `fetchPos( gid, double p[3] )` returns false when the vertex is not held.
//! Used for the PERSISTENT split-edge map recoverSplitEdges() rebuilds, whose
//! keys never travel through refine()'s Phase 2 and so never carry a length in
//! detail::KeyGid. Those endpoints ARE reachable locally: a closed parent's
//! corners are the union of its children's corners, closure siblings are
//! co-resident (repairClosureCohesion()), and after rebuildHalo() every vertex
//! an owned face references is held WITH ITS POSITION -- see Decision 14.
//!
//! A vertex that is nonetheless missing is skipped rather than fatal: the entry
//! is only ever consumed by the BLUE tie-break, and closeFaces() aborts naming
//! the edge if it actually needs one that is absent. Silence here therefore
//! cannot become a wrong diagonal.
template <class FetchPos>
inline void addSplitEdgeLengths( const std::map<EdgeKey, GlobalId>& splitEdges,
                                 int dim, FetchPos&& fetchPos,
                                 std::map<EdgeKey, double>& len2Of )
{
    double pa[3], pb[3];
    for ( const auto& kv : splitEdges )
    {
        const GlobalId a = kv.first.id[0];
        const GlobalId b = kv.first.id[1];
        if ( !fetchPos( a, pa ) || !fetchPos( b, pb ) )
            continue;
        len2Of[kv.first] = edgeLen2Canonical( a, pa, b, pb, dim );
    }
}

//! Number of VISIBLE faces a red face with `nSplit` bisected edges becomes.
//! |S| = 0 -> 1 (passed through, not a closure child); 1 -> 2 (green);
//! 2 -> 3 (blue); 3 -> 4 (red-closure).
inline int closureChildCount( int nSplit ) { return nSplit + 1; }

//! Look up the midpoints of a red face's three edges in the split-edge map.
//! `mid[k]` receives the midpoint gid of edge k = (v[k], v[(k+1)%3]), or
//! invalid_gid when that edge is not bisected. Returns |S|.
inline int faceSplitEdges( const GlobalId v[3],
                           const std::map<EdgeKey, GlobalId>& midpointOf,
                           GlobalId mid[3] )
{
    int n = 0;
    for ( int k = 0; k < 3; ++k )
    {
        auto it = midpointOf.find( makeEdgeKey( v[k], v[( k + 1 ) % 3] ) );
        if ( it == midpointOf.end() )
        {
            mid[k] = invalid_gid;
        }
        else
        {
            mid[k] = it->second;
            ++n;
        }
    }
    return n;
}

//! Number of NEW face gids closeFaces() will consume for the same arguments.
//! Call this before the gid-allocating MPI_Exscan in the distributed path; a
//! passed-through (|S| = 0) face keeps its own gid and consumes none.
inline std::size_t
countClosureChildren( const std::vector<RedFace>& red,
                      const std::map<EdgeKey, GlobalId>& midpointOf )
{
    std::size_t n = 0;
    GlobalId mid[3];
    for ( const RedFace& p : red )
    {
        const int s = faceSplitEdges( p.v, midpointOf, mid );
        if ( s > 0 )
            n += static_cast<std::size_t>( closureChildCount( s ) );
    }
    return n;
}

//! Output of closeFaces().
struct CloseResult
{
    //! The visible face layer.
    std::vector<VisibleFace> visible;
    //! Parallel to `visible`: the index in the red input each visible face
    //! derives its face USER fields from (a closure child inherits its parent's).
    std::vector<int> sourceRed;
    ClosureStats stats;
};

//! Retriangulate a red face list into the visible (conforming) face list.
//!
//! \param red          the red layer.
//! \param midpointOf   EdgeKey -> midpoint gid for EVERY edge bisected in the
//!                     red layer that this rank touches, including edges of KEPT
//!                     faces bisected by a refining neighbour (that is what the
//!                     Phase-2 extension in refine() exists to supply) and edges
//!                     bisected in an EARLIER round and still carrying a hanging
//!                     node (recoverSplitEdges(), unioned in by refine() step
//!                     0c). Being bisected is a persistent property of the red
//!                     layer: a map holding only this round's bisections closes
//!                     the level jumps created now and leaves every inherited one
//!                     open, so the mesh stops being conforming from round 2.
//! \param firstChildGid  first gid to hand out to closure children; they take
//!                     `countClosureChildren()` consecutive gids from here.
//! \param freshChild   optional, parallel to `red`: non-zero marks a red face
//!                     that was just CREATED by this round's 1->4 split. Its
//!                     interior edges join two vertices created this round, so
//!                     nothing can have bisected them; its two boundary edges are
//!                     halves of a parent edge and may be OLD (when the parent
//!                     edge already carried a hanging node, the split reuses that
//!                     midpoint rather than making a new one), in which case they
//!                     can legitimately be bisected this round. Pass it together
//!                     with `firstNewVertexGid` to have that invariant checked
//!                     here, where |S| is computed anyway; pass an empty vector
//!                     to skip the check.
//! \param firstNewVertexGid  lowest vertex gid created by this round's split.
//!                     Vertex gids are dense, so a vertex is new iff its gid is
//!                     at or above this. Only used for the `freshChild` check;
//!                     invalid_gid (the default) disables it.
//! \param len2Of       EdgeKey -> the WHOLE edge's squared length, from
//!                     edgeLen2Canonical(), for the same key set as
//!                     `midpointOf`. This is what the blue diagonal is chosen
//!                     from; see the header note on the tie-break for why the
//!                     length cannot be computed here from corner positions.
//!                     An entry is required only for a BLUE parent's two split
//!                     edges -- green and red-closure need no tie-break -- and a
//!                     missing one is a hard abort naming the edge, never a
//!                     silent fall-back to the gid rule. Defaulted empty so a
//!                     caller that provably closes no blue face (a single green
//!                     parent in a unit test) need not build it; any caller that
//!                     can produce blue must.
inline CloseResult closeFaces(
    const std::vector<RedFace>& red,
    const std::map<EdgeKey, GlobalId>& midpointOf, GlobalId firstChildGid,
    const std::vector<char>& freshChild = std::vector<char>(),
    GlobalId firstNewVertexGid = invalid_gid,
    const std::map<EdgeKey, double>& len2Of = std::map<EdgeKey, double>() )
{
    CloseResult out;
    out.visible.reserve( red.size() );
    out.sourceRed.reserve( red.size() );
    GlobalId nextGid = firstChildGid;

    for ( std::size_t r = 0; r < red.size(); ++r )
    {
        const RedFace& p = red[r];
        GlobalId mid[3];
        const int nSplit = faceSplitEdges( p.v, midpointOf, mid );
        ++out.stats.patternCount[nSplit];

        // A red child of a face refined in THIS round may only have a split edge
        // on a boundary edge it INHERITED whole, i.e. one whose two endpoints
        // both predate this round. Any edge touching a vertex created this round
        // is brand new and cannot have been bisected.
        if ( nSplit > 0 && r < freshChild.size() && freshChild[r] )
            for ( int k = 0; k < 3; ++k )
                if ( mid[k] != invalid_gid &&
                     ( p.v[k] >= firstNewVertexGid ||
                       p.v[( k + 1 ) % 3] >= firstNewVertexGid ) )
                    Kokkos::abort(
                        "Tessera::closeFaces: a red face freshly created by "
                        "this round's 1->4 split has a bisected edge that "
                        "touches a vertex created by this same round. Such an "
                        "edge is brand new, so this cannot happen unless the "
                        "split-edge map contains an edge that was not present "
                        "in the pre-split red layer." );

        // 2:1 balance precondition: at most ONE midpoint per parent edge. If a
        // half-edge (v[k], mid[k]) or (mid[k], v[k+1]) is itself bisected the
        // level jump exceeds 1 and no fixed closure pattern applies -- fail
        // loudly rather than emit a mesh that still has hanging nodes.
        for ( int k = 0; k < 3; ++k )
        {
            if ( mid[k] == invalid_gid )
                continue;
            if ( midpointOf.count( makeEdgeKey( p.v[k], mid[k] ) ) != 0 ||
                 midpointOf.count(
                     makeEdgeKey( mid[k], p.v[( k + 1 ) % 3] ) ) != 0 )
                Kokkos::abort(
                    "Tessera::closeFaces: an edge of a red face is bisected "
                    "more than once (a half-edge of it is also in the "
                    "split-edge map), i.e. the red layer is not 2:1 balanced. "
                    "The closure patterns assume one midpoint per edge; use "
                    "refine(), which enforces the balance, rather than "
                    "refineLocal() for repeated adaptive rounds." );
        }

        // Emit one CLOSURE CHILD: a fresh gid, the parent's level and user
        // fields, and the parent recorded outright.
        auto child = [&]( GlobalId x, GlobalId y, GlobalId z )
        {
            VisibleFace f;
            f.v[0] = x;
            f.v[1] = y;
            f.v[2] = z;
            f.gid = nextGid++;
            f.level = p.level;
            f.parent = p.gid;
            for ( int k = 0; k < 3; ++k )
                f.parentVerts[k] = p.v[k];
            out.visible.push_back( f );
            out.sourceRed.push_back( static_cast<int>( r ) );
            ++out.stats.nClosureChildren;
        };

        if ( nSplit == 0 )
        {
            // Pass the red face through unchanged: same gid, no closure parent.
            VisibleFace f;
            for ( int k = 0; k < 3; ++k )
                f.v[k] = p.v[k];
            f.gid = p.gid;
            f.level = p.level;
            f.parent = invalid_gid;
            out.visible.push_back( f );
            out.sourceRed.push_back( static_cast<int>( r ) );
        }
        else if ( nSplit == 1 )
        {
            // GREEN. Rotate so the split edge is edge 0 of (A,B,C).
            const int k = ( mid[0] != invalid_gid )
                              ? 0
                              : ( ( mid[1] != invalid_gid ) ? 1 : 2 );
            const GlobalId A = p.v[k];
            const GlobalId B = p.v[( k + 1 ) % 3];
            const GlobalId C = p.v[( k + 2 ) % 3];
            const GlobalId m = mid[k];
            child( A, m, C );
            child( m, B, C );
        }
        else if ( nSplit == 2 )
        {
            // BLUE. Rotate so the UNSPLIT edge is edge 2 = (C,A): with `u` the
            // unsplit edge index, rot = (u+1)%3 puts edge u at position 2.
            const int u = ( mid[0] == invalid_gid )
                              ? 0
                              : ( ( mid[1] == invalid_gid ) ? 1 : 2 );
            const int rot = ( u + 1 ) % 3;
            const GlobalId A = p.v[rot];
            const GlobalId B = p.v[( rot + 1 ) % 3];
            const GlobalId C = p.v[( rot + 2 ) % 3];
            const GlobalId q0 = mid[rot];             // midpoint of (A,B)
            const GlobalId q1 = mid[( rot + 1 ) % 3]; // midpoint of (B,C)
            // Cut off the corner triangle at B, then split the remaining quad
            // (A, q0, q1, C) along the SHORTER diagonal -- equivalently (see the
            // header note) connect the midpoint of the LONGER split edge to its
            // opposite corner: q0 <-> C when (A,B) is longer, else q1 <-> A.
            auto len2At = [&]( GlobalId x, GlobalId y ) -> double
            {
                auto lit = len2Of.find( makeEdgeKey( x, y ) );
                if ( lit != len2Of.end() )
                    return lit->second;
                // Name the edge before dying. D6's version of this returned a
                // default on a miss and turned a hard failure into a plausible
                // mesh that failed one check; the version that printed the
                // missing key was diagnosed in one run.
                std::fprintf(
                    stderr,
                    "Tessera::closeFaces: missing len2 for split edge "
                    "(%llu,%llu) of blue parent gid %llu\n",
                    static_cast<unsigned long long>( x ),
                    static_cast<unsigned long long>( y ),
                    static_cast<unsigned long long>( p.gid ) );
                Kokkos::abort(
                    "Tessera::closeFaces: no squared length for a split "
                    "edge of a BLUE closure parent. The blue diagonal is "
                    "chosen geometrically, from the two split edges' "
                    "lengths, so len2Of must cover every split edge of "
                    "every blue parent -- an empty or partial map must NOT "
                    "silently fall back to the midpoint-gid rule, which is "
                    "not rank-count stable. In the distributed path the "
                    "length rides along with the midpoint gid in Phase 2's "
                    "coordinator reply (detail::KeyGid); a persistently "
                    "split edge gets it locally next to "
                    "recoverSplitEdges()." );
                return 0.0; // unreachable; Kokkos::abort() does not return
            };
            const double lAB = len2At( A, B );
            const double lBC = len2At( B, C );
            bool diagQ0C;
            if ( lAB != lBC )
            {
                diagQ0C = ( lAB > lBC );
            }
            else
            {
                // Exact geometric tie (a symmetric mesh -- the undisturbed
                // icosphere has many). Fall back to the old lower-midpoint-gid
                // rule: deterministic and partition-independent, but gid-valued
                // and so rank-count dependent, which is what the counter says.
                diagQ0C = ( q0 < q1 );
                ++out.stats.nBlueDiagTie;
            }
            if ( diagQ0C )
            {
                child( A, q0, C );
                child( q0, B, q1 );
                child( q0, q1, C );
                ++out.stats.nBlueDiagQ0C;
            }
            else
            {
                child( A, q0, q1 );
                child( q0, B, q1 );
                child( A, q1, C );
                ++out.stats.nBlueDiagQ1A;
            }
        }
        else
        {
            // RED-CLOSURE: the 1->4 child convention, but these stay in the
            // CLOSURE layer -- level is NOT incremented and the parent is NOT
            // promoted, so no refinement cascade is triggered.
            child( p.v[0], mid[0], mid[2] );
            child( p.v[1], mid[1], mid[0] );
            child( p.v[2], mid[2], mid[1] );
            child( mid[0], mid[1], mid[2] );
        }
    }

    out.stats.nVisible = static_cast<int>( out.visible.size() );
    return out;
}

//! Output of unclose().
struct UncloseResult
{
    //! The restored red layer, in first-encounter order over the visible input.
    std::vector<RedFace> red;
    //! Parallel to `red`: the visible index its face USER fields come from --
    //! itself for a passed-through red face, else the LOWEST-GID child of the
    //! parent (all children inherited an identical copy, so any would do; the
    //! lowest gid makes the choice deterministic and partition-independent).
    std::vector<int> sourceVisible;
    //! Parallel to the visible input: which entry of `red` each visible face
    //! belongs to. This is what translateMask() uses.
    std::vector<int> redOfVisible;
    //! The PERSISTENT split-edge map, recovered from the closure bookkeeping:
    //! for every red face that was closed, which of its edges is bisected in the
    //! red layer and at which midpoint gid. See recoverSplitEdges() for why this
    //! is exact, and Tessera_RefineParallel.hpp step 0c for why it is needed --
    //! "this edge is bisected" is a persistent property of the red layer, but
    //! refine()'s Phase 2 only ever learns the edges bisected in the CURRENT
    //! round.
    std::map<EdgeKey, GlobalId> splitEdges;
};

//! Recover which edges of one closed red parent are bisected, and at which
//! midpoint, from the parent's closure children alone.
//!
//! Two facts make this exact and communication-free:
//!
//!   * A parent edge is split IFF it is not an edge of any child. Green replaces
//!     (a,b) by (a,m),(m,b) and keeps (b,c) and (c,a); blue keeps only the one
//!     unsplit edge; red-closure keeps none; |S| = 0 emits no children at all.
//!   * The midpoint of a split parent edge (x,y) is the unique child corner `m`
//!     outside {a,b,c} for which (x,m) and (m,y) are both edges of exactly ONE
//!     child. The one-child qualifier is essential: an edge shared by two
//!     children is a DIAGONAL of the parent's fan, not part of its boundary, and
//!     without it the blue pattern is ambiguous -- in the q0 < q1 blue, both
//!     (q0,B) and (q0,C) exist, so q0 would spuriously answer for edge (B,C)
//!     as well as for (A,B). Boundary edges of the fan appear exactly once.
//!
//! Appends into `splitEdges`. Aborts if the recovery is not unique, which means
//! the child set handed in is not one whole closure family (e.g. siblings split
//! across ranks -- see repairClosureCohesion()).
inline void recoverSplitEdges( const GlobalId parentVerts[3],
                               const std::vector<const VisibleFace*>& children,
                               std::map<EdgeKey, GlobalId>& splitEdges )
{
    std::map<EdgeKey, int> childEdgeCount;
    std::set<GlobalId> midCandidates;
    const std::set<GlobalId> corners = { parentVerts[0], parentVerts[1],
                                         parentVerts[2] };
    for ( const VisibleFace* c : children )
        for ( int k = 0; k < 3; ++k )
        {
            ++childEdgeCount[makeEdgeKey( c->v[k], c->v[( k + 1 ) % 3] )];
            if ( corners.count( c->v[k] ) == 0 )
                midCandidates.insert( c->v[k] );
        }

    int nSplit = 0;
    for ( int k = 0; k < 3; ++k )
    {
        const GlobalId x = parentVerts[k];
        const GlobalId y = parentVerts[( k + 1 ) % 3];
        if ( childEdgeCount.count( makeEdgeKey( x, y ) ) != 0 )
            continue; // the parent edge survived => it is not bisected
        ++nSplit;

        GlobalId found = invalid_gid;
        int nFound = 0;
        for ( const GlobalId m : midCandidates )
        {
            auto h0 = childEdgeCount.find( makeEdgeKey( x, m ) );
            auto h1 = childEdgeCount.find( makeEdgeKey( m, y ) );
            if ( h0 != childEdgeCount.end() && h0->second == 1 &&
                 h1 != childEdgeCount.end() && h1->second == 1 )
            {
                found = m;
                ++nFound;
            }
        }
        if ( nFound != 1 )
            Kokkos::abort(
                "Tessera::recoverSplitEdges: could not uniquely recover the "
                "midpoint of a bisected parent edge from its closure children. "
                "The children handed in are not one complete closure family -- "
                "either the sibling set is split across ranks (migrate() must "
                "keep closure siblings co-resident; see "
                "repairClosureCohesion()) or a child's ClosureParentVerts do "
                "not name its actual parent." );
        splitEdges[makeEdgeKey( x, y )] = found;
    }

    if ( static_cast<int>( children.size() ) != closureChildCount( nSplit ) )
        Kokkos::abort( "Tessera::recoverSplitEdges: a closure family's child "
                       "count does not match the |S| implied by which of its "
                       "parent's edges survived in the children." );
}

//! Inverse of closeFaces(): collapse the visible layer back to the red layer.
//! Removes only faces -- never a vertex -- so the transient closure discards no
//! vertex state. Purely local per child: a child determines its parent outright.
//! Also recovers the persistent split-edge map (`splitEdges`) from the closure
//! bookkeeping -- see recoverSplitEdges().
inline UncloseResult unclose( const std::vector<VisibleFace>& visible )
{
    UncloseResult out;
    out.redOfVisible.assign( visible.size(), -1 );
    std::map<GlobalId, int> redOfParent; // retired parent gid -> index in `red`
    std::set<GlobalId> redGids;          // duplicate-gid guard
    // Closure children grouped by their red index, for the split-edge recovery.
    std::map<int, std::vector<const VisibleFace*>> childrenOfRed;

    for ( std::size_t i = 0; i < visible.size(); ++i )
    {
        const VisibleFace& f = visible[i];

        if ( f.parent == invalid_gid )
        {
            // A red face: passes through with its own gid, level, and corners.
            RedFace rf;
            for ( int k = 0; k < 3; ++k )
                rf.v[k] = f.v[k];
            rf.gid = f.gid;
            rf.level = f.level;
            if ( !redGids.insert( rf.gid ).second )
                Kokkos::abort(
                    "Tessera::unclose: two restored red faces share a gid. "
                    "Either a live red face's gid collides with some closure "
                    "child's ClosureParent (a retired parent gid was reissued "
                    "instead of being allocated above the global max face "
                    "gid), "
                    "or the closure bookkeeping members were never initialized "
                    "on a face -- see initClosureFaceMembers()." );
            out.redOfVisible[i] = static_cast<int>( out.red.size() );
            out.red.push_back( rf );
            out.sourceVisible.push_back( static_cast<int>( i ) );
            continue;
        }

        auto it = redOfParent.find( f.parent );
        if ( it == redOfParent.end() )
        {
            RedFace rf;
            for ( int k = 0; k < 3; ++k )
                rf.v[k] = f.parentVerts[k];
            rf.gid = f.parent;
            // Closure children carry the parent's level verbatim.
            rf.level = f.level;
            if ( !redGids.insert( rf.gid ).second )
                Kokkos::abort( "Tessera::unclose: a restored closure parent's "
                               "gid collides with a live red face's gid (a "
                               "retired parent gid was reissued)." );
            const int ri = static_cast<int>( out.red.size() );
            redOfParent.emplace( f.parent, ri );
            out.red.push_back( rf );
            out.sourceVisible.push_back( static_cast<int>( i ) );
            out.redOfVisible[i] = ri;
            childrenOfRed[ri].push_back( &f );
        }
        else
        {
            const int ri = it->second;
            out.redOfVisible[i] = ri;
            childrenOfRed[ri].push_back( &f );
            if ( f.gid < visible[out.sourceVisible[ri]].gid )
                out.sourceVisible[ri] = static_cast<int>( i );
        }
    }

    // Recover the persistent split-edge map from the closure families. A
    // passed-through red face has no children and, by construction, |S| = 0.
    for ( const auto& kv : childrenOfRed )
        recoverSplitEdges( out.red[kv.first].v, kv.second, out.splitEdges );

    return out;
}

//! Translate a refine mask indexed by VISIBLE owned faces (what markByQuality
//! and every caller produce) into one indexed by the RED faces unclose()
//! restored: a red parent is marked iff ANY of its closure children was.
inline std::vector<char> translateMask( const std::vector<char>& visibleMask,
                                        const UncloseResult& un )
{
    std::vector<char> redMask( un.red.size(), 0 );
    const std::size_t n =
        std::min( visibleMask.size(), un.redOfVisible.size() );
    for ( std::size_t i = 0; i < n; ++i )
        if ( visibleMask[i] && un.redOfVisible[i] >= 0 )
            redMask[un.redOfVisible[i]] = 1;
    return redMask;
}

// ----------------------------------------------------------------------------
// Mesh <-> plain-struct bridges
// ----------------------------------------------------------------------------

//! Read faces [0, n) of a host copy of a Conforming mesh's face AoSoA into the
//! plain VisibleFace structs the closure kernel operates on.
template <class MeshT, class HostFaceAoSoA>
std::vector<VisibleFace> readVisibleFaces( HostFaceAoSoA& hf, std::size_t n )
{
    static_assert( MeshT::refinement_mode == RefinementMode::Conforming,
                   "readVisibleFaces() is meaningful only for a "
                   "RefinementMode::Conforming mesh -- a HangingNode2to1 face "
                   "carries no closure bookkeeping members" );
    auto fv = Cabana::slice<FaceField::Verts>( hf );
    auto fg = Cabana::slice<FaceField::Gid>( hf );
    auto fl = Cabana::slice<FaceField::Level>( hf );
    auto cp = Cabana::slice<MeshT::closure_parent_field>( hf );
    auto cv = Cabana::slice<MeshT::closure_parent_verts_field>( hf );

    std::vector<VisibleFace> out( n );
    for ( std::size_t f = 0; f < n; ++f )
    {
        for ( int k = 0; k < 3; ++k )
        {
            out[f].v[k] = fv( f, k );
            out[f].parentVerts[k] = cv( f, k );
        }
        out[f].gid = fg( f );
        out[f].level = fl( f );
        out[f].parent = cp( f );
    }
    return out;
}

//! Write the closure bookkeeping members of host face `f` from a VisibleFace.
//! A no-op in RefinementMode::HangingNode2to1 (the members do not exist), so
//! generic face-materialization code can call it unconditionally.
template <class MeshT, class HostFaceAoSoA>
void writeClosureFace( HostFaceAoSoA& hf, std::size_t f, const VisibleFace& vf )
{
    if constexpr ( MeshT::refinement_mode == RefinementMode::Conforming )
    {
        auto cp = Cabana::slice<MeshT::closure_parent_field>( hf );
        auto cv = Cabana::slice<MeshT::closure_parent_verts_field>( hf );
        cp( f ) = vf.parent;
        for ( int k = 0; k < 3; ++k )
            cv( f, k ) = vf.parentVerts[k];
    }
    else
    {
        (void)hf;
        (void)f;
        (void)vf;
    }
}

//! Initialize faces [begin, end) of a host face AoSoA to "not a closure child":
//! ClosureParent = ClosureParentVerts[k] = invalid_gid. A no-op in
//! RefinementMode::HangingNode2to1.
//!
//! Every site that MATERIALIZES faces without going through the closure kernel
//! must call this. A Cabana AoSoA's backing View is zero-initialized, so an
//! uninitialized ClosureParent reads as gid 0 -- a perfectly valid face gid --
//! and unclose() would silently restore a bogus parent for every face. That is
//! why this is spelled as an explicit call rather than left to the allocator.
template <class MeshT, class HostFaceAoSoA>
void initClosureFaceMembers( HostFaceAoSoA& hf, std::size_t begin,
                             std::size_t end )
{
    if constexpr ( MeshT::refinement_mode == RefinementMode::Conforming )
    {
        auto cp = Cabana::slice<MeshT::closure_parent_field>( hf );
        auto cv = Cabana::slice<MeshT::closure_parent_verts_field>( hf );
        for ( std::size_t f = begin; f < end; ++f )
        {
            cp( f ) = invalid_gid;
            for ( int k = 0; k < 3; ++k )
                cv( f, k ) = invalid_gid;
        }
    }
    else
    {
        (void)hf;
        (void)begin;
        (void)end;
    }
}

} // namespace Tessera

#endif // TESSERA_REFINE_CLOSURE_HPP
