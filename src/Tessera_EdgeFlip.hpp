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

#ifndef TESSERA_EDGE_FLIP_HPP
#define TESSERA_EDGE_FLIP_HPP

#include "Tessera_AllToAllV.hpp"
#include "Tessera_Distribute.hpp"
#include "Tessera_EditFamily.hpp"
#include "Tessera_Fields.hpp"
#include "Tessera_HaloRebuild.hpp"
#include "Tessera_Mesh.hpp"
#include "Tessera_Profiling.hpp"
#include "Tessera_Reduction.hpp"
#include "Tessera_RefineClosure.hpp"
#include "Tessera_RefineParallel.hpp"
#include "Tessera_Types.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Tessera
{

// ============================================================================
// Caller-driven edge flip (REMESH editing family)
// ============================================================================
//
// flipEdges() replaces the diagonal of the quad formed by the two faces
// incident on each marked edge. For an interior manifold edge (a,b) whose
// incident faces are (a,b,c) and (b,a,d), the quad boundary is a->d->b->c and
// the flip swaps the diagonal (a,b) for (c,d):
//
//        c                 c
//       /|\               / \
//      / | \             /   \
//     a  |  b   --->    a-----b
//      \ | /             \   /
//       \|/               \ /
//        d                 d
//
// V, E and F are ALL UNCHANGED. Nothing is created and nothing is destroyed --
// only connectivity moves. In particular:
//
//   * the flipped edge KEEPS ITS GID and gets new endpoints, so its EdgeKey
//     changes and with it its detail::edgeCoordRank routing;
//   * the two faces KEEP THEIR GIDS and get new corners and new edges;
//   * the vertex->edge and vertex->face CSRs change at all four of a, b, c, d.
//
// DECISION 1 -- GIDS ARE PRESERVED, so a gid no longer determines a key.
// Preserving gids means no peer's reference is invalidated by a flip, which is
// what makes the operation cheap; the price is that `gid <-> key` is NOT a
// bijection across a flip. Nothing in Tessera assumes it is -- the key side
// tables (Mesh::edgeKeys()/faceKeys()) are rebuilt from Verts by rebuildHalo()
// round D, and every gid-keyed path (halo plans, migrate(), the HDF5 I/O) is
// keyed on gid alone -- but a CALLER that cached "the edge whose key is K has
// gid G" across a flipEdges() call is wrong afterwards.
//
// ----------------------------------------------------------------------------
// The orientation-preserving rewrite, and which face keeps which gid
// ----------------------------------------------------------------------------
//
// WINDING IS NOT A FREE CHOICE. With the two incident faces written as
// (u,v,w) and (v,u,x) -- i.e. the first traverses the edge u->v and the second
// v->u, which is exactly what a consistently oriented manifold guarantees --
// the quad's boundary cycle is u->x->v->w and the only orientation-preserving
// retriangulation on the diagonal (w,x) is
//
//     (u, x, w)   and   (v, w, x)
//
// (tasks/completed/edge-flip.md's "(a,c,d) and (b,d,c)" is the REVERSED pair; see the
// progress log). The new edge (w,x) is traversed x->w in the first and w->x in
// the second, so the result is manifold and consistently oriented.
//
// WHICH OF THE TWO OLD GIDS LANDS ON WHICH NEW FACE is not free either: check 2
// of the test requires a flip to be its OWN INVERSE down to the face gids, and
// most rules fail that. The rule that works is stated entirely in canonical
// (gid-ordered) terms. Write p = min(u,v), q = max(u,v) -- i.e. the EdgeKey's
// two ids -- and r = min(w,x), s = max(w,x). Then
//
//     the old face {p,q,r}  keeps its gid on the new face {p,r,s}
//     the old face {p,q,s}  keeps its gid on the new face {q,r,s}
//
// "the face opposite the SMALLER opposite-vertex goes to the new face
// containing the SMALLER endpoint". Applying the rule to the flipped edge
// (r,s) maps {r,s,p} back onto {p,q,r} and {r,s,q} back onto {p,q,s}, so a
// second flip of the same edge restores the original (corners, gid) pairing
// exactly. Level and every face user field ride along with the gid.
//
// ----------------------------------------------------------------------------
// Distributed structure -- the owner decides, both face owners apply
// ----------------------------------------------------------------------------
//
// A flip needs BOTH incident faces, and the 1-deep vertex halo does not
// guarantee they are co-resident: an owned face all three of whose corners are
// ghosts can have edge-neighbours held on neither this rank nor on any single
// other rank. So the decision is made at a THIRD rank -- the same
// detail::edgeCoordRank coordinator refine()'s 2:1 balance and splitEdges()'
// mask agreement use -- and is then shipped to whoever has to write it.
//
//   1. ADVERTISE. Every rank sends, per OWNED FACE and per each of its three
//      edges, a detail::FlipAdvert to that edge's coordinator: the EdgeKey, the
//      face's gid and owner, the gid of the edge itself, the gids of the two
//      OTHER edges of that face, the opposite corner's gid, a winding flag, and
//      the POSITIONS of all three corners. Separately, the OWNER of each marked
//      edge sends a detail::FlipVerdict -- presence at the coordinator is the
//      verdict, as in splitEdges().
//
//      CARRYING THE POSITIONS IS THE POINT, and it is where this operation
//      departs from splitEdges(). splitEdges() deliberately does NOT put a
//      length in a message (its progress log, 2026-08-10, departure 2) because
//      the operands of its one geometric decision are the deciding face's own
//      corners, which rebuildHalo() guarantees are held locally. That reasoning
//      does not transfer: the flip coordinator is a third rank that holds
//      NEITHER incident face, and it needs all four of a, b, c, d to evaluate
//      the policy and the priority. The in-tree precedent for a coordinator
//      message with a geometric rider is detail::KeyGid's `len2`
//      (Tessera_RefineParallel.hpp), documented in the comment above it.
//
//   2. DECIDE, AT THE COORDINATOR. For each marked edge the coordinator holds
//      both adverts and therefore a, b, c, d and all four positions:
//        BOUNDARY   -- other than exactly two incident faces: reject.
//        DUPLICATE  -- does (c,d) already exist? Flipping into an existing edge
//                      produces a non-manifold mesh, and on a coarse mesh it
//                      happens routinely (any valence-3 vertex). The (c,d)
//                      coordinator is a DIFFERENT rank in general, so this
//                      costs one extra round: the deciding coordinator asks,
//                      and only a negative answer lets the flip through. The
//                      answer is exact because step 1's advertisement is over
//                      EVERY owned face's three edges, so the (c,d)
//                      coordinator has seen the whole global edge set.
//        GEOMETRIC  -- the Policy's normal-deviation and quality tests, from
//                      the four positions.
//
//   3. INDEPENDENT SET. Two flips sharing a FACE conflict -- that face would be
//      rewritten twice -- and two sharing only a vertex do not. No graph
//      traversal is needed: each face owner sees its own three edges' verdicts
//      and keeps only the highest-priority candidate, then endorses it back
//      through the coordinator, which accepts an edge only when BOTH of its
//      faces endorse it.
//
//      DECISION 2 -- THE PRIORITY IS (squared length DESCENDING, EdgeKey
//      ASCENDING) AND CONTAINS NO GID. Gids come from an MPI_Exscan and are
//      agreed across the ranks of ONE run but are not the same values at a
//      different rank count -- exactly the finding that forced splitEdges()'
//      diagonal tie-break away from the closure's lower-midpoint-gid rule (its
//      progress log, 2026-08-10, departure 1). An EdgeKey is built from
//      pre-existing vertex gids and IS invariant, and every length is computed
//      through edgeLen2Canonical() (Tessera_RefineClosure.hpp) so two ranks
//      comparing the same edge get bit-identical doubles regardless of endpoint
//      order. The pair is a total order on distinct edges, so a face can endorse
//      at most one candidate and the accepted set is a pure function of the
//      global mesh.
//
//      The result is an independent set, NOT a maximal one: an edge that loses
//      on face F is not reconsidered when F's winner is itself vetoed by its
//      other face. That is deliberate (Decision 3 below) -- one round, one
//      bounded cost.
//
//   4. APPLY. Each face owner rewrites its face's Verts/Edges in place and the
//      edge owner rewrites the flipped edge's Verts/Faces; nothing is resized.
//      setOwnedCounts() with the UNCHANGED counts bumps the generation so stale
//      GenerationGuard handles are caught.
//
//   5. rebuildHalo() at the halo's recorded depth. Round D of the rebuild
//      already redoes edgeKeys(), faceKeys() and both CSRs from the owned
//      entities' Verts, so there is nothing for this operation to rebuild by
//      hand -- the finalize sequence is Tessera_EdgeSplit.hpp's, verbatim.
//      Round G is what supplies the position of a corner a face acquired from
//      the far side of the quad and does not yet hold.
//
// DECISION 3 -- ONE INDEPENDENT SET PER CALL, NOT A LOOP TO EXHAUSTION.
// Iterating inside flipEdges() would hide an unbounded number of collectives
// behind one call and make the cost unpredictable. THE CALLER LOOPS:
//
//     for ( int round = 0; round < maxRounds; ++round )
//     {
//         auto mask = myEdgeSelection( mesh );   // e.g. valence equalization
//         auto res = flipEdges( mesh, halo, mask );
//         if ( res.accepted == 0 ) break;
//     }
//
// which is also what lets the caller re-derive valences between rounds.
//
// VALENCE SELECTION IS THE CALLER'S, NOT TESSERA'S (task Decision 2). The
// dominant use of flipping is valence equalization, which needs the FULL
// valence of a, b, c and d. A vertex's valence is a local quantity at its owner
// -- the vertex->edge CSR's owned rows are complete at halo depth 1 -- so the
// caller computes it and encodes it in the mask. Keeping it out means
// flipEdges() itself needs ONLY DEPTH 1: every test above is evaluable from the
// two incident faces, which the coordinator supplies.
//
// THE RESULT DIFFERS FROM A SERIAL PASS. A serial "rebuild the edge map after
// each accepted flip" sweep is inherently sequential and order-dependent; the
// independent-set result is not the same edit set, and both are valid. A
// consumer must compare QUALITY STATISTICS, not flip sets.
//
// EdgeField::Faces IS NOT TRUSTED AS INPUT and is only partly repaired on
// output. It is best-effort by design -- migrate() carries it verbatim, so it
// can name a face no rank holds (Tessera_FaceAdjacency.hpp) -- so incidence
// here is derived from the ADVERTISEMENTS, which are built from owned faces and
// are therefore true. On output the FLIPPED edge's Faces is rewritten exactly
// (step 4). The six SIDE edges of a flipped quad keep the Faces they had, and
// one entry of each may now name the flip's sibling face rather than the face
// that actually holds it -- both gids are live faces of the same quad, which is
// within the field's best-effort contract, and repairing it would cost a round
// on a field no code path in src/ reads for correctness.
//
// Preconditions: `mesh` is distributed (post-distribute), owned-first, entities
// carry global gids, the halo is valid (rebuildHalo() has run), and
// `edgeMask.size() == mesh.numOwnedEdges()`. Collective.

//! Result of flipEdges(). Every counter is GLOBAL and the five verdict
//! counters partition `requested`.
struct FlipResult
{
    //! Marked OWNED edges, summed globally.
    long long requested = 0;
    //! Edges actually flipped, globally.
    long long accepted = 0;
    //! Rejected: other than exactly two incident faces (a boundary edge has
    //! one; a non-manifold edge has more, and cannot occur in a mesh Tessera
    //! built).
    long long rejectedBoundary = 0;
    //! Rejected: the edge the flip would create already exists.
    long long rejectedDuplicateEdge = 0;
    //! Rejected: the Policy's normal-deviation or quality test failed.
    long long rejectedGeometric = 0;
    //! Rejected: passed every test but lost the independent-set round to a
    //! higher-priority candidate on one of its two faces.
    long long rejectedConflict = 0;

    //! THE FLIP MAP: (old EdgeKey, new EdgeKey) for every accepted flip this
    //! rank TOUCHES -- i.e. one whose flipped edge or either incident face it
    //! owns. Sorted and unique by the old key. Lets a caller check the
    //! independent-set and key-bookkeeping properties without instrumenting the
    //! library, the same role SplitResult::midpoints plays for splitEdges().
    std::vector<std::pair<EdgeKey, EdgeKey>> flipped;
};

//! Geometric admissibility of a candidate flip, evaluated at the coordinator
//! from the four corner positions of the quad.
struct DefaultFlipPolicy
{
    //! Reject if either new face's normal points more than this far from the
    //! area-weighted average of the two old normals (radians). Guards against
    //! flipping a nearly-flat pair into a fold.
    double maxNormalDeviation = 0.35;
    //! Reject if either new face's radius ratio (inradius/circumradius, scaled
    //! to 1 for an equilateral triangle) falls below this.
    double minQuality = 0.05;
};

namespace detail
{

//! One (owned face, one of its three edges) advertisement. Carries everything
//! the coordinator needs to decide and to describe the rewrite back to both
//! face owners without a second query: the participant identity, the three edge
//! gids of the face, the opposite corner, the winding, and the three corner
//! POSITIONS (see the header comment on why the positions travel).
struct FlipAdvert
{
    EdgeKey key;        //!< the advertised edge, canonical (lo, hi)
    GlobalId faceGid;   //!< the advertising face
    GlobalId oppGid;    //!< the face corner not on `key`
    GlobalId edgeGid;   //!< gid of `key` as this face holds it
    GlobalId edgeOppLo; //!< gid of the face's edge (oppGid, key.id[0])
    GlobalId edgeOppHi; //!< gid of the face's edge (oppGid, key.id[1])
    Rank owner;         //!< the face's owner (the advertising rank)
    //! 1 if the face traverses key.id[0] -> key.id[1], 0 if the reverse.
    //! Exactly one of an edge's two faces is forward in an oriented manifold.
    unsigned char forward;
    double posLo[3]; //!< position of key.id[0]
    double posHi[3]; //!< position of key.id[1]
    double posOpp[3];
};

//! (edge, its gid, its owner) sent by the OWNER of each MARKED edge. Presence
//! at the coordinator IS the verdict, as in splitEdges()' SplitVerdict; the gid
//! and owner ride along so the coordinator can address the apply message for
//! the edge itself, whose owner need not own either incident face.
struct FlipVerdict
{
    EdgeKey key;
    GlobalId edgeGid;
    Rank owner;
};

//! "Does `target` exist as an edge?", asked by the coordinator of `asker` of
//! the coordinator of `target`. The reply travels back over the same pairing.
struct DupQuery
{
    EdgeKey target;
    EdgeKey asker;
};
//! The answer to a DupQuery, addressed by the asked-about candidate.
struct DupReply
{
    EdgeKey asker;
    unsigned char exists;
};

//! (face, a candidate edge of it, that edge's squared length) delivered to the
//! face's owner, which keeps only the highest-priority candidate per face.
struct FaceCand
{
    GlobalId faceGid;
    EdgeKey key;
    double len2;
};

//! A face owner's endorsement of the one candidate it kept, returned to that
//! edge's coordinator. An edge is accepted iff BOTH its faces endorse it.
struct FaceVote
{
    EdgeKey key;
    GlobalId faceGid;
};

//! The rewrite of one face, delivered to that face's owner. `v` and `e` are the
//! new corner gids and the new edge gids, in matching order (edge k is
//! (v[k], v[(k+1)%3])).
struct FaceApply
{
    GlobalId faceGid;
    GlobalId v[3];
    GlobalId e[3];
};

//! The rewrite of the flipped edge, delivered to the EDGE's owner.
struct EdgeApply
{
    GlobalId edgeGid;
    GlobalId v[2]; //!< the new endpoints, canonical (lo, hi)
    GlobalId f[2]; //!< the two (unchanged) incident face gids, ascending
};

//! (old key, new key) for one accepted flip, delivered to every participant so
//! FlipResult::flipped is complete on each rank that touches the flip.
struct FlipNote
{
    EdgeKey oldKey;
    EdgeKey newKey;
};

//! Twice the area-weighted normal of triangle (p0,p1,p2): (p1-p0) x (p2-p0).
inline void flipTriNormal( const double* p0, const double* p1,
                           const double* p2, double n[3] )
{
    const double u[3] = { p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] };
    const double v[3] = { p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2] };
    n[0] = u[1] * v[2] - u[2] * v[1];
    n[1] = u[2] * v[0] - u[0] * v[2];
    n[2] = u[0] * v[1] - u[1] * v[0];
}

//! Angle in radians between two vectors, or -1 if either is degenerate.
inline double flipAngleBetween( const double a[3], const double b[3] )
{
    const double na = std::sqrt( a[0] * a[0] + a[1] * a[1] + a[2] * a[2] );
    const double nb = std::sqrt( b[0] * b[0] + b[1] * b[1] + b[2] * b[2] );
    if ( !( na > 0.0 ) || !( nb > 0.0 ) )
        return -1.0;
    double c = ( a[0] * b[0] + a[1] * b[1] + a[2] * b[2] ) / ( na * nb );
    c = std::max( -1.0, std::min( 1.0, c ) );
    return std::acos( c );
}

//! Radius ratio (inradius/circumradius) of a triangle, SCALED TO 1 for an
//! equilateral one: 2 * (4 area^2) / (s * abc). 0 for a degenerate triangle.
inline double flipRadiusRatio( const double* p0, const double* p1,
                               const double* p2 )
{
    const double* p[3] = { p0, p1, p2 };
    double side[3];
    for ( int k = 0; k < 3; ++k )
    {
        double s = 0.0;
        for ( int d = 0; d < 3; ++d )
        {
            const double dd = p[( k + 1 ) % 3][d] - p[k][d];
            s += dd * dd;
        }
        side[k] = std::sqrt( s );
    }
    double n[3];
    flipTriNormal( p0, p1, p2, n );
    const double area =
        0.5 * std::sqrt( n[0] * n[0] + n[1] * n[1] + n[2] * n[2] );
    const double abc = side[0] * side[1] * side[2];
    const double s = 0.5 * ( side[0] + side[1] + side[2] );
    if ( !( abc > 0.0 ) || !( s > 0.0 ) )
        return 0.0;
    return 2.0 * ( 4.0 * area * area ) / ( s * abc );
}

//! Implementation of flipEdges(); see the header comment for the algorithm and
//! the guarantees.
template <class MeshT, class Policy>
FlipResult flipEdgesImpl( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
                          const std::vector<char>& edgeMask,
                          const Policy& policy )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_REFINE );
    constexpr int Dim = MeshT::dim;
    constexpr int D3 = Dim < 3 ? Dim : 3;
    const int R = mesh.rank();
    const int size = mesh.commSize();
    MPI_Comm comm = mesh.comm();

    const int nv = static_cast<int>( mesh.numVertices() ); // owned + ghost
    const int nOwnedE = static_cast<int>( mesh.numOwnedEdges() );
    const int nOwnedF = static_cast<int>( mesh.numOwnedFaces() );

    FlipResult result;

    // ---- host copies: all vertices (position lookup) + all edges/faces ------
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", nv );
    Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> he(
        "he", mesh.numEdges() );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( he, mesh.edges() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto v_gid = Cabana::slice<VertexField::Gid>( hv );
    auto v_pos = Cabana::slice<VertexField::Position>( hv );
    auto e_gid = Cabana::slice<EdgeField::Gid>( he );
    auto e_verts = Cabana::slice<EdgeField::Verts>( he );
    auto e_faces = Cabana::slice<EdgeField::Faces>( he );
    auto f_gid = Cabana::slice<FaceField::Gid>( hf );
    auto f_verts = Cabana::slice<FaceField::Verts>( hf );
    auto f_edges = Cabana::slice<FaceField::Edges>( hf );

    std::unordered_map<GlobalId, int> gid2lv; // vertex gid -> local index
    gid2lv.reserve( nv * 2 );
    for ( int i = 0; i < nv; ++i )
        gid2lv[v_gid( i )] = i;

    // The caller's verdict, as (EdgeKey, gid). The OWNER of an edge decides and
    // the mask is indexed by owned edge local index, so this is the whole of
    // this rank's contribution to the global decision.
    std::vector<std::pair<EdgeKey, GlobalId>> myMarked;
    {
        const int n = std::min( nOwnedE, static_cast<int>( edgeMask.size() ) );
        for ( int e = 0; e < n; ++e )
            if ( edgeMask[e] )
                myMarked.push_back(
                    { makeEdgeKey( e_verts( e, 0 ), e_verts( e, 1 ) ),
                      e_gid( e ) } );
    }

    result.requested =
        globalSum( mesh, static_cast<long long>( myMarked.size() ) );

    // Empty mask: a no-op with no communication beyond the collective above.
    // V/E/F, every gid, every key and the halo are untouched.
    if ( result.requested == 0 )
        return result;

    // ---- Step 1: advertise ------------------------------------------------
    std::vector<std::vector<FlipAdvert>> toCoord( size );
    for ( int f = 0; f < nOwnedF; ++f )
    {
        GlobalId v[3];
        for ( int k = 0; k < 3; ++k )
            v[k] = f_verts( f, k );
        for ( int k = 0; k < 3; ++k )
        {
            const GlobalId p0 = v[k];
            const GlobalId p1 = v[( k + 1 ) % 3];
            const GlobalId op = v[( k + 2 ) % 3];
            FlipAdvert m;
            m.key = makeEdgeKey( p0, p1 );
            m.faceGid = f_gid( f );
            m.oppGid = op;
            m.edgeGid = f_edges( f, k );
            m.owner = static_cast<Rank>( R );
            m.forward = ( p0 == m.key.id[0] ) ? 1 : 0;
            // The face's other two edges are (p1, op) at slot (k+1)%3 and
            // (op, p0) at slot (k+2)%3; name them by which ENDPOINT of the
            // canonical key they meet, so the coordinator can address them
            // without knowing this face's rotation.
            if ( m.forward )
            {
                m.edgeOppLo = f_edges( f, ( k + 2 ) % 3 ); // (op, lo)
                m.edgeOppHi = f_edges( f, ( k + 1 ) % 3 ); // (hi, op)
            }
            else
            {
                m.edgeOppLo = f_edges( f, ( k + 1 ) % 3 ); // (lo, op)
                m.edgeOppHi = f_edges( f, ( k + 2 ) % 3 ); // (op, hi)
            }
            const GlobalId ids[3] = { m.key.id[0], m.key.id[1], op };
            double* dst[3] = { m.posLo, m.posHi, m.posOpp };
            for ( int j = 0; j < 3; ++j )
            {
                auto it = gid2lv.find( ids[j] );
                if ( it == gid2lv.end() )
                    Kokkos::abort(
                        "Tessera::flipEdges: a corner of an OWNED face is not "
                        "held locally, so the flip's geometric test cannot be "
                        "evaluated. rebuildHalo() guarantees every vertex an "
                        "owned face references is held WITH its position; a "
                        "mesh reaching flipEdges() without that has an invalid "
                        "halo." );
                for ( int d = 0; d < 3; ++d )
                    dst[j][d] = 0.0;
                for ( int d = 0; d < D3; ++d )
                    dst[j][d] = static_cast<double>( v_pos( it->second, d ) );
            }
            toCoord[edgeCoordRank( m.key, size )].push_back( m );
        }
    }
    std::vector<std::vector<FlipVerdict>> toVerdict( size );
    for ( const auto& km : myMarked )
        toVerdict[edgeCoordRank( km.first, size )].push_back(
            { km.first, km.second, static_cast<Rank>( R ) } );

    auto adverts = allToAllV( comm, toCoord );
    auto verdicts = allToAllV( comm, toVerdict );

    // ---- Step 2: decide, at the coordinator --------------------------------
    //
    // `exists` is the GLOBAL edge set restricted to this coordinator's keys:
    // every owned face advertised all three of its edges, and every edge of the
    // mesh is an edge of some owned face, so it is exact.
    std::map<EdgeKey, std::vector<const FlipAdvert*>> incident;
    std::set<EdgeKey> exists;
    for ( const auto& m : adverts.data )
    {
        incident[m.key].push_back( &m );
        exists.insert( m.key );
    }
    std::map<EdgeKey, FlipVerdict> marked;
    for ( const auto& m : verdicts.data )
        marked[m.key] = m; // an edge has ONE owner, so at most one verdict

    long long locBoundary = 0, locDup = 0, locGeom = 0, locConflict = 0;

    // A candidate that survived the boundary test, with everything the later
    // rounds need. Kept in an ordered map so every loop below is deterministic.
    struct Cand
    {
        const FlipAdvert* fwd; // the face traversing key.id[0] -> key.id[1]
        const FlipAdvert* bwd;
        FlipVerdict edge; // the flipped edge's gid and owner
        EdgeKey newKey;   // makeEdgeKey(w, x)
        double len2;      // priority: descending
        int votes = 0;
    };
    std::map<EdgeKey, Cand> cand;

    for ( const auto& kv : marked )
    {
        auto it = incident.find( kv.first );
        const std::size_t n =
            ( it == incident.end() ) ? 0 : it->second.size();
        if ( n != 2 )
        {
            ++locBoundary;
            continue;
        }
        const FlipAdvert* a0 = it->second[0];
        const FlipAdvert* a1 = it->second[1];
        if ( a0->forward == a1->forward )
            Kokkos::abort(
                "Tessera::flipEdges: the two faces incident on an edge "
                "traverse it in the SAME direction, so the mesh is not "
                "consistently oriented and there is no orientation-preserving "
                "flip. Tessera's builders produce oriented manifolds; this "
                "means the mesh was assembled from an inconsistent soup." );
        Cand c;
        c.fwd = a0->forward ? a0 : a1;
        c.bwd = a0->forward ? a1 : a0;
        c.edge = kv.second;
        c.newKey = makeEdgeKey( c.fwd->oppGid, c.bwd->oppGid );
        c.len2 = edgeLen2Canonical( kv.first.id[0], c.fwd->posLo,
                                    kv.first.id[1], c.fwd->posHi, 3 );
        cand.emplace( kv.first, c );
    }

    // 2b. duplicate-edge round: ask the coordinator of the edge each flip would
    //     CREATE whether it already exists.
    {
        std::vector<std::vector<DupQuery>> ask( size );
        for ( const auto& kv : cand )
            ask[edgeCoordRank( kv.second.newKey, size )].push_back(
                { kv.second.newKey, kv.first } );
        auto asked = allToAllV( comm, ask );
        std::vector<std::vector<DupReply>> back( size );
        for ( int s = 0; s < size; ++s )
        {
            const DupQuery* p = asked.from( s );
            const int cnt = asked.count( s );
            for ( int i = 0; i < cnt; ++i )
                back[s].push_back(
                    { p[i].asker, static_cast<unsigned char>(
                                      exists.count( p[i].target ) ? 1 : 0 ) } );
        }
        auto answers = allToAllV( comm, back );
        std::set<EdgeKey> duplicate;
        for ( const auto& m : answers.data )
            if ( m.exists )
                duplicate.insert( m.asker );
        for ( const EdgeKey& k : duplicate )
        {
            cand.erase( k );
            ++locDup;
        }
    }

    // 2c. geometric test, from the four positions the coordinator holds. The
    //     new faces are (lo, x, w) and (hi, w, x) -- see the header comment on
    //     winding -- and the reference normal is the AREA-WEIGHTED average of
    //     the two old ones, i.e. the sum of the two unnormalized cross
    //     products, summed forward-face-first so the value does not depend on
    //     advertisement arrival order.
    for ( auto it = cand.begin(); it != cand.end(); )
    {
        const Cand& c = it->second;
        const double* pLo = c.fwd->posLo;
        const double* pHi = c.fwd->posHi;
        const double* pW = c.fwd->posOpp;
        const double* pX = c.bwd->posOpp;

        double n1[3], n2[3], np[3], nq[3], nAvg[3];
        flipTriNormal( pLo, pHi, pW, n1 ); // (u,v,w)
        flipTriNormal( pHi, pLo, pX, n2 ); // (v,u,x)
        flipTriNormal( pLo, pX, pW, np );  // (u,x,w)
        flipTriNormal( pHi, pW, pX, nq );  // (v,w,x)
        for ( int d = 0; d < 3; ++d )
            nAvg[d] = n1[d] + n2[d];

        const double dp = flipAngleBetween( np, nAvg );
        const double dq = flipAngleBetween( nq, nAvg );
        const bool okNormal = ( dp >= 0.0 && dq >= 0.0 &&
                                dp <= policy.maxNormalDeviation &&
                                dq <= policy.maxNormalDeviation );
        const bool okQuality =
            ( flipRadiusRatio( pLo, pX, pW ) >= policy.minQuality &&
              flipRadiusRatio( pHi, pW, pX ) >= policy.minQuality );
        if ( okNormal && okQuality )
            ++it;
        else
        {
            it = cand.erase( it );
            ++locGeom;
        }
    }

    // ---- Step 3: the independent set --------------------------------------
    //
    // Offer each surviving candidate to BOTH its face owners; a face keeps the
    // highest-priority candidate among its own edges and endorses only that
    // one; a candidate is accepted iff it collects both endorsements.
    {
        std::vector<std::vector<FaceCand>> offer( size );
        for ( const auto& kv : cand )
        {
            offer[kv.second.fwd->owner].push_back(
                { kv.second.fwd->faceGid, kv.first, kv.second.len2 } );
            offer[kv.second.bwd->owner].push_back(
                { kv.second.bwd->faceGid, kv.first, kv.second.len2 } );
        }
        auto offers = allToAllV( comm, offer );

        // (squared length DESCENDING, EdgeKey ASCENDING) -- no gid anywhere.
        std::map<GlobalId, FaceCand> best;
        for ( const auto& m : offers.data )
        {
            auto it = best.find( m.faceGid );
            if ( it == best.end() )
                best.emplace( m.faceGid, m );
            else if ( m.len2 > it->second.len2 ||
                      ( m.len2 == it->second.len2 && m.key < it->second.key ) )
                it->second = m;
        }
        std::vector<std::vector<FaceVote>> vote( size );
        for ( const auto& kv : best )
            vote[edgeCoordRank( kv.second.key, size )].push_back(
                { kv.second.key, kv.first } );
        auto votes = allToAllV( comm, vote );
        for ( const auto& m : votes.data )
        {
            auto it = cand.find( m.key );
            if ( it != cand.end() )
                ++it->second.votes;
        }
        for ( auto it = cand.begin(); it != cand.end(); )
        {
            if ( it->second.votes == 2 )
                ++it;
            else
            {
                it = cand.erase( it );
                ++locConflict;
            }
        }
    }

    // ---- Step 4: ship the rewrite to whoever has to write it ---------------
    std::vector<std::vector<FaceApply>> faceMsg( size );
    std::vector<std::vector<EdgeApply>> edgeMsg( size );
    std::vector<std::vector<FlipNote>> noteMsg( size );
    for ( const auto& kv : cand )
    {
        const EdgeKey& key = kv.first;
        const Cand& c = kv.second;
        const GlobalId lo = key.id[0], hi = key.id[1];
        const GlobalId w = c.fwd->oppGid, x = c.bwd->oppGid;

        // The two new faces, in the only orientation-preserving pairing:
        //   Tp = (lo, x, w)  with edges (lo,x) (x,w) (w,lo)
        //   Tq = (hi, w, x)  with edges (hi,w) (w,x) (x,hi)
        FaceApply Tp, Tq;
        Tp.v[0] = lo;
        Tp.v[1] = x;
        Tp.v[2] = w;
        Tp.e[0] = c.bwd->edgeOppLo; // (x, lo), held by the backward face
        Tp.e[1] = c.edge.edgeGid;   // the flipped edge, gid preserved
        Tp.e[2] = c.fwd->edgeOppLo; // (w, lo), held by the forward face
        Tq.v[0] = hi;
        Tq.v[1] = w;
        Tq.v[2] = x;
        Tq.e[0] = c.fwd->edgeOppHi; // (hi, w)
        Tq.e[1] = c.edge.edgeGid;
        Tq.e[2] = c.bwd->edgeOppHi; // (x, hi)

        // GID RULE (header comment): with r = min(w,x) and s = max(w,x), the
        // old face whose opposite corner is r keeps its gid on the new face
        // containing lo. That is the pairing under which a second flip of the
        // same edge restores the original (corners, gid) map exactly.
        const FlipAdvert* faceOfR = ( w < x ) ? c.fwd : c.bwd;
        const FlipAdvert* faceOfS = ( w < x ) ? c.bwd : c.fwd;
        Tp.faceGid = faceOfR->faceGid;
        Tq.faceGid = faceOfS->faceGid;
        faceMsg[faceOfR->owner].push_back( Tp );
        faceMsg[faceOfS->owner].push_back( Tq );

        EdgeApply ea;
        ea.edgeGid = c.edge.edgeGid;
        ea.v[0] = c.newKey.id[0];
        ea.v[1] = c.newKey.id[1];
        ea.f[0] = std::min( Tp.faceGid, Tq.faceGid );
        ea.f[1] = std::max( Tp.faceGid, Tq.faceGid );
        edgeMsg[c.edge.owner].push_back( ea );

        const FlipNote note = { key, c.newKey };
        std::set<Rank> touched = { c.fwd->owner, c.bwd->owner, c.edge.owner };
        for ( Rank r : touched )
            noteMsg[r].push_back( note );
    }
    auto faceGot = allToAllV( comm, faceMsg );
    auto edgeGot = allToAllV( comm, edgeMsg );
    auto noteGot = allToAllV( comm, noteMsg );

    // ---- Step 4b: apply, in place -----------------------------------------
    {
        std::unordered_map<GlobalId, const FaceApply*> byFace;
        byFace.reserve( faceGot.data.size() * 2 + 1 );
        for ( const auto& m : faceGot.data )
            byFace[m.faceGid] = &m;
        for ( int f = 0; f < nOwnedF; ++f )
        {
            auto it = byFace.find( f_gid( f ) );
            if ( it == byFace.end() )
                continue;
            for ( int k = 0; k < 3; ++k )
            {
                f_verts( f, k ) = it->second->v[k];
                f_edges( f, k ) = it->second->e[k];
            }
        }

        std::unordered_map<GlobalId, const EdgeApply*> byEdge;
        byEdge.reserve( edgeGot.data.size() * 2 + 1 );
        for ( const auto& m : edgeGot.data )
            byEdge[m.edgeGid] = &m;
        for ( int e = 0; e < nOwnedE; ++e )
        {
            auto it = byEdge.find( e_gid( e ) );
            if ( it == byEdge.end() )
                continue;
            e_verts( e, 0 ) = it->second->v[0];
            e_verts( e, 1 ) = it->second->v[1];
            e_faces( e, 0 ) = it->second->f[0];
            e_faces( e, 1 ) = it->second->f[1];
        }

        Cabana::deep_copy( mesh.edges(), he );
        Cabana::deep_copy( mesh.faces(), hf );
    }

    // DROP THE GHOST EDGES AND FACES BEFORE THE REBUILD. This is not tidiness,
    // it is correctness, and it cost a debugging cycle to find: rebuildHalo()
    // seeds its gid -> tuple map from EVERY locally held edge (owned AND ghost)
    // and round G only fetches the ones that are MISSING, so a rank holding a
    // stale ghost copy of a flipped edge would keep the OLD endpoints and hand
    // them to round D -- which then derives that edge's key, its ownership and
    // the vertex->edge CSR from endpoints the edge no longer has. splitEdges()
    // never meets this because it hands rebuildHalo() a freshly built
    // owned-only mesh; a rewrite-in-place operation has to say so explicitly.
    //
    // Ghost VERTICES are left alone deliberately: a flip moves no vertex, so a
    // ghost position is still exactly its owner's, and round D restamps Owner
    // from the ownership round regardless.
    mesh.resizeEdges( static_cast<std::size_t>( nOwnedE ) );
    mesh.resizeFaces( static_cast<std::size_t>( nOwnedF ) );

    // Counts are UNCHANGED -- this call bumps the generation and nothing else.
    mesh.setOwnedCounts( mesh.numOwnedVertices(),
                         static_cast<std::size_t>( nOwnedE ),
                         static_cast<std::size_t>( nOwnedF ) );

    // Rebuild the ghost layer and the three halo plans at the depth the caller
    // asked for at setup. Round D of the rebuild redoes edgeKeys(), faceKeys()
    // and both CSRs from the owned entities, which is the whole of this
    // operation's side-table bookkeeping; round G fetches the position of a
    // corner a rewritten face acquired from the far side of the quad.
    rebuildHalo( mesh, halo, effectiveHaloDepth( halo ) );

    // ---- the flip map + the counters, in one collective --------------------
    {
        std::set<std::pair<EdgeKey, EdgeKey>> notes;
        for ( const auto& m : noteGot.data )
            notes.insert( { m.oldKey, m.newKey } );
        result.flipped.assign( notes.begin(), notes.end() );
    }
    {
        const long long locAccepted = static_cast<long long>( cand.size() );
        long long loc[5] = { locAccepted, locBoundary, locDup, locGeom,
                             locConflict };
        long long glob[5] = { 0, 0, 0, 0, 0 };
        MPI_Allreduce( loc, glob, 5, MPI_LONG_LONG, MPI_SUM, comm );
        result.accepted = glob[0];
        result.rejectedBoundary = glob[1];
        result.rejectedDuplicateEdge = glob[2];
        result.rejectedGeometric = glob[3];
        result.rejectedConflict = glob[4];
    }

    return result;
}

} // namespace detail

//! Flip exactly the marked edges that survive the tests.
//!
//! `edgeMask.size() == mesh.numOwnedEdges()`, indexed by owned edge local index
//! -- the same host `std::vector<char>` convention refine()'s face mask and
//! splitEdges()' edge mask use. The OWNER of an edge decides; the decision is
//! made at the edge's coordinator, which is a third rank in general, and is
//! applied by both incident faces' owners and by the edge's owner.
//!
//! V, E and F are UNCHANGED and every gid is preserved: the flipped edge keeps
//! its gid with new endpoints and the two faces keep theirs with new corners.
//! A gid therefore no longer determines a key across this call.
//!
//! AT MOST AN INDEPENDENT SET IS APPLIED PER CALL -- two flips sharing a FACE
//! conflict and only the higher-priority one survives, so a caller wanting more
//! progress calls again and watches `accepted`:
//!
//!     while ( flipEdges( mesh, halo, selection( mesh ) ).accepted > 0 ) {}
//!
//! Choosing WHICH edges to flip is the caller's (valence equalization needs the
//! full valence of all four corners, which is a local quantity at a vertex's
//! owner); keeping it out is what makes this operation depth-1-safe.
//!
//! Belongs to the REMESH editing family, so interleaving it with refine() on
//! one mesh throws (Tessera_EditFamily.hpp).
//!
//! Ends by calling rebuildHalo(), so the halo is valid on return: a
//! haloExchange() is meaningful and a second flipEdges() may follow immediately.
//!
//! An EMPTY mask is a no-op fast path: no communication beyond the one
//! collective that establishes the global request count.
//!
//! Collective.
//!
//! INVALIDATION: rebuildHalo() reallocates the AoSoAs, key Views and CSRs and
//! replaces the halo plans, so every slice/CSR/key-View taken out before this
//! call is dangling. Re-slice from the mesh afterwards.
template <class MeshT, class Policy = DefaultFlipPolicy>
FlipResult flipEdges( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
                      const std::vector<char>& edgeMask,
                      const Policy& policy = Policy{} )
{
    requireEditFamily( mesh, EditFamily::Remesh, "flipEdges" );
    return detail::flipEdgesImpl( mesh, halo, edgeMask, policy );
}

} // namespace Tessera

#endif // TESSERA_EDGE_FLIP_HPP
