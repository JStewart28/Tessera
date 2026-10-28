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

#ifndef TESSERA_EDGE_SPLIT_HPP
#define TESSERA_EDGE_SPLIT_HPP

#include "Tessera_AllToAllV.hpp"
#include "Tessera_Distribute.hpp"
#include "Tessera_EditFamily.hpp"
#include "Tessera_Fields.hpp"
#include "Tessera_HaloRebuild.hpp"
#include "Tessera_Mesh.hpp"
#include "Tessera_Profiling.hpp"
#include "Tessera_Reduction.hpp"
#include "Tessera_Refine.hpp"
#include "Tessera_RefineClosure.hpp"
#include "Tessera_RefineParallel.hpp"
#include "Tessera_RefinePolicy.hpp"
#include "Tessera_Types.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Tessera
{

// ============================================================================
// Caller-driven edge split (REMESH editing family)
// ============================================================================
//
// splitEdges() bisects EXACTLY the edges the caller marks. It is the
// edge-addressed counterpart of refine(), which is face-addressed: refine()
// takes a face mask and bisects all three edges of every marked face, so the
// split-edge map is an OUTPUT. Metric-driven remeshing selects EDGES -- an edge
// longer than the local target length is split -- and marking every face
// incident on a wanted edge is strictly the wrong edit: it splits all three of
// that face's edges and then pulls the 2:1 balance closure in on top.
//
// EDITING FAMILY. splitEdges() belongs to the REMESH family and a mesh belongs
// to one family only; interleaving it with refine() throws. See
// Tessera_EditFamily.hpp for the whole statement -- the short version is that a
// child of an edge-addressed split inherits its parent's level, so `Level` is
// advisory afterwards and the 2:1 balance refine() maintains is not a
// meaningful statement about the result.
//
// ----------------------------------------------------------------------------
// Why there is no closure pass and no mark propagation
// ----------------------------------------------------------------------------
//
// This is the pleasant surprise of the edge-addressed form, and it is worth
// stating prominently because it is counter-intuitive next to refine().
//
// Bisecting a SET OF EDGES and subdividing EVERY incident face according to how
// many of its edges were bisected yields a CONFORMING mesh directly. A face with
// 1, 2 or 3 bisected edges becomes 2, 3 or 4 children respectively, and because
// both faces incident on a bisected edge subdivide that edge the same way, no
// hanging node survives.
//
// refine()'s 2:1 machinery exists only because of an ASYMMETRY it creates: it
// bisects all three edges of a marked face and leaves the neighbour untouched,
// so the neighbour is left with a T-junction that must either be tolerated
// (RefinementMode::HangingNode2to1, bounded by the balance fixpoint) or closed
// (RefinementMode::Conforming). splitEdges() has no such asymmetry, so
// refineImpl()'s Phase 1 (the mark-propagation fixpoint) and its step-3b closure
// pass have NO ANALOGUE here and are deliberately not ported.
//
// ----------------------------------------------------------------------------
// The three bit-pattern cases
// ----------------------------------------------------------------------------
//
// For a face with corners (v0,v1,v2) and edges e[k] = (v[k], v[(k+1)%3]), let S
// be the subset of its edges that are bisected, with midpoints mid[k]:
//
//   |S|  children                                                        count
//   ---  ---------------------------------------------------------------  ----
//    0   the face is emitted unchanged, keeping its gid                     1
//    1   (A, m, C), (m, B, C)                     -- the median from the     2
//        with (A,B,C) rotated so the split edge is edge 0     midpoint
//    2   corner triangle (q0, B, q1) plus the quad (A, q0, q1, C) cut        3
//        along its SHORTER diagonal, with (A,B,C) rotated so the UNSPLIT
//        edge is edge 2 = (C,A), q0 = mid(A,B), q1 = mid(B,C)
//    3   (v0,m0,m2) (v1,m1,m0) (v2,m2,m1) (m0,m1,m2) -- the red split        4
//
// so a face with |S| bisected edges becomes |S| + 1 children, exactly as in the
// closure (closureChildCount()); the shapes are the same three patterns, which
// is not a coincidence -- both are "retriangulate a triangle around a known set
// of edge midpoints".
//
// THE TWO-EDGE DIAGONAL is chosen geometrically, through
// edgeLen2Canonical() (Tessera_RefineClosure.hpp) -- the same helper the
// conforming blue closure uses, for the same reason: it orders the endpoints by
// gid before subtracting, so two ranks comparing the same edge get bit-identical
// doubles regardless of which order they hold it in. Reusing it means this
// operation inherits that determinism rather than re-deriving it, and it keeps
// the library's two diagonal rules consistent with each other. As in the closure
// (see its header note), "shorter diagonal" reduces exactly to "connect the
// midpoint of the LONGER split edge to its opposite corner".
//
// Unlike the closure, the choice needs no communication and no length message:
// the operands are the face's OWN corners, and after rebuildHalo() every vertex
// an owned face references is held locally with its position (Decision 14). The
// closure cannot do this because it runs on the un-closed red layer, whose
// corners come from ClosureParentVerts and may name vertices the rank does not
// hold.
//
// EXACT TIES fall back to the smaller of the two split edges' EdgeKeys -- NOT to
// the closure's lower-midpoint-gid rule. Midpoint gids come from an MPI_Exscan,
// so they are agreed across the ranks of one run but are not the same values at
// a different rank count; EdgeKeys are built from pre-existing vertex gids and
// are therefore rank-count invariant, which is what test_split_edges case 4
// asserts. SplitResult::diagTies counts the ties so the rule's exercise is
// measured rather than assumed.
//
// ----------------------------------------------------------------------------
// Distributed structure
// ----------------------------------------------------------------------------
//
// Phase 2 of refineImpl() is reused essentially wholesale, and that is most of
// the work already done. Every cross-rank decision is routed through the same
// EDGE COORDINATOR (detail::edgeCoordRank):
//
//   A. MASK AGREEMENT. An edge is owned by one rank but incident on faces owned
//      by up to two. Every owned face advertises its three edges (key, this
//      rank) and every rank additionally advertises its MARKED owned edges. The
//      coordinator therefore learns the verdict from the edge's OWNER and
//      replies to every co-sharer. Identical routing to refineImpl() Phase 2a
//      with `refining` replaced by the owner's mask bit.
//   B. MIDPOINT GID ASSIGNMENT. As in refineImpl() Phase 2: the
//      midpoint owner is the lowest incident face owner; owners count their
//      midpoints, MPI_Exscan a contiguous global block above the pre-split
//      global MAX VERTEX GID (not above the vertex count -- see the comment at
//      the exscan: once collapseEdges() has removed a vertex the gid space is
//      sparse and the count aliases a live gid), assign, and SEND the gid to all
//      co-sharers, so both sides
//      agree without relying on identical local ordering. That is what makes
//      SplitResult::midpoints a complete split-edge map with the same contract
//      and the same shape as RefineResult::midpoints -- checkMidpointAgreement
//      consumes it unchanged.
//   C. CHILD FACE GIDS from one MPI_Exscan over a contiguous block above the
//      global max face gid, as refine() does. NEW EDGE GIDS from the edge
//      coordinator, which sees each EdgeKey exactly once and so can number its
//      keys densely and hand the same gid to both sides of a shared edge. A new
//      INTERIOR edge lies strictly inside one parent face and is never shared,
//      but the two HALVES of a bisected edge are, so all of them go through the
//      coordinator uniformly -- refineImpl() step 3e verbatim.
//
// Midpoint positions and vertex user fields come from `RefinePolicy`, unchanged:
// DefaultRefinePolicy's linear average is the right rule and applies to every
// user field automatically. Midpoints are NOT projected onto a sphere -- Tessera
// is not a sphere library, and refine() does not either.
//
// Like refine(), splitEdges() ends by calling rebuildHalo(), so the halo is
// VALID on return (its depth preserved): a haloExchange() is meaningful and a
// second splitEdges() may follow immediately with nothing in between.
//
// Preconditions: `mesh` is distributed (post-distribute), owned-first, entities
// carry global gids, and `edgeMask.size() == mesh.numOwnedEdges()`. Collective.

//! Result of splitEdges().
struct SplitResult
{
    //! THE SPLIT-EDGE MAP: (bisected edge, midpoint gid) for every split edge
    //! this rank TOUCHES -- i.e. every edge of any of this rank's owned faces
    //! that was bisected, whether the midpoint was assigned here or received
    //! from its owner. Sorted and unique by EdgeKey.
    //!
    //! Same contract and same shape as RefineResult::midpoints, so
    //! checkMidpointAgreement() consumes it unchanged.
    std::vector<std::pair<EdgeKey, GlobalId>> midpoints;
    //! Marked OWNED edges, summed globally.
    long long requested = 0;
    //! Edges actually bisected, globally. Equal to `requested` -- every marked
    //! edge is split -- and reported separately so a caller sees both.
    long long split = 0;
    //! Global owned face count before and after the split.
    long long facesBefore = 0, facesAfter = 0;
    //! Owned faces with |S| = 0, 1, 2, 3 bisected edges, summed globally.
    long long pattern[4] = { 0, 0, 0, 0 };
    //! Two-edge faces whose two split edges had EXACTLY equal squared length, so
    //! the geometric diagonal rule could not choose and the smaller-EdgeKey
    //! fallback decided. Both are rank-count invariant; the count is published
    //! so the fallback's exercise is measured rather than assumed.
    long long diagTies = 0;
};

namespace detail
{

//! (edge, an incident owned face's owner) advertisement for mask agreement and
//! midpoint ownership. Sent for EVERY owned face's edges: the coordinator needs
//! the full participant set, because the split verdict has to reach BOTH
//! incident faces and the midpoint owner is the lowest of them.
struct SplitAdvert
{
    EdgeKey key;
    Rank owner;
};

//! (edge) verdict advertisement, sent by the edge's OWNER for each edge it
//! marks. Presence at the coordinator IS the verdict, so no flag is needed.
struct SplitVerdict
{
    EdgeKey key;
};

//! (edge, its midpoint owner) reply delivered to every participant of a SPLIT
//! edge. An edge whose owner did not mark it is dropped at the coordinator and
//! generates no reply, so "the edge is in my reply set" == "it is bisected".
struct SplitOwnerReply
{
    EdgeKey key;
    Rank owner;
};

//! (owned split edge, a co-sharer rank) delivered to the midpoint owner.
struct SplitCosharer
{
    EdgeKey key;
    Rank cosharer;
};

//! (edge, assigned midpoint gid) delivered from the midpoint owner to each
//! co-sharer. Unlike refine()'s detail::KeyGid this carries no squared length:
//! the only geometric decision here is the two-edge diagonal, whose operands are
//! the deciding face's own corners and are therefore locally held.
struct SplitKeyGid
{
    EdgeKey key;
    GlobalId gid;
};

//! Implementation of splitEdges(); see the header comment for the algorithm and
//! the guarantees.
template <class MeshT, class Policy>
SplitResult
splitEdgesImpl( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
                const std::vector<char>& edgeMask, const Policy& policy )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_REFINE );
    using memory_space = typename MeshT::memory_space;
    using Scalar = typename MeshT::scalar_type;
    constexpr int Dim = MeshT::dim;
    const int R = mesh.rank();
    const int size = mesh.commSize();
    MPI_Comm comm = mesh.comm();

    const int nv = static_cast<int>( mesh.numVertices() ); // owned + ghost
    const int nOwnedV = static_cast<int>( mesh.numOwnedVertices() );
    const int nOwnedE = static_cast<int>( mesh.numOwnedEdges() );
    const int nOwnedF = static_cast<int>( mesh.numOwnedFaces() );

    SplitResult result;

    // ---- host copies: all vertices (endpoint lookup) + owned edges/faces ----
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
    auto e_verts = Cabana::slice<EdgeField::Verts>( he );
    auto f_gid = Cabana::slice<FaceField::Gid>( hf );
    auto f_verts = Cabana::slice<FaceField::Verts>( hf );
    auto f_lev = Cabana::slice<FaceField::Level>( hf );

    std::unordered_map<GlobalId, int> gid2lv; // vertex gid -> local index
    gid2lv.reserve( nv * 2 );
    for ( int i = 0; i < nv; ++i )
        gid2lv[v_gid( i )] = i;

    auto keyOf = []( GlobalId a, GlobalId b ) { return makeEdgeKey( a, b ); };

    // The caller's verdict, as EdgeKeys. The OWNER of an edge decides, and the
    // mask is indexed by owned edge local index, so this is the whole of this
    // rank's contribution to the global decision.
    std::vector<EdgeKey> myMarked;
    {
        const int n = std::min( nOwnedE, static_cast<int>( edgeMask.size() ) );
        for ( int e = 0; e < n; ++e )
            if ( edgeMask[e] )
                myMarked.push_back( keyOf( e_verts( e, 0 ), e_verts( e, 1 ) ) );
    }

    result.requested =
        globalSum( mesh, static_cast<long long>( myMarked.size() ) );
    result.facesBefore = globalOwnedFaces( mesh );
    result.facesAfter = result.facesBefore;

    // Empty mask: a no-op with no communication beyond the two collectives
    // above. V/E/F, every gid, and the halo are untouched.
    if ( result.requested == 0 )
        return result;

    // ---- Phase A: cross-rank agreement on the mask --------------------------
    // Every owned face advertises its three edges (so the coordinator knows the
    // full participant set) and every rank advertises the owned edges it marked
    // (so the coordinator learns the verdict from the edge's owner).
    std::map<EdgeKey, GlobalId> midGid; // split edges this rank touches
    std::vector<EdgeKey> myMid;         // ...of which these are mine to number
    {
        std::vector<std::vector<detail::SplitAdvert>> toCoord( size );
        for ( int f = 0; f < nOwnedF; ++f )
            for ( int k = 0; k < 3; ++k )
            {
                const EdgeKey key =
                    keyOf( f_verts( f, k ), f_verts( f, ( k + 1 ) % 3 ) );
                toCoord[detail::edgeCoordRank( key, size )].push_back(
                    { key, static_cast<Rank>( R ) } );
            }
        std::vector<std::vector<detail::SplitVerdict>> toVerdict( size );
        for ( const EdgeKey& key : myMarked )
            toVerdict[detail::edgeCoordRank( key, size )].push_back( { key } );

        auto adverts = allToAllV( comm, toCoord );
        auto verdicts = allToAllV( comm, toVerdict );

        // Coordinator: an edge is SPLIT iff its owner marked it. Its midpoint
        // owner is the lowest incident face owner -- every incident face is
        // subdivided, so unlike refine() there is no "refining participants
        // only" qualifier. Reply (key, owner) to every participant and
        // (key, cosharer) to the owner for each other participant.
        std::map<EdgeKey, std::set<Rank>> participants;
        for ( const auto& m : adverts.data )
            participants[m.key].insert( m.owner );
        std::set<EdgeKey> splitKeys;
        for ( const auto& m : verdicts.data )
            splitKeys.insert( m.key );

        std::vector<std::vector<detail::SplitOwnerReply>> reply( size );
        std::vector<std::vector<detail::SplitCosharer>> coshare( size );
        for ( const EdgeKey& key : splitKeys )
        {
            auto it = participants.find( key );
            if ( it == participants.end() || it->second.empty() )
                Kokkos::abort(
                    "Tessera::splitEdges: an edge was marked by its owner but "
                    "no rank advertised an owned face incident on it. The edge "
                    "mask is indexed by OWNED edges, and an owned edge is by "
                    "construction an edge of an owned face, so this means the "
                    "mesh's edge table and face table disagree." );
            const Rank owner = *it->second.begin(); // std::set: ascending
            for ( Rank r : it->second )
            {
                reply[r].push_back( { key, owner } );
                if ( r != owner )
                    coshare[owner].push_back( { key, r } );
            }
        }
        auto replies = allToAllV( comm, reply );
        auto coshares = allToAllV( comm, coshare );

        // ---- Phase B: midpoint-gid assignment -------------------------------
        std::map<EdgeKey, Rank> midOwner;
        for ( const auto& m : replies.data )
            midOwner[m.key] = m.owner;
        std::map<EdgeKey, std::vector<Rank>> myCosharers;
        for ( const auto& m : coshares.data )
            myCosharers[m.key].push_back( m.cosharer );

        for ( const auto& kv : midOwner )
            if ( kv.second == R )
                myMid.push_back( kv.first );
        std::sort( myMid.begin(), myMid.end() );

        // THE BLOCK SITS ABOVE THE GLOBAL MAX VERTEX GID, NOT ABOVE THE VERTEX
        // COUNT -- the same rule step 3c uses for child faces, and for the same
        // reason. The count was equivalent for as long as nothing removed a
        // vertex, and collapseEdges() removes vertices: compact() PRESERVES the
        // surviving gids, so after one collapse the vertex gid space is sparse
        // and its maximum exceeds the count. Basing the block on the count then
        // hands a new midpoint a gid a LIVE vertex still holds, which welds two
        // unrelated parts of the surface together -- V stops growing, Euler
        // breaks by one per split, and an edge appears between two nearly
        // antipodal points. Found by test_collapse_edges' composed
        // split/flip/collapse loop (tasks/completed/edge-collapse.md check 14).
        long long localMaxV = -1;
        for ( int i = 0; i < nOwnedV; ++i )
            localMaxV = std::max( localMaxV, static_cast<long long>( v_gid( i ) ) );
        long long globalMaxV = -1;
        MPI_Allreduce( &localMaxV, &globalMaxV, 1, MPI_LONG_LONG, MPI_MAX,
                       comm );
        long long myCount = static_cast<long long>( myMid.size() );
        long long baseOff = 0;
        MPI_Exscan( &myCount, &baseOff, 1, MPI_LONG_LONG, MPI_SUM, comm );
        if ( R == 0 )
            baseOff = 0;

        for ( std::size_t i = 0; i < myMid.size(); ++i )
            midGid[myMid[i]] = static_cast<GlobalId>(
                globalMaxV + 1 + baseOff + static_cast<long long>( i ) );

        // Deliver owned midpoint gids to co-sharers, so both sides of a shared
        // bisected edge agree with no reliance on identical local ordering.
        std::vector<std::vector<detail::SplitKeyGid>> toShare( size );
        for ( const EdgeKey& key : myMid )
        {
            auto it = myCosharers.find( key );
            if ( it == myCosharers.end() )
                continue;
            for ( Rank c : it->second )
                toShare[c].push_back( { key, midGid[key] } );
        }
        auto shared = allToAllV( comm, toShare );
        for ( const auto& m : shared.data )
            midGid[m.key] = m.gid;
    }

    // Publish the split-edge map: every edge of an owned face that is bisected,
    // with its globally agreed midpoint gid.
    result.midpoints.reserve( midGid.size() );
    for ( const auto& kv : midGid )
        result.midpoints.push_back( { kv.first, kv.second } );

    // ---- Phase C: local topology reconstruction (owned only) ---------------

    // 3a. new owned vertices: originals + owned midpoints.
    {
        const int nNewV = nOwnedV + static_cast<int>( myMid.size() );
        Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace>
            lv( "lv", nNewV );
        auto gid = Cabana::slice<VertexField::Gid>( lv );
        auto own = Cabana::slice<VertexField::Owner>( lv );
        auto flg = Cabana::slice<VertexField::Flags>( lv );
        auto pos = Cabana::slice<VertexField::Position>( lv );
        auto o_flg = Cabana::slice<VertexField::Flags>( hv );
        for ( int i = 0; i < nOwnedV; ++i )
        {
            gid( i ) = v_gid( i );
            own( i ) = static_cast<Rank>( R );
            flg( i ) = o_flg( i );
            for ( int d = 0; d < Dim; ++d )
                pos( i, d ) = v_pos( i, d );
            detail::copyUserFields<VertexField::UserBegin>( lv, i, hv, i );
        }
        for ( std::size_t m = 0; m < myMid.size(); ++m )
        {
            const int idx = nOwnedV + static_cast<int>( m );
            auto ia = gid2lv.find( myMid[m].id[0] );
            auto ib = gid2lv.find( myMid[m].id[1] );
            if ( ia == gid2lv.end() || ib == gid2lv.end() )
                Kokkos::abort(
                    "Tessera::splitEdges: the owner of a midpoint does not "
                    "hold "
                    "both endpoints of the edge it is bisecting. Midpoint "
                    "ownership is 'lowest incident face owner', and that face "
                    "has the whole edge, so this is impossible unless the mesh "
                    "entered splitEdges() without the vertices its owned faces "
                    "reference (see rebuildHalo())." );
            const int la = ia->second;
            const int lb = ib->second;
            gid( idx ) = midGid[myMid[m]];
            own( idx ) = static_cast<Rank>( R );
            flg( idx ) = 0;
            Scalar pa[Dim], pb[Dim], pm[Dim];
            for ( int d = 0; d < Dim; ++d )
            {
                pa[d] = v_pos( la, d );
                pb[d] = v_pos( lb, d );
            }
            policy.interpolatePosition( pm, pa, pb, Dim );
            for ( int d = 0; d < Dim; ++d )
                pos( idx, d ) = pm[d];
            detail::blendVertexUserCross( lv, idx, hv, la, lb, policy );
        }
        // INVALIDATION: this and everything below reallocate and reassign the
        // mesh's storage, invalidating every slice/CSR/key-View handed out
        // before this call. Re-slice from the mesh after splitEdges() returns.
        mesh.resizeVertices( nNewV );
        Cabana::deep_copy( mesh.vertices(), lv );
    }

    // 3b. subdivide each owned face on the bit pattern of its bisected edges.
    //     TOPOLOGY FIRST: the child gids come from the exscan below, and corner
    //     gids do not depend on face gids, so the ordering costs nothing.
    struct NewFace
    {
        GlobalId v[3];
        GlobalId gid;
        Level level;
    };
    std::vector<NewFace> newFace;
    std::vector<char> freshChild; // parallel: created by this call
    std::vector<int> newFaceSrc;  // parallel: `hf` row with the user fields
    newFace.reserve( static_cast<std::size_t>( nOwnedF ) );
    freshChild.reserve( static_cast<std::size_t>( nOwnedF ) );
    newFaceSrc.reserve( static_cast<std::size_t>( nOwnedF ) );
    long long nChildren = 0;
    long long localPattern[4] = { 0, 0, 0, 0 };
    long long localTies = 0;

    // Squared length of a face edge, from the face's OWN corner positions and
    // through the library's one canonical producer, so two ranks evaluating the
    // same edge get bit-identical doubles. Every corner of an owned face is held
    // locally with its position after rebuildHalo() (Decision 14).
    auto edgeLen2 = [&]( GlobalId a, GlobalId b ) -> double
    {
        auto ia = gid2lv.find( a );
        auto ib = gid2lv.find( b );
        if ( ia == gid2lv.end() || ib == gid2lv.end() )
            Kokkos::abort(
                "Tessera::splitEdges: a corner of an OWNED face is not held "
                "locally, so the two-edge diagonal cannot be chosen "
                "geometrically. rebuildHalo() guarantees every vertex an owned "
                "face references is held with its position; a mesh reaching "
                "splitEdges() without that has an invalid halo." );
        double pa[3], pb[3];
        for ( int d = 0; d < Dim; ++d )
        {
            pa[d] = static_cast<double>( v_pos( ia->second, d ) );
            pb[d] = static_cast<double>( v_pos( ib->second, d ) );
        }
        return edgeLen2Canonical( a, pa, b, pb, Dim );
    };

    for ( int f = 0; f < nOwnedF; ++f )
    {
        GlobalId v[3];
        for ( int k = 0; k < 3; ++k )
            v[k] = f_verts( f, k );
        const Level lev = f_lev( f );

        GlobalId mid[3];
        int nSplit = 0;
        for ( int k = 0; k < 3; ++k )
        {
            auto it = midGid.find( keyOf( v[k], v[( k + 1 ) % 3] ) );
            if ( it == midGid.end() )
                mid[k] = invalid_gid;
            else
            {
                mid[k] = it->second;
                ++nSplit;
            }
        }
        ++localPattern[nSplit];

        if ( nSplit == 0 )
        {
            // Emitted unchanged, keeping its gid: no child, no renumbering.
            newFace.push_back( { { v[0], v[1], v[2] }, f_gid( f ), lev } );
            freshChild.push_back( 0 );
            newFaceSrc.push_back( f );
            continue;
        }

        // Every child inherits the parent's LEVEL (Decision 1: `Level` is
        // advisory in the remesh family) and, below, the parent's user fields.
        auto child = [&]( GlobalId x, GlobalId y, GlobalId z )
        {
            newFace.push_back( { { x, y, z }, invalid_gid, lev } );
            freshChild.push_back( 1 );
            newFaceSrc.push_back( f );
            ++nChildren;
        };

        if ( nSplit == 1 )
        {
            // Rotate so the split edge is edge 0 of (A,B,C), then cut the median
            // from its midpoint to the opposite corner. A rotation of a CCW
            // triple is CCW, so both children keep the parent's winding.
            const int k = ( mid[0] != invalid_gid )
                              ? 0
                              : ( ( mid[1] != invalid_gid ) ? 1 : 2 );
            const GlobalId A = v[k];
            const GlobalId B = v[( k + 1 ) % 3];
            const GlobalId C = v[( k + 2 ) % 3];
            child( A, mid[k], C );
            child( mid[k], B, C );
        }
        else if ( nSplit == 2 )
        {
            // Rotate so the UNSPLIT edge is edge 2 = (C,A): with `u` the unsplit
            // edge index, rot = (u+1)%3 puts edge u at position 2.
            const int u = ( mid[0] == invalid_gid )
                              ? 0
                              : ( ( mid[1] == invalid_gid ) ? 1 : 2 );
            const int rot = ( u + 1 ) % 3;
            const GlobalId A = v[rot];
            const GlobalId B = v[( rot + 1 ) % 3];
            const GlobalId C = v[( rot + 2 ) % 3];
            const GlobalId q0 = mid[rot];             // midpoint of (A,B)
            const GlobalId q1 = mid[( rot + 1 ) % 3]; // midpoint of (B,C)
            // Cut the corner triangle at B, then split the quad (A,q0,q1,C)
            // along its SHORTER diagonal -- equivalently connect the midpoint of
            // the LONGER split edge to its opposite corner.
            const double lAB = edgeLen2( A, B );
            const double lBC = edgeLen2( B, C );
            bool diagQ0C;
            if ( lAB != lBC )
            {
                diagQ0C = ( lAB > lBC );
            }
            else
            {
                // Exact geometric tie -- the undisturbed icosphere is highly
                // symmetric and has many. Fall back to the smaller EdgeKey of
                // the two split edges, which is built from PRE-EXISTING vertex
                // gids and so is rank-count invariant; the closure's
                // lower-midpoint-gid fallback is not, because midpoint gids come
                // from an MPI_Exscan.
                diagQ0C = ( keyOf( A, B ) < keyOf( B, C ) );
                ++localTies;
            }
            if ( diagQ0C )
            {
                child( A, q0, C );
                child( q0, B, q1 );
                child( q0, q1, C );
            }
            else
            {
                child( A, q0, q1 );
                child( q0, B, q1 );
                child( A, q1, C );
            }
        }
        else
        {
            // The red 1->4 split, identical to refine()'s existing case.
            child( v[0], mid[0], mid[2] );
            child( v[1], mid[1], mid[0] );
            child( v[2], mid[2], mid[1] );
            child( mid[0], mid[1], mid[2] );
        }
    }

    // 3c. child face gids: ONE exscan over a contiguous global block above the
    //     global max face gid. A subdivided parent's gid is RETIRED (it is
    //     replaced by its children), so live gids are sparse and the max exceeds
    //     the count -- taking the max keeps the bound monotone across calls.
    {
        long long localMaxF = -1;
        for ( int f = 0; f < nOwnedF; ++f )
            localMaxF =
                std::max( localMaxF, static_cast<long long>( f_gid( f ) ) );
        long long globalMaxF = -1;
        MPI_Allreduce( &localMaxF, &globalMaxF, 1, MPI_LONG_LONG, MPI_MAX,
                       comm );
        long long childBase = 0;
        MPI_Exscan( &nChildren, &childBase, 1, MPI_LONG_LONG, MPI_SUM, comm );
        if ( R == 0 )
            childBase = 0;
        GlobalId childGid = static_cast<GlobalId>( globalMaxF + 1 + childBase );
        for ( std::size_t i = 0; i < newFace.size(); ++i )
            if ( freshChild[i] )
                newFace[i].gid = childGid++;
    }
    const int nNewF = static_cast<int>( newFace.size() );

    // 3d. re-derive edges from the new faces (dedup by EdgeKey). Per-edge USER
    //     data cannot be carried through this -- exactly as in refine(); see the
    //     README Known Issue.
    std::map<EdgeKey, int> edge_of;
    std::vector<std::array<GlobalId, 2>> ep;
    std::vector<std::array<int, 2>> efLocal; // incident local face indices
    std::vector<std::array<int, 3>> faceEdge( nNewF );
    for ( int f = 0; f < nNewF; ++f )
        for ( int k = 0; k < 3; ++k )
        {
            const EdgeKey key =
                keyOf( newFace[f].v[k], newFace[f].v[( k + 1 ) % 3] );
            auto it = edge_of.find( key );
            int e;
            if ( it == edge_of.end() )
            {
                e = static_cast<int>( ep.size() );
                edge_of.emplace( key, e );
                ep.push_back( { key.id[0], key.id[1] } );
                efLocal.push_back( { -1, -1 } );
            }
            else
                e = it->second;
            faceEdge[f][k] = e;
            if ( efLocal[e][0] < 0 )
                efLocal[e][0] = f;
            else
                efLocal[e][1] = f;
        }
    const int nLocalE = static_cast<int>( ep.size() );

    // 3e. edge ownership + level + GID via the coordinator (min incident face
    //     owner / level). refineImpl() step 3e verbatim: each EdgeKey is routed
    //     to exactly one coordinator, which numbers its own distinct keys
    //     densely from an MPI_Exscan and replies the SAME gid to every rank that
    //     advertised the key -- so the two sides of a boundary edge agree and
    //     the gids are dense in [0, globalEdges), which distribute(),
    //     MeshBuilder, migrate() and the HDF5 writer/reader all assume.
    std::vector<std::vector<detail::EdgeOwnMsg>> toEC( size );
    for ( int e = 0; e < nLocalE; ++e )
    {
        const EdgeKey key = keyOf( ep[e][0], ep[e][1] );
        Level lv0 = newFace[efLocal[e][0]].level;
        if ( efLocal[e][1] >= 0 )
            lv0 = std::min( lv0, newFace[efLocal[e][1]].level );
        toEC[detail::edgeCoordRank( key, size )].push_back(
            { key, static_cast<Rank>( R ), lv0, invalid_gid } );
    }
    auto ecGot = allToAllV( comm, toEC );
    struct EdgeInfo
    {
        Rank owner;
        Level level;
        GlobalId gid;
    };
    std::map<EdgeKey, EdgeInfo> edgeAgg;
    for ( const auto& m : ecGot.data )
    {
        auto it = edgeAgg.find( m.key );
        if ( it == edgeAgg.end() )
            edgeAgg[m.key] = { m.owner, m.level, invalid_gid };
        else
        {
            it->second.owner = std::min( it->second.owner, m.owner );
            it->second.level = std::min( it->second.level, m.level );
        }
    }
    {
        long long myKeys = static_cast<long long>( edgeAgg.size() );
        long long eBase = 0;
        MPI_Exscan( &myKeys, &eBase, 1, MPI_LONG_LONG, MPI_SUM, comm );
        if ( R == 0 )
            eBase = 0;
        long long n = 0;
        for ( auto& kv : edgeAgg )
            kv.second.gid = static_cast<GlobalId>( eBase + n++ );
    }
    std::vector<std::vector<detail::EdgeOwnMsg>> ecReply( size );
    for ( int s = 0; s < size; ++s )
    {
        const detail::EdgeOwnMsg* p = ecGot.from( s );
        const int cnt = ecGot.count( s );
        for ( int i = 0; i < cnt; ++i )
        {
            const auto& agg = edgeAgg[p[i].key];
            ecReply[s].push_back( { p[i].key, agg.owner, agg.level, agg.gid } );
        }
    }
    auto ecRes = allToAllV( comm, ecReply );
    std::map<EdgeKey, EdgeInfo> edgeOwner;
    for ( const auto& m : ecRes.data )
        edgeOwner[m.key] = { m.owner, m.level, m.gid };

    // 3f. order edges owned-first (local layout only -- the gids come from the
    //     coordinator, so a boundary edge carries the same gid on both ranks).
    std::vector<int> order;
    order.reserve( nLocalE );
    std::vector<char> isOwned( nLocalE, 0 );
    for ( int e = 0; e < nLocalE; ++e )
        if ( edgeOwner[keyOf( ep[e][0], ep[e][1] )].owner == R )
        {
            isOwned[e] = 1;
            order.push_back( e );
        }
    const int nNewOwnedE = static_cast<int>( order.size() );
    for ( int e = 0; e < nLocalE; ++e )
        if ( !isOwned[e] )
            order.push_back( e );

    std::vector<GlobalId> edgeGid( nLocalE );
    std::vector<int> newIndexOf( nLocalE );
    for ( int li = 0; li < nLocalE; ++li )
    {
        const int e = order[li];
        edgeGid[e] = edgeOwner[keyOf( ep[e][0], ep[e][1] )].gid;
        newIndexOf[e] = li;
    }

    // 3g. materialize the edge AoSoA (owned-first).
    {
        Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace> le(
            "le", nLocalE );
        auto gid = Cabana::slice<EdgeField::Gid>( le );
        auto own = Cabana::slice<EdgeField::Owner>( le );
        auto lev = Cabana::slice<EdgeField::Level>( le );
        auto verts = Cabana::slice<EdgeField::Verts>( le );
        auto faces = Cabana::slice<EdgeField::Faces>( le );
        for ( int e = 0; e < nLocalE; ++e )
        {
            const int li = newIndexOf[e];
            const EdgeKey key = keyOf( ep[e][0], ep[e][1] );
            gid( li ) = edgeGid[e];
            own( li ) = edgeOwner[key].owner;
            lev( li ) = edgeOwner[key].level;
            verts( li, 0 ) = ep[e][0];
            verts( li, 1 ) = ep[e][1];
            faces( li, 0 ) = newFace[efLocal[e][0]].gid;
            faces( li, 1 ) =
                efLocal[e][1] >= 0 ? newFace[efLocal[e][1]].gid : invalid_gid;
        }
        mesh.resizeEdges( nLocalE );
        Cabana::deep_copy( mesh.edges(), le );
    }

    // 3h. materialize the face AoSoA (edges now have gids).
    {
        Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> lf(
            "lf", nNewF );
        auto gid = Cabana::slice<FaceField::Gid>( lf );
        auto own = Cabana::slice<FaceField::Owner>( lf );
        auto lev = Cabana::slice<FaceField::Level>( lf );
        auto verts = Cabana::slice<FaceField::Verts>( lf );
        auto edges = Cabana::slice<FaceField::Edges>( lf );
        // Every face here is a plain VISIBLE face with no closure parent. A
        // Cabana AoSoA's backing View is zero-initialized, so an uninitialized
        // ClosureParent would read as face gid 0 -- a perfectly valid gid --
        // and unclose() would restore a bogus parent for every face. Same trap
        // distribute() documents.
        initClosureFaceMembers<MeshT>( lf, 0,
                                       static_cast<std::size_t>( nNewF ) );
        for ( int f = 0; f < nNewF; ++f )
        {
            gid( f ) = newFace[f].gid;
            own( f ) = static_cast<Rank>( R );
            lev( f ) = newFace[f].level;
            for ( int k = 0; k < 3; ++k )
            {
                verts( f, k ) = newFace[f].v[k];
                edges( f, k ) = edgeGid[faceEdge[f][k]];
            }
            detail::copyUserFieldsN<
                FaceField::UserBegin,
                numFaceUserFields<typename MeshT::face_user_fields>()>(
                lf, f, hf, newFaceSrc[f] );
        }
        mesh.resizeFaces( nNewF );
        Cabana::deep_copy( mesh.faces(), lf );
    }

    // owned-only mesh: every local entity is owned until rebuildHalo() below.
    mesh.setOwnedCounts( static_cast<std::size_t>( nOwnedV + myMid.size() ),
                         nNewOwnedE, nNewF );

    // 3i. rebuild key side tables.
    {
        Kokkos::View<EdgeKey*, memory_space> ek(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "edge_keys" ),
            nLocalE );
        auto h_ek = Kokkos::create_mirror_view( ek );
        for ( int e = 0; e < nLocalE; ++e )
            h_ek( newIndexOf[e] ) = keyOf( ep[e][0], ep[e][1] );
        Kokkos::deep_copy( ek, h_ek );
        mesh.setEdgeKeys( ek );

        Kokkos::View<FaceKey*, memory_space> fk(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "face_keys" ),
            nNewF );
        auto h_fk = Kokkos::create_mirror_view( fk );
        for ( int f = 0; f < nNewF; ++f )
            h_fk( f ) = makeFaceKey( newFace[f].v[0], newFace[f].v[1],
                                     newFace[f].v[2] );
        Kokkos::deep_copy( fk, h_fk );
        mesh.setFaceKeys( fk );
    }

    // Rebuild the ghost layer and the three halo plans over the split owned
    // entities, at the depth the caller asked for at setup. Never return with a
    // cleared halo: refine() stopped doing that in a55a8de and splitEdges() must
    // not reintroduce the trap -- a haloExchange() would silently no-op on an
    // empty plan, and a second splitEdges() would fail to find the positions of
    // both endpoints of a midpoint it owns across a partition boundary.
    rebuildHalo( mesh, halo, effectiveHaloDepth( halo ) );

    // ---- diagnostics, in one collective ------------------------------------
    {
        long long loc[6] = {
            localPattern[0], localPattern[1],
            localPattern[2], localPattern[3],
            localTies,       static_cast<long long>( myMid.size() ) };
        long long glob[6] = { 0, 0, 0, 0, 0, 0 };
        MPI_Allreduce( loc, glob, 6, MPI_LONG_LONG, MPI_SUM, comm );
        for ( int i = 0; i < 4; ++i )
            result.pattern[i] = glob[i];
        result.diagTies = glob[4];
        result.split = glob[5];
    }
    result.facesAfter = globalOwnedFaces( mesh );

    return result;
}

} // namespace detail

//! Bisect exactly the marked edges.
//!
//! `edgeMask.size() == mesh.numOwnedEdges()`, indexed by owned edge local index
//! -- the same host `std::vector<char>` convention refine()'s face mask uses, so
//! a device-computed indicator round-trips to the host the same way. The OWNER
//! of an edge decides, and the decision is propagated to every rank holding an
//! incident face. Every incident face is subdivided into 2, 3 or 4 children
//! according to how many of its edges are bisected, so the mesh is CONFORMING on
//! exit with no closure layer and no 2:1 balance pass -- see the header comment
//! for why an edge-addressed split needs neither.
//!
//! Belongs to the REMESH editing family: a child inherits its parent's level and
//! the mesh is thereafter not 2:1-level-meaningful, so interleaving this with
//! refine() on one mesh throws (Tessera_EditFamily.hpp).
//!
//! Ends by calling rebuildHalo(), so the halo is valid on return -- the same
//! postcondition refine() has had since a55a8de.
//!
//! An EMPTY mask is a no-op fast path: no communication beyond the collectives
//! that establish the global request and face counts, and V/E/F unchanged.
//!
//! Collective.
//!
//! INVALIDATION: reallocates the AoSoAs, key Views and CSRs and replaces the
//! halo plans, so every slice/CSR/key-View taken out before this call is
//! dangling. Re-slice from the mesh afterwards.
template <class MeshT,
          class Policy = DefaultRefinePolicy<typename MeshT::scalar_type>>
SplitResult
splitEdges( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
            const std::vector<char>& edgeMask, const Policy& policy = Policy{} )
{
    requireEditFamily( mesh, EditFamily::Remesh, "splitEdges" );
    return detail::splitEdgesImpl( mesh, halo, edgeMask, policy );
}

} // namespace Tessera

#endif // TESSERA_EDGE_SPLIT_HPP
