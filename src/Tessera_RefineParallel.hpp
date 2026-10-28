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

#ifndef TESSERA_REFINE_PARALLEL_HPP
#define TESSERA_REFINE_PARALLEL_HPP

#include "Tessera_AllToAllV.hpp"
#include "Tessera_Distribute.hpp"
#include "Tessera_Fields.hpp"
#include "Tessera_HaloRebuild.hpp"
#include "Tessera_Mesh.hpp"
#include "Tessera_Profiling.hpp"
#include "Tessera_Reduction.hpp"
#include "Tessera_Refine.hpp"
#include "Tessera_RefineClosure.hpp"
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
// Distributed 2:1-balanced red refinement (Step 6b)
// ============================================================================
//
// refine() refines a distributed mesh's OWNED faces (mask indexed by owned face
// local index) with a conforming 2:1 level balance enforced across partition
// boundaries and — the central guarantee — a midpoint vertex whose global id is
// BIT-IDENTICAL on every rank that shares the bisected edge.
//
// It relies on neither the vertex-based 1-ring halo (which does not guarantee a
// face sees its edge-neighbour across a boundary) nor replicated knowledge.
// Instead every cross-rank decision is routed through an EDGE COORDINATOR: the
// rank `edgeCoordRank(EdgeKey)` gathers both incident faces of an edge (each
// owned by some rank and advertised) and drives the decision. Three coordinator
// phases run:
//   1. 2:1 mark-propagation fixpoint: pure function of (level, mark) per face;
//      coordinator flags the coarser face of any edge whose incident final
//      levels would differ by >1; marks only ever turn on (monotone) so the
//      fixpoint terminates, guarded by MPI_Allreduce(changed) and a hard cap.
//   2. Midpoint-gid assignment: every owned face — refining OR kept — advertises
//      its three edges with a `refining` flag. An edge is split iff some incident
//      face refines; its midpoint owner is the lowest incident refining-face
//      owner. Owners count their midpoints, MPI_Exscan a global contiguous block
//      onto the pre-refinement global vertex count, assign, and SEND the gid to
//      ALL co-sharers — so both sides agree with no reliance on identical local
//      ordering, and a rank holding only the KEPT side of a bisected edge still
//      learns the midpoint gid. That last part is what makes RefineResult::
//      midpoints a complete split-edge map, which the conforming closure needs
//      (Task 4); ownership and hence every assigned gid are unaffected. That
//      same reply also carries the bisected edge's SQUARED LENGTH, which the
//      conforming closure's blue diagonal tie-break is chosen from and cannot
//      compute where it is used — see detail::KeyGid and the tie-break note in
//      Tessera_RefineClosure.hpp.
//   3. Edge ownership: each refined edge is owned by the lowest incident (child)
//      face owner, so owned counts remain a global partition (for Euler).
//
// Local topology is then rebuilt exactly as the serial engine (Tessera_Refine),
// reusing the interpolation policy. Phases 1-3 produce an OWNED-ONLY mesh — every
// cross-rank decision went through a coordinator, so no ghost was needed, and the
// acceptance invariants (midpoint agreement, 2:1 balance, owned-only Euler) are
// all owned-entity properties. refine() then finishes with rebuildHalo()
// (Tessera_HaloRebuild.hpp), the general non-replicated 1-deep halo rebuild it
// shares with migrate(), so on return:
//   * the passed halo's three plans are VALID, not cleared — a haloExchange() is
//     meaningful (and a re-sync, since ghost values already equal the owners');
//   * refine() may be called again immediately, with nothing in between;
//   * local ordering is canonical — owned first then ghost, each kind ascending
//     by gid — which a caller can observe through a gid-derived face mask.
//
// Preconditions: `mesh` is distributed (post-distribute), owned-first, entities
// carry global gids, and is 2:1-balanced on entry (true initially — uniform
// level 0 — and maintained by each call). `mask.size() == numOwnedFaces()`.

struct RefineResult
{
    //! 2:1 mark-propagation rounds executed to reach the fixpoint.
    int iterations = 0;
    //! THE SPLIT-EDGE MAP: (bisected edge, midpoint gid) for every split edge
    //! this rank TOUCHES — i.e. every edge of any of this rank's owned faces
    //! (refining *or* kept) that some incident face bisected, whether the
    //! midpoint was assigned here or received from its owner. Sorted and unique
    //! by EdgeKey.
    //!
    //! Widened in Task 3 of tasks/completed/conforming-refinement.md: before, this held
    //! only the edges of this rank's *refining* faces, which is enough to verify
    //! cross-rank gid agreement but not enough to close a kept face whose
    //! neighbour across a partition boundary refined. It is now the input the
    //! conforming closure pass (Task 4) needs, and it remains exactly what
    //! checkMidpointAgreement consumes (a superset of the old contents, so the
    //! agreement check only gets stronger).
    //!
    //! In RefinementMode::Conforming it is widened once more (Task 8 D2) to the
    //! whole split-edge map of the red layer: edges an EARLIER round bisected and
    //! that still carry a hanging node are included, recovered locally by
    //! recoverSplitEdges(). Those entries are consulted by exactly one rank (a
    //! bisected edge has exactly one incident coarse face), so they trivially
    //! agree across ranks.
    std::vector<std::pair<EdgeKey, GlobalId>> midpoints;
    //! Phase-2a edge advertisements this rank sent: 3 per owned face, plus one
    //! extra for each of its already-bisected edges, which are advertised as
    //! their two halves (forEachSubEdge()).
    long long phase2Adverts = 0;
    //! Of those, the ones from refining faces — i.e. the pre-Task-3 Phase-2a
    //! volume. `phase2Adverts - phase2AdvertsRefining` is the added traffic.
    long long phase2AdvertsRefining = 0;
    //! RefinementMode::Conforming only: this rank's closure diagnostics for the
    //! step-3b pass — the |S| histogram over the post-split red layer, the
    //! visible / closure-child counts, the two blue-diagonal tallies and the
    //! count of blue quads whose two split edges were EXACTLY equal in length,
    //! so the geometric tie-break fell back to midpoint gids. Left
    //! zeroed in RefinementMode::HangingNode2to1 (there is no closure layer).
    //! Published so a test can report the closure-face fraction and the pattern
    //! distribution without instrumenting the library at the call site.
    ClosureStats closure;
};

namespace detail
{

//! Deterministic coordinator rank for an edge (routing only; disambiguated by
//! the full EdgeKey at the coordinator).
inline int edgeCoordRank( const EdgeKey& k, int comm_size )
{
    unsigned long long h = k.id[0] * 1099511628211ULL ^ k.id[1];
    return static_cast<int>( h % static_cast<unsigned long long>( comm_size ) );
}

//! One (edge, incident owned face) advertisement for the 2:1 fixpoint.
struct PropMsg
{
    EdgeKey key;
    GlobalId faceGid;
    Rank owner;
    Level level;
    unsigned char mark;
};
//! (edge, incident owned-face owner, is-that-face-refining) advertisement for
//! midpoint ownership. Sent for EVERY owned face's edges, not just refining
//! ones, so a kept face learns the midpoint gid of an edge a neighbour bisected
//! (Task 3 of tasks/completed/conforming-refinement.md). The coordinator still derives
//! ownership from the refining participants alone.
struct OwnerMsg
{
    EdgeKey key;
    Rank owner;
    unsigned char refining;
};
//! (edge, its midpoint owner) reply delivered to each participant.
struct OwnerReply
{
    EdgeKey key;
    Rank owner;
};
//! (owned edge, a co-sharer rank) delivered to the midpoint owner.
struct CosharerMsg
{
    EdgeKey key;
    Rank cosharer;
};
//! (edge, assigned midpoint gid, the WHOLE edge's squared length) delivered from
//! owner to co-sharer.
//!
//! `len2` rides along for the conforming closure's BLUE diagonal tie-break,
//! which is geometric and needs the two split edges' lengths (see the tie-break
//! note in Tessera_RefineClosure.hpp). It cannot be computed where it is used:
//! closeFaces() runs on the un-closed red layer, whose corners may name vertices
//! the rank does not hold. It CAN be computed here for free — the midpoint owner
//! is by construction the owner of an incident REFINING face, so it holds both
//! endpoints of the edge it is bisecting (Phase 3a already relies on exactly
//! that to interpolate the midpoint) — and this message already goes to every
//! co-sharer, so carrying it adds no round. Same shape as the edge-gid fix
//! (Decision 10).
struct KeyGid
{
    EdgeKey key;
    GlobalId gid;
    double len2;
};
//! (edge, incident child-face owner, incident child-face level) for edge
//! ownership + level in the refined mesh. On the coordinator's REPLY the `gid`
//! field additionally carries the edge's globally-assigned gid (step 3e): the
//! coordinator sees each EdgeKey exactly once, so it can number its keys densely
//! from an MPI_Exscan over its own distinct-key count and hand the SAME gid to
//! every rank that advertised the key. On the outbound advertisement `gid` is
//! unused (set to invalid_gid).
struct EdgeOwnMsg
{
    EdgeKey key;
    Rank owner;
    Level level;
    GlobalId gid;
};

//! Implementation of refine() for BOTH refinement modes. Called through the
//! refine() dispatcher below; see the header comment for the algorithm,
//! guarantees, and the (deferred) halo.
//!
//! The two modes share phases 1-3 verbatim: conforming refinement is a closure
//! pass wrapped around the same 2:1-balanced red engine, not a fork of it. What
//! `if constexpr ( Conforming )` adds is three purely local, communication-free
//! steps and one widened count:
//!
//!   0.  UN-CLOSE   the mesh's VISIBLE faces back to the persistent RED layer,
//!                  which is the input phases 1-3 have always expected. A
//!                  closure child names its retired red parent outright, so this
//!                  is a per-face operation with no sibling lookup.
//!   0b. TRANSLATE  `mask` from visible-face indexing (what markByQuality and
//!                  every caller produce) to red-face indexing: a red parent is
//!                  marked iff ANY of its closure children was.
//!   0c. RECOVER    the PERSISTENT split-edge map, also from the closure
//!                  bookkeeping. Being bisected is a property of the red layer
//!                  that outlives the round that caused it, whereas Phase 2 only
//!                  ever learns THIS round's bisections. Phases 1 and 2 key the
//!                  coordinator on the two HALF-edges of a persistently split
//!                  edge (forEachSubEdge()) so the 2:1 propagation and the split
//!                  decision see the true adjacency across a hanging node, and
//!                  the map is unioned into Phase 2's so a refining coarse face
//!                  REUSES the existing midpoint instead of minting a coincident
//!                  second one.
//!   3b. CLOSE      every KEPT red face whose edges are bisected in the red
//!                  layer -- this round's or an earlier one's -- using that
//!                  union. A red child of a face refined in THIS round may have
//!                  |S| > 0 only on a boundary edge it inherited whole (that is,
//!                  around a reused midpoint), which closeFaces() asserts.
//!   3c. the single face-gid MPI_Exscan additionally covers the closure
//!                  children -- countClosureChildren() supplies that count from
//!                  the post-split red topology, before any gid is handed out.
//!
//! In HangingNode2to1 mode the red layer IS the visible layer, the mask needs no
//! translation, and the "closure" is the identity, so the same code path degrades
//! to exactly the pre-conforming behaviour with no extra work and no extra
//! messages.
template <class MeshT, class Policy>
RefineResult refineImpl( MeshT& mesh,
                         MeshHalo<typename MeshT::memory_space>& halo,
                         const std::vector<char>& mask, const Policy& policy )
{
    constexpr bool kConforming =
        ( MeshT::refinement_mode == RefinementMode::Conforming );
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_REFINE );
    using memory_space = typename MeshT::memory_space;
    using Scalar = typename MeshT::scalar_type;
    constexpr int Dim = MeshT::dim;
    const int R = mesh.rank();
    const int size = mesh.commSize();
    MPI_Comm comm = mesh.comm();

    const int nv = static_cast<int>( mesh.numVertices() ); // owned + ghost
    const int nOwnedF = static_cast<int>( mesh.numOwnedFaces() );
    const int nOwnedV = static_cast<int>( mesh.numOwnedVertices() );

    RefineResult result;

    // ---- host copies: all vertices (endpoint lookup) + owned faces ----------
    Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace> hv(
        "hv", nv );
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hv, mesh.vertices() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto v_gid = Cabana::slice<VertexField::Gid>( hv );
    auto v_pos = Cabana::slice<VertexField::Position>( hv );
    auto f_gid = Cabana::slice<FaceField::Gid>( hf );
    auto f_verts = Cabana::slice<FaceField::Verts>( hf );
    auto f_lev = Cabana::slice<FaceField::Level>( hf );

    std::unordered_map<GlobalId, int> gid2lv; // vertex gid -> local index
    gid2lv.reserve( nv * 2 );
    for ( int i = 0; i < nv; ++i )
        gid2lv[v_gid( i )] = i;

    auto keyOf = []( GlobalId a, GlobalId b ) { return makeEdgeKey( a, b ); };

    // New face gids are allocated above the global max of the PRE-REFINE
    // VISIBLE face gids — not above the face count, and not above the max RED
    // gid. A refined parent's gid is retired (replaced by 4 children) and, in
    // Conforming mode, so is a closed parent's (it lives on only in its
    // children's ClosureParent field), so live gids are sparse and the max
    // exceeds the count. Taking the max over the visible layer — which contains
    // every red gid plus the closure children's, allocated above them — makes
    // the bound monotone across rounds, so no allocation can ever collide with
    // a gid that is still referenced.
    long long localMaxF = -1;
    for ( int f = 0; f < nOwnedF; ++f )
        localMaxF = std::max( localMaxF, static_cast<long long>( f_gid( f ) ) );

    // ---- steps 0 / 0b: the RED-layer snapshot phases 1-3 operate on ---------
    // fV / fG / fL are the red faces; fSrc[r] is the row of `hf` that supplies
    // red face r's face USER fields. In HangingNode2to1 mode the red layer is
    // the visible layer and this is the identity snapshot; in Conforming mode it
    // is the un-close of the transient closure layer, and `mark` is the caller's
    // visible-face mask translated onto it.
    std::vector<std::array<GlobalId, 3>> fV;
    std::vector<GlobalId> fG;
    std::vector<Level> fL;
    std::vector<int> fSrc;
    std::vector<char> mark;
    //! step 0c: the PERSISTENT split-edge map -- every red-layer edge this rank
    //! owns a face on that a FORMER round bisected, with its midpoint gid.
    //! Recovered locally from the closure bookkeeping by unclose(); empty in
    //! HangingNode2to1 mode, which keeps no such record.
    std::map<EdgeKey, GlobalId> persistentSplit;

    if constexpr ( kConforming )
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_REFINE_UNCLOSE );
        const std::vector<VisibleFace> visible =
            readVisibleFaces<MeshT>( hf, static_cast<std::size_t>( nOwnedF ) );
        const UncloseResult un = unclose( visible );
        const int nRed = static_cast<int>( un.red.size() );
        fV.resize( nRed );
        fG.resize( nRed );
        fL.resize( nRed );
        fSrc.resize( nRed );
        for ( int r = 0; r < nRed; ++r )
        {
            for ( int k = 0; k < 3; ++k )
                fV[r][k] = un.red[r].v[k];
            fG[r] = un.red[r].gid;
            fL[r] = un.red[r].level;
            fSrc[r] = un.sourceVisible[r];
        }
        mark = translateMask( mask, un );
        persistentSplit = un.splitEdges;
    }
    else
    {
        fV.resize( nOwnedF );
        fG.resize( nOwnedF );
        fL.resize( nOwnedF );
        fSrc.resize( nOwnedF );
        mark.assign( nOwnedF, 0 );
        for ( int f = 0; f < nOwnedF; ++f )
        {
            for ( int k = 0; k < 3; ++k )
                fV[f][k] = f_verts( f, k );
            fG[f] = f_gid( f );
            fL[f] = f_lev( f );
            fSrc[f] = f;
            if ( f < static_cast<int>( mask.size() ) )
                mark[f] = mask[f] ? 1 : 0;
        }
    }
    //! Number of RED faces this rank owns — what phases 1-3 iterate over.
    const int nRedF = static_cast<int>( fV.size() );

    // Advertise a red-layer edge to the edge coordinator as the edge(s) that
    // actually carry the adjacency. An edge a FORMER round bisected is a hanging
    // node: the coarse face still spans (x,y) while the fine faces opposite it
    // carry (x,m) and (m,y), so keying on (x,y) leaves BOTH sides with a single
    // incidence and every coordinator rule that needs two — the Phase-1 2:1
    // propagation, the Phase-2 split decision — silently skips the pair. Sending
    // the two HALF-edges instead makes the coarse face meet its true neighbours,
    // which is what keeps the level jump across a hanging node bounded by one and
    // hence keeps "at most one midpoint per red edge", the precondition of the
    // closure patterns. One level of expansion suffices exactly because that
    // bound holds inductively. A no-op in HangingNode2to1 mode (the map is
    // empty), so that mode's messages are unchanged.
    //
    // `fn` receives (key, isHalf). isHalf matters to Phase 2: a refining face
    // bisects the WHOLE edge (at the midpoint it already has), and does NOT
    // bisect either half — advertising a half as refining makes the coordinator
    // mint a midpoint for it, which is a spurious refinement that then cascades
    // into a red edge carrying two midpoints and no applicable closure pattern.
    auto forEachSubEdge = [&]( GlobalId x, GlobalId y, auto&& fn )
    {
        const EdgeKey key = keyOf( x, y );
        auto it = persistentSplit.find( key );
        if ( it == persistentSplit.end() )
        {
            fn( key, false );
            return;
        }
        fn( keyOf( x, it->second ), true );
        fn( keyOf( it->second, y ), true );
    };

    // ---- Phase 1: 2:1 mark-propagation fixpoint -----------------------------
    std::unordered_map<GlobalId, int> gid2of; // owned red face gid -> index
    gid2of.reserve( nRedF * 2 );
    for ( int f = 0; f < nRedF; ++f )
        gid2of[fG[f]] = f;

    const int kCap = 256;
    int iter = 0;
    {
        TESSERA_SCOPED_TIMER_DETAILED(
            ::Tessera::Profiling::TIMER_REFINE_BALANCE );
        while ( true )
        {
            std::map<EdgeKey, std::vector<detail::PropMsg>> byEdge;
            {
                TESSERA_SCOPED_TIMER_VERBOSE(
                    ::Tessera::Profiling::TIMER_REFINE_ADVERTISE );
                std::vector<std::vector<detail::PropMsg>> toCoord( size );
                for ( int f = 0; f < nRedF; ++f )
                    for ( int k = 0; k < 3; ++k )
                        forEachSubEdge(
                            fV[f][k], fV[f][( k + 1 ) % 3],
                            [&]( const EdgeKey& key, bool )
                            {
                                toCoord[detail::edgeCoordRank( key, size )]
                                    .push_back( { key, fG[f],
                                                  static_cast<Rank>( R ), fL[f],
                                                  static_cast<unsigned char>(
                                                      mark[f] ) } );
                            } );
                auto got = allToAllV( comm, toCoord );
                for ( const auto& m : got.data )
                    byEdge[m.key].push_back( m );
            }

            int changed = 0;
            {
                TESSERA_SCOPED_TIMER_VERBOSE(
                    ::Tessera::Profiling::TIMER_REFINE_MARKREQ );
                std::vector<std::vector<GlobalId>> markReq( size );
                for ( auto& kv : byEdge )
                {
                    auto& inc = kv.second;
                    if ( inc.size() != 2 )
                        continue; // closed surface: exactly two incident faces
                    const int fa = inc[0].level + inc[0].mark;
                    const int fb = inc[1].level + inc[1].mark;
                    if ( fa - fb >= 2 )
                        markReq[inc[1].owner].push_back( inc[1].faceGid );
                    else if ( fb - fa >= 2 )
                        markReq[inc[0].owner].push_back( inc[0].faceGid );
                }
                auto reqs = allToAllV( comm, markReq );

                for ( const GlobalId g : reqs.data )
                {
                    auto it = gid2of.find( g );
                    if ( it != gid2of.end() && !mark[it->second] )
                    {
                        mark[it->second] = 1;
                        ++changed;
                    }
                }
            }
            int globalChanged = 0;
            MPI_Allreduce( &changed, &globalChanged, 1, MPI_INT, MPI_SUM,
                           comm );
            ++iter;
            if ( globalChanged == 0 || iter >= kCap )
                break;
        }
    }
    result.iterations = iter;

    // ---- Phase 2: midpoint-gid assignment with cross-rank agreement ---------
    // 2a. advertise EVERY owned face's edges (owner + refining flag) to the
    //     coordinator. Advertising kept faces too is what lets a rank holding
    //     only the kept side of a bisected edge learn its midpoint gid — the
    //     split-edge map the conforming closure needs. Ownership is still
    //     decided by the refining participants alone (2b), so the midpoint
    //     assignment, its count, and hence every vertex gid are unchanged.
    {
        std::vector<std::vector<detail::OwnerMsg>> toCoord( size );
        for ( int f = 0; f < nRedF; ++f )
        {
            const unsigned char ref = mark[f] ? 1 : 0;
            for ( int k = 0; k < 3; ++k )
                forEachSubEdge(
                    fV[f][k], fV[f][( k + 1 ) % 3],
                    [&]( const EdgeKey& key, bool isHalf )
                    {
                        // A half is advertised to LEARN whether the fine side
                        // bisects it, never to claim this face bisects it.
                        const unsigned char r = isHalf ? 0 : ref;
                        toCoord[detail::edgeCoordRank( key, size )].push_back(
                            { key, static_cast<Rank>( R ), r } );
                        ++result.phase2Adverts;
                        result.phase2AdvertsRefining += r ? 1 : 0;
                    } );
        }
        auto got = allToAllV( comm, toCoord );

        // 2b. coordinator: an edge is SPLIT iff at least one incident face is
        //     refining. For a split edge the midpoint owner is the lowest
        //     incident REFINING-face owner (rule unchanged); reply (key, owner)
        //     to every participant — refining or kept — and (key, cosharer) to
        //     the owner for each other participant. An edge with no refining
        //     incidence is not split and is dropped here, so the extra kept-face
        //     advertisements add no downstream state.
        struct EdgeAgg
        {
            std::set<Rank> participants;
            Rank refOwner = 0;
            bool anyRefining = false;
        };
        std::map<EdgeKey, EdgeAgg> agg;
        for ( const auto& m : got.data )
        {
            EdgeAgg& a = agg[m.key];
            a.participants.insert( m.owner );
            if ( m.refining )
            {
                if ( !a.anyRefining || m.owner < a.refOwner )
                    a.refOwner = m.owner;
                a.anyRefining = true;
            }
        }

        std::vector<std::vector<detail::OwnerReply>> reply( size );
        std::vector<std::vector<detail::CosharerMsg>> coshare( size );
        for ( auto& kv : agg )
        {
            const EdgeKey& key = kv.first;
            const EdgeAgg& a = kv.second;
            if ( !a.anyRefining )
                continue; // edge is not bisected this round
            const Rank owner = a.refOwner;
            for ( Rank r : a.participants )
            {
                reply[r].push_back( { key, owner } );
                if ( r != owner )
                    coshare[owner].push_back( { key, r } );
            }
        }
        auto replies = allToAllV( comm, reply );
        auto coshares = allToAllV( comm, coshare );

        // 2c. this rank now knows the owner of every split edge it TOUCHES —
        //     including edges it only sees from a kept face.
        std::map<EdgeKey, Rank> midOwner;
        for ( const auto& m : replies.data )
            midOwner[m.key] = m.owner;
        std::map<EdgeKey, std::vector<Rank>> myCosharers;
        for ( const auto& m : coshares.data )
            myCosharers[m.key].push_back( m.cosharer );

        // owned midpoints (deterministic order), assign global gids via exscan.
        std::vector<EdgeKey> myMid;
        for ( const auto& kv : midOwner )
            if ( kv.second == R )
                myMid.push_back( kv.first );
        std::sort( myMid.begin(), myMid.end() );

        const long long globalV = globalOwnedVertices( mesh );
        long long myCount = static_cast<long long>( myMid.size() );
        long long baseOff = 0;
        MPI_Exscan( &myCount, &baseOff, 1, MPI_LONG_LONG, MPI_SUM, comm );
        if ( R == 0 )
            baseOff = 0;

        std::map<EdgeKey, GlobalId> midGid;
        for ( std::size_t i = 0; i < myMid.size(); ++i )
            midGid[myMid[i]] = static_cast<GlobalId>(
                globalV + baseOff + static_cast<long long>( i ) );

        // Positions of a held vertex, as doubles, for the closure's blue
        // tie-break. Everything that produces a `len2` — here, the persistent
        // recovery below, and the equivalent in refineLocalConforming() — goes
        // through this same double representation and edgeLen2Canonical(), so the
        // values are bit-identical wherever they are computed. Two ranks that
        // disagreed on one would pick different diagonals for the same quad.
        auto fetchPos = [&]( GlobalId g, double* p ) -> bool
        {
            auto it = gid2lv.find( g );
            if ( it == gid2lv.end() )
                return false;
            for ( int d = 0; d < Dim; ++d )
                p[d] = static_cast<double>( v_pos( it->second, d ) );
            return true;
        };

        //! EdgeKey -> the WHOLE edge's squared length, over exactly `midGid`'s
        //! key set. Consumed by closeFaces() for the blue diagonal.
        std::map<EdgeKey, double> len2Of;
        for ( const EdgeKey& key : myMid )
        {
            double pa[3], pb[3];
            if ( !fetchPos( key.id[0], pa ) || !fetchPos( key.id[1], pb ) )
                Kokkos::abort(
                    "Tessera::refine: the owner of a midpoint does not hold "
                    "both endpoints of the edge it is bisecting. Midpoint "
                    "ownership is 'lowest incident REFINING-face owner', and "
                    "that face has the whole edge, so this is impossible "
                    "unless "
                    "the mesh entered refine() without the vertices its owned "
                    "faces reference (see rebuildHalo())." );
            len2Of[key] =
                edgeLen2Canonical( key.id[0], pa, key.id[1], pb, Dim );
        }

        // deliver owned midpoint gids — and the bisected edge's squared length —
        // to co-sharers (refining or kept).
        std::vector<std::vector<detail::KeyGid>> toShare( size );
        for ( const EdgeKey& key : myMid )
        {
            auto it = myCosharers.find( key );
            if ( it == myCosharers.end() )
                continue;
            for ( Rank c : it->second )
                toShare[c].push_back( { key, midGid[key], len2Of[key] } );
        }
        auto shared = allToAllV( comm, toShare );
        for ( const auto& m : shared.data )
        {
            midGid[m.key] = m.gid;
            len2Of[m.key] = m.len2;
        }

        // Union the PERSISTENT split-edge map in. "This edge is bisected" is a
        // property of the red layer, not of one round: a kept face closed in an
        // earlier round is still the coarse side of a hanging node and must be
        // closed again, and if it REFINES now its 1->4 split must reuse the
        // existing midpoint rather than mint a coincident second vertex. Neither
        // is visible to Phase 2, which only ever learns this round's bisections
        // (the coordinator drops every edge with no refining incidence). The
        // recovered entries need no agreement step: a bisected edge has exactly
        // one incident coarse face, hence exactly one rank that can consult it.
        // No key can collide -- forEachSubEdge() never advertises one.
        for ( const auto& kv : persistentSplit )
            midGid.emplace( kv.first, kv.second );

        // ...and their squared lengths with them. forEachSubEdge() advertises a
        // persistently split edge only as its two HALVES, so its own key never
        // reaches a coordinator and no KeyGid can carry its length — yet a kept
        // face with one is exactly the "must be closed again" case (Decision 8),
        // so blue can arise with one or both split edges persistent. The
        // endpoints are reachable with no communication: they are corners of an
        // owned red face, and after rebuildHalo() every vertex an owned face
        // references is held WITH ITS POSITION (Decision 14) — which is why
        // follow-up 1 had to land before this could.
        addSplitEdgeLengths( persistentSplit, Dim, fetchPos, len2Of );

        // Publish the split-edge map: every edge of an owned face that is
        // bisected in the red layer, with its (globally agreed) midpoint gid.
        // Consumed by checkMidpointAgreement and, from Task 4, by the closure.
        result.midpoints.reserve( midGid.size() );
        for ( const auto& kv : midGid )
            result.midpoints.push_back( { kv.first, kv.second } );

        // ---- Phase 3: local topology reconstruction (owned only) ------------
        auto midOf = [&]( GlobalId a, GlobalId b ) -> GlobalId
        { return midGid.at( keyOf( a, b ) ); };

        // 3a. new owned vertices: originals + owned midpoints.
        const int nNewV = nOwnedV + static_cast<int>( myMid.size() );
        Cabana::AoSoA<typename MeshT::vertex_member_types, Kokkos::HostSpace>
            lv( "lv", nNewV );
        {
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
                const GlobalId a = myMid[m].id[0];
                const GlobalId b = myMid[m].id[1];
                const int la = gid2lv.at( a );
                const int lb = gid2lv.at( b );
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
        }
        // INVALIDATION: the resize/deep_copy calls in this function, the key-
        // View reassignment, and the CSR rebuild below reallocate and reassign
        // this mesh's storage, invalidating every slice/CSR/key-View handed out
        // before this call. Re-slice from the mesh after refine() returns.
        mesh.resizeVertices( nNewV );
        Cabana::deep_copy( mesh.vertices(), lv );

        // 3b. red 1->4 split: kept faces retain their gid, refined faces are
        //     replaced by 4 children. TOPOLOGY FIRST — the gids come from the
        //     exscan below, which in Conforming mode must also cover the closure
        //     children, and their count is a function of exactly this topology
        //     plus the split-edge map. Corner gids do not depend on face gids,
        //     so the ordering costs nothing.
        std::vector<RedFace> newRed;  // the post-split red layer
        std::vector<char> freshChild; // parallel: created by THIS round's split
        std::vector<int> newRedSrc;   // parallel: `hf` row with the user fields
        newRed.reserve( static_cast<std::size_t>( nRedF ) );
        freshChild.reserve( static_cast<std::size_t>( nRedF ) );
        newRedSrc.reserve( static_cast<std::size_t>( nRedF ) );
        int nRefining = 0;
        for ( int f = 0; f < nRedF; ++f )
        {
            const GlobalId a = fV[f][0], b = fV[f][1], c = fV[f][2];
            if ( !mark[f] )
            {
                RedFace rf;
                rf.v[0] = a;
                rf.v[1] = b;
                rf.v[2] = c;
                rf.gid = fG[f];
                rf.level = fL[f];
                newRed.push_back( rf );
                freshChild.push_back( 0 );
                newRedSrc.push_back( fSrc[f] );
                continue;
            }
            ++nRefining;
            const GlobalId ab = midOf( a, b );
            const GlobalId bc = midOf( b, c );
            const GlobalId ca = midOf( c, a );
            const std::array<GlobalId, 3> ch[4] = {
                { a, ab, ca }, { b, bc, ab }, { c, ca, bc }, { ab, bc, ca } };
            const Level clev = static_cast<Level>( fL[f] + 1 );
            for ( const auto& q : ch )
            {
                RedFace rf;
                for ( int k = 0; k < 3; ++k )
                    rf.v[k] = q[k];
                rf.gid = invalid_gid; // assigned from the exscan block below
                rf.level = clev;
                newRed.push_back( rf );
                freshChild.push_back( 1 );
                newRedSrc.push_back( fSrc[f] );
            }
        }

        // 3c. face-gid allocation: ONE exscan over a contiguous global block
        //     above the global max face gid, covering this rank's fresh red
        //     children AND (Conforming only) its closure children.
        long long globalMaxF = -1;
        MPI_Allreduce( &localMaxF, &globalMaxF, 1, MPI_LONG_LONG, MPI_MAX,
                       comm );
        long long nClosureNew = 0;
        if constexpr ( kConforming )
            nClosureNew = static_cast<long long>(
                countClosureChildren( newRed, midGid ) );
        long long myChild = 4LL * nRefining + nClosureNew;
        long long childBase = 0;
        MPI_Exscan( &myChild, &childBase, 1, MPI_LONG_LONG, MPI_SUM, comm );
        if ( R == 0 )
            childBase = 0;
        GlobalId childGid = static_cast<GlobalId>( globalMaxF + 1 + childBase );
        for ( std::size_t i = 0; i < newRed.size(); ++i )
            if ( freshChild[i] )
                newRed[i].gid = childGid++;
        // childGid now names the first gid of this rank's closure-child block.

        // 3b'. CLOSE (Conforming only): retriangulate every kept red face whose
        //      edges a neighbour bisected, so the VISIBLE layer has no
        //      T-junctions. Purely local: a face is owned by exactly one rank
        //      and the closure creates no vertices, so nothing is communicated.
        //      In HangingNode2to1 mode the visible layer IS the red layer.
        std::vector<VisibleFace> newVis;
        std::vector<int> visSrc; // parallel: `hf` row with the user fields
        if constexpr ( kConforming )
        {
            TESSERA_SCOPED_TIMER_DETAILED(
                ::Tessera::Profiling::TIMER_REFINE_CLOSE );
            CloseResult cl;
            {
                TESSERA_SCOPED_TIMER_VERBOSE(
                    ::Tessera::Profiling::TIMER_REFINE_CLOSURE_PATTERNS );
                cl = closeFaces( newRed, midGid, childGid, freshChild,
                                 static_cast<GlobalId>( globalV ), len2Of );
            }
            result.closure = cl.stats;
            newVis = std::move( cl.visible );
            visSrc.reserve( newVis.size() );
            for ( std::size_t i = 0; i < cl.sourceRed.size(); ++i )
                visSrc.push_back( newRedSrc[cl.sourceRed[i]] );
        }
        else
        {
            newVis.reserve( newRed.size() );
            for ( std::size_t i = 0; i < newRed.size(); ++i )
            {
                VisibleFace vf;
                for ( int k = 0; k < 3; ++k )
                    vf.v[k] = newRed[i].v[k];
                vf.gid = newRed[i].gid;
                vf.level = newRed[i].level;
                newVis.push_back( vf ); // parent stays invalid_gid
            }
            visSrc = newRedSrc;
        }
        const int nNewF = static_cast<int>( newVis.size() );

        // 3d. re-derive edges from the new VISIBLE faces (dedup by EdgeKey).
        std::map<EdgeKey, int> edge_of;
        std::vector<std::array<GlobalId, 2>> ep;
        std::vector<std::array<int, 2>> efLocal; // incident local face indices
        std::vector<std::array<int, 3>> faceEdge( nNewF );
        for ( int f = 0; f < nNewF; ++f )
            for ( int k = 0; k < 3; ++k )
            {
                const EdgeKey key =
                    keyOf( newVis[f].v[k], newVis[f].v[( k + 1 ) % 3] );
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

        // 3e. edge ownership + level + GID via coordinator (min incident face
        //     owner / level). Every edge is owned by exactly one rank globally.
        //
        //     The gid is assigned HERE, by the coordinator, rather than from a
        //     local exscan. Each EdgeKey is routed to exactly one coordinator,
        //     so a coordinator can number its own distinct keys densely from an
        //     MPI_Exscan over its key count and reply with that gid to every
        //     rank that advertised the key. That makes the refined edge gids
        //     (a) dense in [0, globalEdges) and (b) IDENTICAL on both sides of a
        //     partition boundary -- which is what every other component
        //     (distribute(), MeshBuilder, migrate()'s gid-keyed edge maps, the
        //     HDF5 writer/reader) already assumes an edge gid to be, and which
        //     vertex and face gids already satisfy. Assigning from a local
        //     exscan over nLocalE instead left gaps in the owned gid space
        //     wherever a non-last rank held a boundary duplicate, and gave the
        //     two sides of a boundary edge different gids for the same edge.
        std::vector<std::vector<detail::EdgeOwnMsg>> toEC( size );
        for ( int e = 0; e < nLocalE; ++e )
        {
            const EdgeKey key = keyOf( ep[e][0], ep[e][1] );
            Level lv0 = newVis[efLocal[e][0]].level;
            if ( efLocal[e][1] >= 0 )
                lv0 = std::min( lv0, newVis[efLocal[e][1]].level );
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
        // Dense global numbering of this coordinator's keys. std::map iteration
        // order is deterministic, so the numbering is reproducible.
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
        // Reply to each participant, grouped by source rank via from()/count().
        std::vector<std::vector<detail::EdgeOwnMsg>> ecReply( size );
        for ( int s = 0; s < size; ++s )
        {
            const detail::EdgeOwnMsg* p = ecGot.from( s );
            const int cnt = ecGot.count( s );
            for ( int i = 0; i < cnt; ++i )
            {
                const auto& agg = edgeAgg[p[i].key];
                ecReply[s].push_back(
                    { p[i].key, agg.owner, agg.level, agg.gid } );
            }
        }
        auto ecRes = allToAllV( comm, ecReply );
        std::map<EdgeKey, EdgeInfo> edgeOwner;
        for ( const auto& m : ecRes.data )
            edgeOwner[m.key] = { m.owner, m.level, m.gid };

        // 3f. order edges owned-first (local layout only -- the gids themselves
        //     come from the coordinator, so a boundary edge carries the same gid
        //     on every rank that holds it).
        std::vector<int> order; // local edge indices, owned first
        order.reserve( nLocalE );
        std::vector<char> isOwned( nLocalE, 0 );
        for ( int e = 0; e < nLocalE; ++e )
        {
            const EdgeKey key = keyOf( ep[e][0], ep[e][1] );
            if ( edgeOwner[key].owner == R )
            {
                isOwned[e] = 1;
                order.push_back( e );
            }
        }
        const int nOwnedE = static_cast<int>( order.size() );
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
            TESSERA_SCOPED_TIMER_DETAILED(
                ::Tessera::Profiling::TIMER_REFINE_REBUILD );
            Cabana::AoSoA<typename MeshT::edge_member_types, Kokkos::HostSpace>
                le( "le", nLocalE );
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
                // Incident faces: local visible-face gids; the cross-rank second
                // face is filled when the halo is rebuilt (Step 7).
                faces( li, 0 ) = newVis[efLocal[e][0]].gid;
                faces( li, 1 ) = efLocal[e][1] >= 0 ? newVis[efLocal[e][1]].gid
                                                    : invalid_gid;
            }
            mesh.resizeEdges( nLocalE );
            Cabana::deep_copy( mesh.edges(), le );
        }

        // 3h. materialize the VISIBLE face AoSoA (edges now have gids).
        {
            TESSERA_SCOPED_TIMER_DETAILED(
                ::Tessera::Profiling::TIMER_REFINE_REBUILD );
            Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace>
                lf( "lf", nNewF );
            auto gid = Cabana::slice<FaceField::Gid>( lf );
            auto own = Cabana::slice<FaceField::Owner>( lf );
            auto lev = Cabana::slice<FaceField::Level>( lf );
            auto verts = Cabana::slice<FaceField::Verts>( lf );
            auto edges = Cabana::slice<FaceField::Edges>( lf );
            for ( int f = 0; f < nNewF; ++f )
            {
                gid( f ) = newVis[f].gid;
                own( f ) = static_cast<Rank>( R );
                lev( f ) = newVis[f].level;
                for ( int k = 0; k < 3; ++k )
                {
                    verts( f, k ) = newVis[f].v[k];
                    edges( f, k ) = edgeGid[faceEdge[f][k]];
                }
                // User fields chase the provenance visible face -> its red face
                // -> the pre-call visible row that red face's fields came from.
                copyUserFieldsN<
                    FaceField::UserBegin,
                    numFaceUserFields<typename MeshT::face_user_fields>()>(
                    lf, f, hf, visSrc[f] );
                // ClosureParent / ClosureParentVerts; no-op in HangingNode2to1.
                writeClosureFace<MeshT>( lf, static_cast<std::size_t>( f ),
                                         newVis[f] );
            }
            mesh.resizeFaces( nNewF );
            Cabana::deep_copy( mesh.faces(), lf );
        }

        // owned-only mesh: every local entity is owned.
        mesh.setOwnedCounts( nNewV, nOwnedE, nNewF );

        // 3i. rebuild key side tables.
        {
            TESSERA_SCOPED_TIMER_DETAILED(
                ::Tessera::Profiling::TIMER_REFINE_REBUILD );
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
                h_fk( f ) = makeFaceKey( newVis[f].v[0], newVis[f].v[1],
                                         newVis[f].v[2] );
            Kokkos::deep_copy( fk, h_fk );
            mesh.setFaceKeys( fk );
        }

        // NOTE: no local 1-ring CSR rebuild here. This used to be step 3j, a
        //       deliberately INCOMPLETE best-effort CSR over local faces and
        //       edges only, with GUARDED gid lookups because a face may
        //       legitimately name a vertex this rank does not hold -- always
        //       true across a partition boundary, and in Conforming mode also
        //       true of a closure child whose midpoint corner is owned by the
        //       refining neighbour. rebuildHalo() below brings those vertices
        //       in and rebuilds both CSRs COMPLETELY over the ghosted 1-ring,
        //       so the partial version was dead work whose only effect was to
        //       leave a container that looked valid and was not.
    }

    // Rebuild the 1-deep ghost layer and the three halo plans over the refined
    // owned entities. Phases 1-3 above deliberately produce an OWNED-ONLY mesh —
    // every cross-rank decision went through a coordinator, so no ghost was
    // needed — but leaving it that way made the library easy to misuse: a
    // haloExchange() would silently no-op on an empty plan, and a second
    // refine() would throw from Phase 3a, which needs the POSITIONS of both
    // endpoints of every midpoint this rank owns and across a partition boundary
    // holds one of them only as a ghost. rebuildHalo() moves nothing; it
    // discovers ownership and the ghost layer by communication (rounds G/B/C/D,
    // shared with migrate()) and leaves ghost values equal to the owners' values.
    //
    // INVALIDATION: this reallocates the AoSoAs, key Views and CSRs and replaces
    // the halo plans, as does everything above it, so every slice/CSR/key-View a
    // caller took out before this refine() call is dangling. Re-slice from the
    // mesh.
    // The depth is PRESERVED, not reset: refine() rebuilds the halo the caller
    // asked for at setup, so a depth-2 mesh is still depth-2 after any number of
    // refine() rounds.
    rebuildHalo( mesh, halo, effectiveHaloDepth( halo ) );

    return result;
}

} // namespace detail

//! Distributed 2:1-balanced red refinement of the owned faces flagged in `mask`.
//! See the header comment for the algorithm, guarantees, and the (deferred) halo.
//!
//! `mask` is indexed by this rank's owned faces as the CALLER sees them — i.e.
//! the VISIBLE faces, which in RefinementMode::Conforming are the closure layer.
//! The mode branch lives inside detail::refineImpl(); see its comment for the
//! three steps conforming refinement adds (un-close, mask translation, close)
//! and why phases 1-3 are shared verbatim.
//!
//! RefinementMode::Conforming post-conditions, beyond the hanging-node ones:
//! every edge of the global visible mesh has exactly two incident faces and the
//! owned-only Euler number is 2 for an ARBITRARY (not just uniform) mask. Face
//! `Level` remains the RED level — a closure child carries its parent's — so in
//! this mode `Level` no longer maps 1:1 to triangle size.
template <class MeshT,
          class Policy = DefaultRefinePolicy<typename MeshT::scalar_type>>
RefineResult refine( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
                     const std::vector<char>& mask,
                     const Policy& policy = Policy{} )
{
    // refine() is the HIERARCHICAL editing family: it maintains the 2:1 level
    // balance and the conforming closure, both of which read Level as
    // authoritative. A mesh already edited by the REMESH family (splitEdges())
    // carries inherited, advisory levels, so balancing against them would be a
    // silent wrong answer -- see Tessera_EditFamily.hpp.
    requireEditFamily( mesh, EditFamily::Hierarchical, "refine" );
    return detail::refineImpl( mesh, halo, mask, policy );
}

} // namespace Tessera

#endif // TESSERA_REFINE_PARALLEL_HPP
