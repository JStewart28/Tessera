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

#ifndef TESSERA_EDGE_COLLAPSE_HPP
#define TESSERA_EDGE_COLLAPSE_HPP

#include "Tessera_AllToAllV.hpp"
#include "Tessera_Compact.hpp"
#include "Tessera_Distribute.hpp"
#include "Tessera_EdgeFlip.hpp"
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
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Tessera
{

// ============================================================================
// Caller-driven edge collapse (REMESH editing family)
// ============================================================================
//
// collapseEdges() merges the two endpoints of each marked edge into one vertex.
// For an interior manifold edge (a,b) whose incident faces are (a,b,c) and
// (b,a,d):
//
//        c                    c
//       /|\                  / \
//      / | \                /   \
//     a--+--b     --->     a=====        the merged vertex sits at parameter
//      \ | /                \   /        policy.t along (a,b)
//       \|/                  \ /
//        d                    d
//
//   * the two incident faces (a,b,c) and (b,a,d) are DELETED;
//   * the edge (a,b) is deleted, and so is one of each COINCIDENT PAIR --
//     (a,c) with (b,c), and (a,d) with (b,d);
//   * every other face and edge that referenced the dying endpoint now
//     references the surviving one.
//
// Net V-1, E-3, F-2, so the Euler number is preserved. THIS IS THE ONLY
// OPERATION IN TESSERA THAT REMOVES DEGREES OF FREEDOM: without it a
// metric-driven remesher is monotone and grows without bound.
//
// THE SURVIVING VERTEX IS min(gid(a), gid(b)) -- deterministic, rank-count
// independent, and the same value every rank derives without an agreement
// round. Because an EdgeKey is canonical (lo, hi) the survivor is always
// key.id[0], which is also what orients policy.t: t = 0 keeps the LOWER-gid
// endpoint, t = 1 keeps the higher one, t = 0.5 is the midpoint.
//
// ----------------------------------------------------------------------------
// Problem 1 -- where coarsening attaches to the data model (DECISION 1)
// ----------------------------------------------------------------------------
//
// COLLAPSE IS A REMESH-FAMILY OPERATION AND IS NOT DEFINED ON A refine()D MESH,
// in either refinement mode. Under RefinementMode::Conforming there is a
// transient closure layer whose children reference retired red parents and
// collapsing across it would have to un-close first; under HangingNode2to1 there
// are T-junctions and the link condition is not even well posed at one. The
// EditFamily guard (Tessera_EditFamily.hpp) enforces this with a message naming
// both families. `Level` is ADVISORY here, as it is for every remesh operation:
// the merged edge of a coincident pair takes the MIN of the two levels and
// nothing downstream may read a 2:1 statement into it.
//
// A true inverse to refine() -- un-refining a red split back to its parent -- is
// a different and much larger feature (it needs the parent's identity retained,
// sibling co-residency and an inverse to the closure) and is explicitly OUT OF
// SCOPE. It is also not what a metric remesher wants, because it can only
// coarsen along the refinement tree.
//
// ----------------------------------------------------------------------------
// Problem 2 -- the link condition, and why it is not answered from residency
// ----------------------------------------------------------------------------
//
// Collapsing (a,b) is topologically valid iff
//
//     link(a) INTERSECT link(b) == { c, d }
//
// -- the two vertices opposite the edge and nothing else. A violation welds two
// distant parts of the surface together: if some x other than c or d is adjacent
// to both endpoints, the edges (a,x) and (b,x) become coincident and the result
// is non-manifold in a way no later check untangles.
//
// Evaluating it needs the FULL ONE-RING OF BOTH ENDPOINTS, i.e. a two-ring
// around the edge, so collapseEdges() requires halo depth >= 2 and THROWS
// otherwise, naming the required and the actual depth. The guard is
//
//     throw iff 0 < mesh.haloDepth() < 2
//
// and not `>= 2` because haloDepth() == 0 means "NEVER DISTRIBUTED" -- a
// replicated mesh straight out of the builder, where every entity is local and
// no ring is missing (Tessera_Mesh.hpp). Rejecting that would reject exactly the
// hand-built soup fixtures. buildVertexStencil() already polices depth this way
// (Tessera_Stencil.hpp), and inventing a second convention would be worse than
// the asymmetry.
//
// CORRECTNESS DOES NOT REST ON LOCAL RING RESIDENCY, ONLY THE PRECONDITION DOES.
// Depth d guarantees the d-ring of every OWNED vertex, and the owner of the edge
// (a,b) need own NEITHER endpoint -- an owned face can have all three corners as
// ghosts, which is the case edge-flip had to design around. So both one-rings
// are assembled from VERTEX-COORDINATOR ADVERTISEMENTS, which are exact by
// construction at any depth: a corner coordinator holds every owned face
// incident on its vertex, and every face of the mesh is owned by exactly one
// rank.
//
// THE VERTEX LINK TEST IS NECESSARY BUT NOT QUITE SUFFICIENT, and the gap is
// closed here rather than left as folklore. On a mesh that is a single
// TETRAHEDRON (or any place where the collapse would fuse two triangles into
// one) every vertex link test passes and yet the result carries TWO FACES WITH
// THE SAME THREE CORNERS. So the coordinator additionally rewrites the faces
// incident on the dying endpoint and rejects the candidate if the rewritten face
// set has a duplicate face key. It holds both endpoints' incident faces already,
// so this costs no extra round; it is counted as rejectedLinkCondition because
// it is the same class of defect -- a topologically invalid merge.
//
// ----------------------------------------------------------------------------
// Problem 3 -- conflicts, and why not shortest-first (DECISION 2)
// ----------------------------------------------------------------------------
//
// Two collapses whose neighbourhoods touch conflict: applying one changes
// whether the other is valid, and applying both can produce garbage. Serial
// reference codes process SHORTEST-FIRST, re-evaluating validity after each
// acceptance; that is inherently sequential and its result depends on the
// processing order.
//
// A DETERMINISTIC INDEPENDENT SET, ONE ROUND PER CALL. Each candidate carries
// the priority
//
//     (squared length ASCENDING, EdgeKey ASCENDING)
//
// -- shortest first, exactly matching the serial PREFERENCE -- and is accepted
// only if it holds the strictly best priority among all candidates incident on
// either endpoint's one-ring. That relation is symmetric, so the accepted set is
// pairwise non-conflicting: no two accepted collapses share a vertex, an edge or
// a face, and their V/E/F deltas are exactly additive.
//
// The priority contains NO GID. Gids come from an MPI_Exscan and are agreed
// across the ranks of one run but are not the same values at a different rank
// count -- the finding that forced splitEdges()' diagonal tie-break away from
// the closure's lower-midpoint-gid rule. An EdgeKey is built from pre-existing
// vertex gids and IS invariant, and every length goes through
// edgeLen2Canonical() (Tessera_RefineClosure.hpp) so two ranks comparing the
// same edge get bit-identical doubles regardless of endpoint order. This is
// flipEdges()' rule with the length ordering REVERSED -- shortest-first here,
// longest-first there.
//
// THE ACCEPTED SET IS NOT THE SERIAL SHORTEST-FIRST SET. It is a subset of it,
// reached in fewer passes. A caller wanting more progress calls again and
// watches `accepted`:
//
//     while ( collapseEdges( mesh, halo, shortEdges( mesh ) ).accepted > 0 ) {}
//
// CONSUMERS MUST COMPARE STATISTICS -- face count, quality distribution,
// edge-length histogram -- NOT EDIT SETS. This is the single most likely source
// of a "why doesn't this match the serial code" question, so it is stated here,
// in the README and in docs/design.md.
//
// ----------------------------------------------------------------------------
// Problem 4 -- cross-rank owner-decides, in five rounds
// ----------------------------------------------------------------------------
//
// The one-rings of a and b may span several ranks even at depth 2, and the
// merged vertex's identity and position must be agreed before any rank rewrites
// connectivity. Everything below is built on the existing coordinators --
// detail::edgeCoordRank for an edge, gid % size for a vertex -- and allToAllV,
// the same machinery refine()'s 2:1 balance and splitEdges() use.
//
//   1. ADVERTISE (one round, five messages). Per OWNED FACE and each of its
//      three edges, a detail::CollapseAdvert to that EDGE's coordinator (the
//      edge key, the face gid and owner, the face's three edge gids named by
//      which endpoint of the canonical key they meet, the opposite corner, the
//      winding, and the three corner POSITIONS). Per owned face and each of its
//      three CORNERS, a detail::CollapseCornerAdvert to that VERTEX's
//      coordinator (the face's three corner gids, its three edge gids, and the
//      three positions). Per owned VERTEX its owner, per owned EDGE its gid,
//      endpoints, owner and level to both endpoints' vertex coordinators. And
//      per MARKED owned edge a detail::CollapseVerdict to the edge's coordinator
//      -- presence at the coordinator IS the verdict, as in splitEdges().
//
//      THERE IS NO SHARED ADVERTISEMENT HELPER TO REUSE. detail::FlipAdvert is
//      private to Tessera_EdgeFlip.hpp and its payload carries one opposite
//      corner, which is not what a collapse needs; refactoring that header to
//      extract a common helper would destabilise an operation that is green at
//      ten registrations for the sake of an abstraction over two payloads that
//      differ. Only its three geometry helpers (detail::flipTriNormal,
//      flipAngleBetween, flipRadiusRatio) are reused, unchanged.
//
//   2. ASSEMBLE THE CANDIDATE FROM COORDINATOR STATE, NOT FROM RESIDENCY. The
//      edge coordinator for (a,b) holds both incident faces' advertisements,
//      hence a, b, c, d and all four positions, so it evaluates the boundary
//      test (other than exactly two incident faces -> reject), computes the
//      priority through edgeLen2Canonical() and the merged position through the
//      policy hook. It then asks BOTH endpoints' VERTEX coordinators one
//      question, and they answer with everything the rest of the operation
//      needs: the endpoint's one-ring, its owner, every owned face and owned
//      edge incident on it (with owners, so the apply round can be addressed
//      directly), and the GEOMETRIC VERDICT.
//
//      THE GEOMETRIC TESTS ARE EVALUATED AT THE VERTEX COORDINATOR, because that
//      is the only place where every surviving face incident on an endpoint is
//      known. Each such face has the endpoint's position replaced by the merged
//      position; the candidate is rejected if any face's normal rotates by more
//      than policy.maxNormalRotation (the fold guard) or if any face's radius
//      ratio falls below policy.minQuality. The two dying faces -- the ones
//      containing both endpoints -- are skipped.
//
//   3. INDEPENDENT-SET ROUND. The coordinator names, for each surviving
//      candidate, every vertex in {a, b} UNION ring(a) UNION ring(b), and sends
//      the priority to each of those vertices' coordinators. A vertex
//      coordinator replies only to the single best candidate that named it. A
//      candidate is accepted iff it collects one reply per named vertex.
//
//   4. APPLY, IN PLACE. The coordinator ships to whoever has to write:
//        * the two dying face gids to their owners, which TOMBSTONE them
//          (Gid = invalid_gid, tombstoneFace()'s convention);
//        * a face fix to the owner of every other face incident on the dying
//          endpoint: rewrite that corner to the surviving gid, and rewrite the
//          two dying edge gids of the coincident pairs to their survivors;
//        * an edge fix to the owner of every edge incident on the dying endpoint
//          other than the three that die: rewrite that endpoint;
//        * a level fix to the owner of each surviving coincident edge: MIN of
//          the pair, per Decision 1;
//        * the merged position to the owner of the surviving vertex, which also
//          blends every vertex user field through the policy hook.
//
//      DO NOT TOMBSTONE THE COLLAPSED EDGE OR THE COINCIDENT PAIRS. compact()
//      drops every vertex and edge that no surviving face references, so getting
//      the FACE set right is sufficient -- and tombstoning an edge that a
//      surviving face still names trips compact()'s GLOBAL closure check and
//      throws. What is genuinely required is the connectivity rewrite: every
//      surviving face's Verts and Edges must name the surviving gid of each
//      merged pair, or the tombstone set does not close.
//
//      DROP THE STALE GHOST TUPLES BEFORE THE REBUILD. Because this operation
//      rewrites entities IN PLACE it resizes all three AoSoAs down to their
//      owned counts first. rebuildHalo() seeds its gid -> tuple map from every
//      locally held entity, owned AND ghost, and only re-fetches the ones that
//      are missing -- so a rank holding a ghost copy of a rewritten edge would
//      keep the OLD endpoints, and the CSR build indexes a dense gid -> local
//      array and writes out of bounds, corrupting the heap. That is not
//      tidiness; it is the bug edge-flip hit as `double free or corruption
//      (out)` at np2 before printing a single case line. UNLIKE flipEdges(),
//      THE GHOST VERTICES GO TOO: a collapse MOVES the surviving vertex, so a
//      ghost copy of it holds a stale position that no round of the rebuild
//      would correct.
//
//   5. compact(), AND NOTHING AFTER IT. compact() already ends with
//      rebuildHalo( mesh, halo, effectiveHaloDepth( halo ) )
//      (Tessera_Compact.hpp), so the incoming depth is preserved without a
//      second call -- adding one is a redundant collective, not a safety net.
//      collapseEdges() returns a compact, fully-haloed mesh at the depth it was
//      handed, and a caller never sees a tombstone.
//
// A NO-OP IS A GENUINE NO-OP. An empty mask costs one collective and returns;
// and if every candidate is rejected, the compaction is skipped too, so the
// entity ordering, every gid and every key are bit-identical to the input.
//
// ENTRY PRECONDITION ON VERTEX DATA. The merged position and every merged user
// field are blended by the SURVIVING VERTEX'S OWNER from its local copies of the
// two endpoints, one of which is in general a GHOST. So vertex positions and
// vertex user fields must be halo-consistent on entry -- call haloExchange()
// after mutating them and before collapsing. splitEdges() reads its endpoints
// the same way and carries the same requirement.
//
// EdgeField::Faces IS NOT TRUSTED AS INPUT and is NOT REPAIRED AT ALL on output.
// It is best-effort by design (migrate() carries it verbatim, so it can name a
// face no rank holds -- Tessera_FaceAdjacency.hpp), so all incidence here is
// derived from the ADVERTISEMENTS, which are built from owned faces and are
// therefore true. Be aware of the output side: a surviving merged edge keeps the
// Faces it had, one entry of which now names a TOMBSTONED face, and compact()'s
// closure check does not look at it (it checks the references of live owned
// FACES only). This is weaker than flipEdges(), which at least rewrites the
// flipped edge's Faces exactly. Repairing it would cost a round on a field no
// code path in src/ reads for correctness; making it exact is
// tasks/completed/face-adjacency.md.
//
// Preconditions: `mesh` is distributed (post-distribute) or replicated
// (haloDepth() == 0), owned-first, entities carry global gids, the halo is valid
// (rebuildHalo() has run), and `edgeMask.size() == mesh.numOwnedEdges()`.
// Collective.

//! Result of collapseEdges(). Every counter is GLOBAL, and the five verdict
//! counters partition `requested`.
struct CollapseResult
{
    //! Marked OWNED edges, summed globally.
    long long requested = 0;
    //! Edges actually collapsed, globally.
    long long accepted = 0;
    //! Rejected: other than exactly two incident faces (a boundary edge has
    //! one; a non-manifold edge has more).
    long long rejectedBoundary = 0;
    //! Rejected: link(a) INTERSECT link(b) != {c,d}, or the merge would fuse
    //! two faces into one (the tetrahedron case) -- see the header comment.
    long long rejectedLinkCondition = 0;
    //! Rejected: a surviving incident face's normal would rotate by more than
    //! policy.maxNormalRotation.
    long long rejectedNormalFlip = 0;
    //! Rejected: a surviving incident face's radius ratio would fall below
    //! policy.minQuality.
    long long rejectedQuality = 0;
    //! Rejected: passed every test but lost the independent-set round to a
    //! higher-priority candidate in one of its endpoints' one-rings.
    long long rejectedConflict = 0;

    //! Entities actually removed, from the compaction: verticesRemoved ==
    //! accepted, edgesRemoved == 3 * accepted, facesRemoved == 2 * accepted for
    //! a closed manifold, and the identity is the cheapest end-to-end check that
    //! the connectivity rewrite closed.
    long long verticesRemoved = 0, edgesRemoved = 0, facesRemoved = 0;

    //! THE COLLAPSED SET: the EdgeKey of every accepted collapse this rank
    //! TOUCHES -- one whose decision it made or whose rewrite it applied.
    //! Sorted and unique. ADDED beyond the API block in tasks/completed/edge-collapse.md,
    //! for the same reason SplitResult::midpoints and FlipResult::flipped exist:
    //! without it a caller (and the test) cannot learn the accepted set without
    //! instrumenting the library at the call site.
    std::vector<EdgeKey> collapsed;
};

//! Geometric admissibility of a candidate collapse, and the blend that produces
//! the merged vertex. Evaluated on the host, so a derived policy may use
//! ordinary host code.
//!
//! PER-FIELD OVERRIDE PATTERN -- the same one DefaultRefinePolicy documents, so
//! a consumer's existing refine policy transfers by inspection: derive, shadow
//! interpolateVertexField, and dispatch on M with `if constexpr` so exactly one
//! field's rule changes while every other falls through to the base.
//!
//!     struct MyPolicy : Tessera::DefaultCollapsePolicy
//!     {
//!         template <std::size_t M>
//!         double interpolateVertexField( double a, double b, double t ) const
//!         {
//!             if constexpr ( M == Tessera::VertexField::UserBegin + 0 )
//!                 return conserve_vorticity( a, b, t );
//!             else
//!                 return Tessera::DefaultCollapsePolicy::
//!                     template interpolateVertexField<M>( a, b, t );
//!         }
//!     };
//!
//! THE HOOKS CARRY `t` AND DefaultRefinePolicy'S DO NOT, which is why this is a
//! separate policy rather than a reuse: DefaultRefinePolicy::
//! interpolatePosition( mid, a, b, dim ) and interpolateVertexField<M>( a, b )
//! are hard-coded 0.5 averages with no t argument (Tessera_RefinePolicy.hpp), so
//! a collapse at t != 0.5 is not expressible through them.
struct DefaultCollapsePolicy
{
    //! 0.5 = midpoint. 0.0 keeps a, 1.0 keeps b, where `a` is the LOWER-GID
    //! endpoint -- so t is oriented by gid and not by local index, which is
    //! what makes the result rank-count invariant.
    double t = 0.5;
    //! Reject if any surviving incident face's normal rotates more than this
    //! (radians) -- the standard fold guard.
    double maxNormalRotation = 0.5;
    //! Reject if any surviving incident face's radius ratio (inradius /
    //! circumradius, scaled to 1 for an equilateral triangle -- the same
    //! convention DefaultFlipPolicy::minQuality uses) falls below this.
    double minQuality = 0.05;

    //! Merged position: out[0..dim) = (1-t)*a[d] + t*b[d].
    void interpolatePosition( double* out, const double* a, const double* b,
                              double t_, int dim ) const
    {
        for ( int d = 0; d < dim; ++d )
            out[d] = ( 1.0 - t_ ) * a[d] + t_ * b[d];
    }

    //! Blend one scalar component of the vertex user field at ABSOLUTE member
    //! index M, same M convention as DefaultRefinePolicy.
    template <std::size_t M>
    double interpolateVertexField( double a, double b, double t_ ) const
    {
        return ( 1.0 - t_ ) * a + t_ * b;
    }
};

namespace detail
{

//! Coordinator rank of a VERTEX gid. The gid-modulo convention every other
//! per-vertex coordinator round in the tree uses (compact()'s closure check,
//! checkOwnershipPartition, the valence tally in test_flip_edges), named here so
//! the collapse rounds read as coordinator rounds rather than as arithmetic.
inline int vertexCoordRank( GlobalId g, int size )
{
    return static_cast<int>( g % static_cast<GlobalId>( size ) );
}

//! One (owned face, one of its three edges) advertisement to that EDGE's
//! coordinator. Same SHAPE as detail::FlipAdvert -- see the header comment on
//! why that type is not reused -- carrying everything needed to identify the
//! quad and to name the four edges that participate in a merge.
struct CollapseAdvert
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
    double posLo[3];
    double posHi[3];
    double posOpp[3];
};

//! One (owned face, one of its three corners) advertisement to that VERTEX's
//! coordinator. The coordinator of v therefore holds EVERY owned face incident
//! on v -- exact at any halo depth, which is what makes the link condition and
//! the geometric tests answerable without local ring residency.
struct CollapseCornerAdvert
{
    GlobalId v;       //!< the corner this advertisement is about
    GlobalId faceGid; //!< the advertising face
    Rank owner;       //!< the face's owner
    GlobalId cv[3];   //!< the face's three corners, in face order
    GlobalId ce[3];   //!< the face's three edges, edge k = (cv[k], cv[k+1])
    double p[3][3];   //!< positions of cv[0..2]
};

//! (owned vertex, its owner), to that vertex's coordinator. The coordinator
//! needs it to address the merged-position message: an edge coordinator learns a
//! vertex's owner only through this reply chain.
struct CollapseVertOwn
{
    GlobalId gid;
    Rank owner;
};

//! (owned edge, its endpoints, owner and level), to BOTH endpoints' vertex
//! coordinators. Faces are advertised by their owner and edges by theirs, and
//! the two need not agree -- so the edge owners cannot be inferred from the
//! corner advertisements and are advertised separately.
struct CollapseEdgeInc
{
    GlobalId v;
    GlobalId edgeGid;
    GlobalId ev[2];
    Rank owner;
    Level level;
};

//! (edge, its gid, its owner) sent by the OWNER of each MARKED edge. Presence at
//! the coordinator IS the verdict, as in splitEdges()' SplitVerdict.
struct CollapseVerdict
{
    EdgeKey key;
    GlobalId edgeGid;
    Rank owner;
};

//! "Tell me about endpoint `which` of candidate `cand`", asked by the edge
//! coordinator of the vertex coordinator. `mergedPos` travels so the geometric
//! test can be evaluated in the reply.
struct CollapseLinkQuery
{
    EdgeKey cand;
    GlobalId which; //!< the endpoint this query is about
    GlobalId other; //!< the far endpoint, so the dying faces can be skipped
    double mergedPos[3];
};

//! One vertex of ring(which). The reply is one message per ring vertex because
//! allToAllV moves fixed-size records and a ring has no fixed size.
struct CollapseRingMsg
{
    EdgeKey cand;
    GlobalId which;
    GlobalId ring;
};

//! One owned face incident on `which`, with its owner, so the coordinator can
//! address the rewrite directly rather than through a second hop.
struct CollapseIncFace
{
    EdgeKey cand;
    GlobalId which;
    GlobalId faceGid;
    Rank owner;
    GlobalId cv[3];
    GlobalId ce[3];
};

//! One owned edge incident on `which`, with its owner and level.
struct CollapseIncEdge
{
    EdgeKey cand;
    GlobalId which;
    GlobalId edgeGid;
    Rank owner;
    GlobalId ev[2];
    Level level;
};

//! The per-endpoint verdict: the endpoint's owner, and whether every SURVIVING
//! face incident on it clears the policy's fold and quality tests once that
//! corner has moved to the merged position.
struct CollapseGeom
{
    EdgeKey cand;
    GlobalId which;
    Rank vOwner;
    unsigned char normalOk;
    unsigned char qualityOk;
};

//! (a vertex in a candidate's two-ring, that candidate, its priority length),
//! delivered to the VERTEX's coordinator, which keeps only the best candidate.
struct CollapseNbr
{
    GlobalId v;
    EdgeKey cand;
    double len2;
};

//! A vertex coordinator's endorsement of the one candidate it kept, returned to
//! that candidate's EDGE coordinator. A candidate is accepted iff it collects
//! one endorsement per vertex it named.
struct CollapseWin
{
    EdgeKey cand;
};

//! A face to tombstone, delivered to its owner.
struct CollapseTomb
{
    GlobalId faceGid;
};

//! The rewrite of one surviving face, delivered to its owner: the dying corner
//! becomes `toV`, and the two dying edge gids of the coincident pairs become
//! their survivors.
struct CollapseFaceFix
{
    GlobalId faceGid;
    GlobalId fromV, toV;
    GlobalId fromE[2], toE[2];
};

//! The rewrite of one surviving edge's endpoint, delivered to its owner.
struct CollapseEdgeFix
{
    GlobalId edgeGid;
    GlobalId fromV, toV;
};

//! The merged level of a surviving coincident edge (Decision 1: advisory).
struct CollapseEdgeLevel
{
    GlobalId edgeGid;
    Level level;
};

//! The merged position, delivered to the SURVIVING vertex's owner, which also
//! blends the vertex user fields from its local copies of the two endpoints.
struct CollapseVertexFix
{
    GlobalId surviving;
    GlobalId dying;
    double pos[3];
};

//! One accepted collapse, delivered to every participant so
//! CollapseResult::collapsed is complete on each rank that touches it.
struct CollapseNote
{
    EdgeKey key;
};

//! Blend one vertex user member (absolute index Mabs) of the SURVIVING vertex
//! from its two endpoint rows through the policy hook, component-wise for an
//! array-valued field. The counterpart of detail::blendVertexMember()
//! (Tessera_Refine.hpp) with the collapse parameter `t` added; it cannot be
//! reused because that hook takes no t.
template <std::size_t Mabs, class VAoSoA, class Policy>
void blendVertexMemberT( VAoSoA& V, int dst, int a, int b, double t,
                         const Policy& policy )
{
    using MT = typename VAoSoA::member_types;
    using FieldT = typename Cabana::MemberTypeAtIndex<Mabs, MT>::type;
    auto s = Cabana::slice<Mabs>( V );
    if constexpr ( std::rank<FieldT>::value == 0 )
    {
        s( dst ) = static_cast<FieldT>(
            policy.template interpolateVertexField<Mabs>(
                static_cast<double>( s( a ) ), static_cast<double>( s( b ) ),
                t ) );
    }
    else
    {
        using ElemT = typename std::remove_extent<FieldT>::type;
        constexpr int C = static_cast<int>( std::extent<FieldT, 0>::value );
        for ( int c = 0; c < C; ++c )
            s( dst, c ) = static_cast<ElemT>(
                policy.template interpolateVertexField<Mabs>(
                    static_cast<double>( s( a, c ) ),
                    static_cast<double>( s( b, c ) ), t ) );
    }
}

template <class VAoSoA, class Policy, std::size_t... Js>
void blendVertexUserTImpl( VAoSoA& V, int dst, int a, int b, double t,
                           const Policy& policy, std::index_sequence<Js...> )
{
    ( blendVertexMemberT<VertexField::UserBegin + Js>( V, dst, a, b, t,
                                                       policy ),
      ... );
}

//! Blend every vertex user field of the surviving vertex from the two endpoints.
template <class VAoSoA, class Policy>
void blendVertexUserFieldsT( VAoSoA& V, int dst, int a, int b, double t,
                             const Policy& policy )
{
    constexpr std::size_t N =
        VAoSoA::member_types::size - VertexField::UserBegin;
    blendVertexUserTImpl( V, dst, a, b, t, policy,
                          std::make_index_sequence<N>{} );
}

//! Implementation of collapseEdges(); see the header comment for the algorithm
//! and the guarantees.
template <class MeshT, class Policy>
CollapseResult
collapseEdgesImpl( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
                   const std::vector<char>& edgeMask, const Policy& policy )
{
    TESSERA_SCOPED_TIMER( ::Tessera::Profiling::TIMER_COLLAPSE );
    constexpr int Dim = MeshT::dim;
    constexpr int D3 = Dim < 3 ? Dim : 3;
    const int R = mesh.rank();
    const int size = mesh.commSize();
    MPI_Comm comm = mesh.comm();

    const int nv = static_cast<int>( mesh.numVertices() ); // owned + ghost
    const int nOwnedV = static_cast<int>( mesh.numOwnedVertices() );
    const int nOwnedE = static_cast<int>( mesh.numOwnedEdges() );
    const int nOwnedF = static_cast<int>( mesh.numOwnedFaces() );

    CollapseResult result;

    // ---- host copies: all vertices (positions/fields) + all edges/faces -----
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
    auto e_lev = Cabana::slice<EdgeField::Level>( he );
    auto e_verts = Cabana::slice<EdgeField::Verts>( he );
    auto f_gid = Cabana::slice<FaceField::Gid>( hf );
    auto f_verts = Cabana::slice<FaceField::Verts>( hf );
    auto f_edges = Cabana::slice<FaceField::Edges>( hf );

    std::unordered_map<GlobalId, int> gid2lv; // vertex gid -> local index
    gid2lv.reserve( nv * 2 + 1 );
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

    // ---- Round 1: advertise -------------------------------------------------
    std::vector<std::vector<CollapseAdvert>> toEdge( size );
    std::vector<std::vector<CollapseCornerAdvert>> toCorner( size );
    std::vector<std::vector<CollapseVertOwn>> toVertOwn( size );
    std::vector<std::vector<CollapseEdgeInc>> toEdgeInc( size );
    std::vector<std::vector<CollapseVerdict>> toVerdict( size );

    auto posOf = [&]( GlobalId g, double out[3] )
    {
        auto it = gid2lv.find( g );
        if ( it == gid2lv.end() )
            Kokkos::abort(
                "Tessera::collapseEdges: a corner of an OWNED face is not held "
                "locally, so the collapse's geometric test cannot be "
                "evaluated. rebuildHalo() guarantees every vertex an owned face "
                "references is held WITH its position; a mesh reaching "
                "collapseEdges() without that has an invalid halo." );
        for ( int d = 0; d < 3; ++d )
            out[d] = 0.0;
        for ( int d = 0; d < D3; ++d )
            out[d] = static_cast<double>( v_pos( it->second, d ) );
    };

    for ( int f = 0; f < nOwnedF; ++f )
    {
        GlobalId v[3], eg[3];
        for ( int k = 0; k < 3; ++k )
        {
            v[k] = f_verts( f, k );
            eg[k] = f_edges( f, k );
        }
        // Per-edge advertisement, to the EDGE coordinators.
        for ( int k = 0; k < 3; ++k )
        {
            const GlobalId p0 = v[k];
            const GlobalId p1 = v[( k + 1 ) % 3];
            const GlobalId op = v[( k + 2 ) % 3];
            CollapseAdvert m;
            m.key = makeEdgeKey( p0, p1 );
            m.faceGid = f_gid( f );
            m.oppGid = op;
            m.edgeGid = eg[k];
            m.owner = static_cast<Rank>( R );
            m.forward = ( p0 == m.key.id[0] ) ? 1 : 0;
            // The face's other two edges are (p1, op) at slot (k+1)%3 and
            // (op, p0) at slot (k+2)%3; name them by which ENDPOINT of the
            // canonical key they meet, so the coordinator can address them
            // without knowing this face's rotation.
            if ( m.forward )
            {
                m.edgeOppLo = eg[( k + 2 ) % 3]; // (op, lo)
                m.edgeOppHi = eg[( k + 1 ) % 3]; // (hi, op)
            }
            else
            {
                m.edgeOppLo = eg[( k + 1 ) % 3]; // (lo, op)
                m.edgeOppHi = eg[( k + 2 ) % 3]; // (op, hi)
            }
            posOf( m.key.id[0], m.posLo );
            posOf( m.key.id[1], m.posHi );
            posOf( op, m.posOpp );
            toEdge[edgeCoordRank( m.key, size )].push_back( m );
        }
        // Per-corner advertisement, to the VERTEX coordinators.
        CollapseCornerAdvert c;
        c.faceGid = f_gid( f );
        c.owner = static_cast<Rank>( R );
        for ( int k = 0; k < 3; ++k )
        {
            c.cv[k] = v[k];
            c.ce[k] = eg[k];
            posOf( v[k], c.p[k] );
        }
        for ( int k = 0; k < 3; ++k )
        {
            c.v = v[k];
            toCorner[vertexCoordRank( v[k], size )].push_back( c );
        }
    }
    for ( int i = 0; i < nOwnedV; ++i )
        toVertOwn[vertexCoordRank( v_gid( i ), size )].push_back(
            { v_gid( i ), static_cast<Rank>( R ) } );
    for ( int e = 0; e < nOwnedE; ++e )
    {
        CollapseEdgeInc m;
        m.edgeGid = e_gid( e );
        m.ev[0] = e_verts( e, 0 );
        m.ev[1] = e_verts( e, 1 );
        m.owner = static_cast<Rank>( R );
        m.level = e_lev( e );
        for ( int j = 0; j < 2; ++j )
        {
            m.v = m.ev[j];
            toEdgeInc[vertexCoordRank( m.v, size )].push_back( m );
        }
    }
    for ( const auto& km : myMarked )
        toVerdict[edgeCoordRank( km.first, size )].push_back(
            { km.first, km.second, static_cast<Rank>( R ) } );

    auto adverts = allToAllV( comm, toEdge );
    auto corners = allToAllV( comm, toCorner );
    auto vertOwns = allToAllV( comm, toVertOwn );
    auto edgeIncs = allToAllV( comm, toEdgeInc );
    auto verdicts = allToAllV( comm, toVerdict );

    // ---- vertex-coordinator state ------------------------------------------
    //
    // Every owned face incident on each of this rank's vertices, every owned
    // edge incident on it, and its owner. Ordered containers throughout so
    // every loop below is deterministic.
    std::map<GlobalId, std::vector<const CollapseCornerAdvert*>> vFaces;
    std::map<GlobalId, std::vector<const CollapseEdgeInc*>> vEdges;
    std::map<GlobalId, Rank> vOwner;
    for ( const auto& m : corners.data )
        vFaces[m.v].push_back( &m );
    for ( const auto& m : edgeIncs.data )
        vEdges[m.v].push_back( &m );
    for ( const auto& m : vertOwns.data )
        vOwner[m.gid] = m.owner;

    // ---- edge-coordinator state: the candidates ----------------------------
    std::map<EdgeKey, std::vector<const CollapseAdvert*>> incident;
    for ( const auto& m : adverts.data )
        incident[m.key].push_back( &m );
    std::map<EdgeKey, CollapseVerdict> marked;
    for ( const auto& m : verdicts.data )
        marked[m.key] = m; // an edge has ONE owner, so at most one verdict

    long long locBoundary = 0, locLink = 0, locNormal = 0, locQuality = 0,
              locConflict = 0;

    struct Cand
    {
        const CollapseAdvert* fwd = nullptr; // traverses key.id[0] -> key.id[1]
        const CollapseAdvert* bwd = nullptr;
        GlobalId lo = invalid_gid, hi = invalid_gid;
        GlobalId c = invalid_gid, d = invalid_gid;
        double mergedPos[3] = { 0.0, 0.0, 0.0 };
        double len2 = 0.0; // priority: ASCENDING (shortest first)
        // filled from the vertex coordinators' replies
        std::set<GlobalId> ringLo, ringHi;
        std::vector<CollapseIncFace> facesLo, facesHi;
        std::vector<CollapseIncEdge> edgesLo, edgesHi;
        Rank ownerLo = -1, ownerHi = -1;
        bool normalOk = true, qualityOk = true;
        int replies = 0;
        std::set<GlobalId> named;
        int votes = 0;
    };
    std::map<EdgeKey, Cand> cand;

    for ( const auto& kv : marked )
    {
        auto it = incident.find( kv.first );
        const std::size_t n = ( it == incident.end() ) ? 0 : it->second.size();
        if ( n != 2 )
        {
            ++locBoundary; // a boundary edge, or a non-manifold one
            continue;
        }
        const CollapseAdvert* a0 = it->second[0];
        const CollapseAdvert* a1 = it->second[1];
        if ( a0->forward == a1->forward )
            Kokkos::abort(
                "Tessera::collapseEdges: the two faces incident on an edge "
                "traverse it in the SAME direction, so the mesh is not "
                "consistently oriented and the merge has no well-defined "
                "orientation. Tessera's builders produce oriented manifolds; "
                "this means the mesh was assembled from an inconsistent soup." );
        Cand c;
        c.fwd = a0->forward ? a0 : a1;
        c.bwd = a0->forward ? a1 : a0;
        c.lo = kv.first.id[0];
        c.hi = kv.first.id[1];
        c.c = c.fwd->oppGid;
        c.d = c.bwd->oppGid;
        c.len2 = edgeLen2Canonical( c.lo, c.fwd->posLo, c.hi, c.fwd->posHi, 3 );
        // The merged position, from the CANONICAL endpoint order: `a` is the
        // lower gid, so t is oriented by gid on every rank alike.
        policy.interpolatePosition( c.mergedPos, c.fwd->posLo, c.fwd->posHi,
                                    policy.t, 3 );
        cand.emplace( kv.first, c );
    }

    // ---- Round 2: ask both endpoints' vertex coordinators -------------------
    {
        std::vector<std::vector<CollapseLinkQuery>> ask( size );
        for ( const auto& kv : cand )
        {
            CollapseLinkQuery q;
            q.cand = kv.first;
            for ( int d = 0; d < 3; ++d )
                q.mergedPos[d] = kv.second.mergedPos[d];
            q.which = kv.second.lo;
            q.other = kv.second.hi;
            ask[vertexCoordRank( q.which, size )].push_back( q );
            q.which = kv.second.hi;
            q.other = kv.second.lo;
            ask[vertexCoordRank( q.which, size )].push_back( q );
        }
        auto asked = allToAllV( comm, ask );

        std::vector<std::vector<CollapseRingMsg>> ringBack( size );
        std::vector<std::vector<CollapseIncFace>> faceBack( size );
        std::vector<std::vector<CollapseIncEdge>> edgeBack( size );
        std::vector<std::vector<CollapseGeom>> geomBack( size );
        for ( int s = 0; s < size; ++s )
        {
            const CollapseLinkQuery* p = asked.from( s );
            const int cnt = asked.count( s );
            for ( int i = 0; i < cnt; ++i )
            {
                const GlobalId v = p[i].which;
                const GlobalId w = p[i].other;
                CollapseGeom g;
                g.cand = p[i].cand;
                g.which = v;
                {
                    auto io = vOwner.find( v );
                    g.vOwner = ( io == vOwner.end() ) ? static_cast<Rank>( -1 )
                                                      : io->second;
                }
                g.normalOk = 1;
                g.qualityOk = 1;

                auto fit = vFaces.find( v );
                if ( fit != vFaces.end() )
                {
                    std::set<GlobalId> ring;
                    for ( const CollapseCornerAdvert* fa : fit->second )
                    {
                        bool hasW = false;
                        for ( int k = 0; k < 3; ++k )
                        {
                            if ( fa->cv[k] != v )
                                ring.insert( fa->cv[k] );
                            if ( fa->cv[k] == w )
                                hasW = true;
                        }
                        // A face containing BOTH endpoints is one of the two
                        // that die: it contributes to the ring (w itself is a
                        // ring vertex, and so are c and d) but not to the
                        // geometric test and not to the rewrite.
                        if ( hasW )
                            continue;

                        CollapseIncFace fm;
                        fm.cand = p[i].cand;
                        fm.which = v;
                        fm.faceGid = fa->faceGid;
                        fm.owner = fa->owner;
                        for ( int k = 0; k < 3; ++k )
                        {
                            fm.cv[k] = fa->cv[k];
                            fm.ce[k] = fa->ce[k];
                        }
                        faceBack[s].push_back( fm );

                        // The fold and quality tests, with this corner moved to
                        // the merged position.
                        double np[3][3];
                        for ( int k = 0; k < 3; ++k )
                            for ( int d = 0; d < 3; ++d )
                                np[k][d] = ( fa->cv[k] == v )
                                               ? p[i].mergedPos[d]
                                               : fa->p[k][d];
                        double nOld[3], nNew[3];
                        flipTriNormal( fa->p[0], fa->p[1], fa->p[2], nOld );
                        flipTriNormal( np[0], np[1], np[2], nNew );
                        const double rot = flipAngleBetween( nOld, nNew );
                        if ( rot < 0.0 || rot > policy.maxNormalRotation )
                            g.normalOk = 0;
                        if ( flipRadiusRatio( np[0], np[1], np[2] ) <
                             policy.minQuality )
                            g.qualityOk = 0;
                    }
                    for ( GlobalId r : ring )
                        ringBack[s].push_back( { p[i].cand, v, r } );
                }

                auto eit = vEdges.find( v );
                if ( eit != vEdges.end() )
                    for ( const CollapseEdgeInc* ea : eit->second )
                    {
                        CollapseIncEdge em;
                        em.cand = p[i].cand;
                        em.which = v;
                        em.edgeGid = ea->edgeGid;
                        em.owner = ea->owner;
                        em.ev[0] = ea->ev[0];
                        em.ev[1] = ea->ev[1];
                        em.level = ea->level;
                        edgeBack[s].push_back( em );
                    }

                geomBack[s].push_back( g );
            }
        }
        auto ringGot = allToAllV( comm, ringBack );
        auto faceGot = allToAllV( comm, faceBack );
        auto edgeGot = allToAllV( comm, edgeBack );
        auto geomGot = allToAllV( comm, geomBack );

        for ( const auto& m : ringGot.data )
        {
            auto it = cand.find( m.cand );
            if ( it == cand.end() )
                continue;
            if ( m.which == it->second.lo )
                it->second.ringLo.insert( m.ring );
            else
                it->second.ringHi.insert( m.ring );
        }
        for ( const auto& m : faceGot.data )
        {
            auto it = cand.find( m.cand );
            if ( it == cand.end() )
                continue;
            if ( m.which == it->second.lo )
                it->second.facesLo.push_back( m );
            else
                it->second.facesHi.push_back( m );
        }
        for ( const auto& m : edgeGot.data )
        {
            auto it = cand.find( m.cand );
            if ( it == cand.end() )
                continue;
            if ( m.which == it->second.lo )
                it->second.edgesLo.push_back( m );
            else
                it->second.edgesHi.push_back( m );
        }
        for ( const auto& m : geomGot.data )
        {
            auto it = cand.find( m.cand );
            if ( it == cand.end() )
                continue;
            ++it->second.replies;
            if ( m.which == it->second.lo )
                it->second.ownerLo = m.vOwner;
            else
                it->second.ownerHi = m.vOwner;
            if ( !m.normalOk )
                it->second.normalOk = false;
            if ( !m.qualityOk )
                it->second.qualityOk = false;
        }
    }

    // ---- the verdicts, in the order the counters name them ------------------
    for ( auto it = cand.begin(); it != cand.end(); )
    {
        Cand& c = it->second;
        bool reject = false;
        if ( c.replies != 2 || c.ownerLo < 0 || c.ownerHi < 0 )
            Kokkos::abort(
                "Tessera::collapseEdges: an endpoint's vertex coordinator did "
                "not answer, or the vertex has no owner. Every owned vertex "
                "advertises its owner and every owned face advertises its three "
                "corners, so a candidate edge whose endpoints are corners of two "
                "owned faces must receive exactly two replies." );

        // THE LINK CONDITION: link(lo) INTERSECT link(hi) == { c, d }.
        if ( !reject )
        {
            std::vector<GlobalId> common;
            std::set_intersection( c.ringLo.begin(), c.ringLo.end(),
                                   c.ringHi.begin(), c.ringHi.end(),
                                   std::back_inserter( common ) );
            std::set<GlobalId> want = { c.c, c.d };
            if ( common.size() != want.size() ||
                 !std::equal( common.begin(), common.end(), want.begin() ) )
            {
                ++locLink;
                reject = true;
            }
        }
        // ... and the fused-face test the vertex link misses (see the header).
        if ( !reject )
        {
            std::set<FaceKey> keys;
            bool dup = false;
            for ( const auto& f : c.facesLo )
                if ( !keys.insert( makeFaceKey( f.cv[0], f.cv[1], f.cv[2] ) )
                          .second )
                    dup = true;
            for ( const auto& f : c.facesHi )
            {
                GlobalId w[3];
                for ( int k = 0; k < 3; ++k )
                    w[k] = ( f.cv[k] == c.hi ) ? c.lo : f.cv[k];
                if ( !keys.insert( makeFaceKey( w[0], w[1], w[2] ) ).second )
                    dup = true;
            }
            if ( dup )
            {
                ++locLink;
                reject = true;
            }
        }
        if ( !reject && !c.normalOk )
        {
            ++locNormal;
            reject = true;
        }
        if ( !reject && !c.qualityOk )
        {
            ++locQuality;
            reject = true;
        }
        if ( reject )
            it = cand.erase( it );
        else
        {
            c.named = c.ringLo;
            c.named.insert( c.ringHi.begin(), c.ringHi.end() );
            c.named.insert( c.lo );
            c.named.insert( c.hi );
            ++it;
        }
    }

    // ---- Round 3: the independent set --------------------------------------
    //
    // Offer each surviving candidate to the coordinator of every vertex in its
    // two-ring; each such coordinator endorses only the best candidate that
    // named it; a candidate is accepted iff it collects one endorsement per
    // named vertex. The relation is symmetric, so of two candidates whose
    // two-rings touch at most one survives.
    {
        std::vector<std::vector<CollapseNbr>> offer( size );
        for ( const auto& kv : cand )
            for ( GlobalId v : kv.second.named )
                offer[vertexCoordRank( v, size )].push_back(
                    { v, kv.first, kv.second.len2 } );
        auto offers = allToAllV( comm, offer );

        // (squared length ASCENDING, EdgeKey ASCENDING) -- no gid anywhere.
        std::map<GlobalId, CollapseNbr> best;
        for ( const auto& m : offers.data )
        {
            auto it = best.find( m.v );
            if ( it == best.end() )
                best.emplace( m.v, m );
            else if ( m.len2 < it->second.len2 ||
                      ( m.len2 == it->second.len2 && m.cand < it->second.cand ) )
                it->second = m;
        }
        std::vector<std::vector<CollapseWin>> vote( size );
        for ( const auto& kv : best )
            vote[edgeCoordRank( kv.second.cand, size )].push_back(
                { kv.second.cand } );
        auto votes = allToAllV( comm, vote );
        for ( const auto& m : votes.data )
        {
            auto it = cand.find( m.cand );
            if ( it != cand.end() )
                ++it->second.votes;
        }
        for ( auto it = cand.begin(); it != cand.end(); )
        {
            if ( it->second.votes == static_cast<int>( it->second.named.size() ) )
                ++it;
            else
            {
                it = cand.erase( it );
                ++locConflict;
            }
        }
    }

    {
        const long long locAccepted = static_cast<long long>( cand.size() );
        long long loc[6] = { locAccepted, locBoundary, locLink, locNormal,
                             locQuality, locConflict };
        long long glob[6] = { 0, 0, 0, 0, 0, 0 };
        MPI_Allreduce( loc, glob, 6, MPI_LONG_LONG, MPI_SUM, comm );
        result.accepted = glob[0];
        result.rejectedBoundary = glob[1];
        result.rejectedLinkCondition = glob[2];
        result.rejectedNormalFlip = glob[3];
        result.rejectedQuality = glob[4];
        result.rejectedConflict = glob[5];
    }

    // NOTHING ACCEPTED IS A GENUINE NO-OP: no rewrite, no tombstone, and no
    // compaction, so the entity ordering, every gid and every key are exactly
    // what they were. Skipping the compaction here is not an optimization -- a
    // rebuild would be observable in the ghost layer's contents.
    if ( result.accepted == 0 )
        return result;

    // ---- Round 4: ship the rewrite to whoever has to write it ---------------
    std::vector<std::vector<CollapseTomb>> tombMsg( size );
    std::vector<std::vector<CollapseFaceFix>> faceMsg( size );
    std::vector<std::vector<CollapseEdgeFix>> edgeMsg( size );
    std::vector<std::vector<CollapseEdgeLevel>> levMsg( size );
    std::vector<std::vector<CollapseVertexFix>> vertMsg( size );
    std::vector<std::vector<CollapseNote>> noteMsg( size );

    for ( const auto& kv : cand )
    {
        const Cand& c = kv.second;
        std::set<Rank> touched;

        // The two coincident pairs: the edge on the DYING endpoint dies and the
        // edge on the SURVIVING endpoint takes its place. Naming them by which
        // endpoint of the canonical key they meet is what makes this readable
        // without knowing either face's rotation.
        const GlobalId fromE[2] = { c.fwd->edgeOppHi, c.bwd->edgeOppHi };
        const GlobalId toE[2] = { c.fwd->edgeOppLo, c.bwd->edgeOppLo };

        // (a) the two incident faces are tombstoned by their owners.
        tombMsg[c.fwd->owner].push_back( { c.fwd->faceGid } );
        tombMsg[c.bwd->owner].push_back( { c.bwd->faceGid } );
        touched.insert( c.fwd->owner );
        touched.insert( c.bwd->owner );

        // (b) every OTHER face incident on the dying endpoint is rewritten. The
        //     two dying faces are already excluded at the coordinator that
        //     supplied this list.
        for ( const auto& f : c.facesHi )
        {
            CollapseFaceFix m;
            m.faceGid = f.faceGid;
            m.fromV = c.hi;
            m.toV = c.lo;
            for ( int k = 0; k < 2; ++k )
            {
                m.fromE[k] = fromE[k];
                m.toE[k] = toE[k];
            }
            faceMsg[f.owner].push_back( m );
            touched.insert( f.owner );
        }

        // (c) every edge incident on the dying endpoint EXCEPT the three that
        //     die -- (lo,hi) and the two dying halves of the coincident pairs.
        for ( const auto& e : c.edgesHi )
        {
            const GlobalId far = ( e.ev[0] == c.hi ) ? e.ev[1] : e.ev[0];
            if ( far == c.lo || far == c.c || far == c.d )
                continue;
            edgeMsg[e.owner].push_back( { e.edgeGid, c.hi, c.lo } );
            touched.insert( e.owner );
        }

        // (d) the surviving edge of each coincident pair takes the MIN level of
        //     the pair (Decision 1: advisory thereafter).
        for ( int k = 0; k < 2; ++k )
        {
            Level lFrom = 0, lTo = 0;
            Rank ownTo = -1;
            bool haveFrom = false, haveTo = false;
            for ( const auto& e : c.edgesHi )
                if ( e.edgeGid == fromE[k] )
                {
                    lFrom = e.level;
                    haveFrom = true;
                }
            for ( const auto& e : c.edgesLo )
                if ( e.edgeGid == toE[k] )
                {
                    lTo = e.level;
                    ownTo = e.owner;
                    haveTo = true;
                }
            if ( haveFrom && haveTo && lFrom < lTo )
            {
                levMsg[ownTo].push_back( { toE[k], lFrom } );
                touched.insert( ownTo );
            }
        }

        // (e) the merged position, to the surviving vertex's owner.
        {
            CollapseVertexFix m;
            m.surviving = c.lo;
            m.dying = c.hi;
            for ( int d = 0; d < 3; ++d )
                m.pos[d] = c.mergedPos[d];
            vertMsg[c.ownerLo].push_back( m );
            touched.insert( c.ownerLo );
        }

        touched.insert( static_cast<Rank>( R ) ); // the deciding coordinator
        for ( Rank r : touched )
            noteMsg[r].push_back( { kv.first } );
    }

    auto tombGot = allToAllV( comm, tombMsg );
    auto faceFixGot = allToAllV( comm, faceMsg );
    auto edgeFixGot = allToAllV( comm, edgeMsg );
    auto levGot = allToAllV( comm, levMsg );
    auto vertGot = allToAllV( comm, vertMsg );
    auto noteGot = allToAllV( comm, noteMsg );

    // ---- Round 4b: apply, in place -----------------------------------------
    {
        // Faces: the connectivity rewrite FIRST, then the tombstones -- a face
        // that dies receives no fix, so the order only matters for reading.
        std::unordered_map<GlobalId, std::vector<const CollapseFaceFix*>>
            byFace;
        byFace.reserve( faceFixGot.data.size() * 2 + 1 );
        for ( const auto& m : faceFixGot.data )
            byFace[m.faceGid].push_back( &m );
        std::set<GlobalId> dead;
        for ( const auto& m : tombGot.data )
            dead.insert( m.faceGid );

        for ( int f = 0; f < nOwnedF; ++f )
        {
            auto it = byFace.find( f_gid( f ) );
            if ( it != byFace.end() )
                for ( const CollapseFaceFix* m : it->second )
                {
                    for ( int k = 0; k < 3; ++k )
                    {
                        if ( f_verts( f, k ) == m->fromV )
                            f_verts( f, k ) = m->toV;
                        for ( int j = 0; j < 2; ++j )
                            if ( f_edges( f, k ) == m->fromE[j] )
                                f_edges( f, k ) = m->toE[j];
                    }
                }
            if ( dead.count( f_gid( f ) ) )
                f_gid( f ) = invalid_gid; // tombstoneFace()'s convention
        }

        std::unordered_map<GlobalId, std::vector<const CollapseEdgeFix*>>
            byEdge;
        byEdge.reserve( edgeFixGot.data.size() * 2 + 1 );
        for ( const auto& m : edgeFixGot.data )
            byEdge[m.edgeGid].push_back( &m );
        std::map<GlobalId, Level> newLevel;
        for ( const auto& m : levGot.data )
        {
            auto it = newLevel.find( m.edgeGid );
            if ( it == newLevel.end() )
                newLevel.emplace( m.edgeGid, m.level );
            else
                it->second = std::min( it->second, m.level );
        }
        for ( int e = 0; e < nOwnedE; ++e )
        {
            auto it = byEdge.find( e_gid( e ) );
            if ( it != byEdge.end() )
                for ( const CollapseEdgeFix* m : it->second )
                    for ( int k = 0; k < 2; ++k )
                        if ( e_verts( e, k ) == m->fromV )
                            e_verts( e, k ) = m->toV;
            auto il = newLevel.find( e_gid( e ) );
            if ( il != newLevel.end() )
                e_lev( e ) = std::min( e_lev( e ), il->second );
        }

        // Vertices: the merged position, and every user field through the
        // policy hook. Both endpoints are read from THIS rank's copies -- the
        // dying one is in general a ghost, which is why the header states the
        // halo-consistency precondition on vertex data.
        for ( const auto& m : vertGot.data )
        {
            auto is = gid2lv.find( m.surviving );
            auto id = gid2lv.find( m.dying );
            if ( is == gid2lv.end() || id == gid2lv.end() )
                Kokkos::abort(
                    "Tessera::collapseEdges: the owner of a surviving vertex "
                    "does not hold both endpoints of the edge being collapsed. "
                    "The two are adjacent, so every halo of depth >= 1 holds "
                    "the far one; a mesh reaching this point without it has an "
                    "invalid halo." );
            for ( int d = 0; d < D3; ++d )
                v_pos( is->second, d ) =
                    static_cast<typename MeshT::scalar_type>( m.pos[d] );
            blendVertexUserFieldsT( hv, is->second, is->second, id->second,
                                    policy.t, policy );
        }

        Cabana::deep_copy( mesh.vertices(), hv );
        Cabana::deep_copy( mesh.edges(), he );
        Cabana::deep_copy( mesh.faces(), hf );
    }

    // DROP THE GHOST TUPLES BEFORE THE REBUILD -- correctness, not tidiness.
    // rebuildHalo() seeds its gid -> tuple map from every locally held entity,
    // owned AND ghost, and round G only fetches the ones that are MISSING, so a
    // stale ghost copy of a rewritten edge would hand round D endpoints the edge
    // no longer has -- and the CSR build indexes a dense gid -> local array, so
    // an endpoint this rank does not hold writes out of bounds and corrupts the
    // heap. That is the bug edge-flip hit as `double free or corruption (out)`
    // at np2. Here the ghost VERTICES go too, unlike flipEdges(): a collapse
    // MOVES the surviving vertex, so a ghost copy of it holds a stale position.
    mesh.resizeVertices( static_cast<std::size_t>( nOwnedV ) );
    mesh.resizeEdges( static_cast<std::size_t>( nOwnedE ) );
    mesh.resizeFaces( static_cast<std::size_t>( nOwnedF ) );
    mesh.setOwnedCounts( static_cast<std::size_t>( nOwnedV ),
                         static_cast<std::size_t>( nOwnedE ),
                         static_cast<std::size_t>( nOwnedF ) );

    // Remove the tombstoned faces and every vertex and edge no surviving face
    // references, restore the owned-first ordering, and rebuild both CSRs, both
    // key tables and the three halo plans. compact() ends with
    // rebuildHalo( mesh, halo, effectiveHaloDepth( halo ) ), so the depth is
    // preserved and a second rebuild here would be a redundant collective.
    // compactImpl() rather than compact() because the editing family is already
    // claimed by this call.
    const CompactStats st = compactImpl( mesh, halo );
    result.verticesRemoved = st.verticesRemoved;
    result.edgesRemoved = st.edgesRemoved;
    result.facesRemoved = st.facesRemoved;

    {
        std::set<EdgeKey> notes;
        for ( const auto& m : noteGot.data )
            notes.insert( m.key );
        result.collapsed.assign( notes.begin(), notes.end() );
    }

    return result;
}

} // namespace detail

//! Collapse exactly the marked edges that survive the tests.
//!
//! `edgeMask.size() == mesh.numOwnedEdges()`, indexed by owned edge local index
//! -- the same host `std::vector<char>` convention refine()'s face mask and
//! splitEdges()'/flipEdges()' edge masks use. The OWNER of an edge decides; the
//! decision is made at the edge's coordinator, which is a third rank in general,
//! with the link condition and the geometric tests answered by the two
//! endpoints' VERTEX coordinators.
//!
//! Each accepted collapse removes 1 vertex, 3 edges and 2 faces, so the Euler
//! number is preserved. The surviving vertex is `min(gid(a), gid(b))` and sits
//! at parameter `policy.t` along the edge measured from that endpoint.
//!
//! AT MOST A MAXIMAL INDEPENDENT SET IS APPLIED PER CALL -- two collapses whose
//! two-rings touch conflict and only the shorter one survives, so a caller
//! wanting more progress calls again and watches `accepted`:
//!
//!     while ( collapseEdges( mesh, halo, shortEdges( mesh ) ).accepted > 0 ) {}
//!
//! THE ACCEPTED SET IS NOT A SERIAL SHORTEST-FIRST PASS' SET -- it is a subset,
//! reached in fewer passes. Compare STATISTICS (face count, quality
//! distribution, edge-length histogram), never edit sets.
//!
//! REQUIRES halo depth >= 2 (the link condition needs both endpoints' full
//! one-rings) and throws std::runtime_error naming the required and the actual
//! depth if `0 < mesh.haloDepth() < 2`. Depth 0 means "never distributed" -- a
//! replicated mesh where every entity is local -- and is accepted.
//!
//! A BOUNDARY EDGE IS ALWAYS REJECTED (`rejectedBoundary`): a
//! boundary-preserving collapse is a separate feature.
//!
//! Belongs to the REMESH editing family, so collapsing a refine()d mesh throws
//! (Tessera_EditFamily.hpp). Collapse is UNDEFINED on a refine()d mesh in either
//! refinement mode, and un-refining one is explicitly out of scope.
//!
//! Vertex positions and vertex user fields must be HALO-CONSISTENT on entry:
//! the merged values are blended by the surviving vertex's owner from its local
//! copies of the two endpoints, one of which is in general a ghost. Call
//! haloExchange() after mutating them and before collapsing.
//!
//! Ends by calling compact(), so the mesh is compact and fully haloed at the
//! depth it was handed: a caller never sees a tombstone, a haloExchange() is
//! meaningful, and a second collapseEdges() may follow immediately.
//!
//! An EMPTY MASK, and a call in which every candidate is rejected, are both
//! GENUINE no-ops: no rewrite, no compaction, and every gid, key and local
//! ordering unchanged.
//!
//! Collective.
//!
//! INVALIDATION: the compaction reallocates the AoSoAs, key Views and CSRs and
//! replaces the halo plans, so every slice/CSR/key-View taken out before this
//! call is dangling. Re-slice from the mesh afterwards.
template <class MeshT, class Policy = DefaultCollapsePolicy>
CollapseResult collapseEdges( MeshT& mesh,
                              MeshHalo<typename MeshT::memory_space>& halo,
                              const std::vector<char>& edgeMask,
                              const Policy& policy = Policy{} )
{
    // THE DEPTH GUARD IS `0 <` AND NOT `>= 2`: depth 0 means "never
    // distributed", where every entity is local and no ring is missing, which is
    // exactly the replicated builder mesh a hand-built fixture uses.
    // buildVertexStencil() polices depth the same way.
    const int depth = mesh.haloDepth();
    if ( depth > 0 && depth < 2 )
        throw std::runtime_error(
            std::string( "Tessera::collapseEdges: requires halo depth >= 2 but "
                         "this mesh has depth " ) +
            std::to_string( depth ) +
            ". The link condition -- link(a) INTERSECT link(b) == {c,d}, the "
            "test that stops a collapse from welding two distant parts of the "
            "surface together -- is a statement about the FULL ONE-RING OF BOTH "
            "ENDPOINTS, i.e. a two-ring around the edge. Rebuild the halo at "
            "depth 2 (distribute(mesh, halo, faceOwner, 2) or rebuildHalo(mesh, "
            "halo, 2)) before collapsing. Depth 0 -- a replicated, "
            "never-distributed mesh -- is accepted, because there no ring is "
            "missing." );
    requireEditFamily( mesh, EditFamily::Remesh, "collapseEdges" );
    return detail::collapseEdgesImpl( mesh, halo, edgeMask, policy );
}

} // namespace Tessera

#endif // TESSERA_EDGE_COLLAPSE_HPP
