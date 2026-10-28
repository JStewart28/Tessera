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

// Regression test: face -> face adjacency through shared edges
// (tasks/completed/face-adjacency.md, src/Tessera_FaceAdjacency.hpp).
//
// GROUND TRUTH IS EXACT AND INDEPENDENT OF TESSERA for the headline case: the
// coarse icosphere soup is generated identically on every rank, face gid ==
// soup face index and vertex gid == soup vertex index, so the true face -> face
// gid map is derived from the replicated soup with a plain
// std::map<EdgeKey, vector<gid>> and compared. Nothing about the reference goes
// through the mesh, the halo, or the coordinator under test.
//
// Cases:
//   1. NEIGHBOUR GID SETS MATCH THE REFERENCE. Subdivision-2 icosphere,
//      distribute(). For every OWNED face the nbrGid row equals the reference
//      row for that face gid -- zero missing, zero extra, at every rank count.
//      This is the definitive check; everything else is a corollary or a
//      different regime.
//   2. DEGREE AND THE CLOSED-SURFACE IDENTITY. Every face has exactly 3
//      edge-neighbours (closed manifold, no boundary), and the sum of owned row
//      lengths == 3 * globalOwnedFaces == 2 * globalOwnedEdges (960 at subdiv 2).
//   3. SYMMETRY. For every owned face f and neighbour gid g, f's gid appears in
//      g's row. Checked against the reference, so it holds even when g is not
//      locally held.
//   4. RANK-COUNT INVARIANCE. Rows are gid-SORTED, so case 1's comparison is an
//      exact list comparison against a rank-count-independent reference; the
//      owned-row map gathered to one rank is therefore identical at ranks 1-5.
//      A checksum over the gathered map is printed so that is visible rather
//      than merely argued.
//   5. numNonResident IS REPORTED, NOT HIDDEN. Printed per rank. Asserted 0 at
//      rank 1. At ranks >= 2 it may be nonzero, and what is asserted is that
//      the flag never lies IN EITHER DIRECTION: every invalid_local entry's
//      nbrGid is genuinely absent from the local face gid set, and every
//      resident entry's local index really names a face with that nbrGid.
//   6. AFTER refine(), Conforming mode. A uniform round, then an ADAPTIVE round
//      so the transient closure layer is really populated. Every VISIBLE face
//      has 3 neighbours, checkConforming still passes, no RETIRED RED PARENT
//      appears anywhere in the adjacency, and the reference is derived from the
//      globally gathered ownedVisibleFaces().
//   7. AFTER refine(), HangingNode2to1 mode. Degree 3 is NOT asserted: under
//      that mode a T-junction is not a shared edge (the coarse side holds (a,b)
//      while the fine side holds (a,m) and (m,b)), so a face on either side of
//      one has degree < 3. That is the mode contract, not a defect. What is
//      asserted is symmetry (via exact equality with the gathered reference),
//      that the distinct edge set is consistent with globalOwnedEdges, that the
//      row total equals twice the number of two-incidence edges, and -- for
//      non-vacuity -- that T-junctions really exist in this workload.
//   8. NON-MANIFOLD INPUT FAILS LOUDLY. A hand-built TriangleSoup with three
//      faces on one edge, on MPI_COMM_SELF so the case is identical at every
//      rank count, must make buildFaceAdjacency() throw naming the EdgeKey.
//  10. SINGLE RANK. numNonResident == 0 and the CSR equals the reference
//      converted to local indices, exactly.
//
// Case 9 (the generation guard) is checked once per process rather than per
// backend -- see staleHandleCase(), which follows test_staleslice_guard.cpp's
// fork-and-check-the-child-died idiom.
//
// Runs on host (Serial) and device (default, HIP), ranks 1-5.

#include "MeshInvariants.hpp"

#include <Tessera.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace Tessera;

// ---------------------------------------------------------------------------
// Reference machinery
// ---------------------------------------------------------------------------

//! One face as the reference needs it: its gid and its three corner vertex gids.
struct FaceRec
{
    GlobalId gid;
    GlobalId v[3];
};

//! Reference face -> face gid map from an edge-keyed grouping of a face list.
//! Rows come out ascending by gid because std::set is ordered, which is exactly
//! the order buildFaceAdjacency() promises. Also reports the incidence
//! histogram: `nOne` edges with a single incident face (a boundary edge, or the
//! coarse side of a hanging node), `nTwo` with two, and aborts on more.
inline std::map<GlobalId, std::vector<GlobalId>>
referenceOf( const std::vector<FaceRec>& faces, int& nOne, int& nTwo )
{
    std::map<EdgeKey, std::vector<GlobalId>> facesOfEdge;
    for ( const FaceRec& f : faces )
        for ( int k = 0; k < 3; ++k )
            facesOfEdge[makeEdgeKey( f.v[k], f.v[( k + 1 ) % 3] )].push_back(
                f.gid );

    std::map<GlobalId, std::set<GlobalId>> nbr;
    for ( const FaceRec& f : faces )
        nbr[f.gid]; // every face gets a row, even a degree-0 one

    nOne = 0;
    nTwo = 0;
    for ( auto& kv : facesOfEdge )
    {
        std::vector<GlobalId>& inc = kv.second;
        std::sort( inc.begin(), inc.end() );
        inc.erase( std::unique( inc.begin(), inc.end() ), inc.end() );
        if ( inc.size() == 1 )
            ++nOne;
        else if ( inc.size() == 2 )
        {
            ++nTwo;
            nbr[inc[0]].insert( inc[1] );
            nbr[inc[1]].insert( inc[0] );
        }
        else
        {
            std::fprintf(
                stderr,
                "test_face_adjacency: the REFERENCE itself is "
                "non-manifold (edge {%llu,%llu} has %zu incident "
                "faces) -- the test input is wrong, not the library\n",
                (unsigned long long)kv.first.id[0],
                (unsigned long long)kv.first.id[1], inc.size() );
            std::abort();
        }
    }

    std::map<GlobalId, std::vector<GlobalId>> out;
    for ( const auto& kv : nbr )
        out[kv.first].assign( kv.second.begin(), kv.second.end() );
    return out;
}

//! Reference from a replicated triangle soup: gid == index for both faces and
//! vertices, straight out of buildFromTriangleSoup(). Independent of Tessera's
//! mesh, halo and coordinator.
template <class Scalar>
std::map<GlobalId, std::vector<GlobalId>>
soupReference( const TriangleSoup<Scalar>& soup, int& nOne, int& nTwo )
{
    std::vector<FaceRec> faces( soup.numFaces() );
    for ( std::size_t f = 0; f < soup.numFaces(); ++f )
    {
        faces[f].gid = static_cast<GlobalId>( f );
        for ( int k = 0; k < 3; ++k )
            faces[f].v[k] = static_cast<GlobalId>( soup.triangles[3 * f + k] );
    }
    return referenceOf( faces, nOne, nTwo );
}

//! This rank's OWNED faces as FaceRecs (the VISIBLE faces, in Conforming mode:
//! a Conforming mesh's face AoSoA stores only those).
template <class MeshT>
std::vector<FaceRec> ownedFaceRecs( MeshT& mesh )
{
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto g = Cabana::slice<FaceField::Gid>( hf );
    auto v = Cabana::slice<FaceField::Verts>( hf );
    std::vector<FaceRec> out( mesh.numOwnedFaces() );
    for ( std::size_t f = 0; f < out.size(); ++f )
    {
        out[f].gid = g( f );
        for ( int k = 0; k < 3; ++k )
            out[f].v[k] = v( f, k );
    }
    return out;
}

//! Gather every rank's owned faces onto every rank, so each can build the full
//! global reference itself. Owned faces partition the global mesh, so the union
//! is the global face list with no duplicates.
inline std::vector<FaceRec>
allgatherFaceRecs( MPI_Comm comm, const std::vector<FaceRec>& mine )
{
    int size = 1;
    MPI_Comm_size( comm, &size );
    const int myBytes = static_cast<int>( mine.size() * sizeof( FaceRec ) );
    std::vector<int> counts( size, 0 );
    MPI_Allgather( &myBytes, 1, MPI_INT, counts.data(), 1, MPI_INT, comm );
    std::vector<int> displs( size + 1, 0 );
    for ( int r = 0; r < size; ++r )
        displs[r + 1] = displs[r] + counts[r];
    std::vector<FaceRec> all( displs[size] / sizeof( FaceRec ) );
    MPI_Allgatherv( mine.data(), myBytes, MPI_BYTE, all.data(), counts.data(),
                    displs.data(), MPI_BYTE, comm );
    return all;
}

//! Host-side unpacked view of a FaceAdjacency: one row per local face, with the
//! three parallel arrays kept parallel.
struct HostRows
{
    std::vector<std::vector<GlobalId>> gid;
    std::vector<std::vector<Rank>> owner;
    std::vector<std::vector<LocalIndex>> local;
    long long totalOwnedEntries = 0;
};

template <class MemSpace>
HostRows readRows( const FaceAdjacency<MemSpace>& adj, int nLocalF,
                   int nOwnedF )
{
    const auto& csr = adj.csr.get();
    auto off =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), csr.offsets );
    auto nbr = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                    csr.neighbors );
    auto ng =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), adj.nbrGid );
    auto no = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                   adj.nbrOwner );
    HostRows h;
    h.gid.resize( nLocalF );
    h.owner.resize( nLocalF );
    h.local.resize( nLocalF );
    for ( int f = 0; f < nLocalF; ++f )
        for ( int p = off( f ); p < off( f + 1 ); ++p )
        {
            h.gid[f].push_back( ng( p ) );
            h.owner[f].push_back( no( p ) );
            h.local[f].push_back( nbr( p ) );
            if ( f < nOwnedF )
                ++h.totalOwnedEntries;
        }
    return h;
}

//! Gid of every LOCAL face (owned + ghost), and the owned gid list.
template <class MeshT>
void localFaceGids( MeshT& mesh, std::vector<GlobalId>& all,
                    std::vector<GlobalId>& owned )
{
    Cabana::AoSoA<typename MeshT::face_member_types, Kokkos::HostSpace> hf(
        "hf", mesh.numFaces() );
    Cabana::deep_copy( hf, mesh.faces() );
    auto g = Cabana::slice<FaceField::Gid>( hf );
    all.resize( mesh.numFaces() );
    for ( std::size_t f = 0; f < all.size(); ++f )
        all[f] = g( f );
    owned.assign( all.begin(), all.begin() + mesh.numOwnedFaces() );
}

// ---------------------------------------------------------------------------
// The shared owned-row assertions: exact equality with the reference, plus the
// resident/non-resident consistency of case 5. Returns local fails.
// ---------------------------------------------------------------------------
template <class MeshT, class MemSpace>
int checkOwnedRows( MeshT& mesh, const FaceAdjacency<MemSpace>& adj,
                    const std::map<GlobalId, std::vector<GlobalId>>& ref,
                    const char* what, bool requireDegree3 )
{
    const int nLocalF = static_cast<int>( mesh.numFaces() );
    const int nOwnedF = static_cast<int>( mesh.numOwnedFaces() );
    int fails = 0;

    HostRows h = readRows( adj, nLocalF, nOwnedF );
    std::vector<GlobalId> allGids, ownedGids;
    localFaceGids( mesh, allGids, ownedGids );
    std::map<GlobalId, int> localOf;
    for ( int f = 0; f < nLocalF; ++f )
        localOf[allGids[f]] = f;

    long long nonResident = 0;
    for ( int f = 0; f < nOwnedF; ++f )
    {
        const GlobalId fg = ownedGids[f];
        auto it = ref.find( fg );
        if ( it == ref.end() )
        {
            std::fprintf( stderr,
                          "  FAIL [%s]: owned face gid %llu has no reference "
                          "row -- it is not part of the reference face set\n",
                          what, (unsigned long long)fg );
            ++fails;
            continue;
        }

        // Case 1 / 3 / 4: exact, ORDERED list equality with the gid-sorted
        // reference. Ordered rather than set equality is what makes the row a
        // rank-count invariant.
        if ( h.gid[f] != it->second )
        {
            std::fprintf( stderr,
                          "  FAIL [%s]: owned face gid %llu row mismatch "
                          "(got %zu entries, reference has %zu)\n",
                          what, (unsigned long long)fg, h.gid[f].size(),
                          it->second.size() );
            ++fails;
        }

        // Case 2: degree on a closed conforming surface.
        if ( requireDegree3 && h.gid[f].size() != 3 )
        {
            std::fprintf( stderr,
                          "  FAIL [%s]: owned face gid %llu has degree %zu, "
                          "expected 3 on a closed conforming surface\n",
                          what, (unsigned long long)fg, h.gid[f].size() );
            ++fails;
        }

        // Case 3, stated directly against the reference so it holds for a
        // neighbour this rank does not hold: f appears in its neighbour's row.
        for ( GlobalId g : h.gid[f] )
        {
            auto rg = ref.find( g );
            if ( rg == ref.end() ||
                 !std::binary_search( rg->second.begin(), rg->second.end(),
                                      fg ) )
            {
                std::fprintf( stderr,
                              "  FAIL [%s]: adjacency is not symmetric -- %llu "
                              "lists %llu but not the reverse\n",
                              what, (unsigned long long)fg,
                              (unsigned long long)g );
                ++fails;
            }
        }

        // Case 5: the flag never lies in either direction.
        for ( std::size_t p = 0; p < h.gid[f].size(); ++p )
        {
            const LocalIndex li = h.local[f][p];
            const GlobalId g = h.gid[f][p];
            const bool held = localOf.count( g ) > 0;
            if ( li == invalid_local )
            {
                ++nonResident;
                if ( held )
                {
                    std::fprintf(
                        stderr,
                        "  FAIL [%s]: entry for neighbour %llu is "
                        "invalid_local but the face IS held locally\n",
                        what, (unsigned long long)g );
                    ++fails;
                }
            }
            else if ( li < 0 || li >= nLocalF || allGids[li] != g )
            {
                std::fprintf( stderr,
                              "  FAIL [%s]: entry for neighbour %llu resolves "
                              "to local index %d, which is not that face\n",
                              what, (unsigned long long)g, li );
                ++fails;
            }
        }

        // Rows are also self-consistent in the owner half.
        for ( std::size_t p = 0; p < h.gid[f].size(); ++p )
            if ( h.owner[f][p] < 0 || h.owner[f][p] >= mesh.commSize() )
            {
                std::fprintf( stderr,
                              "  FAIL [%s]: neighbour %llu carries owner rank "
                              "%d, outside [0,%d)\n",
                              what, (unsigned long long)h.gid[f][p],
                              (int)h.owner[f][p], mesh.commSize() );
                ++fails;
            }
    }

    if ( nonResident != adj.numNonResident )
    {
        std::fprintf( stderr,
                      "  FAIL [%s]: numNonResident reports %lld but %lld "
                      "owned-row entries are invalid_local\n",
                      what, adj.numNonResident, nonResident );
        ++fails;
    }
    if ( mesh.commSize() == 1 && adj.numNonResident != 0 )
    {
        std::fprintf( stderr,
                      "  FAIL [%s]: numNonResident is %lld at a single rank, "
                      "where every face is resident\n",
                      what, adj.numNonResident );
        ++fails;
    }

    return fails;
}

//! Sum of owned row lengths across all ranks.
template <class MemSpace, class MeshT>
long long globalRowEntries( MeshT& mesh, const FaceAdjacency<MemSpace>& adj )
{
    HostRows h = readRows( adj, static_cast<int>( mesh.numFaces() ),
                           static_cast<int>( mesh.numOwnedFaces() ) );
    long long g = 0;
    MPI_Allreduce( &h.totalOwnedEntries, &g, 1, MPI_LONG_LONG, MPI_SUM,
                   mesh.comm() );
    return g;
}

// ---------------------------------------------------------------------------
// Case 9: the generation guard (once per process, host-only mesh).
// ---------------------------------------------------------------------------

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

//! Build the adjacency, refine(), then copy the stale CSR handle: it must abort.
//! On MPI_COMM_SELF so it behaves identically at every rank count, and on a
//! Serial/HostSpace mesh because the hazard under test is host-side generation
//! bookkeeping.
inline int staleHandleCase()
{
    using MeshT = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                       Kokkos::HostSpace, Kokkos::Serial>;
    int fails = 0;

    MeshT mesh( MPI_COMM_SELF );
    buildIcosphere( mesh, 1 );
    MeshHalo<Kokkos::HostSpace> halo;
    auto faceOwner = facePartitionByAxis( mesh );
    distribute( mesh, halo, faceOwner );

    auto adj = buildFaceAdjacency( mesh );
    const std::size_t gen_before = mesh.generation();

    std::vector<char> mask( mesh.numOwnedFaces(), 0 );
    for ( std::size_t f = 0; f < mask.size(); ++f )
        mask[f] = ( f % 3 == 0 ) ? 1 : 0;
    refine( mesh, halo, mask );

    if ( mesh.generation() == gen_before )
    {
        std::fprintf( stderr,
                      "  FAIL [guard]: refine() did not bump generation()\n" );
        ++fails;
    }
    if ( runInChild(
             [&]()
             {
                 auto c = adj.csr; // copy ctor validates -> must abort
                 (void)c;
             } ) )
    {
        std::fprintf( stderr,
                      "  FAIL [guard]: a FaceAdjacency CSR handle held across "
                      "refine() did not trip the generation guard\n" );
        ++fails;
    }
    return fails;
}

// ---------------------------------------------------------------------------
// Case 8: non-manifold input fails loudly.
// ---------------------------------------------------------------------------
inline int nonManifoldCase()
{
    using MeshT = Mesh<double, 3, VertexFields<>, EdgeFields<>, FaceFields<>,
                       Kokkos::HostSpace, Kokkos::Serial>;
    int fails = 0;

    // Three triangles fanned around the single edge (0,1). Hand-built, not a
    // mutated icosphere: the point is that the INPUT is non-manifold, and a
    // mutation would leave that entangled with whatever else changed.
    TriangleSoup<double> soup;
    soup.positions = { 0.0, 0.0,  0.0, // 0
                       1.0, 0.0,  0.0, // 1
                       0.0, 1.0,  0.0, // 2
                       0.0, -1.0, 0.0, // 3
                       0.0, 0.0,  1.0 };
    soup.triangles = { 0, 1, 2, 0, 1, 3, 0, 1, 4 };

    // MPI_COMM_SELF: every rank runs the identical single-rank case, so the
    // result does not depend on the rank count.
    MeshT mesh( MPI_COMM_SELF );
    buildFromTriangleSoup( mesh, soup );

    bool threw = false;
    std::string msg;
    try
    {
        auto adj = buildFaceAdjacency( mesh );
        (void)adj;
    }
    catch ( const std::runtime_error& e )
    {
        threw = true;
        msg = e.what();
    }

    if ( !threw )
    {
        std::fprintf( stderr, "  FAIL [nonmanifold]: buildFaceAdjacency() "
                              "accepted a non-manifold input\n" );
        ++fails;
    }
    else if ( msg.find( "NON-MANIFOLD" ) == std::string::npos ||
              msg.find( "{0, 1}" ) == std::string::npos ||
              msg.find( " 3 distinct" ) == std::string::npos )
    {
        std::fprintf( stderr,
                      "  FAIL [nonmanifold]: the throw does not name the "
                      "offending EdgeKey {0, 1} and its 3 incidences: \"%s\"\n",
                      msg.c_str() );
        ++fails;
    }
    return fails;
}

// ---------------------------------------------------------------------------
// The per-backend body.
// ---------------------------------------------------------------------------
template <class Scalar, class Exec>
int run( const char* tag )
{
    using mem = typename Exec::memory_space;
    using ConfMeshT = Mesh<Scalar, 3, VertexFields<>, EdgeFields<>,
                           FaceFields<>, mem, Exec, RefinementMode::Conforming>;
    using HangMeshT =
        Mesh<Scalar, 3, VertexFields<>, EdgeFields<>, FaceFields<>, mem, Exec,
             RefinementMode::HangingNode2to1>;

    int rank = 0, size = 1;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    int fails = 0;

    // ---- Cases 1-5, 10: the coarse distributed icosphere -------------------
    {
        ConfMeshT mesh( MPI_COMM_WORLD );
        buildIcosphere( mesh, 2 );
        MeshHalo<mem> halo;
        auto faceOwner = facePartitionByAxis( mesh, 2 );
        distribute( mesh, halo, faceOwner, 1 );
        haloExchange( mesh, halo );

        // Reference: from the replicated soup, not from the mesh.
        int nOne = 0, nTwo = 0;
        const auto soup = generateIcosphere<Scalar>( 2 );
        const auto ref = soupReference( soup, nOne, nTwo );
        if ( nOne != 0 )
        {
            std::fprintf( stderr,
                          "  FAIL [%s/coarse]: reference has %d boundary edges "
                          "on a closed surface\n",
                          tag, nOne );
            ++fails;
        }

        auto adj = buildFaceAdjacency( mesh );
        fails += checkOwnedRows( mesh, adj, ref, "coarse", /*deg3=*/true );

        // Case 2: the closed-surface identity, globally.
        const long long entries = globalRowEntries( mesh, adj );
        const long long F = globalOwnedFaces( mesh );
        const long long E = globalOwnedEdges( mesh );
        if ( entries != 3 * F || entries != 2 * E )
        {
            std::fprintf( stderr,
                          "  FAIL [%s/coarse]: row total %lld != 3*F %lld != "
                          "2*E %lld\n",
                          tag, entries, 3 * F, 2 * E );
            ++fails;
        }

        // Case 4: a rank-count-invariant fingerprint of the whole owned-row
        // map, printed so the invariance is visible and not merely argued.
        // (Exact invariance is already pinned by case 1's ordered comparison
        // against the rank-count-independent reference.)
        HostRows h = readRows( adj, static_cast<int>( mesh.numFaces() ),
                               static_cast<int>( mesh.numOwnedFaces() ) );
        std::vector<GlobalId> allGids, ownedGids;
        localFaceGids( mesh, allGids, ownedGids );
        unsigned long long sum = 0;
        for ( std::size_t f = 0; f < ownedGids.size(); ++f )
            for ( std::size_t p = 0; p < h.gid[f].size(); ++p )
                sum +=
                    ( ownedGids[f] * 1000003ULL + h.gid[f][p] ) * ( p + 1ULL );
        unsigned long long gsum = 0;
        MPI_Allreduce( &sum, &gsum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                       MPI_COMM_WORLD );

        // Case 5 / 10: report, and at a single rank compare the CSR against the
        // reference converted to local indices, exactly.
        long long gNonRes = 0;
        MPI_Allreduce( &adj.numNonResident, &gNonRes, 1, MPI_LONG_LONG, MPI_SUM,
                       MPI_COMM_WORLD );
        std::printf( "  [%s] coarse: rank %d numNonResident=%lld "
                     "(global %lld), rowsum=%lld, F=%lld E=%lld, "
                     "checksum=%llu\n",
                     tag, rank, adj.numNonResident, gNonRes, entries, F, E,
                     gsum );

        if ( size == 1 )
        {
            std::map<GlobalId, int> localOf;
            for ( std::size_t f = 0; f < allGids.size(); ++f )
                localOf[allGids[f]] = static_cast<int>( f );
            for ( std::size_t f = 0; f < ownedGids.size(); ++f )
            {
                const auto& r = ref.at( ownedGids[f] );
                std::vector<LocalIndex> want;
                for ( GlobalId g : r )
                    want.push_back( localOf.at( g ) );
                if ( h.local[f] != want )
                {
                    std::fprintf( stderr,
                                  "  FAIL [%s/np1]: CSR row of face gid %llu "
                                  "does not equal the reference in local "
                                  "indices\n",
                                  tag, (unsigned long long)ownedGids[f] );
                    ++fails;
                }
            }
        }
    }

    // ---- Case 6: after refine(), Conforming mode ---------------------------
    {
        ConfMeshT mesh( MPI_COMM_WORLD );
        buildIcosphere( mesh, 2 );
        MeshHalo<mem> halo;
        auto faceOwner = facePartitionByAxis( mesh, 2 );
        distribute( mesh, halo, faceOwner, 1 );
        haloExchange( mesh, halo );

        for ( int round = 0; round < 2; ++round )
        {
            // Round 0 uniform (the closure is inert), round 1 ADAPTIVE so the
            // closure layer is really populated and retired parents exist.
            std::vector<char> mask( mesh.numOwnedFaces(), 1 );
            if ( round == 1 )
            {
                auto fg = mesh.template faceSlice<FaceField::Gid>();
                for ( std::size_t f = 0; f < mask.size(); ++f )
                    mask[f] = ( fg( f ) % 3 == 0 ) ? 1 : 0;
            }
            refine( mesh, halo, mask );

            fails += TesseraTest::checkConforming( mesh );

            const auto mine = ownedFaceRecs( mesh );
            const auto all = allgatherFaceRecs( MPI_COMM_WORLD, mine );
            int nOne = 0, nTwo = 0;
            const auto ref = referenceOf( all, nOne, nTwo );
            if ( nOne != 0 )
            {
                std::fprintf( stderr,
                              "  FAIL [%s/conf r%d]: %d single-incidence edges "
                              "on a conforming mesh\n",
                              tag, round, nOne );
                ++fails;
            }

            auto adj = buildFaceAdjacency( mesh );
            fails += checkOwnedRows( mesh, adj, ref, "conforming",
                                     /*deg3=*/true );

            // No RETIRED RED PARENT may appear anywhere in the adjacency: the
            // face AoSoA of a Conforming mesh holds only the visible layer, so
            // the retired parents unclose() would restore must be absent both as
            // row owners (structurally true) and as neighbour gids.
            const auto visible = TesseraTest::ownedVisibleFaces( mesh );
            const auto un = Tessera::unclose( visible );
            std::set<GlobalId> visibleGids;
            int nClosureChildren = 0;
            for ( const auto& vf : visible )
            {
                visibleGids.insert( vf.gid );
                if ( vf.parent != invalid_gid )
                    ++nClosureChildren;
            }
            std::set<GlobalId> retired;
            for ( const auto& rf : un.red )
                if ( !visibleGids.count( rf.gid ) )
                    retired.insert( rf.gid );

            HostRows h = readRows( adj, static_cast<int>( mesh.numFaces() ),
                                   static_cast<int>( mesh.numOwnedFaces() ) );
            std::vector<GlobalId> allGids, ownedGids;
            localFaceGids( mesh, allGids, ownedGids );
            for ( std::size_t f = 0; f < ownedGids.size(); ++f )
            {
                if ( !visibleGids.count( ownedGids[f] ) )
                {
                    std::fprintf( stderr,
                                  "  FAIL [%s/conf r%d]: face gid %llu has a "
                                  "row but is not a visible face\n",
                                  tag, round,
                                  (unsigned long long)ownedGids[f] );
                    ++fails;
                }
                for ( GlobalId g : h.gid[f] )
                    if ( retired.count( g ) )
                    {
                        std::fprintf(
                            stderr,
                            "  FAIL [%s/conf r%d]: retired red parent "
                            "%llu appears as a neighbour\n",
                            tag, round, (unsigned long long)g );
                        ++fails;
                    }
            }

            int gClosure = 0;
            MPI_Allreduce( &nClosureChildren, &gClosure, 1, MPI_INT, MPI_SUM,
                           MPI_COMM_WORLD );
            std::printf( "  [%s] conforming r%d: rank %d visible=%zu "
                         "closureChildren=%d (global %d) retiredParents=%zu "
                         "numNonResident=%lld\n",
                         tag, round, rank, visible.size(), nClosureChildren,
                         gClosure, retired.size(), adj.numNonResident );

            // NON-VACUITY: the adaptive round must actually produce a closure
            // layer, otherwise the retired-parent assertion proves nothing.
            if ( round == 1 && gClosure == 0 )
            {
                std::fprintf( stderr,
                              "  FAIL [%s/conf r1]: no closure children were "
                              "emitted -- the case is vacuous\n",
                              tag );
                ++fails;
            }
        }
    }

    // ---- Case 7: after refine(), HangingNode2to1 mode ----------------------
    {
        HangMeshT mesh( MPI_COMM_WORLD );
        buildIcosphere( mesh, 2 );
        MeshHalo<mem> halo;
        auto faceOwner = facePartitionByAxis( mesh, 2 );
        distribute( mesh, halo, faceOwner, 1 );
        haloExchange( mesh, halo );

        {
            auto fg = mesh.template faceSlice<FaceField::Gid>();
            std::vector<char> mask( mesh.numOwnedFaces(), 0 );
            for ( std::size_t f = 0; f < mask.size(); ++f )
                mask[f] = ( fg( f ) % 3 == 0 ) ? 1 : 0;
            refine( mesh, halo, mask );
        }

        const auto mine = ownedFaceRecs( mesh );
        const auto all = allgatherFaceRecs( MPI_COMM_WORLD, mine );
        int nOne = 0, nTwo = 0;
        const auto ref = referenceOf( all, nOne, nTwo );

        auto adj = buildFaceAdjacency( mesh );
        // Degree 3 deliberately NOT required: see the header note on T-junctions.
        fails += checkOwnedRows( mesh, adj, ref, "hanging", /*deg3=*/false );

        // Edge consistency: the distinct edge set of the global face list is
        // exactly the global owned edge set, and every two-incidence edge
        // contributes two row entries.
        const long long entries = globalRowEntries( mesh, adj );
        const long long E = globalOwnedEdges( mesh );
        if ( static_cast<long long>( nOne + nTwo ) != E )
        {
            std::fprintf( stderr,
                          "  FAIL [%s/hanging]: %d distinct edges in the face "
                          "list but globalOwnedEdges is %lld\n",
                          tag, nOne + nTwo, E );
            ++fails;
        }
        if ( entries != 2LL * nTwo )
        {
            std::fprintf( stderr,
                          "  FAIL [%s/hanging]: row total %lld != 2 * "
                          "two-incidence edges %d\n",
                          tag, entries, 2 * nTwo );
            ++fails;
        }
        // NON-VACUITY: this workload must really contain T-junctions, otherwise
        // the mode split being tested is not exercised.
        if ( nOne == 0 )
        {
            std::fprintf( stderr,
                          "  FAIL [%s/hanging]: no hanging nodes were created "
                          "-- the case is vacuous\n",
                          tag );
            ++fails;
        }
        if ( rank == 0 )
            std::printf( "  [%s] hanging: edges=%lld twoIncidence=%d "
                         "oneIncidence(T-junction)=%d rowsum=%lld\n",
                         tag, E, nTwo, nOne, entries );
    }

    std::printf( "  [%s] rank %d %s\n", tag, rank, fails == 0 ? "ok" : "FAIL" );
    return fails;
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    int fails = 0;
    {
        int rank = 0;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        if ( rank == 0 )
            std::printf( "test_face_adjacency: face->face adjacency through "
                         "shared edges\n" );

        fails += run<double, Kokkos::Serial>( "Serial" );
        if ( !std::is_same<Kokkos::DefaultExecutionSpace,
                           Kokkos::Serial>::value )
            fails += run<double, Kokkos::DefaultExecutionSpace>( "Default" );

        // Cases 8 and 9 are single-rank by construction (MPI_COMM_SELF) and
        // independent of the execution space, so they run once per process.
        fails += nonManifoldCase();
        fails += staleHandleCase();
    }

    Kokkos::finalize();

    int global_fails = 0;
    MPI_Allreduce( &fails, &global_fails, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    MPI_Finalize();
    return global_fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
