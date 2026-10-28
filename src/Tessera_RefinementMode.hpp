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

#ifndef TESSERA_REFINEMENT_MODE_HPP
#define TESSERA_REFINEMENT_MODE_HPP

namespace Tessera
{

// ============================================================================
// RefinementMode — which conformity contract a Mesh's refinement obeys
// ============================================================================
//
// This is a COMPILE-TIME Mesh template parameter (appended last, so every
// existing Mesh<...> spelling stays source-compatible), not a runtime flag: the
// closure bookkeeping face members exist in the face AoSoA only in `Conforming`
// mode, so a hanging-node mesh pays zero extra memory. refine() / refineLocal()
// dispatch on `MeshT::refinement_mode` with `if constexpr`.
//
//   HangingNode2to1  Red (1->4) refinement with hanging nodes (T-junctions)
//                    permitted but bounded to a single refinement-level jump
//                    across any edge. A partial refine mask therefore leaves a
//                    non-conforming mesh: a bisected edge on a kept face has one
//                    incident face on one side and two on the other, and the
//                    owned-only Euler number V - E + F equals 2 only for a
//                    uniform refine.
//
//   Conforming       The same 2:1-balanced red layer, plus a TRANSIENT
//                    red-green-blue closure pass that retriangulates every kept
//                    face carrying hanging nodes. The visible mesh has no
//                    T-junctions (every edge has exactly two incident faces) for
//                    an arbitrary adaptive mask. The closure creates no new
//                    vertices -- it only reconnects midpoints the neighbouring
//                    red splits already produced -- and is recomputed from
//                    scratch on each refine() call, which bounds the number of
//                    triangle similarity classes.
//
// See docs/design.md (Adaptive refinement) and
// tasks/completed/conforming-refinement.md for the full design.
enum class RefinementMode
{
    HangingNode2to1, //!< 2:1-bounded hanging nodes; opt-in, cheaper.
    Conforming //!< Red-green-blue transient closure; no hanging nodes. The
               //!< Mesh default: a hanging node breaks a surface operator
               //!< SILENTLY, so the safe contract is the one you get by
               //!< default and the cheap one is the one you ask for.
};

} // namespace Tessera

#endif // TESSERA_REFINEMENT_MODE_HPP
