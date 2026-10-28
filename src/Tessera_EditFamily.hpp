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

#ifndef TESSERA_EDIT_FAMILY_HPP
#define TESSERA_EDIT_FAMILY_HPP

#include <stdexcept>
#include <string>

namespace Tessera
{

// ============================================================================
// Editing families
// ============================================================================
//
// Tessera has TWO disjoint families of topological edit, and a mesh belongs to
// exactly one of them (tasks/completed/edge-split.md, Decision 1):
//
//   HIERARCHICAL   refine()  /  refineLocal()
//                  Maintains the 2:1 LEVEL BALANCE and, in
//                  RefinementMode::Conforming, the transient closure layer.
//                  FaceField::Level / EdgeField::Level are AUTHORITATIVE: they
//                  only ever rise, the balance invariant is stated in terms of
//                  level differences, and the closure is keyed off them.
//
//   REMESH         splitEdges()  (and, as they land, collapseEdges(),
//                  flipEdges(), compact())
//                  Maintains CONFORMITY and MANIFOLDNESS only. `Level` is
//                  ADVISORY: a child face inherits its parent's level, so the
//                  mesh is not 2:1-level-meaningful afterwards.
//
// WHY THEY CANNOT BE INTERLEAVED. refine()'s whole design rests on the level
// model, and that model is coherent only because refine() performs the uniform
// 1->4 red split. Bisecting ONE edge of a triangle produces two children whose
// edges have mixed levels; no single integer describes them, and a 2:1
// level-difference invariant is not the right statement about the result. So a
// refine() applied to a mesh splitEdges() has edited would balance against
// levels that no longer mean anything, and a splitEdges() applied to a
// conforming refine()d mesh would edit the transient CLOSURE layer as if it
// were the persistent red one.
//
// That is a subtle wrong answer, so it is ENFORCED rather than documented: the
// mesh carries an EditFamily tag, `None` until its first topological edit and
// then fixed, and each entry point calls requireEditFamily() to turn the
// mistake into an immediate, message-carrying abort.
//
// Extending the level model to anisotropic bisection (per-edge levels with a
// compatible balance rule) is the alternative that would remove the split, and
// it is a much larger design that no known consumer needs. It is recorded as
// future work, not attempted.

//! Which family of topological edit has been applied to a mesh.
enum class EditFamily
{
    None = 0,     //!< no topological edit yet; either family may claim it
    Hierarchical, //!< refine() / refineLocal()
    Remesh        //!< splitEdges() / collapseEdges() / flipEdges() / compact()
};

//! Human-readable family name, for the guard's message.
inline const char* editFamilyName( EditFamily f )
{
    switch ( f )
    {
    case EditFamily::Hierarchical:
        return "Hierarchical (refine/refineLocal)";
    case EditFamily::Remesh:
        return "Remesh (splitEdges/collapseEdges/flipEdges/compact)";
    default:
        return "None";
    }
}

//! Claim `mesh` for editing family `want`, throwing if it already belongs to
//! the other one. A mesh tagged `None` is claimed silently; re-claiming the
//! family it already has is a no-op. `op` names the calling entry point so the
//! message says which call was rejected.
//!
//! Throws std::runtime_error. The tag is rank-local but every rank sets it in
//! the same collective call, so every rank reaches the same verdict and the
//! throw is effectively collective.
template <class MeshT>
void requireEditFamily( MeshT& mesh, EditFamily want, const char* op )
{
    const EditFamily have = mesh.editFamily();
    if ( have != EditFamily::None && have != want )
        throw std::runtime_error(
            std::string( "Tessera::" ) + op + ": this mesh belongs to the " +
            editFamilyName( have ) + " editing family, and " + op +
            " belongs to the " + editFamilyName( want ) +
            " family. The two are DISJOINT and must not be interleaved on one "
            "mesh: the hierarchical family maintains the 2:1 level balance and "
            "the conforming closure with Level authoritative, while the remesh "
            "family maintains conformity and manifoldness only and leaves "
            "Level "
            "advisory (a child inherits its parent's level). Build a fresh "
            "mesh "
            "for the other family." );
    mesh.setEditFamily( want );
}

} // namespace Tessera

#endif // TESSERA_EDIT_FAMILY_HPP
