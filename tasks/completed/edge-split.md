# Caller-driven edge split

**Status:** IMPLEMENTED. First of the four topological-edit tasks
([edge-split](edge-split.md) → [mesh-compaction](mesh-compaction.md) →
[edge-flip](edge-flip.md) → [edge-collapse](edge-collapse.md)). **Read the
"Editing families" section below before starting any of the four** — it is stated
once, here, and cross-referenced from the other three.

**Verified against `08dd346`** (branch `conforming-refinement`) — the code this task
cites was re-read at that commit.

## Problem

Tessera is **split-based only**, and even the split is not addressable by the
caller. `refine()` takes a **face mask** and bisects all three edges of every
marked face; the split-edge map (`RefineResult::midpoints`) is an **output**, not
an input.

There is no way to say "bisect these particular edges". The closest expressible
thing — mark every face incident on a wanted edge — is strictly wrong for the
purpose: it splits all three of that face's edges, not the one wanted, and then
pulls in the 2:1 balance closure on top. The resulting edit is larger than
requested and differently shaped.

**Why it matters.** Metric-driven remeshing selects edges, not faces: an edge
longer than the local target length is split, an edge shorter than a floor is
collapsed, and a flip is applied where it improves valence. The whole family is
edge-addressed. The driving consumer (the Beatnik z-model's dynamic remesher) is
built out of exactly `split_selected_edges` / `collapse_short_edges` /
`flip_edges_for_quality`, and none of the three is expressible today.

## Approach

### Editing families — the decision that scopes all four tasks

Tessera has one topology editor today, `refine()`, and its whole design rests on a
**level model**: `FaceField::Level` and `EdgeField::Level` only ever rise, the 2:1
balance invariant is stated in terms of level differences, and
`RefinementMode::Conforming` maintains a transient closure layer keyed off
`ClosureParent`. That model is coherent because `refine()` only ever performs the
uniform 1→4 red split.

An edge-addressed split does **not** fit that model. Bisecting one edge of a
triangle produces two children whose edges have mixed levels; there is no single
integer that describes them, and a 2:1 level-difference invariant is not the right
statement about the resulting mesh.

**Decision 1 — two disjoint editing families, and a mesh belongs to one.**

| Family | Operations | Invariant maintained | Level semantics |
| --- | --- | --- | --- |
| **Hierarchical** | `refine()` | 2:1 level balance; conforming closure | `Level` is authoritative |
| **Remesh** | `splitEdges()`, `collapseEdges()`, `flipEdges()`, `compact()` | conformity and manifoldness only | `Level` is advisory |

A `splitEdges()` child face **inherits its parent's level** and the mesh is
thereafter not 2:1-level-meaningful. Interleaving `refine()` with any remesh
operation on the same mesh is **unsupported**. Enforce it rather than documenting
it: add a small `EditFamily` tag to the mesh (`None` initially, set on first edit),
and have each entry point throw a message naming both families when the tag
disagrees. A one-line check that turns a subtle wrong answer into an immediate
abort.

The alternative — extending the level model to anisotropic bisection — is a much
larger design (per-edge levels with a compatible balance rule) and is not needed
by any known consumer. Record it as future work; do not attempt it here.

**Decision 3 — `splitEdges()` makes no shape guarantee; the bound is a property
of the caller's MASK.** Added 2026-08-14 after measuring it (see the progress
log), because the opposite is easy to assume from `refine()`.

`refine()`'s conforming closure *is* shape-bounded independently of the round
count, and the reason is specific: the closure is **transient**. Un-close
discards the whole closure layer every round, the red engine sees only red
faces, so every visible triangle is one of finitely many retriangulations of a
red triangle and the similarity classes are bounded by construction
(`tests/test_conforming_quality.cpp` measures this).

`splitEdges()` has no such reset. Its children **persist**: a `|S| = 1`
median-cut child is an ordinary face next round and can be cut again, and the
set of similarity classes reachable in *n* rounds is unbounded in *n*. Whether
shape degrades therefore depends entirely on which edges the caller marks:

| Mask rule | Behaviour, measured to depth |
| --- | --- |
| length-driven (split iff longer than a target) | **bounded** — periodic, period 3, min r/R cycling 0.3780/0.3780/0.2815 with the min angle dead flat at 33.203° over ten rounds |
| anti-length (split the *short* edges) | **unbounded** — min r/R halves every round, 0.1953 → 0.0007 in seven |
| length-blind (metric uncorrelated with length) | **unbounded** — r/R < 1e-4 by round 8, ~96% of faces below 0.30 by round 27 |

A length-driven mask is a coarse relative of Rivara longest-edge bisection and
is self-correcting for the same reason: bisecting the longest edge of a
stretched triangle shortens it, so the rule attacks exactly the anisotropy it
would otherwise accumulate. Nothing in `splitEdges()` supplies that; the caller
does. **Do not quote case 8's floor as a property of `splitEdges()`.**

**Decision 2 — an edge-mask split needs no closure and no mark propagation.**
This is the pleasant surprise and it should be stated prominently, because it is
counter-intuitive next to `refine()`. Bisecting a set of edges and splitting
**every** incident face according to how many of its edges were bisected yields a
**conforming** mesh directly: a face with 1, 2 or 3 bisected edges becomes 2, 3 or
4 children respectively, and no hanging node survives. The 2:1 closure machinery
exists because `refine()` bisects all three edges of a marked face and leaves the
neighbour untouched. `splitEdges()` has no such asymmetry. So Phase 1 of
`refineImpl` (the mark-propagation fixpoint) and the closure pass have **no
analogue here** — do not port them.

### API

New header `src/Tessera_EdgeSplit.hpp`.

```cpp
struct SplitResult
{
    //! (bisected edge, midpoint gid) for every split edge this rank TOUCHES —
    //! same contract and same shape as RefineResult::midpoints, so
    //! checkMidpointAgreement consumes it unchanged. Sorted, unique by EdgeKey.
    std::vector<std::pair<EdgeKey, GlobalId>> midpoints;
    long long requested = 0;   //!< marked owned edges, summed globally
    long long split = 0;       //!< edges actually bisected, globally
    long long facesBefore = 0, facesAfter = 0;
};

//! Bisect exactly the marked edges. `edgeMask.size() == mesh.numOwnedEdges()`;
//! the OWNER of an edge decides, and the decision is propagated to every rank
//! holding an incident face. Every incident face is subdivided into 2, 3 or 4
//! children according to how many of its edges are bisected. Conforming on exit
//! with no closure layer and no 2:1 balance pass.
//!
//! Ends by calling rebuildHalo(), so the halo is valid on return — the same
//! postcondition refine() has had since `a55a8de`. Collective.
template <class MeshT, class Policy = DefaultRefinePolicy>
SplitResult splitEdges( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo,
                        const std::vector<char>& edgeMask,
                        const Policy& policy = Policy{} );
```

Host `std::vector<char>` sized `numOwnedEdges()`, matching `refine()`'s mask
convention — a device-computed indicator round-trips to the host. Keep the
convention rather than inventing a second one.

### Implementation

Reuse `refine()`'s Phase 2 wholesale; that is most of the work already done.

1. **Cross-rank agreement on the mask.** An edge is owned by one rank but incident
   on faces owned by up to two. Advertise, per owned face, its three
   `(EdgeKey, faceGid, ownerRank)` to `detail::edgeCoordRank`. The coordinator
   learns the split verdict from the edge's **owner** and replies to every
   co-sharer. Identical routing to `refineImpl` Phase 2a, with `refining` replaced
   by the owner's mask bit.
2. **Midpoint gid assignment.** Unchanged from `refineImpl` Phase 2: the midpoint
   owner is the lowest incident face owner; owners count their midpoints,
   `MPI_Exscan` a contiguous global block onto the pre-split global vertex count,
   assign, and **send the gid to all co-sharers** so both sides agree without
   relying on identical local ordering. This is what makes `midpoints` a complete
   split-edge map and what makes `checkMidpointAgreement` applicable.
3. **Midpoint position and field transfer.** `RefinePolicy` unchanged —
   `DefaultRefinePolicy`'s linear average is the correct rule and applies to every
   user field automatically. **Do not project midpoints onto a sphere**; Tessera
   is not a sphere library and `refine()` does not either.
4. **Local topology reconstruction.** New code, and the only genuinely new logic.
   Per local face, dispatch on the bit pattern of its three bisected edges:
   - **1 bisected edge** → 2 children. Split along the median from the midpoint to
     the opposite corner. One new interior edge.
   - **2 bisected edges** → 3 children. Two diagonals are possible; **pick the
     shorter**, tie-broken by the smaller `EdgeKey`. Both operands are locally
     available (the face's own corners and the two midpoints), so the tie-break is
     evaluable locally and is rank-count invariant.
     **Use `Tessera::edgeLen2Canonical()` (`Tessera_RefineClosure.hpp:212`) for the
     comparison, not a hand-rolled squared length.** The conforming blue closure
     chooses its diagonal geometrically through that exact helper (Decision 15,
     `4cee602`), and the helper exists precisely so two ranks comparing the same
     edge get bit-identical doubles regardless of endpoint order. Reusing it means
     this task inherits that determinism instead of re-deriving it, and it keeps
     the two diagonal rules in the library consistent with each other. Follow the
     same exact-tie fallback the closure uses, and report the tie count as
     `ClosureStats` does.
   - **3 bisected edges** → 4 children, the red split. Identical to `refine()`'s
     existing 1→4 case; reuse it.
   Child face gids: `MPI_Exscan` a contiguous block per rank onto the pre-split
   global face count, as `refine()` does. New interior edge gids likewise, via the
   edge coordinator so the two sides of a shared new edge agree (a new interior
   edge is never shared — it lies strictly inside one parent face — so it can be
   assigned locally; the *halves* of a bisected edge **are** shared and must go
   through the coordinator).
5. **Child levels.** Parent's level, per Decision 1. Set the `Conforming`
   closure members to "visible, no parent" via `initClosureFaceMembers` so a
   zero-filled `ClosureParent` cannot read as face gid 0 — the same trap
   `distribute()` already documents.
6. **Halo.** Call `rebuildHalo()` (`Tessera_HaloRebuild.hpp`) before returning,
   preserving `halo.depth` if [halo-depth.md](halo-depth.md) has landed. Never
   return with a cleared halo — `refine()` stopped doing that in `a55a8de` and
   `splitEdges` must not reintroduce the trap.
7. **Edge user fields.** Edges are re-derived from the new face connectivity here,
   exactly as in `refine()`, so per-edge user data cannot be carried through.
   README *Known Issues* already records this for `refine()`/`refineLocal()`
   ("Edge user fields are reset by `refine()`/`refineLocal()`"); **extend that
   entry to name `splitEdges` rather than adding a second one.** Vertex and face
   user fields are preserved and interpolated as usual.
8. **Empty mask** is a no-op fast path: no communication beyond the collective
   that establishes the global request count, and V/E/F unchanged.

### Non-goals

- Anisotropic level bookkeeping (Decision 1's alternative).
- Choosing *which* edges to split. That is the consumer's metric.
- Collapse, flip, compaction — the three sibling tasks.

### Tests

New `tests/test_split_edges.cpp`, registered at **TIER `regression`**, backends
**SERIAL and HIP**, ranks **`${TESSERA_TEST_MPI_RANKS}` (1–5)**. Gate promotion is
**pre-authorized for this task**.

Every count assertion below is a closed-form identity, so the test checks
arithmetic rather than checking Tessera against itself. All are asserted via the
global owned-count reductions and hold at every rank count.

1. **One edge.** Mark the single edge with the globally smallest `EdgeKey` (a
   rank-count-invariant choice). Deltas: `V+1`, `E+3`, `F+2`. Euler
   `1 - 3 + 2 = 0`, so `V-E+F == 2` still. `checkConforming`,
   `checkNoInteriorVertex`, `checkOwnedEuler == 2`.
2. **Three edges of one face.** Deltas: `V+3`, `E+9`, `F+6` (that face → 4, its
   three neighbours → 2 each). Euler delta 0.
3. **All edges — equivalence with `refine()`.** Mark every owned edge. Result must
   have `V' = V+E = 642`, `E' = 2E+3F = 1920`, `F' = 4F = 1280` — **the same
   counts as a uniform `refine()`** on a fresh mesh. Stronger: the two meshes'
   **vertex position multisets** (sorted lexicographically, so gid numbering is
   irrelevant) must be **bitwise identical**, and so must the multiset of face
   corner-position triples. This is the sharpest available correctness check on
   the 3-edge path, because it pins the new code against machinery already
   verified 150/150.
4. **Two-edge pattern determinism.** Mark edges by a rank-count-invariant rule
   (`EdgeKey.id[0]` even) so faces with all of 1, 2 and 3 bisected edges occur.
   Assert the resulting V/E/F and the face corner-position triple multiset are
   **identical at ranks 1–5**. This is the check that the two-diagonal tie-break
   really is local and rank-count invariant.
5. **Midpoint agreement.** `checkMidpointAgreement( result.midpoints )` on every
   case above. Plus: an edge appears in the map iff some rank touching it reports
   it, and no rank reports an edge it does not touch — reuse the globally-decided
   ground-truth pattern from `test_refine_splitedges.cpp`.
6. **Midpoint positions.** Each midpoint equals the exact average of its endpoints
   (bitwise for the default policy, since `(a+b)/2` is one operation), and is
   **not** on the unit sphere — asserting the negative pins Decision 3.
7. **User-field transfer.** A vertex user field seeded to a linear function of
   position is reproduced at every midpoint to `1e-15` relative, for both a
   `double` and a `double[3]` field.
8. **Repeated rounds.** Seven successive `splitEdges` calls with a
   length-threshold mask (split any edge above the current mean length), no
   intervening `migrate()`. Euler `== 2` and `checkConforming` after each round;
   minimum triangle radius ratio **and minimum angle** reported per round and
   asserted above floors measured rather than guessed; plus **saturation** — the
   final two rounds must set no new worst. Seven rounds and not five: the
   measured sequence has period 3, so five rounds show one dip and one recovery,
   which is consistent with a bound but does not establish one. Depth beyond
   seven is `tests/test_split_edges_depth.cpp`'s job (TIER `unit`), which also
   drives the anti-length and length-blind masks that Decision 3 tabulates.
   Override the round count with `TESSERA_SPLIT_ROUNDS`.
9. **Empty mask.** No-op: V/E/F and `topologyChecksum` unchanged.
10. **Halo valid on return.** Immediately after `splitEdges`, `haloExchange()`
    leaves every ghost vertex position equal to its owner's, and a **second**
    `splitEdges` with no intervening `migrate()` succeeds. Confirms step 6.
11. **Family guard.** `refine()` on a mesh already edited by `splitEdges` throws,
    naming both families; and `splitEdges` on a `refine()`d mesh throws. Pins
    Decision 1.

## Exit criterion

- `test_split_edges` green at **SERIAL and HIP, ranks 1–5**, and the full gate
  still green with nothing relabelled.
- Check 3 passes **bitwise** — the all-edges split and a uniform `refine()`
  produce the same geometry.
- Check 4 passes: the edit is rank-count invariant, including the two-edge
  diagonal choice.
- README gains `splitEdges` in the API section and an **Editing families**
  subsection carrying Decision 1's table; the family guard's error message is
  quoted there.
- `docs/design.md` gains an *Edge-addressed splitting* subsection recording
  Decision 2 (no closure, no mark propagation, and why) and the three bit-pattern
  cases with the two-edge tie-break rule.

## Where this sits

**No hard prerequisite.** `rebuildHalo()` already exists in tree
(`Tessera_HaloRebuild.hpp`, `25980f2`), so step 6 is available today —
[halo-depth.md](halo-depth.md) is only needed if a caller wants a halo deeper than
1, which `splitEdges` itself does not. Prerequisite in spirit for the other three
remesh operations, since Decision 1 and the coordinator idiom are established here.
See the ordering diagram in [halo-depth.md](halo-depth.md).

## Progress log

- 2026-08-07 — Task written, then re-checked against `08dd346` after pulling
  `../tessera`. Nothing implemented.
- 2026-08-10 — Implemented. New `src/Tessera_EdgeSplit.hpp` (`splitEdges()`,
  `SplitResult`) and `src/Tessera_EditFamily.hpp` (`EditFamily`,
  `requireEditFamily()`); `Mesh` carries the tag; `refine()` and `refineLocal()`
  claim `Hierarchical`, `splitEdges()` claims `Remesh`. New
  `tests/test_split_edges.cpp` at TIER `regression`, SERIAL + HIP, ranks 1–5.
  `edgeSetOf()`/`checkSplitEdgeCoverage()` moved from
  `test_refine_splitedges.cpp` into `tests/MeshInvariants.hpp` and are now shared
  by both tests. README gains the API line, the *Editing families* subsection
  with Decision 1's table and the quoted guard message, and the edge-user-field
  Known Issue is extended to name `splitEdges` rather than duplicated;
  `docs/design.md` gains *Edge-addressed splitting*.

  **Two deliberate departures from the task text, both recorded here:**

  1. **The two-edge exact-tie fallback is the smaller `EdgeKey`, not the
     closure's lower-midpoint-gid rule.** The task says both ("tie-broken by the
     smaller `EdgeKey`" and "follow the same exact-tie fallback the closure
     uses"), and they are not the same rule. The `EdgeKey` rule is the one that
     satisfies the exit criterion: midpoint gids come from an `MPI_Exscan`, so
     the closure's fallback is agreed across the ranks of one run but is *not*
     rank-count invariant — which is exactly the Decision-11 finding that made
     the closure's main rule geometric in the first place. An `EdgeKey` is built
     from pre-existing vertex gids and is invariant. Ties are common on the
     icosphere (22 on the case-4 workload), so this is not a corner: with the
     gid rule, check 4 would be expected to fail. The count is published as
     `SplitResult::diagTies` and asserted equal between the world run and the
     `MPI_COMM_SELF` reference.
  2. **The whole-edge squared length is computed locally, not carried in a
     message.** `edgeLen2Canonical()` is used as instructed, but its operands are
     the deciding face's own corners, which `rebuildHalo()` guarantees are held
     with their positions — so no analogue of `detail::KeyGid`'s `len2` rider is
     needed. The closure needs the message only because it runs on the un-closed
     red layer, whose corners come from `ClosureParentVerts`.

  **Test 8's floor, measured rather than guessed.** "Minimum radius ratio" is
  read as inradius/circumradius (0.5 for an equilateral triangle, → 0 for a
  sliver), which is the reading under which "asserted above a floor" is the
  meaningful statement. First-run per-round values over the five
  length-threshold rounds:

  | round | 1 | 2 | 3 | 4 | 5 |
  |---|---|---|---|---|---|
  | min r/R | 0.3780 | 0.3780 | **0.2815** | 0.3780 | 0.3780 |

  byte-identical at np1–5 on both backends and in both execution spaces, so the
  floor is set to **0.25**. The sequence does not drift downward — rounds 4 and 5
  recover to round 1's value while F grows 320 → 28160 — which is the substantive
  result.

  **First-run measurements** (all byte-identical at np1–5, SERIAL and HIP,
  `Serial` and `Default` execution spaces):

  | check | result |
  |---|---|
  | 1. one edge | V 162→163, E 480→483, F 320→322; \|S\|=(318,2,0,0) |
  | 2. one face's three edges | V 162→165, E 480→489, F 320→326; \|S\|=(316,3,0,1) |
  | 3. all edges vs uniform `refine()` | V=642 E=1920 F=1280 both; vertex-position and face-corner-triple multisets **bitwise** equal |
  | 4. parity mask vs `MPI_COMM_SELF` | V=395 E=1179 F=786; \|S\|=(81,82,87,70); diagTies=22 — all identical |
  | 6. midpoints | every one the exact bitwise average of its endpoints and strictly inside the unit sphere |
  | 7. user fields | worst relative error 8.88e-16 over `double` and `double[3]` |
  | 9./10. empty mask, halo | checksum unchanged; plans non-empty and ghost resync clean at np>1 |
  | 11. family guard | throws in both directions, message names both families |

- 2026-08-14 — **Risk R12 investigated: the case-8 floor was five rounds deep.**
  Raised against a downstream (Beatnik) phase that rests on `splitEdges()`
  holding triangle shape over many rounds, where the only evidence was case 8's
  five rounds. `tests/test_conforming_quality.cpp` records the trap directly:
  eight rounds there could not distinguish saturation from a maximum being
  discovered slowly, sixteen could. Five rounds are consistent with a bound but
  do not establish one, and a Beatnik run refines far more than five times.

  **New `tests/test_split_edges_depth.cpp`, TIER `unit`, SERIAL + HIP, ranks
  1 and 4.** A diagnostic, not a gate assertion: it drives four mask families to
  whatever depth a face budget allows and prints a table — per round the global
  min r/R, the global min angle, and the POPULATION below five r/R thresholds
  (the tail is what a minimum cannot tell you: one bad triangle versus bad
  triangles becoming a fixed fraction of the mesh). Every rule is a pure function
  of the global geometry — the two length rules by construction, the two hash
  rules because the hash is over the raw IEEE bits of the edge MIDPOINT POSITION
  and not over gids, which come from an `MPI_Exscan` and are not rank-count
  invariant. Knobs: `TESSERA_SPLIT_DEPTH_ROUNDS`, `TESSERA_SPLIT_DEPTH_FACES`.

  **Result: the risk is real, but it lands on the MASK, not on `splitEdges()`** —
  written up as Decision 3 above. Case 8's mask is fine at depth; two plausible
  alternatives are not.

  | family | rule | rounds | min r/R trajectory | verdict |
  |---|---|---|---|---|
  | `above-mean` | split iff longer than the global mean (case 8's mask) | 10, F 320 → 3 276 800 | 0.3780 0.3780 **0.2815** 0.3780 0.3780 **0.2815** 0.3780 0.3780 **0.2815** 0.3780 | **bounded — exactly periodic, period 3**; min angle 33.203° in *every* round; the tail below 0.30 is 0 except in the dip rounds |
  | `below-mean` | split iff shorter than the global mean | 7, F → 1 179 680 | 0.1953 0.0568 0.0169 0.0068 0.0031 0.0015 0.0007 | **unbounded** — halves per round; min angle 24.96° → 0.21°; ~17% of faces below 0.25 and stable, so it is a fixed fraction, not a few bad cells |
  | `hash-third` | length-blind: hash(midpoint position) ≡ 0 mod 3 | 27, F → 2 340 916 | 0.1953 → < 1e-4 by round 7, 0.0000 from round 8 | **unbounded** — min angle 24.96° → 0.000°; 96.7% of faces below 0.30 by round 27 |
  | `cap-hash` | the same rule inside a fixed geodesic cap | 30, F → 8 096 | 0.2238 → 0.0000 by round 11 | **unbounded**, and localised: a small region collapses while the rest of the mesh is untouched |

  Every printed round line is **byte-identical** across np 1, 2, 4, 5 × {SERIAL,
  HIP} × {`Serial`, `Default`} — 7 configurations reduce to exactly 39 distinct
  round lines, which is 8 + 7 + 12 + 12, i.e. one line per round per family and
  no spread at all. So the trajectories above are properties of the global mesh,
  not of a decomposition. Cost: 1m14s at np4, 4m45s at np1 for a 12-round,
  400k-face budget.

  **Why `above-mean` is periodic and not merely flat.** Splitting every
  above-mean edge of a near-uniform icosphere is a coarse relative of Rivara
  longest-edge bisection: it attacks the long edge of a stretched triangle, which
  is the one whose bisection *improves* the shape. The mesh cycles through three
  states — uniform, half-split, three-quarters-split — and 0.2815 is the shape of
  the transient state, re-entered identically every third round. `below-mean`
  is the same machinery driven backwards and it degrades geometrically.

  **Changes to `tests/test_split_edges.cpp` case 8:** five rounds → **seven**
  (two complete periods rather than one; F reaches 204 800, and 10 rounds would
  reach 3.3M, which is a diagnostic's job not a gate's), round count overridable
  with `TESSERA_SPLIT_ROUNDS`; the **minimum angle** is now measured, printed and
  asserted above 30.0° (measured 33.203° flat); and a **saturation assertion**
  was added — the worst over the final two rounds must not be below the worst
  over the earlier ones. That last one is the assertion five rounds could not
  support and is what actually retires R12 inside the gate: a monotone decline
  fails it at every depth, a periodic sequence passes it as soon as the drive
  exceeds one period. `kMinRadiusRatioFloor` stays at 0.25 — the measured worst
  is unchanged at 0.2815 — but its comment now carries the ten-round table and an
  explicit SCOPE paragraph saying the floor is a statement about the mask.
  Case 8's per-round `checkAll()` was also broken out into its six named
  post-conditions, printed only on failure, so a deep round says WHICH one moved
  rather than reporting an aggregate count.

  **The saturation comparison needs a relative tolerance, and finding that out
  cost a debugging cycle worth recording.** The first version asserted
  `worstLate >= worstEarly` bit-exactly and FAILED at exactly six rounds — the
  first depth at which the late window contains a dip round. The periodicity is a
  statement about shape, not about bits: round 6's dip is reached by three more
  rounds of arithmetic than round 3's, so the two agree to ~1e-13
  (0.281541949162 to twelve places) but not in the last bits. The assertion now
  compares to 1e-6 relative, which is far below anything worth detecting given
  that the unbounded families halve the value every round. If a future change
  makes this fail, check the printed twelve-place values before assuming a
  regression.

  **Verification:** `test_split_edges` green at SERIAL and HIP, ranks 1-5, in
  both execution spaces — 180 case lines all `ok`, zero failures, and case 8's
  seven round lines reduce to exactly 7 distinct lines over all 20 instances,
  i.e. byte-identical everywhere. 14 s at np5 to 41 s at np1 (was ~5 s at five
  rounds); comparable to `conforming_quality`'s 11-16 s. The full gate was NOT
  re-run: nothing under `src/` was touched, and the gate definition (label,
  backends, ranks) is unchanged — `test_split_edges_depth` is TIER `unit`.

  **Not done, deliberately:** no quality constraint was added to `splitEdges()`
  itself. R12's suggested fix — refuse to split an edge whose child would fall
  below a shape floor — would make the operation's output depend on a geometric
  predicate the caller cannot see, i.e. `splitEdges()` would silently bisect
  fewer edges than asked, which contradicts the "bisects EXACTLY the marked
  edges" contract that cases 1-4 pin. The right place for the constraint is the
  caller's metric, and the right library-side offering (if a consumer needs it)
  is a separate opt-in *mask filter* that returns which of the caller's marks it
  dropped. Not needed by any consumer today.
