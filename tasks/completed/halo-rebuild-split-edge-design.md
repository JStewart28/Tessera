# Halo rebuild + geometric blue tie-break

> **This file is self-contained.** A new session can be told *"read
> `tasks/completed/halo-rebuild-split-edge-design.md` and implement follow-up 1"* and needs
> nothing else to start. Background lives in
> [tasks/completed/conforming-refinement.md](conforming-refinement.md) (design + Decision
> record) and [tasks/completed/conforming-refinement-debug.md](conforming-refinement-debug.md)
> (Task 8's verification evidence) — read those when a section says to, not before.
>
> **Both items are pre-existing library warts surfaced by Task 8, not conforming-mode
> defects.** Conforming refinement is complete and green (gate 140/140, unit 62/62 at
> `cc219f8`). Neither item blocks anything; both make the library harder to misuse.
>
> **Do them in order: 1 then 2.** Follow-up 1 is independently valuable and materially
> simplifies follow-up 2 — see *Why 1 before 2* at the end. Do not start 2 first.
>
> **Both follow-ups have landed** — 1 on 2026-08-05 (`Tessera_HaloRebuild.hpp`,
> Decision 14), 2 on 2026-08-06 (geometric blue tie-break, Decision 15). The design
> sections below are kept as the design of record; the **progress log at the end says
> what actually happened and where it differed from the design** — read that first if
> you are here to find out what the code does today.

---

## Status

| # | Item | Priority | Status |
|---|------|----------|--------|
| 1 | Factor the halo rebuild into a standalone `rebuildHalo()` that `refine()` calls | **High** — caused 6 of Task 8's 9 defects | **Done** (2026-08-05) — gate **150/150**, unit **62/62**. See the progress log. |
| 2 | Carry the split edge's squared length in Phase 2's coordinator reply so the blue tie-break can be geometric | Medium — closes Decision 11 | **Done** (2026-08-06) — `blueDiagMismatch` **0** at np1–5 on both backends (was 4 of 20 blue parents at np5); full sweep **193/193**. Decision 15. See the progress log. |

**Both were recorded in README *Known Issues* and in the Decision record** (Decision 11
for item 2, Decision 13's second bullet for item 1). **Both entries are now deleted** —
Decisions 14 and 15 resolved them, and neither accepted limit of the `Conforming`
default survives.

**This file is closed out.** Nothing here is open work.

---

## Environment recipe (the whole thing)

The Bash tool runs a **non-login** shell, where no `PrgEnv-*` module is loaded and the
Cray `CC` wrapper fails at configure. Everything goes through `bash -lc`, and
`SPACK_USER_CONFIG_PATH` must be exported or spack silently drops every Tuolumne
system external.

```bash
bash -lc '
  export SPACK_USER_CONFIG_PATH=$HOME/.spack/tuolumne
  . /usr/WS2/stewartj/spack/share/spack/setup-env.sh
  spack env activate ~/spack_envs/tuolumne_trilinos/
  cd /g/g20/stewartj/research-bridges/tessera-dev/Tessera/build-tuolumne && make -j 48'
```

`build-tuolumne/` exists and is current. If `CMakeCache.txt` is ever lost, reconfigure
with `bash ../run_cmake_toulumne.sh` (never a bare `cmake .`, which deletes the cache).

**Running tests.** `TESSERA_REPO` **must be exported** before `flux batch`, or the job
dies in ~7 s with `TESSERA_REPO: unbound variable` — the committed runners
`source "${TESSERA_REPO}/scripts/lib/tessera_env.sh"` under `set -u`, and the
resolver's own self-location (`tessera_env.sh:32`) cannot help because the script
needs the variable to *find* the resolver.

```bash
export TESSERA_REPO=/g/g20/stewartj/research-bridges/tessera-dev/Tessera
flux batch scripts/tuolumne/run_regression_minset.flux   # gate: 140 instances, ~14 min
flux batch scripts/tuolumne/run_unit_minset.flux         # unit: 62 instances, ~5 min
```

Two `flux batch`ed 1-node exclusive jobs run **concurrently on separate pdebug nodes**,
so submit both at once. For a handful of tests, copy the `probe.flux` template from
[conforming-refinement-debug.md](conforming-refinement-debug.md) — and heed its three
harness traps, especially that **a ctest timeout poisons every test after it** unless
the `flux run` is wrapped in `timeout -k 5 <sec>`.

---

## Code map — what a new session needs to know before touching either item

| Thing | Where |
|---|---|
| `refine()` dispatcher → `detail::refineImpl()` | [Tessera_RefineParallel.hpp:1066](../../src/Tessera_RefineParallel.hpp#L1066) → [:192](../../src/Tessera_RefineParallel.hpp#L192) |
| `migrate()` — one ~600-line function, rounds S/A/B/C/D | [Tessera_MeshMigrate.hpp:276-874](../../src/Tessera_MeshMigrate.hpp#L276-L874) |
| `MeshHalo` (the three plans) + `detail::buildKindPlan()` | [Tessera_Distribute.hpp:39](../../src/Tessera_Distribute.hpp#L39), [:56](../../src/Tessera_Distribute.hpp#L56) |
| `closeFaces()` / `unclose()` / `recoverSplitEdges()` | [Tessera_RefineClosure.hpp:265](../../src/Tessera_RefineClosure.hpp#L265) |
| The blue tie-break itself | [Tessera_RefineClosure.hpp:374-392](../../src/Tessera_RefineClosure.hpp#L374-L392) |
| Aggregate header (a new header must be added here) | [Tessera.hpp:18-45](../../src/Tessera.hpp#L18-L45) |

**Header install is a glob** (`FILES_MATCHING PATTERN "*.hpp"`,
[CMakeLists.txt:118](../../CMakeLists.txt#L118)), so adding a header needs **no CMake
edit** — only an `#include` line in `Tessera.hpp`.

**Include graph is clean for a new shared header.** `RefineParallel.hpp` does *not*
include `MeshMigrate.hpp` and vice versa; both include `Distribute.hpp` and
`Mesh.hpp`. So a new `Tessera_HaloRebuild.hpp` can be included by both with no cycle.

---

# Follow-up 1 — standalone `rebuildHalo()` that `refine()` calls

## Problem

`refine()` drops every ghost and clears the halo plans
([Tessera_RefineParallel.hpp:1043-1045](../../src/Tessera_RefineParallel.hpp#L1043-L1045)),
and nothing rebuilds them. The general (non-replicated) 1-deep halo rebuild **exists**
but is welded inside `migrate()`. Two consequences:

1. `haloExchange()` on a freshly-refined mesh is a **silent no-op** on an empty plan,
   not a synced ghost layer.
2. **A second `refine()` throws.** `refineImpl()` builds `gid2lv` from
   `mesh.numVertices()` — owned **+ ghost** — at entry
   ([:265-268](../../src/Tessera_RefineParallel.hpp#L265-L268)), and Phase 3a needs the
   *positions* of both endpoints of every midpoint the rank owns in order to
   interpolate it ([:617-618](../../src/Tessera_RefineParallel.hpp#L617-L618)). Midpoint
   ownership is "lowest incident refining-face owner", so a rank can own the midpoint
   of an edge one of whose endpoints it holds only as a **ghost** — which the previous
   `refine()` already dropped. Result at np ≥ 2: `std::out_of_range:
   unordered_map::at`, on a non-zero rank, before any output flushes.

The documented workaround is an **identity `migrate()`** (`dest[f] == rank`) followed
by `haloExchange()`, which works only because rounds B/C/D of `migrate()` *are* the
halo rebuild.

**Why this is the top item.** It accounted for **six of Task 8's nine defects** (D1,
D3, D5, and three separate sites in D6) and **zero** of them were closure defects. It
fails in the worst possible way: not at the call the user got wrong, but as a throw
from *inside the next* `refine()`, on a non-zero rank, with no output. Every
multi-round driver must know an idiom that reads like a no-op to survive at all.

It also drags a **benign but confusing side effect** along: the identity `migrate()`
permutes local face ordering, so the next round's children get different gids, so a
**gid-derived mask selects a different face set**. That is why `refine_conforming`
round 3 at np1 reports `F=2372` today and `F=2386` before D3 added the re-halo, and
why `conforming_migrate`'s `F` varies with rank count. Fixing item 1 properly can
retire this too — see *Decision to make* below.

## Where the seam is

`migrate()` is six phases, not five — there is an unlabelled **gather** between S and
A that the round letters hide, and it is the one most easily missed:

| Round | Line | Job | Belongs to |
|---|---|---|---|
| S | [310](../../src/Tessera_MeshMigrate.hpp#L310) | sibling-cohesion fixup on `dest` (Conforming only) | **migrate** |
| **G** | [393-465](../../src/Tessera_MeshMigrate.hpp#L393-L465) | **gather referenced-but-non-held vertex/edge tuples from their owners** via a gid coordinator | **rebuildHalo** |
| A | [468](../../src/Tessera_MeshMigrate.hpp#L468) | move owned faces + their verts/edges to destinations | **migrate** |
| B | [515](../../src/Tessera_MeshMigrate.hpp#L515) | ownership (lowest-rank) + ghost discovery via coordinators | **rebuildHalo** |
| C | [587](../../src/Tessera_MeshMigrate.hpp#L587) | fetch ghost faces (+ their verts/edges) from face owners | **rebuildHalo** |
| D | [659](../../src/Tessera_MeshMigrate.hpp#L659) | assemble owned-first AoSoAs + CSR + keys + the three halo plans | **rebuildHalo** |

**Round G is not optional and it is the heart of the matter.** Its own comment
([:382-385](../../src/Tessera_MeshMigrate.hpp#L382-L385)) says outright that `migrate()`
"is the deferred post-refine ghost builder ... so it must recover those referenced
tuples before round A can move a face with its full vertex/edge pack." After
`refine()`, an owned face can reference a vertex the rank does not hold at all — always
true across a partition boundary, and in Conforming mode also true of a closure child
whose midpoint corner is owned by the refining neighbour (**risk point 4**). G fetches
those full tuples, **positions included**. A `rebuildHalo()` that skipped G would fail
precisely on the common Conforming-mode case.

Rounds B/C/D consume exactly three things produced by A: `faceById`, `vById`, `eById`
(gid-keyed `std::map`s of the tuples this rank now owns/references,
[:471-473](../../src/Tessera_MeshMigrate.hpp#L471-L473)). **That is the entire interface**
between the halves. Note G feeds `heldV`/`heldE`
([:386-391](../../src/Tessera_MeshMigrate.hpp#L386-L391), built from the host copies taken
at [:297](../../src/Tessera_MeshMigrate.hpp#L297)), which A then packs into those maps —
so `rebuildHalo()` needs **G, then a local fill of the three maps from owned faces,
then B/C/D**. Ordering differs between the two callers: `migrate()` runs S→G→A→BCD,
`rebuildHalo()` runs G→fill→BCD.

## Design

New header `src/Tessera_HaloRebuild.hpp` (BSD/SPDX header per CLAUDE.md), holding
rounds B/C/D as:

```cpp
//! Rebuild the 1-deep ghost layer and the three halo plans in place, from the
//! mesh's current OWNED entities. No entity changes rank. After this returns,
//! haloExchange() is meaningful and refine() may be called again.
//!
//! Postconditions: every vertex and edge referenced by an owned face is held
//! locally (owned or ghost); every face sharing a vertex with an owned face is
//! held as a ghost; halo.{v,e,f}plan are consistent with the new local indices.
//!
//! INVALIDATION: reallocates the AoSoAs, key Views and CSRs — every slice, CSR
//! handle and key View taken out before this call is dangling. Re-slice after.
template <class MeshT>
void rebuildHalo( MeshT& mesh, MeshHalo<typename MeshT::memory_space>& halo );
```

Then:

* `migrate()` becomes rounds S + A, ending with a call to the shared B/C/D code.
* `refineImpl()` replaces the three `halo.*plan.clear()` calls at
  [:1043-1045](../../src/Tessera_RefineParallel.hpp#L1043-L1045) with `rebuildHalo(mesh,
  halo)`.

**Factor G and B/C/D into two `detail::` helpers** and have both callers use them. Do
*not* implement `rebuildHalo()` as "call `migrate()` with an identity `dest`" — that
keeps round A's all-to-all, keeps the reordering, and runs round S's fixup pointlessly.

```
detail::gatherReferencedTuples( mesh, heldV, heldE )                 // round G
detail::finishHaloAndAssemble( mesh, halo, faceById, vById, eById )  // rounds B/C/D
        ^                                        ^
migrate()      = S -> G -> A ---------------------|
rebuildHalo()  = G -> fill maps from owned faces -|
```

`rebuildHalo()`'s "fill" step is a local loop: for each owned face, insert its tuple
into `faceById` and its three vertices/edges (now guaranteed present by G) into
`vById`/`eById`. No communication beyond G's coordinator round trip.

**Consequence worth stating explicitly, because follow-up 2 depends on it:** after
`rebuildHalo()`, *every vertex referenced by an owned face is held locally with its
position*. That is a stronger postcondition than "a 1-deep ghost layer exists", and it
is what makes follow-up 2's persistent-split-edge case solvable locally.

### Decision to make: does `rebuildHalo()` preserve local ordering?

Round D currently canonicalises to **owned-first, each kind ascending by gid**
([:662-695](../../src/Tessera_MeshMigrate.hpp#L662-L695)). Two options:

**(A) Keep the gid-sort (recommended).** Reuse round D verbatim; least code, least
risk. The payoff is a genuinely useful new invariant: **local index order becomes a
pure function of the owned gid set**, so a mesh's layout is reproducible regardless of
how it was reached. Because `refine()` would now *always* end with this
canonicalisation, the "with vs without re-halo differ" ambiguity disappears — there is
one behaviour. Note this *changes* `refine()`'s current output ordering, so expect
gid-masked tests' per-round counts to shift once (see *Acceptance*).

**(B) Preserve the incoming owned order, append ghosts.** Avoids a sort and avoids
perturbing today's numbers, but needs round D generalised to take an explicit owned
order, and leaves two orderings in the codebase.

Recommend **(A)**. Record the choice in the Decision record of
[conforming-refinement.md](conforming-refinement.md) either way, and state in
`rebuildHalo()`'s doc block that local ordering is canonical-by-gid, since callers can
observe it through gid-derived masks.

### Bonus cleanup this enables

`refineImpl()` step 3j builds a deliberately **incomplete** owned-vertex 1-ring CSR
"best-effort ... because ghost faces/edges are dropped"
([:976-1036](../../src/Tessera_RefineParallel.hpp#L976-L1036)). Round D rebuilds the CSR
completely, so 3j becomes dead work. Delete it in the same change and say so in the
commit — but only after the suite is green with `rebuildHalo()` in place, so a failure
is attributable.

## Implementation steps

1. **Create `src/Tessera_HaloRebuild.hpp`** with the license header. Move round G
   ([:393-465](../../src/Tessera_MeshMigrate.hpp#L393-L465)) into
   `detail::gatherReferencedTuples()` and rounds B/C/D
   ([:515-872](../../src/Tessera_MeshMigrate.hpp#L515-L872)) into
   `detail::finishHaloAndAssemble( mesh, halo, faceById, vById, eById )`. Pure code
   motion — **make this its own commit and verify the gate is still green before
   changing behaviour.** The `MigrateStats` fields that rounds G and B/C/D populate
   must be threaded out (pass the stats struct by reference, or return a small struct).
2. **Add `rebuildHalo()`** to the same header: `gatherReferencedTuples()`, then fill
   the three maps from the owned faces, then `finishHaloAndAssemble()`. Add
   `#include "Tessera_HaloRebuild.hpp"` to `Tessera.hpp` and to
   `Tessera_MeshMigrate.hpp`.
3. **Rewrite `migrate()`** as S → `gatherReferencedTuples()` → A →
   `finishHaloAndAssemble()`. Gate must stay green — still no behaviour change.
4. **Call it from `refineImpl()`**: replace the three `clear()` calls at
   [:1043-1045](../../src/Tessera_RefineParallel.hpp#L1043-L1045). Update the
   INVALIDATION comment there and the header contract at
   [:78](../../src/Tessera_RefineParallel.hpp#L78) ("CLEARS the passed halo. A
   haloExchange() must not be...") — that text becomes wrong.
5. **Delete the workaround from the tests** (this is the acceptance test, below).
6. **Delete step 3j** and simplify.
7. **Docs:** README *Known Issues* — the re-halo entry goes away entirely (do not
   reword it; the wart is gone). `docs/design.md`'s refine section, the
   `Tessera_RefineParallel.hpp` header block, and Decision 13's second bullet all
   claim a caller must re-halo. Add a Decision-record entry.

## Acceptance

**The acceptance test is deleting the workaround.** These sites exist only because of
this defect; every one must be removable with the suite still green:

| Test | Sites |
|---|---|
| `test_refine_splitedges.cpp` | [415-418](../../tests/test_refine_splitedges.cpp#L415-L418) |
| `test_refine_conforming.cpp` | [293-296](../../tests/test_refine_conforming.cpp#L293-L296) (mesh), [299-302](../../tests/test_refine_conforming.cpp#L299-L302) (control) |
| `test_conforming_migrate.cpp` | [203-207](../../tests/test_conforming_migrate.cpp#L203-L207) (fixture) |
| `test_conforming_operators.cpp` | [312-320](../../tests/test_conforming_operators.cpp#L312-L320) (fixture) |
| `test_conforming_determinism.cpp` | [302-305](../../tests/test_conforming_determinism.cpp#L302-L305), [558-561](../../tests/test_conforming_determinism.cpp#L558-L561), [668-671](../../tests/test_conforming_determinism.cpp#L668-L671) |
| `test_conforming_quality.cpp` | [467](../../tests/test_conforming_quality.cpp#L467) |
| `test_markquality_conforming.cpp` | [265](../../tests/test_markquality_conforming.cpp#L265) |

Do **not** touch the `dest`-vector uses in `test_migrate_mesh.cpp`,
`test_loadbalance.cpp` or `test_io.cpp` that perform a *genuine* migration — check
each site's `dest` is the identity before deleting.

Then: gate **140/140**, `ctest -L unit` **62/62**, `format-check` clean on touched
files under **both** clang-format v21 (`/usr/bin`) and v19
(`/opt/rocm-6.4.2/llvm/bin`).

**Add one new test** — the whole point is that the naive sequence now works:

```cpp
// refine() twice back-to-back with NOTHING in between, at ranks 1-5.
refine( mesh, halo, mask1 );
refine( mesh, halo, mask2 );   // must not throw; mesh must be valid
haloExchange( mesh, halo );    // must be meaningful, not a no-op on an empty plan
```

Assert `checkOwnershipPartition`, `checkConforming` (Conforming mode) and a non-empty
halo plan at ranks ≥ 2. Register it `regression` at ranks 1–5 on both backends — but
**promoting a test into the gate requires explicit user confirmation of backends and
ranks** (CLAUDE.md), so ask.

**Expect gid-masked per-round counts to move once** under option (A), because
`refine()`'s output ordering changes. That is the documented benign permutation
effect, not a regression: `F` totals, `euler`, conformity and closure fractions must
be unchanged in *character*, and `conforming_determinism` (which uses a **geometric**
mask precisely to be ordering-immune) must be **byte-identical**. Treat any change in
`conforming_determinism` as a real failure. Record the before/after numbers.

## Risks

* **`MigrateStats` threading** is the fiddliest mechanical part; rounds B/C/D touch
  stats fields that `migrate()` returns.
* **Cost.** `refine()` gains a full ghost-discovery round trip. Every multi-round
  caller already paid it via the identity `migrate()`, so real drivers get *faster*
  (no round A all-to-all), but a single-shot `refine()` on a mesh nobody will
  re-halo now pays for a halo it may not need. Measure the gate's wall time before and
  after (baseline: 842 s\*proc / 14.3 min). If it matters, a `bool rebuild = true`
  parameter is the escape hatch — but do not add it speculatively.
* **`loadBalance()`** calls `migrate()` internally
  ([Tessera_Zoltan2Balancer.hpp](../../src/Tessera_Zoltan2Balancer.hpp)); confirm it goes
  through the refactored path.
* Sequence steps 1→3 as **no-behaviour-change commits** and run the gate between them.
  The edge-gid defect in D3 hid behind an XOR checksum for two rank counts; do not
  assume a green spot-check means a green gate on a change in shared code.

---

# Follow-up 2 — geometric blue tie-break (closing Decision 11)

## Problem

The blue closure pattern splits the quad `(A, q0, q1, C)` along one of **two** valid
diagonals. The tie-break is *"connect the midpoint with the lower GID to its opposite
corner"* ([Tessera_RefineClosure.hpp:374-392](../../src/Tessera_RefineClosure.hpp#L374-L392)).
Midpoint gids are globally agreed within one run, so the closure is
**partition-independent at a fixed rank count** — but they come from an `MPI_Exscan`,
so they are **not the same values at a different rank count**. The same red mesh
therefore closes with a different blue diagonal at np5 than at np1: measured, **4 of
20 blue parents flip**, with np1–4 all agreeing.

Everything else is rank-count invariant and asserted as such by
`conforming_determinism` case A (red layer, `|S|` histogram, closure-vertex set,
V/E/F). The case now asserts the sharper statement that the visible layer differs
**only** through blue diagonals, in both directions. So this is a *recorded design
limit*, not a failure — README *Known Issues* tells consumers not to compare a
conforming mesh's visible face set bitwise across rank counts.

## Why no purely local rule can work — do not re-derive this

D6 implemented the geometric rule in full and **reverted it**. The reasoning, worth
not rediscovering:

* **The rule is simpler than it looks.** With `q0 = (A+B)/2`, `q1 = (B+C)/2`,
  `dQ0C² - dAQ1² = (3/4)(|C-B|² - |B-A|²)`. So "shorter diagonal" is exactly
  **"connect the midpoint of the longer split edge"** — the standard rule, needing
  only the two split edges' lengths, never the diagonals'.
* **It cannot be evaluated from corner positions.** `closeFaces()` runs on the
  **un-closed** red layer, whose corners come from closure children's
  `ClosureParentVerts`, and a child may name a vertex gid its rank does not hold —
  the documented **risk point 4**, benign only because the closure never needed
  positions. A 1-deep halo does not help: those are *parent* corners, not neighbours.
  Instrumented at np2, one `closeFaces()` call asked for **12** positions the rank did
  not have, including original icosphere vertices (gids 1, 3, 21, 41).
* **Two traps it exposed.** (a) The regression was *invisible* structurally —
  `refine_conforming` np2 reported `euler=2`, correct `F`, correct `|S|` histogram,
  both diagonals populated, and failed on `checkClosureInverse` alone, because *a
  silently-wrong diagonal yields a mesh that is perfectly conforming and perfectly
  wrong*. (b) A `std::function` position lookup that returns a default on a miss
  converts a hard failure into a plausible mesh; the version that **printed the
  missing gid** solved it in one run.

**The D6 patch is gone** (`/tmp/geometric-tiebreak-d6-keep.patch` was reaped from
scratch). Re-implement from this section, not from the patch.

The conclusion is *not* "geometry is impossible" — it is **"the length must be
attached to the edge, not computed from corner positions."**

## Design

### The mechanism: one extra field on an existing round trip

Phase 2 step 2c already delivers each split edge's midpoint gid from the midpoint
owner to every co-sharer, via `detail::KeyGid`
([Tessera_RefineParallel.hpp:172-176](../../src/Tessera_RefineParallel.hpp#L172-L176))
over an `allToAllV` at [:565](../../src/Tessera_RefineParallel.hpp#L565):

```cpp
struct KeyGid { EdgeKey key; GlobalId gid; };          // today
struct KeyGid { EdgeKey key; GlobalId gid; double len2; }; // proposed
```

The midpoint owner is by construction the owner of an incident **refining** face, so
it holds both endpoints of the whole edge it is bisecting and can compute `len2`. **No
extra message round** — the same shape as Decision 10's edge-gid fix.

`refineImpl()` then carries a second map alongside `midGid`:

```cpp
std::map<EdgeKey, double> len2Of;   // whole-edge squared length, per split edge
```

and `closeFaces()` gains a parameter mirroring `midpointOf`:

```cpp
closeFaces( red, midpointOf, firstChildGid, freshChild, firstNewVertexGid,
            /*len2Of=*/ const std::map<EdgeKey,double>& = {} );
```

The tie-break becomes: for parent edges `(A,B)` and `(B,C)`, look up
`len2Of[key(A,B)]` and `len2Of[key(B,C)]` and connect the midpoint of the **longer**
one to its opposite corner. **Corner positions are never needed** — which is precisely
what makes it evaluable where D6's attempt was not.

**`len2` must be bit-identical wherever computed**, or two ranks can disagree and
produce a cracked mesh. Provide one canonical helper and use it everywhere:

```cpp
//! Squared length of edge (a,b). Endpoints are ordered by GID before subtracting
//! so the floating-point result is independent of which rank evaluates it and in
//! which order it holds the two vertices.
inline double edgeLen2Canonical( GlobalId ga, const double* pa,
                                 GlobalId gb, const double* pb );
```

### The one open sub-problem: persistent split edges

`forEachSubEdge()` ([:364-375](../../src/Tessera_RefineParallel.hpp#L364-L375)) advertises
a **persistently** split edge `(x,y)` *only as its two halves* `(x,m)`, `(m,y)` —
never as `(x,y)`. So Phase 2 never sees the key `(x,y)` and cannot deliver its `len2`.
But a red face with a persistently split edge is exactly the "kept face must be closed
again" case (Decision 8), so **blue can arise with one or both split edges
persistent.** This must be answered before the rule is correct.

Three options:

1. **Recover `len2` locally where the midpoint is recovered (recommended).**
   `recoverSplitEdges()` already reconstructs `(a,b) → m` from a closed parent's
   **closure children**, which are *owned* faces. Three facts make the positions
   reachable, and they compose exactly:
   * A closed parent's corners are the **union of its children's corners** — that is
     what makes `recoverSplitEdges()` possible in the first place.
   * **Sibling co-residency is guaranteed** (Task 5's `dest` fixup +
     `repairClosureCohesion()`, asserted at every rank count by
     `conforming_migrate`'s `checkSiblingCoresidency`), so *all* children of a closed
     parent are on this rank.
   * After **follow-up 1**, every vertex referenced by an owned face is held **with
     its position** — that is round G's postcondition, not merely a 1-deep ghost
     layer.

   So `a`, `b` and `m` are all held, and `len2` is computable with no communication.
   **This is why follow-up 1 comes first:** the argument fails without round G, and
   round G is today reachable only by going through `migrate()`. Note this is exactly
   the case D6 hit as "12 missing positions" — the difference is that D6 needed the
   corners of the *un-closed red layer* at `closeFaces()` time, whereas this needs
   them at `recoverSplitEdges()` time, where the children are still in hand.
2. **Announce the whole key with its `len2`** in `forEachSubEdge()`, as a third
   emission flagged "length only — do not treat as an incidence." Workable but
   dangerous: this is exactly where D2's subtlest bug lived. Advertising a half as
   refining made the coordinator mint a midpoint *for the half*, which produced a
   plausible round (`euler=2`, `F=1494` where the answer is `F=1222`) and aborted a
   round later. Any new emission must not perturb the split decision or the level
   propagation. Prefer option 1.
3. **Persist `len2` in a field.** Rejected: edge user fields are reset by `refine()`
   (README *Known Issues*), so it needs a new face field — new state for a derived
   quantity.

Whichever is chosen, **verify options 1 and 2 agree** on an edge visible to both, and
record the choice in the Decision record.

### Exact ties

Two split edges of exactly equal length is not hypothetical — the icosphere is highly
symmetric and round 1 is the undisturbed icosphere. `len2` equality must resolve
deterministically and, ideally, rank-count stably.

**Recommended first cut:** on exact tie, fall back to today's lower-midpoint-gid rule
and **count the ties** in `ClosureStats` (add `nBlueDiagTie`). This keeps the change
minimal and is honest: it reduces the rank-count dependence from "4 of 20 blue
parents" to "only exact geometric ties", and the counter says whether any remain.
`conforming_determinism` case A's assertion can then be tightened from *"the visible
layer may differ through blue diagonals"* to *"...only through blue diagonals that are
exact geometric ties"*, failing if `diagMismatch > 0` while `nBlueDiagTie == 0`.

**If the tie count is non-zero on the test workload**, the stable resolution is to
compare the two midpoints' **quantised positions** lexicographically — positions are
rank-count stable, which is exactly why `conforming_determinism` compares by them.
That needs the midpoint *position* delivered in `KeyGid` (3 more scalars) rather than
just `len2`. Reuse the quantisation from
[test_conforming_determinism.cpp:154-163](../../tests/test_conforming_determinism.cpp#L154-L163)
(`llround(p[d] * 1e9)` + `hashCombine`) so test and library agree. **Measure before
paying for this.**

## Implementation steps

1. **Land follow-up 1 first.** Option 1 above depends on it.
2. Add `edgeLen2Canonical()` and `nBlueDiagTie` to `ClosureStats`
   ([Tessera_RefineClosure.hpp:170-173](../../src/Tessera_RefineClosure.hpp#L170-L173)).
3. Extend `detail::KeyGid` with `len2`; populate it at
   [:556-563](../../src/Tessera_RefineParallel.hpp#L556-L563) from the owner's held
   endpoints; build `len2Of` from `midGid`'s own entries **and** the received ones at
   [:566-567](../../src/Tessera_RefineParallel.hpp#L566-L567).
4. Fill `len2Of` for persistent split edges via option 1, next to
   `recoverSplitEdges()`, and union it in exactly where `persistentSplit` is unioned
   into `midGid` ([:569+](../../src/Tessera_RefineParallel.hpp#L569)).
5. Add the `len2Of` parameter to `closeFaces()` and switch the tie-break. **Seven call
   sites** must be updated — [RefineParallel.hpp:726](../../src/Tessera_RefineParallel.hpp#L726),
   [Refine.hpp:643](../../src/Tessera_Refine.hpp#L643),
   [MeshInvariants.hpp:613](../../tests/MeshInvariants.hpp#L613), and
   `test_refine_closure.cpp` [145](../../tests/test_refine_closure.cpp#L145),
   [352](../../tests/test_refine_closure.cpp#L352),
   [369](../../tests/test_refine_closure.cpp#L369),
   [492](../../tests/test_refine_closure.cpp#L492). Defaulting the parameter to `{}`
   keeps them compiling, but **an empty map must not silently mean "gid rule"** —
   that is exactly D6's default-on-miss footgun. Make a missing `len2` for an edge
   that *is* split a hard `Kokkos::abort` naming the edge, and have the serial-only
   call sites pass real lengths.
6. `refineLocalConforming()` ([Tessera_Refine.hpp:580](../../src/Tessera_Refine.hpp#L580))
   needs the same treatment — it has all positions locally, so this is easy, but do
   not skip it: `refine_closure` covers that path and would otherwise diverge from the
   distributed one.
7. Rewrite the header note at
   [Tessera_RefineClosure.hpp:87-118](../../src/Tessera_RefineClosure.hpp#L87-L118),
   which currently explains at length why this *cannot* be done.

## Acceptance

* `conforming_determinism` case A: **`vis` agrees at np1–5** with
  `blueDiagMismatch == 0` (today np5 gives 4). This is the whole point.
* Gate 140/140 and unit 62/62; `refine_closure`'s blue unit tests rewritten to drive
  **both** diagonals from **geometry** (cut a different corner of the reference
  triangle), plus a case pinning that **relabelling the midpoint gids does not move
  the diagonal** — that is the regression test for the property being bought.
* `checkClosureInverse` must pass. **It is the only check that catches a
  silently-wrong diagonal** — `euler`, `F` and the `|S|` histogram will all look
  correct. Never conclude "the closure is fine" from `euler == 2`.
* Both blue diagonals must stay populated (`nBlueDiagLowFirst`/`LowSecond`, now
  length-derived); a rule that collapses to one diagonal is wrong.
* Re-run `conforming_quality` at 16 rounds (`TESSERA_QUALITY_ROUNDS=16`) and confirm
  the shape bounds still hold. The geometric rule should **improve** blue's radius
  ratio — it picks the shorter diagonal — so if `maxQ` gets *worse* than Decision 12's
  2.5254, the rule is inverted. Recalibrate Decision 12's numbers if it improves.
* Update Decision 11 to resolved (preserve the original analysis), README *Known
  Issues* (delete the blue-diagonal entry), and `docs/design.md`'s closure section.

## Risks

* **A wrong diagonal is structurally invisible.** See above. Lean on
  `checkClosureInverse` and on the gid-relabelling test.
* **Float determinism across backends.** `len2` is computed on host in both paths, but
  SERIAL and HIP must agree bit-for-bit or the two backends will pick different
  diagonals. `conforming_determinism` compares against an `MPI_COMM_SELF` reference
  within one backend; add a cross-backend check of the diagonal counts, or at minimum
  confirm `nBlueDiagLowFirst`/`LowSecond` match across backends in the gate output.
* **Scope creep into the tie case.** Ship the counter first, measure, and only then
  decide whether to send positions.

---

## Why 1 before 2

1. **Item 2's recommended solution to its one open sub-problem depends on item 1.**
   Recovering a persistent split edge's `len2` locally is sound only if every vertex
   referenced by an owned face is held — which is exactly what `rebuildHalo()`
   guarantees and what `refine()` currently does not.
2. **Item 1 is worth more.** Six defects versus one recorded, bounded, documented
   limit that no consumer has hit.
3. **Item 2 needs a trustworthy suite to land against.** Its failure mode is a mesh
   that passes every structural check, so it must be validated on a suite where the
   only moving part is the diagonal. Doing item 1 first — with its own
   ordering-permutation shift absorbed and its numbers re-recorded — gives that.

---

## Progress log

- 2026-08-05 — File created. Both items scoped from Task 8's close-out (D8); no code
  written. Established by reading the code: the `migrate()` seam is S+A (move) vs
  G+B/C/D (halo), whose entire interface between halves is the three gid-keyed maps
  `faceById` / `vById` / `eById`; a new `Tessera_HaloRebuild.hpp` creates no include
  cycle and needs no CMake edit (header install is a glob); and item 2's real obstacle
  is that the **length must be attached to the edge**, not computed from corner
  positions, which makes `KeyGid` the right carrier.

  **The one thing an initial read of `migrate()` gets wrong, corrected here before it
  cost anyone a debugging cycle:** there is an unlabelled **round G** between S and A
  ([:393-465](../../src/Tessera_MeshMigrate.hpp#L393-L465)) that gathers
  referenced-but-non-held vertex/edge tuples from their owners, and it belongs to the
  halo half, not the move half. The five round *letters* hide it. It is not optional —
  after `refine()` an owned face can reference a vertex the rank does not hold at all
  (risk point 4; in Conforming mode, routinely, via a closure child's midpoint
  corner), and a `rebuildHalo()` without G would fail exactly on the common case. Its
  postcondition is also stronger than "a 1-deep ghost layer exists": *every vertex
  referenced by an owned face is held with its position.* That is what makes item 2's
  persistent-split-edge sub-problem solvable **locally** — compose it with sibling
  co-residency (already guaranteed and asserted) and the fact that a closed parent's
  corners are the union of its children's. So the 1-then-2 ordering is **load-bearing,
  not merely preferable**.

  Noted that `/tmp/geometric-tiebreak-d6-keep.patch` has been reaped from scratch, so
  item 2 must be rebuilt from the reasoning preserved here and in Decision 11.

- 2026-08-05 — **Follow-up 1 landed**, in three commits: the code motion, the
  behaviour change, and the step-3j deletion. Gate **150/150** and `ctest -L unit`
  **62/62**; `format-check` clean on every touched file under clang-format v21
  (`/usr/bin`) and v19 (`/opt/rocm-6.4.2/llvm/bin`). Recorded as **Decision 14** in
  [conforming-refinement.md](conforming-refinement.md); the README *Known Issues*
  re-halo entry is deleted outright.

  **The design held up; four things are worth correcting or pinning for follow-up 2.**

  1. **`MigrateStats` needed no threading at all** — listed as "the fiddliest
     mechanical part" under *Risks*, it was a non-issue. Both fields
     (`siblingFixups`, `siblingGroups`) are populated by round S, which stays in
     `migrate()`; rounds G and B/C/D touch no stats. The two shared helpers have
     `void` return.
  2. **`rebuildHalo()` is provably the identity `migrate()` minus rounds A and S**,
     which made the acceptance far cheaper to trust than *Acceptance* anticipated.
     Round A with an identity `dest` is content-preserving on the three maps, so
     every test that already used the workaround is **byte-identical**. The predicted
     "expect gid-masked per-round counts to move once" did **not** materialise for
     those tests: probed at np1 and np5 across the eight gid-masked/conforming tests,
     seven were byte-identical and exactly one number moved —
     `conforming_migrate`'s `loadBalance destFixups` 26 → 25, a Zoltan2 cut shifted
     by the now-canonical owned-face ordering, with `F`, `maxFaces`, `maxWeight`,
     `ideal` and `inv=0` unchanged. `conforming_determinism` was byte-identical, as
     required. Option **(A)** (keep round D's gid-sort) was taken.
  3. **Cost is small and in the predicted direction.** The pre-existing 140 gate
     instances went 814 → 837 s\*proc (+2.8%); `refine_rehalo` adds 69 s\*proc for
     10 instances (gate total 906). No `bool rebuild` escape hatch was added.
  4. **The one real bug written during this work was in the new test, not the
     library, and it was a deadlock:** the per-round `printf` called
     `TesseraTest::globalOwned{Vertices,Edges,Faces}()` — all `MPI_Allreduce` —
     **inside the `rank == 0` guard**. np1 passed and np2/np5 hung with no output.
     Reduce on every rank, then print on rank 0. Worth remembering because it looks
     exactly like a library hang and the probe harness reports it as `exit=124`.

  **What this buys follow-up 2**, restated now that it is a fact rather than a plan:
  after `refine()` returns, *every vertex referenced by an owned face is held locally
  with its position* (round G's postcondition, not merely "a 1-deep ghost layer
  exists"). Compose that with sibling co-residency and "a closed parent's corners are
  the union of its children's" and option 1 of follow-up 2's persistent-split-edge
  sub-problem is sound with no communication. `rebuildHalo()` is also public API, so
  a `recoverSplitEdges()`-adjacent helper can rely on the postcondition by name.

- 2026-08-06 — **Follow-up 2 landed. This file is closed out.** Recorded as
  **Decision 15** in [conforming-refinement.md](conforming-refinement.md), which
  resolves **Decision 11** and the first bullet of **Decision 13**; the README
  *Known Issues* blue-diagonal entry is deleted outright. The previous session
  wrote every edit but compiled and ran **nothing**; this session verified it.

  **The acceptance criterion is met, and it is the whole point of the change:**
  `conforming_determinism` case A reports `vis=ok blueDiagMismatch=0
  parentMissing=0 blueTie=4` at **np1–5 SERIAL and np1–4 HIP**, in both the
  `[Serial]` and `[Default]` exec spaces, where Decision 11 measured **4 of 20
  blue parents flipping at np5**. Full sweep (`run_all_tests.flux`) **193/193** —
  135 regression + 58 unit instances, zero failures, nothing relabelled.

  **The design was right about the hard part and wrong about one number.**

  1. **The "code complete, unverified" state compiled and passed as written.**
     None of the seven `closeFaces()` call sites, the `fetchPos` return-type
     deduction, or the six-parameter signature needed a fix. The predicted
     first-pass fallout did not materialise. The load-bearing design claim — that
     Decision 14's round-G postcondition makes the persistent-split-edge case
     (option 1) solvable with **no communication** — held exactly as argued. The
     1-then-2 ordering was worth what the file claimed it was.
  2. **Exact ties are NOT zero, and the doc above asserted they were.** *Exact
     ties* said to ship the counter and measure. Measured: **4** blue parents tie
     on `conforming_determinism`'s workload — identical at every rank count and on
     both backends — and **11 of 202** on `refine_closure`'s inverse case. The
     already-written prose in `docs/design.md` and in the test's own comment
     claimed the count was *zero on the test workload*; both were corrected. That
     was the one substantive error in the handoff, and it was in the documentation
     rather than the code.
  3. **The escalation the doc reserved is not needed, for a reason the measurement
     had to supply.** Delivering the midpoint *position* to break ties by quantised
     coordinates was contingent on a non-zero tie count — which happened. But
     `blueDiagMismatch` is **0 anyway**: the gid fallback agrees at every rank
     count tested, so the escalation would buy nothing measurable. Not
     implemented. The residual dependence is real in principle, unobserved in
     practice, and the counter keeps it a measurement instead of an assumption.
     Note the two 4s are a coincidence, not a correspondence: the 4 parents that
     flipped at np5 under the old rule are the ones geometry now decides, and the
     4 that tie are gid-stable.
  4. **Shape improved, which is the second independent check that the comparison
     is not inverted.** Re-measured at 16 rounds: blue's worst radius ratio
     **2.5254 → 2.2344**, worst amplification **2.4906 → 2.2310**, saturating at
     round **8** instead of 11 and flat through 16. A rule that took the *longer*
     diagonal would have made `maxQ` worse. Red, green, min angle, closure
     fraction and every `F` are identical to D7's. Decision 12's **bounds were
     left alone** (`Q ≤ 2.8`) — they hold with more margin now, and re-tightening
     a gate bound onto a just-measured number only manufactures a future false
     failure; only the recorded measurements were updated.
  5. **Both diagonals stay populated**, so the rule has not collapsed:
     `refine_closure` inverse 101/101/11, `refine_conforming` 6/9/3 at round 1 to
     41/41/11 at round 3, and `q0C + q1A` equals the `|S|=2` count exactly at
     every round checked. SERIAL and HIP are byte-identical at equal rank count —
     that is the cross-backend float-determinism check the *Risks* section asked
     for. `F`, `euler`, the `|S|` histogram and the closure fraction are
     unchanged, which is the signature that says only the diagonal moved.
  6. **No rebuild was needed for the formatting pass.** `format-check` failed on
     six files under **both** clang-format v21 and v19 — the identical violation
     set, so applying the formatter was safe under both. The tested binaries
     predate the reformat, so equivalence was established by comparing
     whitespace-stripped files rather than by re-running: five matched exactly and
     `Tessera_RefineParallel.hpp`'s single difference is clang-format reflowing a
     string literal across a break, where the concatenated string is unchanged.
     Worth reusing — it converts "is this reformat safe?" from a 25-minute rebuild
     plus a 17-minute sweep into one `tr -d '[:space:]' | diff`.
