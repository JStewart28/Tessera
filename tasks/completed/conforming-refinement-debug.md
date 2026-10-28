# Conforming Refinement — Task 8 debugging

> **This file is self-contained.** A new session can be told *"read
> `tasks/completed/conforming-refinement-debug.md` and complete the next task"* and needs
> nothing else to start. Background on the feature's design lives in
> [tasks/completed/conforming-refinement.md](conforming-refinement.md) — read it when a
> debug task tells you to, not before.
>
> **How to work this file.** Tasks are ordered by dependency, D1 first. Do the
> lowest-numbered task whose status is not Done. When a task completes: update
> its status, record what the fix actually was under *What landed*, append to the
> *Findings log* at the bottom (append-only), commit, and push. If a fix changes
> a **design** decision, also rewrite the affected section of
> `tasks/completed/conforming-refinement.md` — that file is the design's source of truth
> and must never describe something the code no longer does.
>
> **Rules inherited from Task 8 (non-negotiable).**
> - Never delete a test or weaken an assertion to get green.
> - A test that cannot be made green is relabelled `unit`, recorded in README
>   *Known Issues*, listed under *Open failures* here, and reported.
> - The gate definition (label `regression` × {SERIAL, HIP} × ranks 1–5) does not
>   change. Changing it requires updating `CLAUDE.md`, `tests/CMakeLists.txt`,
>   `scripts/tuolumne/run_regression_minset.flux`, and `.github/workflows/ci.yml`
>   together.

---

> **Task 8 is complete as of 2026-08-05 (D0–D8 all done).** Gate 140/140, unit
> 62/62, zero open failures. Nothing here is a live work item; the file is now
> evidence and handoff. The two follow-ups it identifies are named at the end of D8.

## Status

| # | Task | Status |
|---|------|--------|
| D0 | Triage sweep — run the suite, collect first failures | **Done** (2026-08-04) |
| D1 | Fix the `refine_splitedges` np≥2 hang | **Done** (2026-08-04) |
| D2 | Fix risk point 9 — the persistent split-edge map | **Done** (2026-08-04) |
| D3 | Fix the `refine_conforming` np≥2 `unordered_map::at` abort | **Done** (2026-08-04) |
| D4 | Re-sweep: get the remaining nine never-executed tests to a first verdict | **Done** (2026-08-04) |
| D5 | Downstream conforming tests (`conforming_migrate`, `io`, `markquality_conforming`) | **Done** (2026-08-04) |
| D6 | `conforming_operators` and `conforming_determinism` | **Done** (2026-08-04) |
| D7 | `conforming_quality` — calibrate the provisional bounds | **Done** (2026-08-05) |
| D8 | Full gate at ranks 1–5, both tiers, `format-check`; close out Task 8 | **Done** (2026-08-05) |

**Open failures.** **None**, confirmed by D8's full-gate and full-unit runs at HEAD
(140/140 and 62/62). All 28 registrations pass at ranks 1–5 on both backends.
D1, D2, D3, D5, D6 and D7 are fixed.

One recorded **design limit**, not a failure: in `Conforming` mode the *visible*
layer is rank-count dependent up to blue closure diagonals (np1–4 agree, np5 flips
4 of 20 blue parents). Everything provably a function of the global mesh — red
layer, `|S|` histogram, closure-vertex set, V/E/F — is invariant and asserted as
such, and `conforming_determinism` now asserts the sharper "the visible layer
differs *only* through blue diagonals". Decision 11 in
[tasks/completed/conforming-refinement.md](conforming-refinement.md); README *Known Issues*.

D7 answered the last open question: the shape bound **is** fixed in the round count —
`maxQ` saturates at 2.5254 by round 11 and is flat through round 16 — so
`conforming_quality` is recalibrated, extended to 16 rounds, and **promoted into the
gate** (Decision 12).

**Final tier counts, confirmed by D8 at HEAD: 140 `regression` + 62 `unit` = 202
instances over 28 registrations.** D7's promotion of `conforming_quality`'s two
registrations moved 10 instances from `unit` to `regression` (the tiers were 130 + 72
before it), and D7's predicted split was exact. Against the pre-conforming baseline
at `f50a91b^` — 70 `regression` + 28 `unit` = 98 instances over 19 registrations —
this work added **9 registrations and 104 instances**.

---

## How to build and run (this is the whole environment recipe)

The Bash tool runs a **non-login** shell, where no `PrgEnv-*` module is loaded and
the Cray `CC` wrapper fails at configure. Everything below must go through
`bash -lc`, and `SPACK_USER_CONFIG_PATH` must be exported or spack silently drops
every Tuolumne system external.

```bash
bash -lc '
  export SPACK_USER_CONFIG_PATH=$HOME/.spack/tuolumne
  . /usr/WS2/stewartj/spack/share/spack/setup-env.sh
  spack env activate ~/spack_envs/tuolumne_trilinos/
  cd /g/g20/stewartj/research-bridges/tessera-dev/Tessera/build-tuolumne && make -j 48'
```

Build dir `build-tuolumne/` exists and is current. If `CMakeCache.txt` is ever
lost, reconfigure with `bash ../run_cmake_toulumne.sh` (never a bare `cmake .`,
which deletes the cache).

### Running tests — two harnesses

**Full sweep** (submit, then read `tessera-conforming-tests.<id>.out` in the repo
root):

```bash
flux batch scripts/tuolumne/run_conforming_tests.flux
```

That script is committed. It runs both tiers × {SERIAL, HIP} at np1–4 (`_np5`
excluded), with a 100 s per-test ctest timeout, on 1 pdebug node for 20 minutes.

**Isolated probe** — use this while debugging a single test. Copy
`probe.flux` below to the scratchpad, edit the `for` list, `flux batch` it, and
read `probe.<id>.out` in the repo root.

```bash
#!/usr/bin/env bash
#FLUX: --job-name=tessera-probe
#FLUX: --nodes=1
#FLUX: --exclusive
#FLUX: --queue=pdebug
#FLUX: --time-limit=15m
#FLUX: --output=/g/g20/stewartj/research-bridges/tessera-dev/Tessera/probe.{{id}}.out
set -uo pipefail
source "${TESSERA_REPO}/scripts/lib/tessera_env.sh"
cd "${TESSERA_BUILD_DIR}/tests"
P="--nodes=1 --exclusive --cores-per-task=1 --env=GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8388608 --unbuffered"
for spec in "tessera_test_refine_splitedges_SERIAL 2" "tessera_test_refine_conforming_SERIAL 1"; do
  set -- $spec
  echo "########## $1 np$2 ##########"
  timeout -k 5 100 flux run --ntasks $2 $P ./$1 2>&1 | grep -v "no version information"
  echo "########## exit=${PIPESTATUS[0]} ##########"
done
```

### Three harness traps that cost D0 a run — do not rediscover these

1. **A ctest timeout poisons every test after it.** ctest kills its own child,
   but the `flux run` sub-job keeps holding the node `--exclusive`, so every
   later `flux run` queues forever and also times out. In the D0 sweep this
   turned one genuine hang into eight bogus ones. **Always wrap a
   possibly-hanging `flux run` in `timeout -k 5 <sec>`** (as `probe.flux` does)
   — `timeout` signals `flux run`, which tears the sub-job down properly.
2. **A killed test loses all its stdout.** ctest reports `<end of output>` with
   nothing before it. That is *not* evidence the test hung before its first
   print. Use `--unbuffered` on `flux run` when you need partial output from a
   hang.
3. **`flux run` from the login node just queues** — it goes to the system
   instance and waits for a pdebug node, so an interactive one-off appears to
   hang. Always `flux batch`.

---

## D0 — Triage sweep — **DONE (2026-08-04)**

`scripts/tuolumne/run_conforming_tests.flux` (new, committed) submitted as job
`f3QH9jz6D1aw`; it hit its 20-minute limit after 86 of 165 tests, having been
poisoned by trap 1 above from test 79 onwards. An isolated re-probe (job
`f3QHQtUmFtpX`) established which of those failures were real.

### Verdicts — everything that actually ran

**Passing, np1–4, both backends** (32 registrations, 78 test instances):
`keys`, `data_model`, `connectivity`, `migrate`, `halo`, `refine`,
`refinement_mode`, `refine_closure`, `staleslice_guard`, `geometry`,
`stencil_topology`, `apply_stencil`, `reduce_faces`, `global_reduce`,
`distribute`, `migrate_mesh`, `refine_parallel`.

That is a substantial result on its own: **Tasks 1 and 2 are clean** —
`refinement_mode` and `refine_closure` both pass, so the `FaceField::UserBegin`
shift (risk point 1) and the closure patterns/tie-break/inverse are all sound.
`staleslice_guard` passing clears risk point 8. `distribute`, `migrate_mesh`,
`halo`, `apply_stencil` and `reduce_faces` all now run on a **`Conforming`**-mode
mesh by default and pass, so the wider face tuple does not disturb construction,
distribution, halo, geometry, or the operators (risk point 6 clear on HIP too).

**Failing:**

| Test | np1 | np≥2 | Nature |
|---|---|---|---|
| `refine_splitedges` (SERIAL) | Pass | **Hang** (100 s timeout) | D1 — **fixed** |
| `refine_splitedges` (HIP) | *unknown* — poisoned | *unknown* | D1 — **fixed**, np1–5 |
| `refine_conforming` (SERIAL) | **Fail, rounds 2–3** | **Abort** `std::out_of_range` | D2 — **fixed** (np1) / D3 — open (np≥2) |

**Never executed at all** (they sit after `refine_conforming` in ctest order and
the sweep never reached them): `conforming_migrate`, `loadbalance`, `io`,
`markquality_edge`, `markquality_curv`, `conforming_operators`,
`conforming_determinism`, `conforming_quality`, `markquality_conforming`.
**All nine were resolved by D4** — see that section for the verdicts; six pass,
three are the *Open failures* at the top of this file.

### The `refine_conforming` np1 output, in full — this is the primary evidence

```
test_refine_conforming: distributed conforming refinement (size 1)
  [Serial] adaptive round1 ok   (it=1 F=596  euler=2    closure=261/596 =0.438 |S| hist=[335,108,15,0] blue lo1/lo2=6/9   | control euler=-136 badInc=414  tjunc=138)
  [Serial] adaptive round2 FAIL (it=1 F=1013 euler=-136 closure=357/1013=0.352 |S| hist=[656,140,23,2] blue lo1/lo2=12/11 | control euler=-314 badInc=1031 tjunc=248)
  [Serial] adaptive round3 FAIL (it=1 F=1606 euler=-328 closure=493/1606=0.307 |S| hist=[1113,225,13,1] blue lo1/lo2=7/6  | control euler=-426 badInc=1474 tjunc=295)
  [Serial] empty-mask   ok (V=162 E=480 F=320 closureChildren=0)
  [Serial] uniform-mask ok (V=642 E=1920 F=1280 closureChildren=0 |S| hist=[1280,0,0,0])
```

(`[Default]` repeats `[Serial]` identically — the same two rounds fail.)

**Risk point 9 is confirmed, exactly as Task 7 predicted it:** round 1 is
conforming (`euler=2`), rounds 2 and 3 are not. Round 2's `euler=-136` is
*numerically identical* to round 1's hanging-node **control** — i.e. from round 2
on, the conforming mesh has degraded to exactly the hanging-node mesh's defect
count. The closure still fires (`|S| hist` has non-zero entries every round), it
just only closes the level jumps created *this* round, leaving every jump
inherited from an earlier round open. Non-vacuity holds throughout (the control
fails conformity every round), so the test is measuring something real.

The `empty-mask` and `uniform-mask` cases pass, but note they run on a **fresh**
mesh, not on the refined one — so `empty-mask` here is *not* the idempotence
probe. `conforming_determinism`'s idempotence case is, and it has not run yet.

---

## D1 — Fix the `refine_splitedges` np≥2 hang

**Status: DONE (2026-08-04).** `refine_splitedges` passes SERIAL and HIP at
np1–5. Two independent test-side defects, the second only visible once the first
was fixed. **The second one very likely explains D3 as well — read *What
landed* before starting D3.**

**Reproduce.**

```
probe.flux with: "tessera_test_refine_splitedges_SERIAL 2"
→ exit=124 (timeout at 100 s), no output captured
```

np1 passes in 3.8 s and prints a complete, plausible report:

```
[Serial] round1 ok (refining=46 splitEdges=138 keptOnlyDiscovered=0 phase2a: 960 total vs 138 refining-only, x6.96)
[Serial] round2 ok (it=1 faces=731 localMidsSum=267 phase2a: 1374 vs 273, x5.03)
[Serial] round3 ok (it=1 faces=950 localMidsSum=219 phase2a: 2193 vs 219, x10.01)
[Serial] round4 ok (it=1 faces=1124 localMidsSum=174 phase2a: 2850 vs 174, x16.38)
```

**Leading hypothesis: the hang is in the test, not in the library.**
`refine_splitedges` is pinned to `RefinementMode::HangingNode2to1` (Task 7A), and
`refine_parallel` — same mode, same `refine()`, same extended Phase 2 — **passes
at np1, 2, 3 and 4**. So the shared Phase-2 code path does not deadlock. What
`refine_splitedges` has that `refine_parallel` does not is its own multi-rank
reference machinery: the partition-free round-1 reference, the per-rank
kept-side-discovery reduction, and `checkSplitEdgeCoverage()`'s coordinator
round trip. A collective inside a rank-conditional branch (`if (rank == 0)`, an
early `continue` on an empty local set, or a `++fails` early-out that skips a
later `MPI_Allreduce`) deadlocks at np≥2 and is invisible at np1.

**Procedure.**

1. Read `tests/test_refine_splitedges.cpp` and list every MPI call. For each,
   confirm it is reached by all ranks unconditionally — in particular that no
   `MPI_Allreduce`/`allToAllV` sits inside a loop whose trip count is a local
   quantity, and that no failure path returns before a collective.
2. If that audit is clean, get a stack. Rebuild with
   `-DCMAKE_BUILD_TYPE=RelWithDebInfo` (already the default) and attach:
   inside a `flux batch` allocation, `flux run` the np2 job in the background,
   then `gdb -p <pid> -batch -ex bt` on both ranks. Cheaper first step: bisect
   by adding flushed per-phase prints to the test and re-probing with
   `--unbuffered`.
3. Also probe `tessera_test_refine_splitedges_HIP` at np1 and np2 — its D0
   verdicts were all poisoned and are unknown. If HIP np1 also hangs while
   SERIAL np1 passes, the cause is different and belongs to risk point 6, not
   here.

**Acceptance.** `refine_splitedges` passes SERIAL and HIP at np1–5. **Met.**

**What landed.**

The leading hypothesis above was right that the bug was in the test rather than
in the library, and right about the mechanism — *a collective inside a
rank-conditional branch* — but the audit in step 1 of the procedure misses it,
because the collective is not a bare `MPI_*` call: it is hidden in an **argument
to the `printf`**.

**Defect 1 — the hang.** `tests/test_refine_splitedges.cpp`, the rounds-2-4
loop:

```cpp
if ( rank == 0 )
    std::printf( "  [%s] round%d %s (it=%d faces=%lld ...",
                 ..., TesseraTest::globalOwnedFaces( mesh ), ... );
                 //   ^^^^^^^^^^^^^^^^ MPI_Allreduce, rank 0 only
```

`globalOwnedFaces()` (`tests/MeshInvariants.hpp:222`) is an `MPI_Allreduce`.
Rank 0 entered it at the end of round 1 and every other rank walked on into
round 2's `refine()`, whose first collective is Phase 1's `allToAllV` — a
deadlock. np1 cannot see it. Fixed by hoisting the call to a `const long long
gFaces` above the `if`, evaluated by every rank.

Localised by tracing: temporary per-rank `fprintf(stderr)` markers around every
phase of the test, run with `TESSERA_PROBE=1`. The decisive clue was that
rank 0's *round-1 print itself* never appeared while rank 1 had already printed
`round2 refine enter` — i.e. rank 0 was stuck **inside** the print, not before
or after it.

A scan of every `if ( rank == 0 )` block in `tests/` and `examples/` for a
collective (including ones reached through a helper) found **this as the only
occurrence**, so no sibling fix was needed.

**Defect 2 — a `std::out_of_range` abort, unmasked by the fix.** With the
deadlock gone, np2-5 got further and aborted in round 2 with
`unordered_map::at` on a non-zero rank — *the same exception as D3, in
`HangingNode2to1` mode, in a test with no closure at all.*

Root cause: `refine()` drops every ghost and clears the halo plans
(`src/Tessera_RefineParallel.hpp:926`), but its own Phase 3a needs the
**positions of both endpoints** of every midpoint the rank owns, in order to
interpolate that midpoint (`gid2lv.at( a )`, line 533). Midpoint ownership is
"lowest incident refining-face owner", so a rank can own the midpoint of an edge
one of whose endpoints it holds only as a *ghost* — and after the previous
`refine()` that ghost is gone. Refining twice with no rebuild in between
therefore throws at np >= 2. This is the already-documented README *Known
Issue* ("a distributed mesh must be re-haloed after `refine()`"), one step
sharper than it was recorded: the consequence is not just an inert
`haloExchange()`, it is a **throw from inside the next `refine()`**.

Fixed in the test, not the library, because the library's contract is already
the documented one and two other tests already follow it
(`test_conforming_determinism.cpp:289` and `test_conforming_quality.cpp:324`
both re-halo between rounds with exactly this idiom):

```cpp
std::vector<Rank> dest( mesh.numOwnedFaces(), static_cast<Rank>( rank ) );
migrate( mesh, halo, dest );   // identity: the Step-7 halo rebuild rides along
haloExchange( mesh, halo );
```

placed at the *end* of the round body — after every check and the print — so the
invariants still measure exactly what `refine()` produced, with no ghosts
present.

**Side effect, benign, worth knowing.** The identity `migrate()` permutes the
local face ordering, so round 2's children are assigned gids in a different
order, so a **gid-derived mask picks a different face set** in later rounds.
Visible as round 3's `localMidsSum` moving 219 -> 218 at np1. Nothing regressed:
the mesh is the same size (`faces=` is identical round-for-round at every rank
count) and all of this test's checks are reference-free coordinator-decided
ground truth, so they hold under any such permutation. But do **not** treat a
gid-masked test's per-round counts as rank-count invariants — that is
`conforming_determinism`'s job, and it uses a *geometric* mask for exactly this
reason.

**Result, np1-5 x {SERIAL, HIP}, all `exit=0`** (job `f3QHvwB5yGRu`).
Non-vacuity holds and strengthens with rank count — round 1's
`keptOnlyDiscovered` is 0 / 6 / 13 / 24 / 27 at np 1 / 2 / 3 / 4 / 5, so the
kept-side cross-boundary discovery that Task 3 exists to provide is genuinely
exercised. Global face counts are identical at every rank count
(458 / 731 / 950 / 1124 over the four rounds). Phase-2a message volume, for the
Task-8 measurement table: **x6.96, x5.03, x10.01, x16.38** total-vs-refining-only
over rounds 1-4 (rank-count independent).

**Also landed.** README *Known Issues*: the re-halo entry now states the
second-`refine()` throw, gives the identity-`migrate()` idiom as a code block,
and notes the gid-permutation side effect. The per-round `printf`s in this test
now `fflush( stdout )` so a later hang cannot swallow them (harness trap 2).

**Steer for D3 — read this before debugging D3 separately.** D3 is an
`unordered_map::at` abort on a non-zero rank at np>=2 in `refine_conforming`.
Defect 2 above is the *same exception, same rank-count threshold, same
round-2 onset*, and `test_refine_conforming.cpp`'s round loop (line 180) has
**no re-halo between rounds** either. So the first thing to try for D3 is adding
the identity-`migrate()` + `haloExchange()` idiom to that loop — it may well be
the whole of D3, and it is not a closure bug. Two more loops need the same
check when their turn comes: `test_conforming_determinism.cpp:516-517` refines
twice back-to-back with nothing in between, and `test_markquality_edge.cpp:361`.
D3's note that "the fix is to make the map complete (D2), not to soften the
`at()`" still stands for `midGid.at()` — but the map that is actually throwing
here is `gid2lv`, which is a different map and a different problem.

---

## D2 — Fix risk point 9: the split-edge map is this-round-only

**Status: DONE (2026-08-04).** `refine_conforming` passes SERIAL and HIP at np1:
all three adaptive rounds `euler=2`, plus empty-mask and uniform-mask. **The
central defect of Task 8, and it was three coupled defects, not one — read *What
landed*.** np≥2 is still D3.

**Reproduce.** `probe.flux` with `"tessera_test_refine_conforming_SERIAL 1"` →
`exit=1`, rounds 2 and 3 FAIL as quoted under D0.

**Root cause (established by reading the code in Task 7, confirmed by D0's
numbers).** Step 3b′ in `detail::refineImpl()`
(`src/Tessera_RefineParallel.hpp`) calls `closeFaces( newRed, midGid, … )`.
`midGid` is built entirely inside Phase 2 from **this round's** refinements: the
coordinator drops every edge with `!anyRefining` (see the 2b block, `if
( !a.anyRefining ) continue;`). But a kept face needs closing whenever its edge
is bisected **in the red layer**, which is a *persistent* condition.

Concretely: round 1 refines A, so its kept neighbour B's edge `(a,b)` is bisected
at `m` and B closes into two green children — correct. Round 2 un-closes B back
to red corners `(a,b,c)`; A's children are level 1, B is level 0, and the 2:1
fixpoint permits that, so B does not refine. In Phase 2a, B advertises `(a,b)`
with `refining = 0`; A's children advertise `(a,m)` and `(m,b)`, which are
*different keys*. `(a,b)` therefore has no refining incidence, is dropped, and is
absent from `midGid`. `closeFaces()` computes `|S| = 0` for B and passes it
through **unclosed** — the T-junction returns.

**Proposed fix (from Task 7; weigh it, do not apply blindly).** The persistent
split-edge map is recoverable **locally, with no communication and no new field**,
from the closure bookkeeping already stored. For a parent `P` with corners
`(a,b,c)` and its children, a parent edge is split **iff it does not appear as an
edge of any child**: green replaces `(a,b)` by `(a,m),(m,b)` while `(b,c)` and
`(c,a)` survive; blue removes two; red-closure removes all three; `|S| = 0`
removes none. The midpoint of a split parent edge `(x,y)` is then the unique
child corner `m` outside `{a,b,c}` with both `(x,m)` and `(m,y)` present as child
edges. So `unclose()` (in `src/Tessera_RefineClosure.hpp`) can return an
`EdgeKey → GlobalId` map alongside the red layer, and step 3b′ closes against
`midGid` **unioned with** it.

Three things to get right — the first two are called out in the design, the
third follows from D3 and is new:

1. The recovered map must be consulted for **kept faces only**. A face refined
   this round has its edges replaced anyway.
2. `countClosureChildren()` in step 3c must be given **the same union**, or the
   face-gid `MPI_Exscan` under-allocates and children collide.
3. The union must be applied in `refineLocal()`'s conforming path
   (`detail::refineLocalConforming()` in `src/Tessera_Refine.hpp`) too, not just
   the distributed one — `refine_closure` passes today only because its
   conforming case is a *single* `refineLocal()` call from a fresh mesh.

An alternative worth one paragraph of thought before committing to the local
reconstruction: have the Phase-2 coordinator treat a red edge with a **single**
incidence as split. It needs the midpoint gid, which only the finer side knows
and which the local reconstruction already has — so it is probably strictly
worse. Record whichever you choose, and why, in the *Decision record* of
`tasks/completed/conforming-refinement.md`.

**Acceptance.** `refine_conforming` passes all three adaptive rounds plus the
empty-mask and uniform-mask cases, SERIAL, np1. The non-vacuity control must
still **fail** conformity every round (`control euler` stays negative) — if the
control starts passing, the fix has made the mask too weak and the test is
proving nothing.

**Also update.** `tasks/completed/conforming-refinement.md`: rewrite the *Design →
Distributed algorithm* step 3b description and the *Phase 2 extension*
paragraph to describe the union, and add a Decision-record entry. Risk point 9
becomes a resolved finding. **Done.**

**What landed.**

The local reconstruction was the right call and the alternative is strictly worse
(Decision 8 in `tasks/completed/conforming-refinement.md` records why). But the closure union
by itself gets round 2 only, and **two further defects had to be fixed for round 3**;
both were invisible before the union existed, and both are about the *red engine*
rather than the closure. All three are in the library, none in the tests.

**Defect 1 — the union (risk point 9 proper).** `unclose()` now also returns
`UncloseResult::splitEdges`, recovered by the new `recoverSplitEdges()` in
`src/Tessera_RefineClosure.hpp`. `refineImpl()` captures it as `persistentSplit`
(step 0c) and `emplace`s it into Phase 2's `midGid` before `countClosureChildren()`
and `closeFaces()`; `refineLocalConforming()` seeds its `midpointOf` with it.
`RefineResult::midpoints` therefore publishes the union, which is what makes
`checkClosureInverse` (it re-closes from `res.midpoints`) agree without any test
change.

**The uniqueness rule in the proposed fix above is wrong as written** — worth
knowing, because it reads convincingly. "The midpoint of `(x,y)` is the unique child
corner `m ∉ {a,b,c}` with `(x,m)` and `(m,y)` both present as child edges" is
ambiguous for **blue**: in the `q0 < q1` diagonal the children are `(A,q0,C)`,
`(q0,B,q1)`, `(q0,q1,C)`, so for parent edge `(B,C)` both `q1` (correct) and `q0`
(via `(q0,B)` and `(q0,C)`) satisfy it. The repair is to require each half-edge to
belong to **exactly one** child: an edge shared by two children is a *diagonal* of
the parent's fan, and fan-*boundary* edges — which is what the two halves of a split
parent edge are — appear exactly once. Verified by hand against all four patterns
including both blue diagonals. `recoverSplitEdges()` aborts if the answer is not
unique, which is also a cheap detector for a closure family split across ranks.

**Defect 2 — a refining coarse face minted a second, coincident midpoint.** Not
predicted anywhere, and *pre-existing since Step 6b*. If `(a,b)` is bisected at `m`
and the coarse face `K` owning `(a,b)` refines, Phase 2 allocates a **fresh**
midpoint for `(a,b)` — the fine side's `m` is invisible to it, because the fine
faces carry `(a,m)`/`(m,b)`, which are different keys. Two coincident vertices, and
the mesh is cracked. Fixed by the same union: `midOf( a, b )` now finds the
persistent entry, so the split reuses `m` and the red layer becomes *conforming*
there (edge `(a,m)` gains its second incidence from `K`'s child). Consequence for
`closeFaces()`: a fresh red child's two inherited boundary half-edges can now be
**old** edges and so can legitimately be bisected this round, which is why the
`freshChild` assertion is no longer "|S| must be 0" but "no split edge may touch a
vertex created this round" — hence the new `firstNewVertexGid` parameter.

**Defect 3 — the 2:1 propagation was blind across a hanging node.** Also
pre-existing. Both coordinator rules that matter (Phase 1's level comparison, Phase
2's split decision) require **two** incident faces for an edge key, and a hanging
node leaves exactly one on each side: `(a,b)` is advertised only by the coarse face,
`(a,m)` only by one fine face. So a level jump across a hanging node was never
bounded, and round 3 hit `closeFaces()`'s "an edge of a red face is bisected more
than once" abort. Fixed with `forEachSubEdge()` in `refineImpl()`: a persistently
split edge is advertised to the coordinator as its **two half-edges**, in Phase 1
and Phase 2 both, so the coarse face meets its true neighbours. Visible in the run
as the fixpoint doing real work for the first time — `it=1/2/3` over rounds 1/2/3,
where it was `it=1/1/1` before.

**The subtle half of defect 3, which cost the second build.** A half must be
advertised with `refining = 0` **regardless of the advertising face's own mark**. A
refining coarse face bisects the *whole* edge, at the midpoint it already has, and
bisects neither half. Advertising a half as refining makes the coordinator mint a
midpoint *for the half* — a spurious refinement that propagates and lands as
exactly the same "bisected more than once" abort one round later. The first attempt
had this wrong and produced a plausible-looking round 2 (`euler=2`, `F=1494`) that
aborted in round 3; with it fixed, round 2 is `F=1222`. **A conforming round that
passes every check can still be over-refining — compare `F` against the
hanging-node control's growth, not just `euler`.**

**Result, np1, both backends** (jobs `f3QJLhrfQLKh`, `f3QJMiKVuhm9`):

```
round1 ok (it=1 F=596  euler=2 closure=261/596 =0.438 |S| hist=[335,108,15,0]  blue lo1/lo2=6/9   | control euler=-136 badInc=414  tjunc=138)
round2 ok (it=2 F=1222 euler=2 closure=557/1222=0.456 |S| hist=[665,194,39,13] blue lo1/lo2=19/20 | control euler=-314 badInc=1031 tjunc=248)
round3 ok (it=3 F=2386 euler=2 closure=1100/2386=0.461 |S| hist=[1286,392,84,16] blue lo1/lo2=47/37 | control euler=-426 badInc=1474 tjunc=295)
empty-mask   ok (V=162 E=480 F=320 closureChildren=0)
uniform-mask ok (V=642 E=1920 F=1280 closureChildren=0 |S| hist=[1280,0,0,0])
```

Non-vacuity holds and is not weakened: the control still fails conformity every
round and its defect counts still *grow* (414 → 1031 → 1474). **For D7:** the
closure fraction is essentially flat across rounds (0.438 / 0.456 / 0.461) and both
blue diagonals stay populated, which is the first real evidence that the closure
cost and the shape bound are round-independent — the premise D7 has to confirm.

**No regressions.** 14 spot-checked test instances all `exit=0`, chosen to cover
everything the change could touch: `refine_closure` (SERIAL+HIP, the
`refineLocalConforming` path), `refine`, `refinement_mode`, `refine_parallel`,
`refine_splitedges`, `migrate_mesh`, `halo`, `distribute`, `apply_stencil`,
`staleslice_guard` at np1–4 across both backends. `HangingNode2to1` is provably
untouched: `persistentSplit` is always empty there, so `forEachSubEdge()` degrades
to the old single advertisement and the union is a no-op — its Phase-2a volumes are
byte-identical (`x6.96 / x5.03 / x10.01 / x16.38`, unchanged from D1).

**Also landed.** `tasks/completed/conforming-refinement.md`: *Persistent split edges* section
under the distributed algorithm, step 0c/1/3/3b in the pseudocode, Decisions 8 and
9, risk point 9 marked resolved (with the original analysis preserved), and a new
*Known limit* recording that defects 2 and 3 remain in `HangingNode2to1` mode —
they cannot be fixed there without a new field or a new message round, and that
mode's tests cannot see them because they assert non-conformity anyway.

---

## D3 — Fix the `refine_conforming` np≥2 `unordered_map::at` abort

**Status: DONE (2026-08-04).** `refine_conforming` passes SERIAL and HIP at np1–5.
Two defects: D1's steer was right about the abort, and fixing it exposed a second,
**pre-existing library** defect in how `refine()` assigns edge gids. **Read *What
landed*.**

**Status when D3 started:** **D2 did not fix it and did not change it** — re-probed
after D2 at np2 and np3, still `std::out_of_range: unordered_map::at`, still on a
non-zero rank, still before any output flushes. So it is *not* the missing
persistent split edge; **start from D1's steer instead** (the missing re-halo in
`test_refine_conforming.cpp`'s round loop, line ~180 — `gid2lv.at()` in Phase 3a
needs both endpoint positions of every midpoint the rank owns, and one of them is
a ghost `refine()` already dropped). The "candidate sites" paragraph below was
written before D1 established that, and `midGid` — the map it points at — is now
complete, so treat `gid2lv` as the prime suspect rather than `midGid`.

**Reproduce.** `probe.flux` with `"tessera_test_refine_conforming_SERIAL 2"`:

```
terminate called after throwing an instance of 'std::out_of_range'
  what():  unordered_map::at
flux-shell[0]: FATAL: task-exit: task rank 1 ... exited (Aborted)
########## exit=134 ##########
```

Note it aborts on **rank 1**, and before any test output flushes, so the round
it dies in is unknown. Re-run with `--unbuffered` and an explicit `fflush` (or
`std::endl`) after each round's print to find out.

**Candidate sites.** Search for `.at(` on a midpoint or gid map:
`midOf()` in `detail::refineImpl()` (`midGid.at( keyOf( a, b ) )`) is the prime
suspect — it is an `at()` on the very map D2 shows to be incomplete. Also check
the `midpointOf` lambda handed to `closeFaces()` and any `.at()` in
`tests/MeshInvariants.hpp`'s new checks. Whatever the site, the *fix* is to make
the map complete (D2), not to soften the `at()` into a `find()` — an `at()`
throwing here is the map being wrong, and silently substituting `invalid_gid`
would turn a loud failure into a corrupt mesh.

Also note this fires at np2 but not np1, so there is a partition-dependent
component: a rank holding only the kept side of an edge is the case np1 cannot
produce. That is exactly what Task 3 built the extended Phase 2 to cover, so
check whether the failing lookup is on a rank that should have received the gid
in round 2c.

**Acceptance.** `refine_conforming` passes SERIAL and HIP at np1–5. **Met**, at
both tiers' expense of nothing: no test was relabelled, and one assertion got
*stronger* (below).

**What landed.**

**Defect 1 — the abort was exactly D1's re-halo defect.** `gid2lv.at()` at
`src/Tessera_RefineParallel.hpp:611` (Phase 3a) needs the *positions* of both
endpoints of every midpoint the rank owns; midpoint ownership is "lowest incident
refining-face owner", so a rank can own the midpoint of an edge one of whose
endpoints it holds only as a ghost — and `refine()` has already dropped every ghost.
`test_refine_conforming.cpp`'s round loop had no rebuild between rounds, so round 2
threw at np ≥ 2. Fixed test-side with the documented identity-`migrate()` +
`haloExchange()` idiom, applied to **both** the conforming mesh and the
hanging-node control, placed after every check and the print so the invariants still
measure exactly what `refine()` produced. `midGid` was never involved — D3's
original "candidate sites" paragraph pointed at the wrong map, as D1 suspected.

**Defect 2 — `refine()` minted edge gids from a local exscan.** Unmasked by
defect 1's fix: `empty-mask` then failed at **np3, np4 and np5** on the topology
**checksum** alone — every count, `euler`, conformity, closure-inertness and
`closureInverse` held. Diagnosis, and the reason this is worth reading: only the
**edge** gids moved, and by an exactly explicable amount. Step 3f gid'd edges from an
`MPI_Exscan` over `nLocalE` — each rank's *local* edge count, which includes the
boundary edges it holds but does not own (edge owner = min incident face owner, so
the duplicate always sits on the higher rank). Every non-last rank holding such a
duplicate therefore opens a **gap** in the owned gid space, shifting every later
rank's block up: 23 / 44 / 76 gids pushed above the global max at np3 / np4 / np5,
exactly the boundary-edge count of ranks 1..n-2. np1 and np2 pass structurally —
at np2 every duplicate lands on the last rank, where it does no harm.

Worse than the gaps: the two sides of a boundary edge got **different gids for the
same edge** (each from its own rank's block). The old step-3f comment stated this as
if it were fine ("boundary edges carry distinct gids per rank but are matched by
EdgeKey"). It is not fine — `distribute()`, `MeshBuilder`, `migrate()`'s gid-keyed
`eById`/`eOwner` maps and the HDF5 writer/reader all treat an edge gid as a *global*
identifier, and vertex and face gids already are one. Edges were the only kind whose
gids were locally minted.

Fixed in the library, **for free**: each `EdgeKey` is routed to exactly one
coordinator, so the coordinator numbers its own distinct keys densely from an
`MPI_Exscan` over its key count and returns the gid in the reply of the round trip
step 3e already performs for ownership and level. One extra field on `EdgeOwnMsg`,
no extra message round. Refined edge gids are now dense in `[0, globalEdges)` and
identical on every rank holding the edge. Recorded as **Decision 10** in
`tasks/completed/conforming-refinement.md`, and `docs/design.md`'s refine section now
describes the coordinator as assigning the gids.

**One assertion got stronger, deliberately.** The gid half of `empty-mask` was a
global **XOR** of owned gids (`topologyChecksum`), which is permutation-invariant and
cancels: at np4 and np5 it was **passing while the gid set was in fact wrong** —
shifting a contiguous, even-sized block by a fixed amount can cancel in XOR. It now
also compares the globally gathered, sorted owned-gid **set** per kind, and on
failure prints which kind and which gids were lost/gained. `case_empty` also reports
*which* condition failed (a named bit per condition) instead of only bumping a
count; that is what turned this from "np3 fails somehow" into a one-line diagnosis.
Had the weak checksum not been strengthened, defect 2 would have shipped hidden at
two of the five gate rank counts.

**Result** (final probe job `f3QJwCBao2jR`; gate job `f3QJgiXFh3Ef`):
`refine_conforming` `exit=0` at np1–5 × {SERIAL, HIP}, all three adaptive rounds
`euler=2`, plus empty-mask and uniform-mask. Non-vacuity intact — the
hanging-node control still fails conformity every round with growing defect counts
(414 → 1031 → 1474 at np1). The adaptive-round numbers are **byte-identical before
and after the edge-gid change** at every rank count, which is the cleanest evidence
that the change moved gids only and not a single refinement decision.

**No regressions.** The full gate (`regression` × {SERIAL, HIP} × ranks 1–5) was
run rather than spot-checks, because the edge-gid change is in the path *both* modes
share: 130 instances, everything green through test #172 except the three
never-before-executed conforming tests recorded under *Open failures*, whose
failures were shown to be pre-existing by rebuilding with the change stashed.
`loadbalance`, `io`, `markquality_edge` and `markquality_curv` — all of which
consume edge gids — pass at ranks 1–5 on both backends for the first time.

---

## D4 — Re-sweep and get a first verdict on the nine never-executed tests

**Status: DONE (2026-08-04).** All nine have verdicts; **six pass, three fail**, and
the three failures are the ones already tabulated under *Open failures*. Seven of the
nine were answered by D3's gate run; D4 itself ran the last two —
**`conforming_quality` and `markquality_conforming` both pass at np1–5 on both
backends**, first execution ever, no code change required. **D4 changed no code.**
The suite now has **zero unexecuted registrations** (28 of 28), so D5–D7 are scoped
for the first time — and D5's scope shrinks to one test.

Nine registrations had **never been executed**: `conforming_migrate`,
`loadbalance`, `io`, `markquality_edge`, `markquality_curv`,
`conforming_operators`, `conforming_determinism`, `conforming_quality`,
`markquality_conforming`. Until they run, the size of the remaining work is
unknown and D5–D7 cannot be scoped.

**Procedure.** After D1–D3 land, resubmit
`flux batch scripts/tuolumne/run_conforming_tests.flux`. Expect it to need more
than one 20-minute allocation now that it will get further: if it times out
again, either raise `--time-limit` (pdebug's cap permitting) or split it — add a
`-R`/`-E` filter to run only the tests downstream of `refine_conforming`.

Record every verdict in the table under D0 and split the failures across D5–D7.
Do **not** start fixing before the whole list is in hand: Task 8's dependency
ordering exists because one upstream bug manifests as a dozen unrelated
downstream failures, and D2 in particular is expected to change most of these
verdicts on its own.

**What landed.**

No code change. D4 is a measurement task and the measurement came out clean.

**The two remaining tests, np1–5 × {SERIAL, HIP}, all `exit=0`** (jobs
`f3QJyix3uKMH` SERIAL, `f3QJyj547vZm` HIP). The sweep script was not reused: with
only two registrations left, two `probe.flux` jobs — one per backend, ten `flux run`s
each at a 110 s `timeout` — answered it in **1.4 minutes per job** rather than
contending for a 20-minute allocation. Both jobs ran concurrently on separate pdebug
nodes, which is worth knowing as a harness fact: `flux batch`ing two 1-node exclusive
jobs does not serialise them.

`markquality_conforming` (SERIAL, identical at every rank count and on HIP):

```
markquality/edge ok: conforming marked=25 F=430 closure=50 euler=2 badInc=0 tjunc=0 | control marked=25 F=395 euler=-3  badInc=25  tjunc=0
markquality/curv ok: conforming marked=35 F=500 closure=95 euler=2 badInc=0 tjunc=0 | control marked=35 F=425 euler=-33 badInc=115 tjunc=25
```

Non-vacuity holds on both criteria: the `HangingNode2to1` control fails conformity
(`euler=-3`, `-33`) with the *same* mark count, so the difference is the closure and
nothing else. Task 6's specific worry is also visibly answered — `CurvatureCriterion`
now sees two incident faces where the control leaves 25 T-junctions, and it still
marks exactly 35 faces, so the criterion's edge coordinator is not disturbed by the
wider face tuple. This closes **risk point 7's marking half** and completes D5's
`markquality_conforming` item; **D5 reduces to `conforming_migrate` alone**, since
`io` already passed at ranks 1–5 in D3's gate.

`conforming_quality` (8 adaptive rounds on a shrinking geodesic cap; `unit` tier, so
outside the gate by design):

| round | marked | F | minAngle° | maxQ | closure | closureFrac |
|---|---|---|---|---|---|---|
| 1 | 6 | 344 | 26.094 | 1.5672 | 12 | 0.0349 |
| 2 | 12 | 392 | 25.987 | 1.5672 | 36 | 0.0918 |
| 3 | 24 | 476 | 25.987 | 1.5672 | 60 | 0.1261 |
| 4 | 28 | 576 | 25.987 | 1.5672 | 92 | 0.1597 |
| 5 | 46 | 732 | 25.987 | 1.5672 | 128 | 0.1749 |
| 6 | 62 | 944 | 25.987 | 1.7759 | 176 | **0.1864** |
| 7 | 88 | 1232 | 25.987 | 1.7759 | 224 | 0.1818 |
| 8 | 134 | 1672 | 25.987 | **2.2344** | 292 | 0.1746 |

`inv=0` (closure-inverse mismatches) every round. All three provisional bounds hold
with wide margin — worst minAngle 25.987° vs 20°, worst `Q` 2.2344 vs 4.0, worst
closure fraction 0.1864 vs 0.50.

**Rank-count invariance is total, and this is the strongest single result in D4.**
Across np1–5 × {SERIAL, HIP} × {`[Serial]`, `[Default]`} — 40 executions of the
8-round loop — there are exactly **8 distinct round lines**, i.e. every printed
quantity above is byte-identical at every rank count on both backends. Same for
`markquality_conforming`. That is not something D4 had to be true; it says the
geometric mask, the closure patterns and the blue tie-break all reproduce across
partitions on this workload.

**Steer for D7 — the honest reading, which is not "the bounds are confirmed".**
Two of the three quantities are bounded on this evidence and one is **not yet shown
to be**:

- `minAngle` is *dead flat* at 25.987° from round 2 on. Round 1's 26.094° is the
  undisturbed icosphere; 25.987° is the worst shape the closure patterns can make,
  reached immediately and never worsened. This is exactly the round-independence D2's
  flat closure fraction hinted at, now confirmed over 8 rounds.
- `closureFrac` **peaks and then declines** (0.1864 at round 6, then 0.1818, 0.1746).
  A peak followed by a decline is real evidence of a bound, not just a slow climb.
- `maxQ` **grows monotonically, in steps, and has not plateaued**: 1.5672 (rounds
  1–5) → 1.7759 (6–7) → 2.2344 (8). The step pattern — flat for several rounds, then
  a jump — reads like a new worst-case nesting appearing each time the cap's refined
  region gains a level, and the *last* observed round is a jump. **8 rounds cannot
  distinguish a bounded sequence from an unbounded one here.** Do not calibrate the
  `Q` bound down to ~2.5 on this data; extend the test to 12–16 rounds first and see
  whether `maxQ` plateaus. If it keeps stepping, that is the design finding D7 is
  told to stop and report, not a tolerance to loosen — and it would bear directly on
  D8's default-flip decision.

Also for D7: the test is **stable across repeated runs** in the sense the promotion
criterion asks about (identical output in 40 executions), and it costs ~4 s, so
nothing about cost or flakiness blocks promotion to `regression` — only the
unresolved `Q` question does. Promotion still needs explicit user confirmation of
backends and ranks (`CLAUDE.md`).

**Cross-check that nothing was missed.** `ctest -N` lists **28 distinct
registrations** (72 `unit` instances + 130 `regression` instances, all
SERIAL|HIP). D0's 19 (17 passing + `refine_splitedges` + `refine_conforming`) plus
D4's 9 is exactly 28, so the "never executed" list was complete and is now empty.
Caveat for D8, stated so it is not mistaken for full coverage: 14 of the 15 `unit`
registrations have their evidence from **D0's np1–4** sweep, not from a post-D3 run —
D8's `ctest -L unit` is what confirms them at current HEAD.

---

## D5 — Downstream conforming tests: `conforming_migrate` (~~`io`~~, ~~`markquality_conforming`~~)

**Status: DONE (2026-08-04).** `conforming_migrate` passes SERIAL and HIP at np1–5.
**One defect, and it was D1's re-halo defect for the third time** — a two-line
test-side fix, no library change. All four measurements D5 was told to collect are
below, including the predicted +32 B per face, which lands *exactly*. **Scope was
reduced to `conforming_migrate` alone** before D5 started. D4
unblocked this: `io` passes at ranks 1–5 on both backends (D3's gate run) and
`markquality_conforming` passes at np1–5 on both backends (D4), so **risk point 7 is
clear in both halves** — the writer/reader agree on the two closure datasets, and the
mask translation composes with both quality criteria. What remains is
`conforming_migrate`'s np2–5 `unordered_map::at` abort, i.e. risk points 3, 4 and 5.
Read D1's *What landed* first: the same exception with the same np≥2 threshold has
twice now been the missing re-halo between rounds, not a closure defect.

These exercise the closure composed with redistribution, persistence and
marking. Their known risk points, from
`tasks/completed/conforming-refinement.md` → Task 8:

- **Risk 3 — face gid collisions across rounds.** Retired parent gids are reused
  by un-close while children are allocated above the global max visible gid.
  Surfaces as duplicate gids in `checkOwnershipPartition` *only after* a
  `migrate()` brings two ranks together, so only in a multi-round test.
- **Risk 4 — non-local vertices named by closure children.** By design a closure
  child may name a vertex gid its rank does not hold until the halo is rebuilt.
  Anything assuming post-refine locality will abort or read garbage.
- **Risk 5 — sibling co-residency.** If `migrate()`'s round S fixup or
  `repairClosureCohesion()` misses a case, two ranks each restore the same
  parent → duplicate face, caught by `checkOwnershipPartition`. Non-vacuity:
  `conforming_migrate` fails if the fixup count is zero at ranks ≥ 2.
- **Risk 7 — I/O field-set mismatch.** Writer and reader disagreeing about the
  two closure datasets fails as a round-trip checksum mismatch, not a read
  error. `io` also asserts red-layer identity and then refines the read-back
  mesh once more.

Also collect here, from the test printouts: the `dest`-fixup counts (adversarial
and Zoltan2's), pre/post `loadBalance` max face count and max weighted load vs
ideal, and the actual on-disk size delta for a conforming vs hanging-node file
(predicted +32 B per owned face).

**What landed.**

**The defect — D1's re-halo defect, third occurrence.** The abort fired *before any
output*, and the reason is that it was not in either case at all: it was in the
shared fixture `refinedConformingMesh()`
([test_conforming_migrate.cpp:198-199](../../tests/test_conforming_migrate.cpp#L198-L199)),
which calls `refine()` **twice back-to-back with nothing in between** — the exact
shape D1 documented and D3 confirmed. `gid2lv.at()` in Phase 3a needs the
*positions* of both endpoints of every midpoint the rank owns, and across a
partition boundary one of those endpoints is a ghost that the first `refine()` has
already dropped. Fixed with the documented identity-`migrate()` + `haloExchange()`
idiom between the two rounds. Nothing in the library changed; **the whole of D5 is a
ten-line test edit** (plus two `fflush( stdout )` calls after the case prints, per
harness trap 2 — their absence is exactly why the abort looked like it had no
locus).

Worth stating plainly because D5's brief pointed elsewhere: **risk points 3, 4 and 5
were never broken.** The moment the fixture stopped throwing, every one of their
checks passed at every rank count on the first run — `checkOwnershipPartition` (risk
3, gid collisions across rounds, which only a post-`migrate()` multi-round test can
see), `owned1RingLocal` (risk 4, closure children naming non-local vertices) and
`checkSiblingCoresidency` with a *positive* fixup count (risk 5). Task 5's sibling
fixup and `repairClosureCohesion()` are correct as written. The lesson is the one D1
and D3 already taught, now with a third data point: **an `unordered_map::at` abort at
np≥2 in a conforming test has meant the missing re-halo every single time, and never
a closure defect.** Check the fixture for two `refine()`s before reading any of the
risk-point analysis.

**Result, np1–5 × {SERIAL, HIP}, all `exit=0`** (jobs `f3QK5ndGJFS7` SERIAL,
`f3QK5nkAauXD` HIP; baseline `f3QK3ZfePcoh` reproduced the abort at np2 and np3
first). `inv=0` — i.e. zero invariant failures out of
`checkSiblingCoresidency` + `checkOwnershipPartition` + `owned1RingLocal` +
`checkConforming` + `checkNoInteriorVertex` + `check21BalanceRed` + the topology
checksum, plus the three-plan ghost corrupt-resync — and `euler=2`, at every rank
count on both backends.

**Measurement 1–3 — fixups and load balance** (SERIAL; HIP identical except where
noted):

| np | F | siblingGroups | case A destFixups | maxFaces before→after | maxWeight before→after | ideal | case B destFixups |
|---|---|---|---|---|---|---|---|
| 1 | 1222 | 246 | 0 *(size 1: nothing to scatter)* | 1222→1222 | 911.0→911.0 | 911.0 | 0 |
| 2 | 1194 | 242 | **248** | 1194→604 | 893.0→447.0 | 446.5 | 4 |
| 3 | 1230 | 244 | **306** | 1230→417 | 908.0→304.0 | 302.7 | 20 |
| 4 | 1220 | 248 | **318** | 1220→307 | 902.0→226.0 | 225.5 | 14 |
| 5 | 1232 | 242 | **318** | 1232→251 | 914.0→184.0 | 182.8 | 25–26 |

Non-vacuity is strong in both cases. Case A's adversarial `dest = gid % size`
scatters **essentially every** sibling group — 248 fixups against 242 groups at np2 —
so the hazard the case exists to create is created in full. Case B needed no help:
Zoltan2 partitions by centroid and splits siblings **on its own** (4–26 fixups), which
is the prediction in `Tessera_Zoltan2Balancer.hpp:180` confirmed. The weighted load
lands within **0.6 %** of ideal at every rank count (447.0 vs 446.5; 184.0 vs 182.8)
against a bound of 2× ideal, so the per-parent weighting really is what makes the
sibling fixup load-neutral — the fixup moves faces without moving weight.

**Measurement 4 — on-disk cost, and the +32 B prediction is exact** (`io`, job
`f3QK7YNayTao`):

| np | redFaces | visibleF | closureChildren | h5 total | closure payload | share | B / visible face |
|---|---|---|---|---|---|---|---|
| 1 | 911 | 1222 | 818 | 201590 B | 39104 B | 19.4 % | **32.000** |
| 2 | 893 | 1194 | 804 | 197194 B | 38208 B | 19.4 % | **32.000** |
| 4 | 902 | 1220 | 827 | 201276 B | 39040 B | 19.4 % | **32.000** |

39104/1222, 38208/1194 and 39040/1220 are all exactly 32, at three different rank
counts — the two extra closure members of the conforming face tuple cost precisely
what the design predicted, and 19.4 % of the file.

**One thing not to mistake for a regression.** `F` now *varies with rank count*
(1222 / 1194 / 1230 / 1220 / 1232 at np1–5) where a naive reading would want it
invariant. That is D1's documented, benign side effect: the identity `migrate()`
permutes the local face ordering, so round 1's children get gids in a different
order, so round 2's **gid-derived** mask (`gidMask( mesh, 5 )`) selects a different
face set at each rank count. This test's assertions are all structural — partition
integrity, conformity, Euler, checksum-vs-itself — so they hold under any such
permutation, and `io` (which has always had the re-halo) prints the *same* F values,
1222 at np1 and 1194 at np2, which is the cross-check that the numbers are right.
**Rank-count invariance of per-round counts is `conforming_determinism`'s job, and it
uses a geometric mask for exactly this reason** — do not "fix" the F spread here.

**A cosmetic instability, recorded so D8 does not trip on it.** Case B's
`destFixups` differs by ±1 between backends and between the `[Serial]` and
`[Default]` passes at np3 and np5 (20 vs 21, 25 vs 26), and `maxWeight` with it
(184.0 vs 185.0). That is Zoltan2's partitioner, not the closure: it is a *reported
count*, no assertion depends on it, and the weight bound it feeds has a 2× margin
against a 0.6 % actual. Non-vacuity for case B needs only "Zoltan2 splits siblings
unaided", which holds at every rank count.

**No regressions to check for.** The change touches one test file and no library
code, so nothing outside `conforming_migrate` can be affected; D8's full gate is the
confirmation at HEAD.

---

## D6 — `conforming_operators` and `conforming_determinism`

**Status: DONE (2026-08-04).** Both pass SERIAL and HIP at np1–5 — **70/70 test
instances green** (jobs `f3QKnDU4D5qZ` SERIAL, `f3QKnDaehtYX` HIP), together with
`refine_closure`, `refine_conforming`, `conforming_quality`, `conforming_migrate`
and `markquality_conforming` re-run alongside them. **The suite now has zero open
failures.** Three defects, all test-side, plus one **design decision** (Decision 11)
that cost the most work and produced the most durable finding. **Read *What
landed*** — in particular, the geometric tie-break the brief and I both expected to
be the right answer turns out to be *impossible locally*, and the measurement that
shows why is worth not repeating.

**Original brief follows.**

**Status:** Not started, but **no longer blocked** — D4 gave both a verdict and D5
sharpens the steer. **Do the re-halo check first.** Both tests abort with
`unordered_map::at` at np≥2, and that abort has now been the missing
identity-`migrate()` + `haloExchange()` between two `refine()` calls **three times
running** (D1, D3, D5) and a closure defect zero times.
`test_conforming_determinism.cpp:516-517` refines twice back-to-back with nothing in
between, so it is the same shape D5 just fixed; check
`test_conforming_operators.cpp`'s fixture for the same. Do not start from the risk
points below until that is ruled out. **But note the np1 idempotence failure is a
separate, genuine defect** — no re-halo is involved at np1, so that one needs the
risk-point-9 reading in case 1 below.

`conforming_operators` is the payoff test — closed 1-ring fan, stencil topology,
`applyStencil` and `reduceVertexFromFaces` error at closure vs interior vertices
asserted separately. An empty closure-vertex set is a hard failure by design.

`conforming_determinism` has three cases and they need different judgement:

1. **Idempotence** — the sharpest probe of risk point 9, and the one case that
   should flip from failing to passing purely as a result of D2. If it still
   fails after D2, D2 is incomplete.
2. **Rank-count independence** vs an `MPI_COMM_SELF` reference, compared by
   quantised **position**. Read risk point 10 in
   `tasks/completed/conforming-refinement.md` before touching this. If it fails *only* on
   the visible layer while the red layer, the `|S|` histogram, the
   closure-vertex set and V/E/F all agree, and `blueDiagMismatch > 0`, that is
   **not** a bug in the risk-point-2 sense — it is the blue tie-break comparing
   midpoint gids that the `MPI_Exscan` ordered differently at different rank
   counts. That is a **design decision to make and record**: leave it
   (documenting that the visible layer is rank-count dependent up to blue
   diagonals) or replace the tie-break with a geometric rule. No *gid*-based
   rule can be rank-count stable. Record the decision in
   `tasks/completed/conforming-refinement.md`'s *Decision record* either way.
3. **Cross-mode equivalence under a uniform mask** — full topology, not counts.
   D0 already showed the uniform-mask closure is inert at np1
   (`|S| hist=[1280,0,0,0]`, `closureChildren=0`), which is a good sign for this
   case.

**What landed.**

**The re-halo steer was right, for the fourth, fifth and sixth time.** All three
aborts were the missing identity-`migrate()` + `haloExchange()` between two
back-to-back `refine()` calls, in three separate places:
`test_conforming_operators.cpp:311-312`, `test_conforming_determinism.cpp:516-517`
(case B's fixture, the site the brief named), and — not predicted anywhere —
**case C's round loop**, which refines *both* the conforming and the hanging-node
mesh twice with nothing in between. The tally over the whole of Task 8 is now
**six occurrences, zero closure defects.** Fixing case B moved the abort *past* the
idempotence print into case C, which is how the third one surfaced: **when an abort
moves rather than disappears, re-read the output for what now prints, don't assume
the fix failed.**

`conforming_operators` needed nothing else: `closureVerts` 311/301/322/318/318 at
np1–5 (non-vacuous by design — an empty closure-vertex set is a hard failure),
`fanBad=0/0`, `ringBad=0/0`, `stencilMaxErr` and `areaMaxErr` at closure vertices
identical to interior ones (0 exactly on Serial, ~7e-18 on HIP), and
`vArea == fArea` to 9 digits. **Risk points 3–5 and the operator half of 6 are
clear.**

**The np1 idempotence failure was a stale assertion, not a defect — and the
diagnostic found it in one run.** Following D3's lesson, the first thing I did was
give case B a named bit per condition instead of a bare `++fails`. It printed
`why=0x08: mids` with `dVis=+0 dRed=+0 dV=+0 dE=+0 dF=+0` — i.e. *every* structural
quantity identical and only `!res.midpoints.empty()` failing. That assertion
contradicts the contract D2 established: in `Conforming` mode
`RefineResult::midpoints` is deliberately the **whole** split-edge map of the red
layer, persistent hanging nodes included, and `RefineParallel.hpp:105-110` says so
outright. It also contradicted its own neighbour — `checkClosureInverse`, two lines
below, *needs* those entries to pass. Replaced with the correct and sharper
statement: `midpoints` must equal **exactly** the split-edge map recovered from the
pre-refine mesh, so an empty mask must have *invented nothing* rather than reported
nothing. **When two assertions in the same block disagree about a contract, one of
them is stale — check the library's own doc block before believing either.**

**Decision 11 — the blue tie-break stays gid-valued.** Case A then failed at
**np5 only**, with exactly the signature risk point 10 predicted:
`vis=DIFF blueDiagMismatch=4`, and `counts / hist / red / closureVerts /
parentMissing` all ok. np1–4 agree.

The brief and the user both preferred replacing the tie-break with a geometric
rule, so I implemented it in full: shorter diagonal with a positional
lexicographic fallback for exact ties, a `ClosurePosOf` position lookup threaded
through all six `closeFaces()` call sites, and the blue unit tests rewritten to
drive **both** diagonals from geometry (by cutting a different corner of the
reference triangle) plus a new case pinning that relabelling the midpoint *gids*
does not move the diagonal. **It does not work, and the reason is worth keeping:**

* The rule is simpler than it looks. With `q0 = (A+B)/2`, `q1 = (B+C)/2`,
  `dQ0C - dAQ1 = (3/4)(|C-B|^2 - |B-A|^2)`, so "shorter diagonal" is exactly
  "connect the midpoint of the **longer split edge**" — the standard rule, needing
  only the two split edges' lengths.
* It cannot be evaluated locally. `closeFaces()` runs on the **un-closed** red
  layer, whose corners come from closure children's `ClosureParentVerts`, and a
  child may name a vertex gid its rank does not hold — **the already-documented
  risk point 4**, which was benign only because the closure never needed positions.
  A 1-deep halo does not help: those are *parent* corners, not neighbours.
  Instrumented at np2, one `closeFaces()` call asked for **12** positions the rank
  did not have, including original icosphere vertices (gids 1, 3, 21, 41).

So: every gid-valued quantity is exscan-derived and positions are unreachable,
therefore **no purely local rule can be rank-count stable on the data the closure
currently has.** Making a geometric rule work needs communication — the split
edge's squared length carried in Phase 2's existing coordinator reply, contributed
by a rank holding both endpoints (the refining side always does; Decision 8's
*persistent* split edges have only a coarse incidence and need a separate answer).
Deferred, not rejected: it is one extra field on an existing round trip, the same
shape as Decision 10's fix. The implementation is preserved at
`/tmp/geometric-tiebreak-d6-keep.patch` (not committed) and the reasoning in
Decision 11 and in `Tessera_RefineClosure.hpp`'s header note, which used to claim
partition-independence in a way that quietly conflated the two senses.

**Two traps this exposed, both worth remembering:**

1. **The regression it caused was *invisible* in the structural checks.** With
   positions missing, `refine_conforming` np2 reported `euler=2`, correct `F`,
   correct `|S|` histogram, populated both diagonals — and still FAILed, on
   `checkClosureInverse` alone. A silently-wrong diagonal produces a mesh that is
   perfectly conforming and perfectly wrong. *`euler=2` does not mean the closure
   agreed with itself.*
2. **A `std::function` position lookup that returns a default on a miss is a
   footgun.** My first version returned the origin, which turned a hard failure
   into a plausible mesh. The instrumented version that *printed* the missing gid
   is what solved this in one run.

**What case A asserts now — narrowed, but not to nothing.** Rather than dropping
the visible-layer check, it asserts the sharper statement that the visible layer
differs **only** through blue diagonals, in both directions: a `vis` difference
with `diagMismatch == 0` fails (the closure diverged for some other reason), and a
`diagMismatch != 0` that leaves `vis` identical also fails (`blueDiag` is not
measuring what the closure emitted). Everything provably a function of the global
mesh still has to agree exactly. This still bites: at np1–4 `vis=ok` *requires*
`blueDiagMismatch=0`, which is what the run shows. np5 prints
`vis=DIFF-by-blue-diag blueDiagMismatch=4` and passes.

**Result, np1–5 x {SERIAL, HIP}, 70/70 `exit=0`.** Case A: `V=676 E=2022 F=1348`
and `|S| hist=[1820,60,20,0]` **byte-identical at every rank count**, `closureVerts=74`,
`parentMissing=0`. Case B: `siblingGroups` 246/242/244/248/242 (non-vacuous — there
is real closure to be idempotent about) with every structural delta zero. Case C:
`closureChildren=0` both rounds at every rank count, so the uniform-mask closure is
provably inert and the two modes agree bit-for-bit on face gids, corners and levels.

**Also landed.** `docs/design.md`'s closure section and `Tessera_RefineClosure.hpp`'s
header note now distinguish partition-independence from rank-count independence
instead of asserting the first and implying the second; README *Known Issues* gains
the blue-diagonal entry (with the practical consequence: do not compare a conforming
mesh's *visible* face set bitwise across rank counts — compare the red layer or
compare by position) and its stale "conforming is green at one rank only, several
tests never executed" entry is corrected to the current state. Risk point 10 is
marked resolved with its original analysis preserved.

**No library code changed in D6.** Every fix is test-side, so the only tests that
can be affected are the ones re-run above.

---

## D7 — `conforming_quality`: calibrate the provisional bounds

**Status: DONE (2026-08-05).** The bound **is** fixed in the round count: extended to
16 rounds, `maxQ` saturates at 2.5254 by round 11 and is flat through round 16 while
the mesh grows 4348 → 24608 faces. So this is *not* the design finding D7 was told to
stop and report. Bounds recalibrated from measurement, two new assertions added
(the red layer's own shape, and saturation itself), and the test **promoted to
`regression` × {SERIAL, HIP} × ranks 1–5** with explicit user confirmation. **No
library code changed.** Decision 12 in
[tasks/completed/conforming-refinement.md](conforming-refinement.md). **Read *What landed*** —
the per-pattern instrumentation is what turned "is `maxQ` bounded?" into a
one-run answer, and it is worth reusing.

**Original brief follows.**

**Status:** Not started, but **no longer blocked** — D4 ran it: green at np1–5 on
both backends, with the full per-round measurement table and a rank-count-invariance
result. **Read D4's *Steer for D7* before starting.** The short version: `minAngle`
and `closureFrac` are demonstrably round-independent over 8 rounds, but `maxQ` is
still *stepping upward at the last round measured* (1.5672 → 1.7759 → 2.2344), so
D7's first job is to extend the test to 12–16 rounds and find out whether it
plateaus — **not** to tighten the bound to the observed 2.23.

Task 7 registered this as **`unit`, deliberately outside the gate**, with bounds
derived from the patterns' ideal-parent geometry rather than measured: **min
angle ≥ 20°, radius ratio `Q` ≤ 4.0, closure fraction ≤ 0.50** over 8 adaptive
rounds on a shrinking geodesic cap.

Replace them with values justified by the measured per-round data the test
prints, then promote to `regression` at ranks 1–5 **only if** it is stable across
repeated runs. Promoting a test into the gate requires explicit user
confirmation of backends and ranks (`CLAUDE.md`).

**If the measured quality is genuinely unbounded with round count, that is a
design finding, not a tolerance to loosen** — stop, record it in
`tasks/completed/conforming-refinement.md`, and report it. The whole justification for
making the closure transient is that the bound is fixed.

**What landed.**

**The instrumentation came first, and it is what made this a one-run task.** D4's
handoff framed D7 as "run 12–16 rounds and see whether `maxQ` plateaus", which would
have produced a yes/no with no explanation. Instead the measurement was first split
by **which closure pattern emitted the face** — `|S|` is recoverable with no library
change, since a closed parent's `|S|` is (number of its children) − 1 and siblings are
co-resident — and extended to measure the **red parent** of every closure child
(reachable for the same reason) and the amplification `Q(child)/Q(parent)`. One
16-round run then answered not just *whether* the bound is fixed but *which family
moves and why*:

| round | F | minAngle | maxQ | closureFrac | `\|S\|`=0 (red) | `\|S\|`=1 (green) | `\|S\|`=2 (blue) | amp |
|---|---|---|---|---|---|---|---|---|
| 1 | 344 | 26.094 | 1.5672 | 0.0349 | 1.0278 | 1.5672 | — | 1.5426 |
| 5 | 732 | 25.987 | 1.5672 | 0.1749 | 1.0278 | 1.5672 | — | 1.5426 |
| 6 | 944 | 25.987 | 1.7759 | **0.1864** | 1.0278 | 1.5672 | 1.7759 | 1.7515 |
| 8 | 1672 | 25.987 | 2.2344 | 0.1746 | 1.0278 | 1.5672 | 2.2344 | 2.2310 |
| 10 | 3124 | 25.987 | 2.2344 | 0.1434 | 1.0278 | 1.5672 | 2.2344 | 2.2310 |
| 11 | 4348 | 25.987 | **2.5254** | 0.1316 | 1.0278 | 1.5672 | 2.5254 | 2.4906 |
| 12–15 | 6052–17296 | 25.987 | 2.5254 | 0.1143–0.0726 | 1.0278 | 1.5672 | 2.5254 | 2.4906 |
| 16 | 24608 | 25.987 | 2.5254 | 0.0623 | 1.0278 | 1.5672 | 2.5254 | 2.4906 |

Four things fall out that eight rounds could not show:

1. **The red layer — the closure's *input* — never moves at all.** `Q` = 1.0278 and
   min angle 54.397° in every one of the 16 rounds. So none of the movement in the
   aggregate is the red engine degrading; it is entirely the closure's own patterns,
   which is the split the aggregate number cannot make.
2. **Green never moves either** (1.5672 from round 1), and `|S|`=3 (red-closure) is
   **never realised on this workload** — worth knowing, because it means the
   red-closure pattern's shape is *not* covered by this test's evidence.
3. **Blue is the only family that moves, and it saturates.** 1.7759 (r6) → 2.2344
   (r8) → 2.5254 (r11), then identical for rounds 11–16 while F grows 5.7× and the
   marked set 6.2×. The step-then-flat pattern is a maximum over a **finite** set
   being progressively discovered — (red similarity class) × (which edges are split)
   × (which blue diagonal) — and each new combination the growing cap reaches can
   raise the maximum once. D4 was right to refuse to call it bounded at round 8: the
   last round it measured was itself a step, and the *next* step was still to come at
   round 11.
4. **The closure fraction peaks and then declines monotonically**, 0.1864 at r6 down
   to 0.0623 at r16 — an O(perimeter) set inside an O(area) mesh, now visible over
   enough rounds to be unambiguous rather than a two-point hint.

**Bounds, each the measured worst plus ~10%:** min angle ≥ **24.0°** (was 20.0),
`Q` ≤ **2.8** (was 4.0), closure fraction ≤ **0.25** (was 0.50), and a new
amplification bound ≤ **2.8**. The margin is thin on purpose and it is affordable
because there is no run-to-run spread to absorb (see the invariance result below).
Task 7's derivation was right in *direction* — blue is the worst family and green
lands below its ideal-parent value — and loose by ~1.6× on `Q`.

**Two new assertions, which are the real deliverable.** Calibrating numbers alone
would have left the *claim* ("the bound is fixed in the round count") as prose in a
comment with a printed table underneath it. It is now checked:

* **The red layer is bounded separately** (`Q` ≤ 1.10, min angle ≥ 50°). If this ever
  fires, the defect is in the red engine and the closure bounds are measuring someone
  else's damage — a distinction no aggregate can make.
* **Saturation:** the worst `Q` over the final 4 rounds may not exceed the worst over
  the preceding rounds (relative tolerance 1e-6, since the tail's worst face is a
  different face from the head's). **This is why the round count is 16 and not 12** —
  the last step is at round 11, so a 12-round test cannot assert saturation at all.
  The assertion is skipped, with a printed note, below 12 rounds.

**Also added:** `TESSERA_QUALITY_ROUNDS` overrides the round count, so the
round-independence claim can be re-measured to any depth without editing and
rebuilding — that is how the 16-round evidence was collected, and it is reproducible
in one command.

**Promotion, with the user's explicit confirmation** (`CLAUDE.md` requires it for
backends and ranks): `regression` × {SERIAL, HIP} × ranks 1–5. The gate
*definition* is unchanged — label, backends and ranks all already match — so none of
the four gate-definition files needed an edit. **Tier counts move: `regression`
130 → 140 instances, `unit` 72 → 62** (verified with `ctest -N -L`), which D8's
recount should start from.

**Result, np1–5 × {SERIAL, HIP}, 10/10 `exit=0`** (calibration run
`f3QTpdVZrSHu` / `f3QTpdcX74wh`; final binaries re-run at `f3QTtqth7Ypo` /
`f3QTtr1vg3aP`; the 16-round exploration was `f3QL4AAzVZYj`). 11–16 s per instance.
**Rank-count and backend invariance is total and now covers the new quantities:**
320 printed round lines over np1–5 × {SERIAL, HIP} × {`[Serial]`, `[Default]`} reduce
to exactly **16 distinct lines**. Non-vacuity is unchanged and still asserted — a
round that marked nothing or everything fails, and zero closure children over the
whole run fails.

**No regressions to check for.** One test file plus its CMake tier; no library code.
D8's full gate is the confirmation at HEAD — and it will now *include* this test.

---

## D8 — Close out Task 8

**Status: DONE (2026-08-05).** Gate **140/140** (job `f3QTwKZkEkSK`, 14.3 min), unit
**62/62** (job `f3QTwKgRBWzK`, 4.9 min), `format-check` clean on every touched file
under both clang-format binaries. **No code changed** — D8 is a verification and
close-out task and everything it checked was already green. `Conforming` stays the
default (Decision 13). **Read *What landed*** for the two harness facts and the one
number that moved since D2 and is *not* a regression.

**Original checklist follows.**

1. `flux batch scripts/tuolumne/run_regression_minset.flux` — the real gate,
   `regression` × {SERIAL, HIP} × ranks **1–5**. Green.
2. `ctest -L unit --output-on-failure` — green.
3. `cmake --build build-tuolumne --target format-check` — clean. Note that
   several *untouched* files (e.g. `src/Tessera_Geometry.hpp`) already violate
   clang-format v21 and that predates this work; check touched files under both
   the v21 (`/usr/bin`) and v19 (`/opt/rocm-6.4.2/llvm/bin`) binaries.
4. Recompute the **new test totals** (they were 70/70 regression and 28/28 unit
   before the conforming work; at HEAD `ctest -N -L` reports **140 `regression`
   and 62 `unit` instances**, D7's promotion having moved 10 across) and record
   them in the *Status* section of
   `tasks/completed/conforming-refinement.md` and in the `CLAUDE.md` task-log row.
5. Decide the **default flip**. Task 7 flipped `Mesh`'s `Mode` default to
   `Conforming` before anything had run. If conforming cannot be made green,
   revert the default to `HangingNode2to1` (keeping conforming opt-in and fully
   built) and record why in README *Known Issues* and under *Open failures*.
   Reverting is preferable to leaving a broken default.
6. Fill in Task 8's *Report back* in `tasks/completed/conforming-refinement.md`, including
   its deferred-measurements table, from the run output.
7. Housekeeping: `tuolumne2150-claude-3638145.core` is an untracked crash dump
   in the repo root from an earlier session — delete it or gitignore `*.core`.
   `.claude/settings.json` and `systems/tuolumne/spack.yaml` also had uncommitted
   modifications at the start of D0; check whether they should land.

**What landed.**

**No code changed.** Every check D8 was told to run came back green on the first
attempt, so this is a measurement and close-out record.

**1–2. Both tiers, at HEAD.**

| run | job | result | wall |
|---|---|---|---|
| `regression` × {SERIAL, HIP} × ranks 1–5 | `f3QTwKZkEkSK` | **140/140**, 0 failed | 14.3 min (842 s\*proc) |
| `ctest -L unit` | `f3QTwKgRBWzK` | **62/62**, 0 failed | 4.9 min (293 s\*proc) |

**202 instances over 28 registrations, zero failures.** The two jobs ran
**concurrently on separate pdebug nodes** (D4's harness fact, reconfirmed), so both
tiers cost 14 minutes of wall clock, not 19. Cost is dominated by
`conforming_quality`, which holds all eight of the slowest slots at 10.5–12.6 s;
nothing else exceeds ~8 s. This also retires D4's recorded caveat that 14 of the 15
`unit` registrations still had their evidence from D0's np1–4 sweep rather than from
current HEAD — `ctest -L unit` covers them at ranks 1–5 at HEAD now.

**A harness trap that cost the first submission, and is not in the list above.**
`run_regression_minset.flux` and `run_unit_minset.flux` both locate the resolver via
`source "${TESSERA_REPO}/scripts/lib/tessera_env.sh"` under `set -u`, so **the
submitting shell must export `TESSERA_REPO`** or the job dies in 7.6 s with
`TESSERA_REPO: unbound variable` (job `f3QTvj6QXJ2b`). The resolver itself
self-locates — `tessera_env.sh:32` falls back to its own directory — but the batch
script needs the variable to *find* the resolver, so the fallback cannot help.
Earlier sessions had it exported and never saw this. Submit with:

```bash
export TESSERA_REPO=/g/g20/stewartj/research-bridges/tessera-dev/Tessera
flux batch scripts/tuolumne/run_regression_minset.flux
```

The scripts were left alone — this is environment plumbing, not a gate-definition
issue, and the gate definition is single-sourced and must not be edited casually.

**3. `format-check`.** The whole-tree target fails on **`src/Tessera_Geometry.hpp`
only** (3 violations, lines 184/185/209), exactly as this brief predicted. Confirmed
pre-existing rather than assumed: that file is **not** in the conforming work's diff
(`git diff f50a91b^..HEAD`) and was last touched by `5dcb202`, which precedes Task 1.
All **34** `.cpp`/`.hpp` files the conforming work touched (Tasks 1–8) are clean under
**both** binaries — v21.1.8 (`/usr/bin`) and v19.0.0 (`/opt/rocm-6.4.2/llvm/bin`),
0 violations each. Checking the two versions separately matters because they disagree
about this codebase: v21 is what the `format-check` target uses and v19 is what a
ROCm-flavoured shell picks up first.

**4. Test totals.** `ctest -N -L` at HEAD: **140 `regression` + 62 `unit` = 202
instances**, over **14 + 14 = 28 registrations** — matching D7's handoff prediction
exactly. Before the conforming work (at `f50a91b^`) the suite was **70 `regression` +
28 `unit` = 98 instances over 19 registrations** (7 `regression` + 12 `unit`), so this
work added **9 registrations and 104 instances** — more than doubling the instance
count, with the gate itself going 7 → 14 registrations and 70 → 140 instances. The
nine new registrations are `refinement_mode` and `refine_closure` (`unit`), and
`refine_splitedges`, `refine_conforming`, `conforming_migrate`,
`conforming_operators`, `conforming_determinism`, `conforming_quality` and
`markquality_conforming` (`regression`). Recorded in `tasks/completed/conforming-refinement.md`'s
*Status* section and *Report back*, and in the `CLAUDE.md` task-log row. Worth noting
the `unit` tier is **not** symmetric across backends — 34 SERIAL + 28 HIP — because
two of its registrations, `global_reduce` (5 instances) and `staleslice_guard` (1),
are **SERIAL-only**, which is exactly the 6-instance gap. The gate tier *is*
symmetric, 70 + 70.

**5. The default flip: `Conforming` stays.** Decision 13 in
`tasks/completed/conforming-refinement.md`. The escape hatch existed for "if conforming cannot
be made green"; conforming is green at 202/202 with *stronger* assertions than the
hanging-node path had before this work, so its precondition never arose. The two
accepted limits are recorded in README *Known Issues* rather than being reasons to
revert: the blue-diagonal rank-count dependence (Decision 11), and the re-halo
contract — which is **pre-existing and mode-independent**, so reverting the default
would not have addressed it.

**6. *Report back* filled in**, including the deferred-measurements table, which now
carries a results column for all six tasks. Task 4's row is the only measurement D8
had to actually collect (job `f3QUMwrCeAU7`, `refine_conforming` at np1–5) — the rest
were already in D1/D5/D6/D7. The Task 4 result is worth having: **the closure
fraction is flat in the round and near-flat in the rank count** (round 1 is 0.438 at
every np; rounds 2–3 stay in 0.450–0.469 across all five rank counts) and equals
**≈3× the mask fraction** (round 1 marks 46 of 320 faces = 0.144). Task 2's row also
answers a question in the negative-sounding direction that is worth stating plainly:
the `closeFaces()` "red child of a refined face has `|S| = 0`" assert **did fire**,
and correctly — D2's defect 4 made a fresh red child's `|S| > 0` legitimate, so the
assert was *narrowed* rather than deleted.

**7. Housekeeping.** The `.core` crash dump was already gone; `*.core` is now
gitignored so the next MPI abort cannot be committed (`*.out` already was, which is
why none of the job logs show up as untracked). Both start-of-D0 modifications
landed. `systems/tuolumne/spack.yaml` **had to** land: it adds the `repos:` block that
the live env carries, and CLAUDE.md's invariant is that env snapshots stay in sync —
verified by diffing the snapshot against
`~/spack_envs/tuolumne_trilinos/spack.yaml` (identical after the change, divergent
before). It also documents the `SPACK_USER_CONFIG_PATH` / `$CLUSTER` non-login-shell
trap in a comment header, which is the same trap this file's *How to build and run*
section warns about. `.claude/settings.json` landed as-is at the user's explicit
direction, after I flagged that two of its entries (`Bash(bash -lc ' *)` and
`Bash(git stash *)`) are blanket wildcards in a committed allowlist and that the
first permits arbitrary command execution for any checkout.

**One number moved since D2, and it is not a regression.** `refine_conforming`
round 3 at np1 now reports `F=2372` where D2 recorded `F=2386` (and `|S|` hist
`[1278,388,82,18]` vs `[1286,392,84,16]`). Cause: **D3** added the identity-`migrate()`
+ `haloExchange()` re-halo to this test's round loop, and this test's mask is
**gid-derived** (`gidMask`), so the permuted face ordering assigns round-2 children
different gids and round 3 marks a different — equally valid — face set. That is D1's
documented benign side effect, and it does not contradict D3's "byte-identical before
and after" claim, which was about the *edge-gid change* measured in isolation against
a stashed build, not about the re-halo landing in the same commit. Round 3 is
`euler=2` at every rank count with the closure fraction unchanged at 0.450–0.469, and
the control still fails every round with growing defects. **Do not read D2's round-3
numbers as the current expected values** — D2 predates the re-halo.

**Two follow-ups this work identifies, neither in scope here.** Both are already
tracked in README *Known Issues* and in the design's decision record; naming them
together is the useful handoff:

1. **Factor the halo rebuild out of `migrate()` into a standalone `rebuildHalo()`
   that `refine()` calls.** This is the highest-value one: it caused **six of Task 8's
   nine defects**, it throws from *inside* the next `refine()` rather than failing
   visibly at the call the user got wrong, and every multi-round driver has to know
   the identity-`migrate()` idiom to work at all.
2. **Carry the split edge's squared length in Phase 2's existing coordinator reply**,
   which is what a geometric blue tie-break needs to be rank-count stable (Decision
   11). One extra field on a round trip that already happens — the same shape as
   Decision 10's fix. The implementation of the tie-break itself is preserved at
   `/tmp/geometric-tiebreak-d6-keep.patch` (not committed, and on scratch that may be
   reaped — D6's Decision 11 has the full reasoning if it is gone).

---

## Findings log

*(append-only)*

- **2026-08-05 — D8. Task 8 is closed and conforming refinement is done.** Gate
  **140/140** (`f3QTwKZkEkSK`, 14.3 min) and `ctest -L unit` **62/62**
  (`f3QTwKgRBWzK`, 4.9 min) at HEAD — **202 instances over 28 registrations, both
  backends, ranks 1–5, zero failures.** **No code changed**; every check came back
  green first attempt. The two jobs ran **concurrently on separate pdebug nodes**, so
  both tiers cost 14 min of wall clock rather than 19. `format-check`: the only tree
  violation is `src/Tessera_Geometry.hpp`, *proved* pre-existing rather than assumed
  (absent from `git diff f50a91b^..HEAD`, last touched by `5dcb202`, which precedes
  Task 1); all **34** touched source files are clean under **both** clang-format
  v21.1.8 (`/usr/bin`) and v19.0.0 (`/opt/rocm-6.4.2/llvm/bin`) — worth checking both
  because they disagree about this codebase. Totals moved **98 → 202 instances** and
  **19 → 28 registrations** (the gate itself 70 → 140 and 7 → 14), i.e. this work more
  than doubled the instance count; D7's predicted 140/62 split was exact. **`Conforming` stays the default (Decision 13)** — the
  escape hatch's precondition ("if conforming cannot be made green") never arose, and
  neither accepted limit is a reason to revert, since the re-halo one is
  *mode-independent and pre-existing*. **Harness trap worth remembering, which cost
  the first submission:** the committed batch runners `source
  "${TESSERA_REPO}/scripts/lib/tessera_env.sh"` under `set -u`, so **the submitting
  shell must export `TESSERA_REPO`** or the job dies in 7.6 s with `unbound variable`
  (`f3QTvj6QXJ2b`); the resolver self-locates at `tessera_env.sh:32`, but the script
  needs the variable to *find* the resolver, so that fallback can't help. **The one
  number that moved since D2 is not a regression:** `refine_conforming` round 3 at np1
  is now `F=2372`, not D2's `F=2386`, because **D3** added the re-halo to this test's
  round loop and its mask is **gid**-derived — D1's documented permutation effect. This
  does not contradict D3's "byte-identical" claim, which was about the edge-gid change
  measured against a stashed build, not the re-halo landing in the same commit; *treat
  a gid-masked test's per-round counts as valid-but-arbitrary, and don't read D2's
  round-3 numbers as current expected values.* Deferred measurements: only Task 4's
  needed collecting (`f3QUMwrCeAU7`) — **closure fraction is flat in the round and
  near-flat in the rank count** (0.438 at every np in round 1; 0.450–0.469 across all
  five rank counts in rounds 2–3) and is **≈3× the mask fraction** (46 of 320 faces =
  0.144), the O(perimeter) scaling that D7's 16 rounds then show *declining* to 0.0623.
  Task 2's row resolves in the direction easy to misreport: the `closeFaces()`
  "`|S| = 0` for a fresh red child" assert **did fire**, correctly, and was *narrowed*
  by D2 rather than deleted. Housekeeping: `*.core` now gitignored (dump already gone);
  `systems/tuolumne/spack.yaml` **had** to land — it carries the live env's `repos:`
  block and CLAUDE.md requires snapshots stay in sync, verified by diffing against
  `~/spack_envs/tuolumne_trilinos/spack.yaml`; `.claude/settings.json` landed as-is at
  the user's explicit direction after I flagged that `Bash(bash -lc ' *)` in a
  committed allowlist is effectively pre-approved arbitrary command execution. **Two
  follow-ups, in priority order:** (1) factor the halo rebuild into a standalone
  `rebuildHalo()` that `refine()` calls — it caused **six of Task 8's nine defects**
  and throws from *inside* the next `refine()` rather than at the call the user got
  wrong; (2) carry the split edge's squared length in Phase 2's existing coordinator
  reply so a geometric blue tie-break becomes possible (Decision 11).
- **2026-08-05 — D7.** `conforming_quality` recalibrated from measurement, extended
  from 8 rounds to **16**, and **promoted to `regression` × {SERIAL, HIP} × ranks
  1–5** with the user's explicit confirmation. Green 10/10, 11–16 s per instance
  (calibration `f3QTpdVZrSHu`/`f3QTpdcX74wh`, final `f3QTtqth7Ypo`/`f3QTtr1vg3aP`,
  16-round exploration `f3QL4AAzVZYj`). **No library code changed.** **The answer to
  D4's open question is that the bound IS fixed:** `maxQ` steps 1.7759 (r6) → 2.2344
  (r8) → 2.5254 (r11) and is then **identical for rounds 11–16** while F grows
  4348 → 24608 and the marked set 384 → 2388 — step-then-flat, i.e. a maximum over a
  finite set being progressively discovered, not drift with depth. D4 was right to
  refuse to call it bounded at 8 rounds: the last round it saw was itself a step and
  another step was still to come at r11. **The method is the transferable part.** The
  brief said "run 12–16 rounds and see whether it plateaus", which yields a yes/no
  with no explanation; instead the measurement was first split by **which closure
  pattern emitted the face** — free, since a closed parent's `|S|` is (its child
  count) − 1 and siblings are co-resident — plus the **red parent** of every closure
  child and the amplification `Q(child)/Q(parent)`. One run then said *which* family
  moves and why: the red layer, the closure's own INPUT, is **dead flat at Q=1.0278 /
  54.397° in all 16 rounds**, green is flat at 1.5672 from round 1, `|S|`=3 is *never
  realised on this workload* (so its shape is not covered by this evidence), and blue
  is the sole mover. *When a trend question has more than one possible cause, split
  the measurement by cause before extending the run — the aggregate number cannot
  tell you whether the closure degraded shape or the red engine did.* Bounds are now
  the measured worst plus ~10%: min angle ≥ **24.0°** (was 20), `Q` ≤ **2.8** (was
  4.0), closure fraction ≤ **0.25** (was 0.50, peak measured 0.1864 at r6 then
  declining monotonically to 0.0623 — O(perimeter) inside O(area), unambiguous over
  16 rounds where 8 gave only a hint), plus a new amplification bound ≤ **2.8**.
  Task 7's derivation was right in direction (blue worst, green below its
  ideal-parent value) and loose by ~1.6× on `Q`. **The real deliverable is two new
  assertions**, because calibrated numbers alone would leave the *claim* as prose:
  the red layer is bounded **separately** (Q ≤ 1.10, angle ≥ 50°), so a red-engine
  regression can never be mistaken for closure damage; and **saturation** — the worst
  Q over the final 4 rounds may not exceed the worst over the preceding rounds. *That
  second one is why the round count is 16 and not 12: the last step is at round 11,
  so a 12-round test cannot assert saturation at all* (it is skipped, with a printed
  note, below 12). `TESSERA_QUALITY_ROUNDS` now overrides the count so the claim is
  re-measurable to any depth without an edit. Invariance is total and now covers the
  new per-pattern quantities: **320 round lines → exactly 16 distinct** over np1–5 ×
  {SERIAL, HIP} × {`[Serial]`, `[Default]`}, which is what made a ~10% margin
  affordable. Gate *definition* unchanged (label/backends/ranks already matched), so
  no gate-definition file needed editing; tier counts move to **140 `regression` /
  62 `unit`** instances for D8's recount. Decision 12 records it; README's
  provisional-bounds Known Issue is deleted rather than reworded, and
  `docs/design.md`'s closure section now carries the measured table so the
  "similarity classes are bounded" claim cites numbers.
- **2026-08-04 — D6.** `conforming_operators` and `conforming_determinism` green
  SERIAL+HIP at **np1–5**; re-run alongside `refine_closure`, `refine_conforming`,
  `conforming_quality`, `conforming_migrate`, `markquality_conforming` for
  **70/70 instances green** (jobs `f3QKnDU4D5qZ`, `f3QKnDaehtYX`). **The suite now
  has zero open failures — 28/28 registrations pass.** No library code changed.
  Three defects plus one design decision. (1) All three aborts were the missing
  re-halo, in *three* places: operators' fixture, determinism case B's fixture (the
  site the brief named), and — unpredicted — **case C's round loop**, which refines
  both meshes twice with nothing between. Task 8's tally is now **six occurrences of
  this one defect and zero closure defects.** Fixing case B moved the abort *past*
  the idempotence print into case C: *when an abort moves instead of disappearing,
  re-read what now prints rather than assuming the fix failed.* (2) The np1
  idempotence failure was a **stale assertion**, found in one run by adding a named
  bit per condition: `why=0x08: mids` alone, every structural delta zero.
  `!res.midpoints.empty()` contradicts the contract D2 established — in Conforming
  mode `midpoints` is deliberately the whole split-edge map including persistent
  hanging nodes, as `RefineParallel.hpp:105-110` states — and contradicted
  `checkClosureInverse` two lines below, which *needs* those entries. Replaced with
  the sharper "invented nothing": `midpoints` must equal exactly the pre-refine
  recovered split-edge map. *When two assertions in one block disagree about a
  contract, one is stale — read the library's doc block before believing either.*
  (3) **Decision 11, and the most durable finding of D6:** case A failed at **np5
  only** with exactly risk point 10's predicted signature (`vis=DIFF`,
  `blueDiagMismatch=4`, everything else ok; np1–4 agree). The geometric tie-break
  everyone expected to be the fix was implemented in full — shorter diagonal with a
  positional tie fallback, a position lookup threaded through all six `closeFaces()`
  call sites, blue unit tests rewritten to drive both diagonals from geometry — and
  **cannot work locally.** The rule itself is clean (`dQ0C - dAQ1 =
  (3/4)(|C-B|² - |B-A|²)`, so it is just "connect the midpoint of the longer split
  edge"), but `closeFaces()` runs on the **un-closed** red layer whose corners come
  from `ClosureParentVerts`, and a child may name a vertex the rank does not hold —
  **risk point 4**, benign until something needed positions. Instrumented at np2:
  **12 missing positions in one call**, including original icosphere vertices. Since
  all gids are exscan-derived and positions are unreachable, *no purely local rule
  can be rank-count stable on the closure's current data*; a geometric one needs the
  split edge's length in Phase 2's existing coordinator reply (one extra field, same
  shape as Decision 10 — deferred, not rejected; patch kept out-of-tree). **Two
  traps:** the regression this caused was invisible structurally — `refine_conforming`
  np2 reported `euler=2`, correct `F`, correct `|S|` histogram, both diagonals
  populated, and failed on `checkClosureInverse` alone, because *a silently-wrong
  diagonal yields a mesh that is perfectly conforming and perfectly wrong*; and a
  position lookup that returns a default on a miss converts a hard failure into a
  plausible mesh — the version that *printed* the missing gid solved it in one run.
  Case A was narrowed but **not** gutted: it now asserts the visible layer differs
  *only* through blue diagonals, failing both a `vis` difference with no diagonal
  mismatch to explain it and a diagonal mismatch that leaves `vis` identical, so it
  still bites at np1–4 where `vis=ok` requires `blueDiagMismatch=0`. Evidence of
  strength elsewhere: case A's `V=676 E=2022 F=1348` and `|S| hist=[1820,60,20,0]`
  are byte-identical at every rank count, case C's `closureChildren=0` proves the
  uniform-mask closure inert, and `conforming_operators` shows closure-vertex stencil
  and area errors identical to interior ones with `closureVerts` 301–322.
- **2026-08-04 — D5.** `conforming_migrate` green SERIAL+HIP at **np1–5** (jobs
  `f3QK5ndGJFS7`, `f3QK5nkAauXD`; baseline `f3QK3ZfePcoh` reproduced the abort first).
  **One defect, no library change, a ten-line test edit.** The abort was D1's re-halo
  defect for the **third** time, and it was in neither case — it was in the shared
  fixture `refinedConformingMesh()`, which refines twice back-to-back with nothing in
  between, so `gid2lv.at()` in Phase 3a wanted a ghost endpoint position the first
  `refine()` had dropped. Fixed with the documented identity-`migrate()` +
  `haloExchange()` idiom, plus `fflush( stdout )` after each case print — their absence
  is why the abort appeared to have no locus. **Risk points 3, 4 and 5 were never
  broken:** the moment the fixture stopped throwing, `checkOwnershipPartition` (gid
  collisions across rounds), `owned1RingLocal` (closure children naming non-local
  vertices) and `checkSiblingCoresidency` with a positive fixup count all passed at
  every rank count on the first run, so Task 5's sibling fixup and
  `repairClosureCohesion()` are correct as written. *An `unordered_map::at` abort at
  np≥2 in a conforming test has meant the missing re-halo every time and a closure
  defect never — check the fixture for two `refine()`s before reading any risk-point
  analysis.* Non-vacuity is strong in both cases: case A's adversarial `dest = gid %
  size` scatters essentially every sibling group (**248 fixups against 242 groups** at
  np2; 306/318/318 at np3/4/5), and case B needed no help at all — **Zoltan2 splits
  siblings unaided** (4/20/14/26 fixups at np2–5), confirming
  `Tessera_Zoltan2Balancer.hpp:180`. `loadBalance` cuts maxFaces 1232→251 at np5 and
  lands the **weighted** load within **0.6 %** of ideal at every rank count (447.0 vs
  446.5; 184.0 vs 182.8) against a 2× bound — so the per-parent weighting is what makes
  the sibling fixup load-neutral. **The +32 B prediction is exact:** `io`'s closure
  payload is 39104 B / 1222 faces, 38208 / 1194, 39040 / 1220 — **32.000 B per visible
  face at three different rank counts**, 19.4 % of the file. Two traps for later: (a)
  `F` now varies with rank count (1222/1194/1230/1220/1232) because the identity
  migrate permutes face ordering and this test's round-2 mask is **gid**-derived —
  benign, D1 documented it, `io` prints the same F values as a cross-check, and
  rank-count invariance is `conforming_determinism`'s job with its *geometric* mask;
  (b) case B's `destFixups` wobbles ±1 across backends and passes (Zoltan2
  nondeterminism) — a reported count only, no assertion reads it. Open failures are
  down to **two**, both D6's, and D5's steer for them is: check the fixture first.
- **2026-08-04 — D4.** The last two never-executed registrations,
  `conforming_quality` and `markquality_conforming`, are **green at np1–5 on both
  backends on first execution** (jobs `f3QJyix3uKMH` SERIAL, `f3QJyj547vZm` HIP).
  **No code change.** The suite now has **zero unexecuted registrations — 28 of 28
  have verdicts, 25 pass, 3 fail**, and the three are the pre-existing np≥2 aborts
  already owned by D5/D6. D5's scope shrinks to `conforming_migrate` alone (`io`
  passed in D3, `markquality_conforming` passes here), so **risk point 7 is clear in
  both halves**. `markquality_conforming` is non-vacuous on both criteria — the
  control fails conformity (`euler=-3`, `-33`) with the *identical* mark count, and
  `CurvatureCriterion` still marks exactly 35 faces where the control leaves 25
  T-junctions, so seeing two incident faces instead of one does not disturb it.
  **The strongest result:** across np1–5 × {SERIAL, HIP} × {`[Serial]`, `[Default]`}
  — 40 executions of `conforming_quality`'s 8-round loop — there are exactly **8
  distinct round lines**, i.e. every measured quantity is byte-identical at every rank
  count on both backends. **The D7 handoff, stated carefully because the pass is
  easy to over-read:** `minAngle` is dead flat at 25.987° from round 2 on and
  `closureFrac` *peaks at round 6 (0.1864) and then declines*, so both are bounded on
  this evidence; but `maxQ` grows in **steps and has not plateaued** — 1.5672 (rounds
  1–5) → 1.7759 (6–7) → 2.2344 (8), with the last observed round being a jump. All
  three provisional bounds hold with wide margin (20°, 4.0, 0.50), yet *8 rounds
  cannot distinguish a bounded `Q` from an unbounded one*, so D7 must extend to 12–16
  rounds before calibrating, and an unbounded `Q` is the design finding it is told to
  stop and report rather than a tolerance to loosen. *A test passing all its bounds is
  not the same as its bounds being justified — read the trend, not just the verdict.*
  Harness facts: two `flux batch`ed 1-node exclusive jobs run **concurrently** on
  separate pdebug nodes (1.4 min each here), so for a handful of tests two per-backend
  `probe.flux` jobs beat re-running the 20-minute sweep. Caveat recorded for D8: 14 of
  the 15 `unit` registrations still have their evidence from D0's np1–4 sweep rather
  than from current HEAD.
- **2026-08-04 — D3.** `refine_conforming` green SERIAL+HIP at **np1–5** (probe
  `f3QJwCBao2jR`, gate `f3QJgiXFh3Ef`). Two defects. (1) The abort was exactly D1's
  re-halo defect — `gid2lv.at()` in Phase 3a needs both endpoint *positions* of every
  midpoint the rank owns and one is a ghost `refine()` already dropped; the round loop
  had no rebuild between rounds. Fixed with the identity-`migrate()` +
  `haloExchange()` idiom on both the conforming mesh and the control. `midGid` was
  never involved. (2) *Unmasked by that fix, pre-existing, library:* `refine()` gid'd
  its re-derived edges from an `MPI_Exscan` over each rank's **local** edge count,
  which includes the boundary duplicates a rank holds but does not own. Two
  consequences — the owned gid space gained a gap per duplicate on any non-last rank
  (an **empty-mask** refine changed the global owned edge-gid set at np3/4/5, by
  exactly 23/44/76 gids; np2 escapes because with two ranks every duplicate lands on
  the last rank), and the two sides of a boundary edge carried **different gids for
  the same edge** — which `distribute()`, `MeshBuilder`, `migrate()`'s gid-keyed edge
  maps and the HDF5 writer/reader all assume they do not, and which vertex and face
  gids already satisfied. Edges were the only kind gid'd locally. Fixed with **no
  extra message round**: each `EdgeKey` reaches exactly one coordinator, so the
  coordinator numbers its keys densely from an exscan over its key count and returns
  the gid in step 3e's existing ownership/level reply (Decision 10). **Trap worth
  remembering:** the check that should have caught this was a global **XOR** of owned
  gids, and XOR *cancels* — it was passing at np4 and np5 while the gid set was
  wrong, because shifting a contiguous even-sized block by a fixed amount is
  XOR-invisible. It now compares the gathered sorted gid **set** per kind and names
  the failing condition and the lost/gained gids; without that strengthening defect 2
  would have shipped hidden at two of the five gate rank counts. *A permutation- and
  cancellation-invariant checksum is not a set comparison — do not trust an XOR to
  detect a structured shift.* Adaptive-round numbers are byte-identical before and
  after the edge-gid change at every rank count (it moved gids, not decisions).
  Regression evidence is the **full gate** rather than spot-checks (both modes share
  the changed path): 130 instances green through #172 except the three
  never-before-executed conforming tests, which were proven pre-existing by
  rebuilding with the change stashed. D4 is now largely answered — `loadbalance`,
  `io`, `markquality_edge`, `markquality_curv` pass at ranks 1–5 both backends;
  `conforming_migrate` / `conforming_operators` / `conforming_determinism` abort at
  np≥2 (plus a np1 idempotence failure) and are D5/D6; only `conforming_quality` and
  `markquality_conforming` remain unexecuted.
- **2026-08-04 — D2.** `refine_conforming` green SERIAL+HIP at **np1** — all three
  adaptive rounds `euler=2` (jobs `f3QJLhrfQLKh`, `f3QJMiKVuhm9`). Risk point 9 was
  real and was **three coupled library defects**, all in the red engine's handling of
  a persistent hanging node, only the first of which Task 7 predicted. (1) The
  closure's split-edge map was this-round-only; `unclose()` now recovers the
  persistent map locally from the closure children (`recoverSplitEdges()`) and it is
  unioned into Phase 2's. Task 7's proposed uniqueness rule is **ambiguous for the
  blue pattern** and needed an "each half-edge belongs to exactly one child"
  qualifier — fan-boundary edges appear once, diagonals twice. (2) *Pre-existing:*
  a coarse face refining across a hanging node minted a fresh midpoint coincident
  with the fine side's, cracking the mesh; the same union makes it reuse the existing
  one, which also means a fresh red child can now legitimately have |S| > 0 on an
  inherited half-edge, so `closeFaces()`'s `freshChild` assertion was narrowed via a
  new `firstNewVertexGid` parameter. (3) *Pre-existing:* both coordinator rules need
  two incident faces for an edge key and a hanging node leaves one on each side, so
  the **2:1 propagation was blind across hanging nodes** and round 3 aborted on
  "bisected more than once"; a persistently split edge is now advertised as its two
  **half-edges** in Phases 1 and 2 (`forEachSubEdge()`). The fixpoint does real work
  for the first time: `it=1/2/3` over the three rounds, was `1/1/1`. **Trap worth
  remembering:** a half must be advertised with `refining = 0` even when the
  advertising face is refining — otherwise the coordinator mints a midpoint for the
  half, and the resulting over-refinement still passes every check for one round
  (`euler=2`, but `F=1494` where the correct answer is `F=1222`) before aborting in
  the next. *A conforming round that passes every check can still be over-refining;
  watch `F`, not just `euler`.* Non-vacuity unaffected (control defects still grow
  414→1031→1474). For D7: closure fraction is flat at 0.438/0.456/0.461 across
  rounds. No regressions in 14 spot-checked instances; `HangingNode2to1` is provably
  untouched (empty persistent map ⇒ both changes are no-ops, and its Phase-2a volumes
  are byte-identical). **D3 is unchanged by all of this** — re-probed, same abort, so
  follow D1's re-halo steer, not the `midGid` steer.
- **2026-08-04 — D1.** `refine_splitedges` green SERIAL+HIP np1-5 (job
  `f3QHvwB5yGRu`). Two test-side defects, the second hidden behind the first.
  (1) The hang: `TesseraTest::globalOwnedFaces( mesh )` — an `MPI_Allreduce` —
  was an *argument to a `printf` guarded by `if ( rank == 0 )`*, so rank 0
  deadlocked against everyone else's round-2 `refine()`. Hoisted above the
  branch. A sweep of every rank-0 block in `tests/` and `examples/` found no
  other instance. (2) Unmasked by that fix, an `std::out_of_range:
  unordered_map::at` in round 2 — **the same exception as D3, but in
  `HangingNode2to1` mode with no closure involved**: `refine()` drops all ghosts
  and clears the halo, yet Phase 3a needs both endpoint *positions* of every
  midpoint the rank owns, and across a partition boundary one of those endpoints
  is a ghost. So `refine()` twice with no rebuild in between throws at np>=2.
  Fixed test-side with the documented identity-`migrate()` + `haloExchange()`
  re-halo idiom (already used by `conforming_determinism` and
  `conforming_quality`); README's Known Issue sharpened to say the consequence
  is a throw, not just an inert `haloExchange()`. **This is very likely all or
  most of D3** — `test_refine_conforming.cpp`'s round loop lacks the same
  re-halo. Side effect: the identity migrate permutes face ordering and hence
  child gid assignment, so gid-derived masks select different faces in later
  rounds (round 3 `localMidsSum` 219 -> 218 at np1) — benign, but gid-masked
  per-round counts are not rank-count invariants. Non-vacuity strengthens with
  rank count (`keptOnlyDiscovered` 0/6/13/24/27 at np1-5); phase2a ratios
  x6.96/x5.03/x10.01/x16.38 for the Task-8 table.
- **2026-08-04 — D0.** Added `scripts/tuolumne/run_conforming_tests.flux` (both
  tiers × {SERIAL, HIP} × np1–4, 100 s per-test timeout, 1 pdebug node, 20 min).
  First execution of any Task 1–7 code. Job `f3QH9jz6D1aw` reached 86/165 before
  its wall limit; job `f3QHQtUmFtpX` re-probed the failures in isolation.
  **Result: 78 test instances pass** — all of Tasks 1 and 2 (`refinement_mode`,
  `refine_closure`), the whole pre-existing suite including `refine_parallel`
  and `migrate_mesh` at np1–4, and `staleslice_guard`. Risk points 1, 6 and 8
  are clear. **Three failures**, all downstream: `refine_splitedges` hangs at
  np≥2 (D1, probably in the test's own collectives — `refine_parallel` shares
  the same Phase 2 and passes), `refine_conforming` fails rounds 2 and 3 at np1
  (D2 — **risk point 9 confirmed exactly as predicted**, round 2's `euler=-136`
  equals round 1's hanging-node control), and `refine_conforming` aborts on rank
  1 at np2 with `std::out_of_range: unordered_map::at` (D3). Nine registrations
  never executed. Also learned three harness traps, documented above; the first
  — a ctest timeout leaving its `flux run` sub-job holding the exclusive node,
  which then times out every subsequent test — turned one real hang into eight
  bogus verdicts and is worth remembering.
