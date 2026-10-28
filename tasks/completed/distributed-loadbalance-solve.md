# Distributed load-balance solve

**Status:** COMPLETE. Implemented on branch `conforming-refinement`.
Check 4's verdict went **against** `Distributed`, so the default is `Sampled` —
see the progress log at the bottom for every measured number.

**Verified against `08dd346`** (branch `conforming-refinement`) — the code this task
cites was re-read at that commit.

## Problem

`computeLoadBalance()` is **rank-0-bound in the global face count**. It
`MPI_Gather`s every rank's owned-face count, then `MPI_Gatherv`s every centroid and
weight to rank 0, solves the whole partitioning problem there over a
`Teuchos::SerialComm`, and `MPI_Scatterv`s the part assignment back.

So rank 0's memory and solve time scale with the **global** face count while every
other rank idles, and the gather/scatter pair is a full global data movement on top
of the migration that follows. At the 16-rank, subdivision-4-ish scale the suite
tests, that is fine. At production scale — millions of faces, hundreds of ranks,
rebalancing every few steps — it is the bottleneck, and it is a memory ceiling on
one rank rather than a distributed cost.

**The rationale for rank-0-only is worth re-reading, because it does not say what it
looks like it says.** The header explains: MultiJagged "is not guaranteed
deterministic across ranks, so — following the same pattern [as
`Canopy_TreePartitioner.hpp`] — only rank 0 solves". That concern is about **each
rank solving the same problem independently and getting different answers**, which
would produce an inconsistent global partition. It does **not** apply to a **single
distributed solve** over an `MpiComm`: there is one solve and therefore one answer,
distributed by construction. The SerialComm choice was inherited from a
per-rank-solve pattern that this code does not use.

That observation is the whole task: the fix is probably to stop working around a
problem that is not present. It must be *verified*, not assumed.

**Why it matters.** An adaptively refining surface concentrates work: after a few
hundred steps a refined feature sits on a handful of ranks, and per-rank cost is
linear in local entity count, so throughput is set by the worst-loaded rank.
Rebalancing has to be cheap enough to do often. The driving consumer (the Beatnik
z-model, whose refined roll-up migrates across the surface over a long run) needs
exactly that.

## Approach

Add a mode parameter rather than replacing the existing path, so the old behaviour
stays available as the reference the test compares against.

```cpp
enum class LoadBalanceMode
{
    GatherRoot,   //!< today's behaviour: gather all, solve on rank 0, scatter back.
                  //!< Retained as the reference implementation and the fallback.
    Distributed,  //!< one Zoltan2 solve over a Teuchos::MpiComm on mesh.comm().
    Sampled       //!< rank 0 solves a deterministic sample to get cut planes,
                  //!< broadcasts them, each rank classifies its own faces locally.
};

template <class MeshT>
std::vector<Rank> computeLoadBalance( MeshT& mesh,
                                      double imbalanceTolerance = 0.05,
                                      LoadBalanceMode mode
                                          = LoadBalanceMode::Distributed );

template <class MeshT>
MigrateStats loadBalance( MeshT& mesh,
                          MeshHalo<typename MeshT::memory_space>& halo,
                          double imbalanceTolerance = 0.05,
                          LoadBalanceMode mode = LoadBalanceMode::Distributed );
```

### `Distributed` — the primary path

Replace the gather/solve/scatter with a single distributed solve:

- `Zoltan2::BasicVectorAdapter` over **this rank's own** owned-face centroids,
  weights and gids (use the real face gids from `ownedFaceGids( mesh )` as the
  adapter's global ids rather than the current `0..total-1` synthesized on rank 0 —
  they are already globally unique, which is exactly what the adapter wants).
- `Teuchos::rcp( new Teuchos::MpiComm<int>( Teuchos::opaqueWrapper( mesh.comm() ) ) )`
  instead of `SerialComm`.
- Same `ParameterList`: `algorithm = multijagged`, `num_global_parts = size`,
  `imbalance_tolerance`, `debug_level = no_status`. **Keep `multijagged` and never
  `rcb`** — Zoltan2's deterministic RCB breaks on Tuolumne (recorded in
  `Canopy_TreePartitioner.hpp`), and that finding stands regardless of the comm.
- `solution.getPartListView()` is then already in this rank's owned-face order; no
  scatter needed. Keep the returned `dest` in exactly the order
  `ownedFaceCentroids/Gids/Weights` produce, since `migrate()` depends on it.
- Keep the `size == 1` fast path.

**The risk to measure, not assume:** whether one distributed MultiJagged solve is
**deterministic run to run at a fixed rank count**. The determinism concern in the
existing header is about cross-rank agreement, which a single solve makes moot, but
run-to-run reproducibility is a separate property and Zoltan2 does not promise it.
Check 4 below measures it. If it fails, `Sampled` is the fallback and
`Distributed` gets a README *Known Issues* entry rather than being the default.

### `Sampled` — the deterministic fallback

If `Distributed` proves non-reproducible, or as a cheaper option at very large
scale:

1. Each rank selects a deterministic sample of its owned faces — every `k`-th face
   in **gid order** (not local order, so the sample does not depend on the current
   partition), with `k` chosen so the global sample size is a target like
   `64 * size`.
2. Gather the sample to rank 0 — `O(sample)`, independent of the global face count.
3. Rank 0 solves the sample with MultiJagged over `SerialComm`, then extracts the
   resulting **axis-aligned cut structure** (MultiJagged is a multi-section method,
   so the solution is a set of per-dimension cut coordinates) and broadcasts it.
4. Every rank classifies **its own** faces against the cuts locally, so the
   classification is exact arithmetic on the rank's own centroids and is trivially
   deterministic and reproducible.

This trades partition quality (the cuts are fitted to a sample) for determinism and
for a rank-0 cost independent of global size. Record the quality difference measured
in check 5 rather than asserting a bound guessed in advance.

### Non-goals

- Replacing Zoltan2 or adding a graph/hypergraph partitioner. Geometric only.
- Changing `migrate()`. This task produces a `dest` array; migration is unchanged.
- Removing `GatherRoot`. It stays as the reference and the fallback.
- The conforming-mode sibling-fixup behaviour, which is `migrate()`'s and is already
  documented (`ownedFaceWeights`' per-parent weighting keeps the perturbation
  load-neutral). Do not touch it; do assert it still holds (check 3).

### Tests

New `tests/test_loadbalance_distributed.cpp`, registered at **TIER `regression`**,
backends **SERIAL and HIP**, ranks **`${TESSERA_TEST_MPI_RANKS}` (1–5)**. Gate
promotion is **pre-authorized for this task**. Leave the existing
`test_loadbalance.cpp` registration alone as a `GatherRoot` regression guard.

The fixture that makes the test meaningful is a **deliberately skewed** starting
partition: build a subdivision-4 icosphere (5 120 faces) and `migrate()` everything
to rank 0, so the balancer has real work to do rather than being handed an
already-good partition.

1. **Validity.** For every mode: `dest.size() == numOwnedFaces()`, every entry in
   `[0, size)`. After `migrate( mesh, halo, dest )`:
   `checkOwnershipPartition`, `globalOwnedEuler == 2`, `owned1RingLocal`,
   `checkConforming`, and `haloExchange()` leaves ghost positions equal to owners'.
2. **Balance achieved.** From the skewed start, after `loadBalance` in `Distributed`
   mode, `max_r numOwnedFaces(r) <= (1 + tol + slack) * mean` with
   `tol = 0.05`; pick `slack` from the value **measured** in the first
   implementation run and record it here rather than guessing. Also assert the
   post-balance max is dramatically better than the pre-balance max (which is the
   whole mesh on rank 0), so the check cannot pass vacuously.
3. **Conforming mode.** Repeat checks 1–2 on a `refine()`d `Conforming` mesh.
   `MigrateStats::siblingFixups` may be nonzero — that is expected and documented,
   not a defect — so assert only that the invariants hold and report the count.
4. **Reproducibility — the deciding measurement.** Build two identical meshes in the
   same run, balance both in `Distributed` mode, and assert the two `dest` arrays
   are **element-wise identical**. Then repeat the whole test binary twice via ctest
   and compare a printed checksum of `dest` across the two runs (write it to stdout
   and have the test assert against a value it recomputes, not against a hardcoded
   one — the cross-run comparison is done by reading the two logs). If this fails,
   `Distributed` is not the default; see the exit criterion.
5. **Quality is not worse than `GatherRoot`.** Same input, both modes: the
   `Distributed` (and `Sampled`) imbalance ratio is no worse than `GatherRoot`'s
   plus a small margin. Do **not** assert the partitions are identical — different
   comms legitimately give different valid partitions. Report all three ratios.
6. **Rank-0 input size is bounded — the point of the task.** Instrument each mode to
   report the number of faces rank 0 receives for its solve. Assert:
   `GatherRoot` receives the global count; `Distributed` receives **zero** (no
   gather); `Sampled` receives `O(size)` and in particular fewer than
   `global / 4` at subdivision 4. Print all three. Without this check the test
   cannot distinguish the new path from the old.
7. **`Sampled` correctness.** Checks 1, 2 and 5 in `Sampled` mode, plus: the sample
   is chosen in gid order so the *same* mesh in two different starting partitions
   yields the **same** cuts — balance the skewed mesh and an axis-partitioned mesh
   of the same subdivision and assert the cut coordinates broadcast by rank 0 are
   identical. That is the property gid-order sampling buys, and it is the reason
   `Sampled` is deterministic where `Distributed` might not be.
8. **Single rank.** All three modes take the fast path, return all-zeros, and
   `loadBalance` is a no-op migrate.
9. **Idempotence.** `loadBalance` twice in a row: the second call's imbalance is no
   worse than the first's, and the number of faces that change owner on the second
   call is small (report it; assert below a fraction measured in the first run).
   A balancer that reshuffles everything on an already-balanced mesh is a
   performance bug that no invariant check catches.
10. **Scaling evidence.** At subdivision 5 (20 480 faces) and ranks 2–5, report
    wall time for the solve phase in each mode via the existing
    `TESSERA_SCOPED_TIMER_DETAILED( TIMER_LB_SOLVE )` instrumentation. Do not assert
    a time bound — machine-dependent and flaky — but do print it, and record the
    numbers in the progress log. This is the evidence the task worked.

## Exit criterion

- `test_loadbalance_distributed` green at **SERIAL and HIP, ranks 1–5**, and the
  full gate still green with nothing relabelled. `test_loadbalance` (the
  `GatherRoot` guard) still green.
- Check 6 passes: rank 0 gathers **zero** faces in `Distributed` mode. That single
  assertion is the task's deliverable.
- Check 4 has a **verdict**, and the default follows it:
  - reproducible → `Distributed` is the default, as written above;
  - not reproducible → the default is `Sampled`, `Distributed` remains available,
    and README *Known Issues* records that one distributed MultiJagged solve is not
    run-to-run reproducible, with the observed evidence.
  Either outcome is a successful completion; record which in the progress log with
  the measurement.
- Check 5 shows the new default's partition quality is not materially worse than
  `GatherRoot`'s, with all three measured ratios recorded here.
- Checks 9 and 10's measured numbers recorded in the progress log.
- README API section documents `LoadBalanceMode`, the default, and the trade-off
  between the three. The `Zoltan2_MultiJagged`-not-`rcb` note is preserved.
- `docs/design.md` *Load balancing* section is corrected: it currently reflects the
  rank-0-only design and its rationale. Rewrite it to state that the cross-rank
  determinism concern applies to per-rank independent solves and not to a single
  distributed solve, and record what check 4 measured about run-to-run
  reproducibility.

## Where this sits

Fully independent — no prerequisites and nothing depends on it. Complements
[distributed-coarse-build.md](distributed-coarse-build.md): that task removes the
replicated *build*, this one removes the gathered *partition solve*, and together
they leave no step whose cost scales with the global mesh on one rank. See the
ordering diagram in [halo-depth.md](halo-depth.md).

## Progress log

- 2026-08-07 — Task written, then re-checked against `08dd346` after pulling
  `../tessera`. Nothing implemented.
- 2026-08-10 — **Implemented and verified.** `LoadBalanceMode`
  (`GatherRoot`/`Distributed`/`Sampled`) plus a `LoadBalanceStats` out-parameter
  added to `computeLoadBalance()`/`loadBalance()` in
  `src/Tessera_Zoltan2Balancer.hpp`; the three solves share one
  `detail::lbMultiJagged<Dim>()` helper so `multijagged`, the parameter list and
  the `Dim`-generic adapter construction are single-sourced.
  `tests/test_loadbalance_distributed.cpp` registered at TIER `regression`,
  SERIAL + HIP, ranks 1–5 (gate is now **220/220**, nothing relabelled;
  `test_loadbalance` still green as the `GatherRoot` guard).

  **Check 6 — the deliverable — passes.** At subdivision 4 (global 5120 faces),
  faces rank 0 receives for its solve: `GatherRoot` **5120**, `Distributed`
  **0**, `Sampled` **128** (np2) to **320** (np5). Asserted, not just printed.

  **Check 4 — the deciding measurement — verdict: `Distributed` is NOT
  run-to-run reproducible, so the default is `Sampled`.** Two identically-built
  subdivision-4 icospheres balanced in one run, measured over **two ctest
  invocations × both backend registrations × both execution spaces** (8
  invocations per rank count): `Distributed` agreed at np1–np4 in every
  invocation and disagreed at np5 on **0, 4, 8, 16 or 18 of 5120** faces
  depending on the invocation. `Sampled` agreed exactly everywhere, and its
  printed `DEST_CHECKSUM` was **bit-identical in all 8 invocations at every rank
  count** — np1 `11499402685461970266`, np2 `3590200686806674594`, np3
  `7075996803214070821`, np4 `8549360387791187482`, np5
  `10270847821036491696` — which is the cross-**run** half of the check, done by
  comparing the two logs. Mechanism: MJ's cut coordinates come from floating-point reductions
  with no fixed partial-sum order, and an icosphere is symmetric enough that many
  centroids sit on a cut and flip on a last-bit change. Nothing about the
  partition is wrong (every invariant, the balance bound and the topology
  checksum hold in every run) — `dest` is simply not a function of the mesh
  alone. Recorded in README → *Known Issues* and `docs/design.md` → *Load
  balancing*. Check 4 now asserts zero mismatches for the default and **reports**
  `Distributed`'s count rather than asserting it, since asserting it would pin a
  property Zoltan2 does not have.

  **Check 5 — quality.** Face-count imbalance (max part / mean) of the `dest`
  each mode produces from the all-on-rank-0 start, subdivision 4, identical on
  SERIAL and HIP: np2 `GatherRoot` 1.0000 / `Distributed` 1.0000 / `Sampled`
  1.0125; np5 1.0000 / 1.0000 / **1.0615**. The default is therefore ~6% looser
  than the reference at np5 and equal at np2 — not materially worse, as the
  criterion requires. Asserted as `≤ GatherRoot + 0.15`; the partitions are
  deliberately not compared for equality.

  **Check 2 — balance slack, measured not guessed.** `slack = 0.10` over
  `1 + tol`, from a worst observed post-balance imbalance of **1.0615** at
  subdivision 4 over ranks 2–5 on both backends (excess over `1+tol` = 0.0115).
  Recorded as `kBalanceSlack` in the test with that derivation. A first draft
  also asserted `maxAfter <= NfG/2`, which is arithmetically impossible at np2
  (a perfect partition *is* `NfG/2`); it was removed in favour of the imbalance
  bound, which is the scale-correct statement at any rank count.

  **Check 3 — conforming.** Subdivision-3 icosphere, one `gid % 3` adaptive
  round, 3340 visible faces, then dumped onto rank 0 and balanced in
  `Distributed` mode: max owned faces 3340 → 1685 (np2) / 685 (np5), imbalance
  1.0090 / 1.0254, `siblingFixups` **18** (np2) / **22** (np5) — expected and
  reported, not asserted — and `rootSolveFaces` still 0. Every invariant holds
  (`checkSiblingCoresidency`, `checkOwnershipPartition`, `owned1RingLocal`,
  `checkConforming`, `check21BalanceRed`, `checkNoInteriorVertex`, Euler == 2,
  topology checksum unchanged, ghost corrupt/resync over both the vertex and the
  face plan).

  **Check 7 — `Sampled` cuts are partition-independent.** The same subdivision-4
  mesh in two different starting partitions (everything-on-rank-0 vs the axis
  partition) broadcasts **bit-identical** cut structures: 14 doubles at np2, 35
  at np5, stride 40 / 16, sample 128 / 320, diff 0. This required sorting the
  gathered sample **by gid** on rank 0 — the gather arrives in *rank* order,
  which is partition-dependent — and selecting the sample by `gid % stride == 0`
  rather than by local index.

  **Check 9 — idempotence, measured.** A second `loadBalance()` on the
  already-balanced subdivision-4 mesh moves **0 of 5120** faces under `Sampled`
  in **every** invocation and rank count, and **0 to 42 of 5120 (up to 0.0082)**
  under `GatherRoot` and `Distributed`, with the imbalance never worsening in any
  mode. Threshold set at `kIdempotenceFrac = 0.05` from those numbers.

  **A finding the task did not anticipate:** `GatherRoot` is not run-to-run
  reproducible either. Solving on one rank over a `SerialComm` removes the
  *communicator* but not the parallelism — MultiJagged is Kokkos-parallel and its
  reductions run on the default execution space whatever comm it is handed — so
  its `dest` varies between invocations by the same mechanism, visible in check
  9's `moved` count. That is why `Sampled` is the right default rather than
  merely the cautious one: it is the only mode measured reproducible, and the
  reason is structural (a broadcast cut structure plus exact local arithmetic),
  not incidental.

  **Check 10 — scaling evidence.** Subdivision 5 (20480 faces), max over ranks
  of the `computeLoadBalance()` wall time, SERIAL / HIP:

  | ranks | `GatherRoot` | `Distributed` | `Sampled` |
  |---|---|---|---|
  | np2 | 0.0552 / 0.0537 s | 0.0558 / 0.0521 s | 0.0517 / 0.0487 s |
  | np5 | 0.0745 / 0.0805 s | 0.0878 / 0.0988 s | 0.0678 / 0.0782 s |

  At this scale the three are within noise of each other, which is expected and
  is *not* the point: 20480 faces fit comfortably on one rank, so `GatherRoot`
  is not yet paying for the memory ceiling it imposes. What the task removes is
  the *scaling* term, and check 6 is what measures that directly — rank 0's solve
  input goes from `O(global)` to `0` or `O(nparts)`. The wall times are recorded
  as the baseline, not as evidence of a speedup at test scale.

  **Implementation notes worth keeping.** `Sampled` recovers the cut structure
  from MultiJagged's own `mj_keep_part_boxes` / `getPartBoxesView()` axis-aligned
  per-part boxes rather than from a private cut array, and classifies a centroid
  as: inside one box → that part; inside several (exactly on a cut) → lowest part
  id; inside none (MJ's boxes span the *sample's* bounding box, not the mesh's) →
  nearest box by squared distance, lowest id on a tie. The sample stride starts
  from the global **gid range** rather than the face count and halves by
  collective agreement until the sample can be partitioned, because live face
  gids are sparse in `Conforming` mode. Every adapter array is padded to at least
  one element so a rank owning nothing (routine after the skewed migrate) never
  hands Zoltan2 a null pointer, while the advertised length stays the true local
  count.
