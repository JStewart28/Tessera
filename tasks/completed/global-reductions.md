# Global reductions beyond `globalMin`

**Status:** DONE (2026-08-09). See the progress log at the bottom.

**Verified against `08dd346`** (branch `conforming-refinement`) — the code this task
cites was re-read at that commit.

## Problem

`Tessera_Reduction.hpp` provides exactly one collective: `globalMin`. A sum, a
max, and an "is this quantity finite everywhere" check are all missing, and the
library's own consumers already work around it:

- `tests/MeshInvariants.hpp` hand-rolls `MPI_Allreduce(MPI_SUM)` over
  `numOwnedVertices()/Edges()/Faces()` in `globalOwnedVertices`,
  `globalOwnedEdges`, `globalOwnedFaces` and `ownedEulerGlobal`.
- Several library internals hand-roll their own `MPI_Allreduce` guards
  (the 2:1 mark-propagation fixpoint, the acquisition-count early exits).

So the capability exists everywhere and is single-sourced nowhere. Any consumer
needs the same four one-liners, and each copy is a place where the wrong
`MPI_Datatype`, the wrong communicator, or a rank-0-only shortcut can hide.

`detail::mpiTypeOf` is also incomplete: it maps `double`, `float`, `int` and
`long` only. A count reduced as `long long` — the natural type for a global entity
count on a production-scale mesh — hits the `static_assert`-guarded
`MPI_DATATYPE_NULL` fallthrough and produces an MPI error at runtime rather than a
compile error, because `std::is_arithmetic<long long>` is true.

**Why it matters.** This is a missing convenience, not a missing capability — but
it is Tessera's to single-source, and the finiteness check in particular is not
something a consumer should be inventing. An evolving-surface solver (the driving
consumer is the Beatnik z-model) needs a global sum for enclosed volume, a global
min for an adaptive timestep, a global max for a CFL-style bound, and a global
all-finite check as its NaN tripwire — the last being the difference between
aborting on the step that produced the NaN and aborting fifty steps later with no
idea where it came from.

## Approach

All in `src/Tessera_Reduction.hpp`. Keep the existing style: thin, single-sourced
wrappers over one `MPI_Allreduce` on `mesh.comm()`, taking a per-rank scalar and
returning the global result on every rank. Tessera owns the collective; the caller
owns the local value.

### Step 1 — Complete `detail::mpiTypeOf`

Add `long long`, `unsigned long long`, `unsigned int`, `unsigned long`, `short`,
`char`. Turn the `MPI_DATATYPE_NULL` fallthrough into a hard compile error for any
arithmetic type with no mapping: replace the trailing `static_assert(
std::is_arithmetic<T>::value )` — which passes for `long long` and lets the null
datatype escape — with a dependent-false `static_assert` in an unmatched
specialization, so an unmapped type fails to compile instead of failing at
runtime. This is a correctness fix, not a convenience: the current code returns a
null datatype for a perfectly ordinary type.

While here, note in the header that the `if` chain over `std::is_same` is resolved
at runtime today; a small `struct MpiType<T>` specialization set is both cheaper
and what makes the compile-time rejection above expressible. Convert it.

### Step 2 — The three new scalar collectives

```cpp
//! Collective sum of a per-rank scalar over the mesh comm. MPI_SUM is not
//! associative in floating point, so a double result is NOT bitwise reproducible
//! across rank counts or across runs on a GPU-partial-sum path. Integer sums ARE
//! exact and reproducible. Callers comparing results across rank counts must
//! account for this.
template <class MeshT, class Scalar>
Scalar globalSum( const MeshT& mesh, Scalar local );

//! Collective maximum. Mirror of globalMin.
template <class MeshT, class Scalar>
Scalar globalMax( const MeshT& mesh, Scalar local );

//! Collective logical AND of a per-rank "everything I hold is finite" verdict.
//! Returns true iff every rank passed true. Implemented as MPI_Allreduce(MPI_LAND)
//! on an int, because MPI_CXX_BOOL is not universally available.
//!
//! This is the NaN/Inf tripwire: the caller checks its own owned data (Kokkos
//! reduction over the fields it cares about) and this call turns a local verdict
//! into a global one, so every rank aborts on the same step rather than one rank
//! diverging silently.
template <class MeshT>
bool globalAllFinite( const MeshT& mesh, bool local_all_finite );
```

The reproducibility warning on `globalSum` is the important part of the comment,
not boilerplate: the driving consumer carries a reduced enclosed volume and a
reduced minimum edge length for the whole run, and *every* adaptive timestep
scales off them — so a 4-rank run and a 5-rank run take slightly different
timesteps and diverge. A consumer that does not know this writes a cross-rank
bitwise comparison test and then spends a week on it.

### Step 3 — Single-source the owned-count reductions

Move out of `tests/MeshInvariants.hpp` into the library, beside the scalar
collectives:

```cpp
template <class MeshT> long long globalOwnedVertices( const MeshT& mesh );
template <class MeshT> long long globalOwnedEdges( const MeshT& mesh );
template <class MeshT> long long globalOwnedFaces( const MeshT& mesh );
//! V - E + F over owned entities. 2 for a closed conforming surface.
template <class MeshT> long long globalOwnedEuler( const MeshT& mesh );
```

Each is `globalSum( mesh, (long long) mesh.numOwnedX() )` — exact, since integer.
`MeshInvariants.hpp`'s versions become one-line forwards to these (keep the
names, so no test churn), and `ownedEulerGlobal` forwards to
`globalOwnedEuler`. Any library-internal hand-rolled `MPI_Allreduce` that is
literally one of these four is replaced.

### Non-goals

- A device-side "is this slice all finite" helper. The local verdict is the
  caller's Kokkos reduction over whichever fields it cares about; Tessera cannot
  know which. `globalAllFinite` takes the verdict, not the data. Say so in the
  header so nobody looks for the missing half.
- Reproducible (fixed-order / compensated) floating-point summation. Out of scope;
  the reproducibility caveat is documented instead. Record it in README *Future
  Optimizations* only after asking.
- Reductions over a slice or a field. Scalars only.

### Tests

Extend `tests/test_global_reduce.cpp` and **re-register it at TIER `regression`,
backends SERIAL and HIP, ranks `${TESSERA_TEST_MPI_RANKS}` (1–5)**. It is
currently a single-rank `unit` registration; the promotion is **pre-authorized for
this task**. Keep the existing `globalMin` checks verbatim as a regression guard.

Every check places the extremum or the odd rank **away from rank 0**, following
the existing file's stated intent — a rank-0 shortcut must not be able to pass.

1. **`globalSum`, exact integers.** `local = rank + 1` → `size*(size+1)/2`.
   `local = 1` → `size`. `local = -rank` → `-size*(size-1)/2`. Assert equality.
2. **`globalSum`, `long long`.** `local = (long long)rank * 1'000'000'007` →
   the closed-form sum. Catches the `mpiTypeOf` gap directly: this is the case
   that fails today.
3. **`globalSum`, double.** `local = 0.5` → `0.5 * size` exactly (dyadic, so
   exact in binary floating point at these sizes).
4. **`globalMax`.** `local = rank` → `size - 1` (max at the last rank).
   `local = size - rank` → `size` (max at rank 0). Integer and double overloads.
5. **`globalAllFinite`, positive.** Every rank passes `true` → `true` on every
   rank.
6. **`globalAllFinite`, negative, three ways.** The **last** rank passes `false`
   because its local check saw (a) a `NaN`, (b) a `+Inf`, (c) a `-Inf`; every
   other rank passes `true`. Every rank must receive `false` in all three cases.
   Build the local verdict with a real `std::isfinite` sweep over a small array
   containing the poisoned value, so the test exercises the intended usage rather
   than passing a literal `false`.
7. **`globalAllFinite` at one rank** returns the local verdict unchanged, both
   polarities.
8. **Owned-count reductions on a real mesh.** Build a subdivision-2 icosphere,
   `distribute`, and assert `globalOwnedVertices == 162`,
   `globalOwnedEdges == 480`, `globalOwnedFaces == 320`,
   `globalOwnedEuler == 2`, at **every** rank count 1–5. These are the numbers
   the closed form gives (`V - E + F = 2`, `3F = 2E` for a closed triangle mesh),
   so the check is against arithmetic, not against Tessera.
9. **Owned-count reductions after `refine()`.** Uniform refine of the same mesh:
   `V' = V + E = 642`, `E' = 2E + 3F = 1920`, `F' = 4F = 1280`, Euler still 2.
   Confirms the helpers survive a topology change and a generation bump.
10. **`globalMin` unchanged.** The three pre-existing checks still pass bitwise.

## Exit criterion

- `test_global_reduce` green at **SERIAL and HIP, ranks 1–5**, and the full gate
  still green with nothing relabelled.
- An unmapped arithmetic type (e.g. `long double`) passed to any of the four
  collectives **fails to compile**, and `long long` compiles and runs correctly.
  Verify the former once by hand and record the diagnostic in the progress log;
  do not commit a deliberately-broken test.
- `tests/MeshInvariants.hpp` contains **no hand-rolled `MPI_Allreduce`** for the
  four owned-count reductions, and no library internal duplicates them.
- README API section documents the four new collectives, and states the
  `globalSum` floating-point reproducibility caveat in the same place — that
  caveat is the one thing a consumer must read before writing a cross-rank test.
- `docs/design.md` gains a short *Global reductions* note: Tessera owns the
  collective, the caller owns the local value, and the finiteness check takes a
  verdict rather than data.

## Where this sits

Fully independent — no prerequisites and nothing depends on it. See the ordering
diagram in [halo-depth.md](halo-depth.md).

## Progress log

- 2026-08-07 — Task written, then re-checked against `08dd346` after pulling
  `../tessera`. Nothing implemented.

- 2026-08-09 — Implemented in full.

  **Step 1.** `detail::mpiTypeOf`'s `if`-chain over `std::is_same` replaced by a
  `detail::MpiType<T>` specialization set (`double`, `float`, `int`, `long`,
  `long long`, `unsigned int`, `unsigned long`, `unsigned long long`, `short`,
  `char`), exposed as a static function because `MPI_Datatype` handles are not
  guaranteed constant expressions. The unmatched primary template carries a
  dependent-false `static_assert` via `detail::AlwaysFalse<T>`. `mpiTypeOf<T>()`
  survives as a one-line forward, so no call site churned.

  Verified by hand (not committed as a test, per the exit criterion) with a
  throwaway TU calling `globalSum( mesh, (long double)1.0 )`:

  ```
  Tessera_Reduction.hpp:47:20: error: static assertion failed due to requirement
  'AlwaysFalse<long double>::value': Tessera: no MPI_Datatype mapping for this
  scalar type. Add a detail::MpiType specialization in Tessera_Reduction.hpp if
  the type is genuinely needed (note that long double has no portable MPI
  mapping).
     47 |     static_assert( AlwaysFalse<T>::value,
        |                    ^~~~~~~~~~~~~~~~~~~~~
  ...in instantiation of 'Tessera::detail::mpiTypeOf<long double>'
  ...in instantiation of 'Tessera::globalSum<FakeMesh, long double>'
  1 error generated.
  ```

  One error, naming the type and the fix. `long long` compiles and runs
  correctly (test 2 below).

  **Step 2.** `globalSum`, `globalMax`, `globalAllFinite` added. `globalMin`,
  `globalMax`, `globalSum` all route through one new
  `detail::meshAllreduce( mesh, local, MPI_Op )`, so the Allreduce itself is
  written once. `globalAllFinite` is `MPI_LAND` on an `int`.

  **Step 3.** `globalOwnedVertices/Edges/Faces/Euler` added to the library.
  `tests/MeshInvariants.hpp`'s four became one-line forwards (names kept, zero
  test churn) and hand-roll no `MPI_Allreduce`. One library internal duplicated
  one of the four — `Tessera_RefineParallel.hpp`'s owned-vertex sum feeding the
  midpoint-gid exscan base — and now calls `globalOwnedVertices( mesh )`. The
  remaining library `MPI_Allreduce` sites are *not* among the four (acquisition
  and need-count early exits, the mark-propagation fixpoint guard, a face-gid
  MAX, `exscanCount`'s paired Allreduce+Exscan) and were left alone.

  **Name collision, worth knowing.** Adding `globalSum`/`globalMax` to namespace
  `Tessera` broke three test files that defined their own two-argument
  `globalSum(long long, MPI_Comm)` / `globalMax(...)` helpers in the global
  namespace under `using namespace Tessera;` — the new templates deduce
  `MeshT = long long, Scalar = MPI_Comm` and win overload resolution whenever the
  non-template needs a conversion, then fail in the body on `mesh.comm()`. The
  local helpers in `test_conforming_migrate.cpp`,
  `test_conforming_determinism.cpp` and `test_halo_depth.cpp` were renamed to
  `commSum`/`commMax` (they are comm-based, not mesh-based). The single-argument
  helpers in the other conforming tests do not collide and were left as they are.

  **Tests.** `test_global_reduce.cpp` rewritten to cover all ten listed checks,
  and re-registered at TIER `regression`, backends SERIAL and HIP, ranks 1-5
  (pre-authorized by this task). The three original `globalMin` checks are kept
  verbatim as check 10. Every extremum and the one dissenting rank sit away from
  rank 0 where the pattern allows; the finiteness negatives build the local
  verdict with a real `std::isfinite` sweep over an array holding the poisoned
  value, one run each for `NaN`, `+Inf`, `-Inf`. Checks 8 and 9 run against a
  distributed subdivision-2 icosphere before and after a uniform `refine()`.

  All 10 registrations pass (SERIAL and HIP, np1-5). Full regression gate green
  with nothing relabelled.

  **Docs.** README's API section documents the four new collectives, the
  verdict-not-data contract, and the `globalSum` floating-point reproducibility
  caveat in the same place. `docs/design.md` gains a *Global reductions* section.
  Nothing added to README *Future Optimizations* — reproducible summation stays
  out of scope, and the task requires asking before adding an entry there.
