# XDMF time-series grouping — progress log

Session record for fix-file-grouping-io. Companion to `fix-file-grouping-io.md`,
which holds the design, the task sequence and the risks; this file holds what
actually happened, in order.

**Read this when** you need the reasoning behind a decision the design states
flatly, the measured numbers behind a claim, or the history of a file you are
about to change. The design says *what is true now*; the log says *how it got
that way and what was tried on the route*.

**Append to it** at the end of any task that makes a decision, changes a
signature, measures something, or finds a bug. Add a new `## <task ID>` section
at the bottom, named for the task it records, so `fix-file-grouping-io.md` can
cite it by ID. No dates: the order of the sections is the chronology. If a
session covers more than one task, name them all; if it belongs to no task, name
the topic.

**End each section with `**Affects:**`** — the later task IDs whose stated plan
this entry changes, one clause each on how, or `none`. A finding that invalidates
a later task is worthless if the session starting that task has to read the whole
log to notice it; this line is the index that makes it findable.

Worth recording, because none of it is recoverable from the code afterwards:
semantic decisions and what forced them, signature changes and why they could not
stay as they were, bugs that only running revealed, measured numbers, and
approaches tried that did not work. Record too where the implementation departed
from the task's stated **Do** steps, and why — a task marked `**DONE**` that was
done differently than it was written is the quietest way for a design to stop
describing the code.

V1 in particular exists to be recorded here: the Paraview version, every reader
tried, and the timestep count each one reported.

## T1

**Decisions taken as given** (handed down with the task, recorded so a later
session does not relitigate them):

- **The byte-comparison vehicle is `examples/02_mesh_pipeline`, not `test_io`.**
  `test_io.cpp` deletes its sidecar
  ([test_io.cpp:294-295](../tests/test_io.cpp#L294-L295),
  [:523-524](../tests/test_io.cpp#L523-L524)), so it leaves nothing to `cmp`;
  the example leaves one `.xmf` per frame per exec space on disk.
- **Determinism was established before anything was attributed to the change.**
- **The failure-direction check was temporary and is reverted.** No timed call
  site is left in the example — T3 owns that.
- **Scope of test running:** all targets built; only `io_SERIAL` np1-5 run.
  `io_HIP` was not run — the sidecar is rank-0 text with no device involvement.

**Final shape.** `XdmfField` moved from `Tessera::detail` to `Tessera`
unchanged. New in `Tessera`:

```cpp
struct XdmfFrame {
    std::string h5name;   // BASENAME of the .h5, never a path
    int dim; int scalarBytes;
    unsigned long long Nv, Nf;
    std::vector<XdmfField> vFields, fFields;
};
```

Emitters, exactly as the design stated them:

```cpp
detail::writeXdmfGrid( std::FILE*, const XdmfFrame&, int indent );
detail::writeXdmfGrid( std::FILE*, const XdmfFrame&, int indent, double time );
Tessera::writeXdmf( const std::string& stem, const XdmfFrame& );
Tessera::writeXdmf( const std::string& stem, const XdmfFrame&, double time );
detail::writeMeshH5( const MeshT&, const std::string& stem ) -> XdmfFrame;
Tessera::writeMesh( const MeshT&, const std::string& stem ) -> XdmfFrame;
Tessera::writeMesh( const MeshT&, const std::string& stem, double ) -> XdmfFrame;
```

`indent` is the column the `<Grid>` line starts at in spaces — 2 standalone, 4
nested in a collection. Every line inside is written as `pad + relative indent`,
so the format strings differ from the originals only by a leading `%s`.

**Departures from the stated Do steps**, both internal and neither changing a
signature the design names:

1. Each overload pair is one emitter plus a thin pair of wrappers. The two
   timed/untimed spellings funnel into `detail::writeXdmfGridImpl( fp, frame,
   indent, const double* time )` and `detail::writeXdmfFile( stem, frame, const
   double* time )`. The *API* is an overload pair as the design requires; the
   pointer is private, and it is what makes byte-identity structural rather than
   a thing to keep re-checking. T2 must call the public
   `detail::writeXdmfGrid()` overloads, not `...Impl`.
2. `TESSERA_SCOPED_TIMER( TIMER_WRITE_MESH )` stayed in the two public
   `writeMesh()` overloads rather than moving into `writeMeshH5()`, so the timer
   still spans the HDF5 write *and* the sidecar exactly as before. A caller that
   calls `writeMeshH5()` directly is therefore untimed — T2's `MeshSeries`
   should go through `writeMesh()`.
3. `const int R = mesh.rank()` was deleted from `writeMeshH5()`: its only use
   was the rank-0 sidecar guard, which now lives in `writeMesh()` and reads
   `mesh.rank()` there. `comm` and `size` are still used inside.

**Verification, and one gap the prescribed check does not cover.** The example
run (4 ranks, `flux run --ntasks 4 --nodes=1 --exclusive --cores-per-task=16`,
the committed binding) produces 6 sidecars per run — Serial/Default/OpenMP ×
frame0/frame1, conforming — and all 6 are byte-identical to their pre-change
copies. But `mesh_pipeline` instantiates its mesh with `VertexFields<>`,
`EdgeFields<>`, `FaceFields<>` and `Dim=3`, so that comparison exercises
**neither** the user-field attribute loop **nor** the `dim != 2 && dim != 3`
attribute-only branch — the two places the reindentation could have gone wrong
and nothing in the repo would have noticed. No existing test leaves a
user-field `.xmf` on disk to compare (`test_io` is the only mesh with user
fields and it deletes its sidecar). Closed with a throwaway header-only probe
(`/usr/WS2/stewartj/t1_runs/probe/`): two 1-task programs including
`git show HEAD:src/Tessera_Xdmf.hpp` and the post-change header, emitting the
same four cases — dim 3, dim 2, dim 4, and empty field lists, each with a scalar
and a multi-extent user field on both centerings. All four `.xmf` byte-identical
old vs new. The same probe printed `detail::writeXdmfGrid( fp, frame, 4 )` and
its timed overload, confirming the nesting T2 needs.

**No bug was revealed by running.** Two things running *did* reveal, both
about the harness rather than the code:

- `/tmp` is node-local on Tuolumne. A job submitted from a `/tmp` directory
  fails to `chdir`, silently lands in the compute node's own `/tmp`, and still
  exits 0 — its output is simply unreachable afterwards. Job working
  directories must be on a shared filesystem; these runs used
  `/usr/WS2/stewartj/t1_runs`.
- `flux job wait` returns `Request requires owner credentials` for these jobs.
  `flux job attach <id>` blocks until exit and `flux job status <id>` then
  yields the exit code.
- Two `--out` stems differing in their **last** component cannot be compared
  byte-for-byte at all: the `.xmf` embeds the `.h5` basename, so the text
  differs for that reason alone. The determinism probe used `--out runA/mesh`
  and `--out runB/mesh` — distinct stems, shared basename.

**Affects:** T2 — `XdmfFrame` and both `detail::writeXdmfGrid()` overloads
landed exactly as the design stated them, so T2 needs no adjustment on that
front; note only that `writeXdmfGridImpl`/`writeXdmfFile` are private plumbing
to call *through*, not directly, and that `MeshSeries` should call `writeMesh()`
(timed) rather than `writeMeshH5()` so `TIMER_WRITE_MESH` still covers the
frame. T2's `tests/test_xdmf_series.cpp` is the first place a user-field `.xmf`
gets asserted-on in-repo — worth asserting the attribute block there, since the
throwaway probe that covered it in T1 is not committed. V1 — R3's fix stays
one-function-wide: every child grid's element spellings now come from
`detail::writeXdmfGridImpl()` alone.

## T2

Tested at commit `0645f88` plus this change; spack env
`~/spack_envs/tuolumne_trilinos/`; jobs `f3T2qc15CLWB` (first np1-3 pass),
`f3T2rdKgDbd1` (np2, cleanup temporarily suppressed, for hand inspection) and
`f3T2sWHo5ZMy` (final np1-3 pass on the reverted source). All three are 100%
pass; the whole build tree compiles.

**Final signatures.** Exactly as the design stated them, with no deviation:

```cpp
// Tessera_Xdmf.hpp
struct Tessera::XdmfTimeStep { XdmfFrame frame; double time; };
void Tessera::writeXdmfSeries( const std::string& masterStem,
                               const std::vector<XdmfTimeStep>& steps );

// Tessera_XdmfSeries.hpp  (new; included from Tessera.hpp after Tessera_Xdmf.hpp)
std::string detail::xdmfDirname( const std::string& path );   // "" when no dir

class Tessera::MeshSeries
{
  public:
    explicit MeshSeries( std::string masterStem );
    template <class MeshT>
    void write( const MeshT& mesh, const std::string& frameStem, double time );
    std::size_t numFrames() const;
    const std::string& masterStem() const;
  private:
    void appendIndexLine( const std::string& frameStem, double time ) const;
    std::string _masterStem;
    std::vector<XdmfTimeStep> _steps;
};
```

`writeXdmfSeries()` goes through the public `detail::writeXdmfGrid( fp, frame, 4,
s.time )` overload, never `writeXdmfGridImpl`, per T1's note. `MeshSeries::write()`
calls the public timed `writeMesh( mesh, frameStem, time )`, never
`detail::writeMeshH5()`, so `TESSERA_SCOPED_TIMER( TIMER_WRITE_MESH )` still spans
each frame. The master rewrite and the `.xmfindex` append sit **outside** that
timer, which is the measurement R4 asks for: rewrite creep shows up as rank-0
wall-clock with `TIMER_WRITE_MESH` flat.

**Decisions.**

- **`xdmfDirname()` is a new `detail` helper, not a reuse of `xdmfBasename()`.**
  It lives in `Tessera_XdmfSeries.hpp` beside its only caller rather than in
  `Tessera_Xdmf.hpp` beside `xdmfBasename()`, because touching the single-grid
  path was out of scope. It returns `""` for a bare stem, so a bare master stem
  and a bare frame stem compare equal (the common in-cwd case) and any `sub/`
  prefix on either side compares unequal.
- **The `.xmfindex` append is loud on failure too**, throwing rather than
  degrading quietly, and is opened/closed per frame so the line is on disk before
  `write()` returns. It shares its directory with the master, so a filesystem
  that cannot take the index cannot take the master either — the two failures do
  not come apart in practice.
- **`fclose` is checked, not just `fopen`.** A short write on a full filesystem
  surfaces at `fclose`, and a silently truncated master is exactly the
  quiet-degradation this design forbids. So `writeXdmfSeries()` throws on
  `fopen`, `fclose` and `rename`, each message naming the path.
- **The collection grid keeps `Name="Tessera"`** — same name as every child, per
  the Conventions table.

**Bugs only running revealed: none in the new code.** One environment trap did
cost a cycle: the checked-out `build-tuolumne/` held a `CMakeCache.txt` created in
a *different* checkout (`.../Tessera-ai-test/build-tuolumne`), so `cmake .` refused
with "current CMakeCache.txt directory ... is different than the directory ...
where CMakeCache.txt was created" and `make <newtarget>` reported "No rule to make
target". A stale build dir carried between checkouts must be cleared and
re-configured with `bash ../run_cmake_toulumne.sh`; it is not a code or CMake
error.

**Departures from the stated Do steps.** One, in the test rather than the library:
the exit criterion's positive assertions are all count- or set-based, and a
count-based assertion can pass on XML that is subtly wrong in shape. So the master
was also **read by hand once**, by temporarily replacing the test's rank-0 cleanup
guard with `if ( rank == 0 && false )`, rebuilding, running np2 alone (job
`f3T2rdKgDbd1`), and inspecting `build-tuolumne/tests/*_np2.xmf`. It is the
specified shape: one `Collection`/`Temporal` wrapper at indent 2, three
`GridType="Uniform"` children at indent 4 with `<Time Value="0"/>`,
`<Time Value="0.5"/>`, `<Time Value="1.25"/>`, each naming only its own
`..._frame<i>.h5`, and each carrying the same four attributes in the same order —
`v_gid`, `f_level`, `vu0`, `fu0`. Only `_frame0..2` files existed, confirming the
two rejected `write()` calls threw before any I/O. The `.xmfindex` held the three
`"<stem> <time>"` lines T4 will consume. That edit is reverted, the final np1-3
job ran on the reverted source, and the test leaves no files behind.

Worth recording for T4: the user-field display names the writer derives are
`vu0` and `fu0` — the `"v"`/`"f"` centering prefix plus `u<j>`, as the design's
T4 step 3 states — now confirmed against emitted XML rather than read off the
writer.

**Affects:** T3 — `MeshSeries` is exactly the API T3 was written against, so its
**Do** steps need no adjustment; one addition, that `MeshSeries` also writes
`<masterStem>.xmfindex`, so T3's exit criterion should expect that file beside
each master and the example's output directory now holds one extra file per
series. T4 — the `.xmfindex` format is now pinned: one line per frame,
`"<frameStem> %.17g\n"`, frame stem verbatim as the caller passed it (a path, not
a basename), appended in write order. The vertex/face user-field display-name rule
is confirmed as `vu<j>`/`fu<j>`. V1 — nothing changes: the master reuses
`detail::writeXdmfGridImpl()` for every child, so R3's fix stays one function
wide, and `Version="3.0"`-with-XDMF2-spellings is untouched and still V1's to
judge. The master's timestep count for R2's distinguishing measurement is
`grep -c '<Time Value=' <master>.xmf`, and the test already pins that count to
the number of frames written. R1 — its stated diagnostic (identical
`<Attribute Name=` set, same order, across children) is now a committed
assertion, so a drift would be caught in CI rather than in Paraview.

## T3

Run at commit `c4fed74` (the example and flux-runner change, committed before
the run so the SHA the job echoes is the code that ran); spack env
`~/spack_envs/tuolumne_trilinos/`. Verification job `f3gprsqZkFZH` in the fresh
directory `/usr/WS2/stewartj/t3_runs/run1`; jobs `f3gpsNLM7bDq` then
`f3gpscSfKB19`, both into the same fresh `/usr/WS2/stewartj/t3_runs/run2` with
the same `--out`, to observe the rerun behaviour below. All three exit 0. No
library source changed; only `mesh_pipeline` was built and run, no ctest.

**Decisions taken as given** (handed down with the task):

- **Master stem is `opt.outStem + "_" + tag + "_np" + std::to_string(size)`**
  — `frameStem()` without `_frame<i>`, so it carries both the execution space
  and the refine mode. It lives in a new `seriesStem()` helper and `frameStem()`
  is `seriesStem() + "_frame<i>"`, so the string is built once and the frames
  sit in the master's directory by construction (R5).
- **One `MeshSeries` per `run()` call**, constructed just before frame 0. Each
  mesh type and execution space gets its own series — R1's structural
  guarantee.
- **Time is the frame index as a `double`**: `0.0` for frame 0, `double( it )`
  for iteration `it`, said so in a comment at both call sites.

**Implementation.** Both `writeMesh()` calls became `series.write( mesh,
frameStem( ... ), time )`; the per-frame `printf` lines are unchanged and one
rank-0 line at the end of `run()` names the master (`[<tag>] series: 4 frames
-> open <stem>.xmf in Paraview`). The flux runner's header now says the example
writes a `MeshSeries` and names the master as the file to open; it also echoes
`git rev-parse HEAD` and `spack env status` before the `flux run`, so job output
carries its own provenance. No departure from the **Do** steps.

**Measured.** Three tags ran (`Serial_conforming`, `Default_conforming` = HIP,
`OpenMP_conforming`) and produced exactly three masters, each with 4 child grids,
`<Time Value=` `0,1,2,3`, attributes `v_gid`, `f_level` on every child (the
example mesh has no user fields, so this is a smaller R1 check than T2's), 16
`Format="HDF"` references to 4 distinct `.h5` basenames all present, a 4-line
`.xmfindex` with times `0,1,2,3`, and `xml.etree` parses each. Per-frame owned
counts are identical across the three spaces (rank 0: `ownedF` 80 → 145 → 189 →
319). The output directory holds 10 files per series: 4 `.h5`, 4 sidecar `.xmf`,
the master and its index.

**Bugs only running revealed: none in the code.** Three harness facts:

- `flux batch --flags=waitable` is refused here: `only the instance owner can
  submit with FLUX_JOB_WAITABLE`. Plain `flux batch` then `flux job status
  <id>` blocks to completion and returns the exit code, which is all that was
  needed.
- A fresh checkout's configure must run with the spack env **activated**
  (`spack env activate ~/spack_envs/tuolumne_trilinos/` inside the same
  `bash -lc`), not only with `SPACK_USER_CONFIG_PATH` exported:
  `run_cmake_toulumne.sh` does not activate it and `find_package(Kokkos)` fails
  at `CMakeLists.txt:50`. Re-running configure in the same build dir after
  activating succeeds; the failed cache needs no clearing.
- The example's build warns `-Wnontrivial-memcall` at
  `src/Tessera_HaloRebuild.hpp:99`; pre-existing, untouched.

**Observed: a rerun with the same `--out` appends to the index.** After two
runs into `run2`, each master is correct — 4 timesteps, `0,1,2,3`, since it is
rewritten from the in-memory steps — and every frame `.h5`/`.xmf` is simply
overwritten. But each `.xmfindex` has **8** lines, times `0,1,2,3,0,1,2,3`:
a fresh `MeshSeries` opens it with `"a"` and never truncates. Not fixed here.

**Affects:** T4 — an `.xmfindex` can legitimately hold more than one run's
frames: a rerun into the same stem leaves `0,1,2,3,0,1,2,3`, which the reopen
constructor's strictly-increasing check (step 2) would reject with a throw even
though the master on disk is valid. T4's restart design must decide what a
fresh (non-reopen) `MeshSeries` does to an existing index — truncate it on the
first `write()` is the obvious candidate — and its test should cover a fresh
series over a stale index. V1 — the masters to open are
`/usr/WS2/stewartj/t3_runs/run1/mesh_{Serial,Default,OpenMP}_conforming_np4.xmf`
(4 timesteps each, attributes `v_gid`, `f_level`; no `vu0`/`fu0` — those exist
only in a T2 test master). R1 — the example's masters are a live instance of its
diagnostic and pass it.
