# XDMF time-series grouping: one dataset in Paraview instead of N

**Status:** IN PROGRESS

## Problem

A downstream solver (Beantik) calls `writeMesh( mesh, stem )` once per output
frame, each call with a different stem. Every call produces `<stem>.h5` plus a
standalone `<stem>.xmf`. In Paraview the result is N unrelated datasets: each
`.xmf` must be opened separately, there is no time slider over the sequence, and
stepping through the mesh evolution means loading and toggling N pipeline
objects by hand.

The cause is not a Paraview setting and not the filenames. The sidecar Tessera
writes carries **no time coordinate and no collection wrapper**, and
`writeMesh()` is handed no time and no series identity, so the library has no
way to state that N files are N timesteps of one thing. Paraview's XDMF readers
are not file-series readers — they will not infer a series from numeric
filenames the way `vtkFileSeriesReader`-backed readers do — so nothing
downstream can recover the grouping. See **Read this first** for the evidence.

End state: a run that writes N frames also writes **one master `.xmf`** which
Paraview opens as a single dataset with N timesteps on the time slider,
animatable, with the same attribute set (`v_gid`, `f_level`, `vu0…`, `fu0…`)
selectable across every step. The per-frame `.h5` layout on disk does not
change, so `readMesh()` keeps working on any individual frame.

**Out of scope**

- Changing the HDF5 layout, `readMesh()`, or `format_version` (still 2). The
  master `.xmf` is pure light data derived from what the writer already emits.
- Writing edges to XDMF. Still deliberately omitted
  ([src/Tessera_Xdmf.hpp:15-21](../src/Tessera_Xdmf.hpp#L15-L21)).
- Any change to the ship gate. The new test is `unit` tier, so the four-location
  gate definition (`docs/testing.md`, `tests/CMakeLists.txt`,
  `scripts/<system>/run_regression_minset.*`, `.github/workflows/ci.yml`) is
  untouched by this work.
- Spatial collections (one grid per rank). Tessera writes one dense global grid
  per frame; there is nothing to collect spatially.
- Changes inside Beantik. This design delivers the library API and updates
  `examples/02_mesh_pipeline` as the reference caller; adopting it in Beantik is
  that project's change.

## Read this first

The brief's framing — "they are not grouped together in paraview" — is accurate
about the symptom but suggests grouping is something that failed. Nothing ever
attempted it. Three specific facts a later session should not re-derive:

1. **There is no collection wrapper in a single-shot `writeMesh()`'s emitted
   XML: the sidecar is a lone `GridType="Uniform"` grid.** (Still true after T2 —
   the collection lives in the *master*, which only `MeshSeries` writes.)
   [src/Tessera_Xdmf.hpp:211](../src/Tessera_Xdmf.hpp#L211) writes
   `<Xdmf Version="3.0"><Domain>` and
   [src/Tessera_Xdmf.hpp:213](../src/Tessera_Xdmf.hpp#L213) closes it, with a
   single `<Grid Name="Tessera" GridType="Uniform">`
   ([src/Tessera_Xdmf.hpp:112](../src/Tessera_Xdmf.hpp#L112)) between them. A
   `<Time>` child can be emitted since T1, but one timed grid in one file is
   still not a series: a reader handed the file sees one grid, so there is
   nothing for a time slider to step through.

2. **`writeMesh()` cannot know a series exists.** Its signatures are
   `writeMesh( const MeshT& mesh, const std::string& stem )` and the timed
   overload ([src/Tessera_HDF5Writer.hpp:430](../src/Tessera_HDF5Writer.hpp#L430),
   [:446](../src/Tessera_HDF5Writer.hpp#L446)). A `time` argument exists, but no
   frame index and no state carried between calls. The caller invents the stem
   (`frameStem()` in the example,
   [examples/02_mesh_pipeline/mesh_pipeline.cpp:107-114](../examples/02_mesh_pipeline/mesh_pipeline.cpp#L107-L114)),
   so the *only* place in the system that knows which files form a sequence, and
   in what time order, is the caller. That is why the fix necessarily adds a
   caller-facing series object rather than being a pure change inside
   `writeXdmf()`.

3. **Paraview will not group the files for us.** The idiomatic and reliable XDMF
   mechanism is a temporal collection —
   `<Grid GridType="Collection" CollectionType="Temporal">` with one child
   `<Grid>` per timestep, each carrying its own `<Time Value="…"/>` and its own
   full `<Topology>`/`<Geometry>` — which is exactly what an adaptively
   remeshing series needs, since `Nv`/`Nf` differ frame to frame. Paraview's
   generic `.series` JSON sidecar is supported only by readers that declare
   file-series support, which the XDMF readers do not. A child grid with no
   `<Time>` has been reported to crash Paraview, so the `<Time>` element is
   mandatory, not decorative.

A fourth observation, deliberately **not** acted on by this design: the emitted
XML declares `Xdmf Version="3.0"` while using XDMF2 element spellings
(`TopologyType=`, `GeometryType=`, `NumberOfElements=`, `NumberType=`/
`Precision=`, e.g.
[src/Tessera_Xdmf.hpp:112-123](../src/Tessera_Xdmf.hpp#L112-L123)). This currently
works — the readers accept the legacy spellings — and changing it is a separate
risk with no bearing on grouping. It is why **V1 must confirm the master file in
more than one Paraview reader** (see R2).

## Approach

Keep the frame files exactly as they are and add a second, higher-level artifact
that names them in time order.

Three pieces, in the same header-only style as the rest of the library:

1. **A frame record.** The metadata the XML needs is one `XdmfFrame` struct, and
   `writeMesh()` **returns** it. Returning it is what makes the series possible
   without duplicating the writer's metadata derivation (`gcV.N`, `gcF.N`, the
   `vXdmf`/`fXdmf` field lists built at
   [src/Tessera_HDF5Writer.hpp:250](../src/Tessera_HDF5Writer.hpp#L250)
   and [src/Tessera_HDF5Writer.hpp:361](../src/Tessera_HDF5Writer.hpp#L361)) in a
   second place, where it would silently drift from the writer.

2. **A stateless master-file emitter.** `writeXdmfSeries( masterStem, steps )`
   takes a vector of `{ XdmfFrame, double time }` and writes the temporal
   collection. Stateless and MPI-free, so it is directly testable without
   building a mesh, and a caller that keeps its own frame bookkeeping can call it
   directly.

3. **A caller-held series handle.** `MeshSeries` accumulates frames and rewrites
   the master after every frame, so the master on disk always describes the
   frames that actually exist. A run killed at frame 40 leaves a master with 40
   timesteps, not a missing or truncated file.

### Why the master is rewritten every frame, not once at the end

An `.xmf` temporal collection is not append-friendly: the closing
`</Grid></Domain></Xdmf>` must move. The alternatives were (a) leave the file
unterminated until finalize — a crashed run then leaves invalid XML that
Paraview refuses outright, which is worse than the status quo; (b) write the
master only in a finalize call — a crashed or killed run yields no master at
all, and long HPC runs are killed routinely. Rewriting is O(frames) text on rank
0 per frame, immaterial against a collective HDF5 write (R4). The rewrite goes
to `<master>.xmf.tmp` and is `std::rename`d over the real path, so a Paraview
reload never observes a half-written file.

### Why frame stems stay caller-owned

`MeshSeries::write()` takes the frame stem from the caller rather than
generating `<base>_%06u`. The master lists each frame's file explicitly, so
frame names need not be numeric or even ordered — and Beantik keeps whatever
naming it already uses. The cost is one real constraint that must be enforced
loudly, not worked around: the XML references the HDF5 file by **basename** —
that is what `XdmfFrame::h5name` holds and documents
([src/Tessera_Xdmf.hpp:43](../src/Tessera_Xdmf.hpp#L43)) — so an `.xmf` can only
reference `.h5` files sitting beside it. The master must therefore live in the
same directory as every frame it names, and `MeshSeries` hard-fails on a frame
stem whose directory differs from the master's.

### Conventions

| Choice | Decision |
|---|---|
| Header | New public API in a new header `src/Tessera_XdmfSeries.hpp`; add it to `src/Tessera.hpp` in alphabetical position (after `Tessera_Xdmf.hpp`, [src/Tessera.hpp:52](../src/Tessera.hpp#L52)) |
| Namespace | `Tessera::` for `XdmfFrame`, `XdmfTimeStep`, `XdmfField`, `writeXdmf`, `writeXdmfSeries`, `MeshSeries`. `XdmfField` lives in `Tessera`, not `Tessera::detail`, because it is reachable through a public return type; its only two uses are in `Tessera_Xdmf.hpp` and [src/Tessera_HDF5Writer.hpp:202](../src/Tessera_HDF5Writer.hpp#L202) |
| Time argument | Always an explicit `double time` from the caller. No default, no frame-index fallback, no NaN sentinel. A caller with no physical time passes the step index — an explicit choice at the call site |
| Optional time | Expressed as an **overload pair**, not a pointer or sentinel: `writeXdmf( stem, frame )` emits no `<Time>`; `writeXdmf( stem, frame, time )` emits one. Same for the two `writeMesh` overloads |
| Failure behavior | Loud. `MeshSeries::write()` throws `std::runtime_error` on a non-increasing time or a frame stem in a different directory from the master. `writeXdmfSeries()` throws if it cannot open the temp file or if `steps` is empty. No best-effort partial master |
| Time formatting | `%.17g`, so a `double` round-trips exactly through the XML |
| Child grid names | Every child keeps `Name="Tessera"`, as today — a stable grid name across timesteps is what lets Paraview track the same object through the collection |
| Comments | Units/ownership/monotonicity contracts on the declarations. Cite the XDMF temporal-collection shape as the provenance on `writeXdmfSeries()` |
| Test tier | `unit`, `SERIAL`, ranks `1;2;3` — new file `tests/test_xdmf_series.cpp`. The gate is untouched. Assertions are on XML text plus file existence; a Python `xml.etree` well-formedness check runs from the test via `python3` (present at `/usr/tce/bin/python3`; `pvpython` and `paraview` are **not** installed on this system, so no in-test Paraview check is possible) |
| Test output paths | Same convention as [tests/test_io.cpp:553-559](../tests/test_io.cpp#L553-L559): the stem is the executable basename plus `_np<size>`, in the cwd, prefixed by `$TESSERA_IO_TMPDIR` when that variable is set. Whatever directory is used must be on a **shared** filesystem — `/tmp` is node-local on Tuolumne, so a job that writes there leaves output unreachable and still exits 0 (see [the T1 log](fix-file-grouping-io-progress-log.md#t1)) |
| License header | Every new file carries the project BSD-3-Clause `/* */` block with `SPDX-License-Identifier: BSD-3-Clause`, per `CLAUDE.md` |
| Formatting | Never run the formatter. Match surrounding style by hand |

### Deliberate deviations

- **Per-frame `.xmf` sidecars are kept.** The series writes a master *and* leaves
  each frame's own sidecar in place, so a single frame stays individually
  inspectable and the existing single-shot `writeMesh( mesh, stem )` behavior is
  unchanged for every current caller. The consequence is that the output
  directory still contains N+1 openable `.xmf` files; the master is the one to
  open, and it is the only one whose name has no frame index in it. This was
  chosen explicitly over suppressing sidecars in series mode.
- **No XInclude / XPointer reuse of topology between child grids.** Each child
  grid repeats its full `<Topology>` and `<Geometry>` pointing at its own `.h5`.
  This is correct for adaptive meshes (topology genuinely changes) and is the
  path with the fewest reader bugs; hoisting shared data into the `<Domain>` is
  reported to break. The XML is O(frames) but each child is ~10 lines of text
  referencing HDF5 — heavy data never enters the XML.
- **`writeMesh()` returns `XdmfFrame` rather than `void`.** This is
  source-compatible with every existing caller (a discarded return value), so no
  caller was modified by T1 — but it is a change to a documented public signature
  and is mirrored into `README.md` and `docs/design.md`.

## Current state

T1 and T2 are done; T3 onward are unbuilt. What exists:

- `Tessera::XdmfField` ([src/Tessera_Xdmf.hpp:32](../src/Tessera_Xdmf.hpp#L32))
  and `Tessera::XdmfFrame` ([:43](../src/Tessera_Xdmf.hpp#L43)), the latter
  documenting `h5name` as a basename.
- `detail::writeXdmfGridImpl( fp, frame, indent, const double* time )`
  ([src/Tessera_Xdmf.hpp:106](../src/Tessera_Xdmf.hpp#L106)) emits one whole
  `<Grid>` and is the single place any child grid's element spellings come from —
  which is what keeps R3's fix one function wide. It is private plumbing: callers
  go through the public `detail::writeXdmfGrid( fp, frame, indent )`
  ([:186](../src/Tessera_Xdmf.hpp#L186)) and
  `detail::writeXdmfGrid( fp, frame, indent, double time )`
  ([:194](../src/Tessera_Xdmf.hpp#L194)).
- `detail::writeXdmfFile( stem, frame, const double* )`
  ([src/Tessera_Xdmf.hpp:202](../src/Tessera_Xdmf.hpp#L202)) wraps a single grid
  in `<Xdmf><Domain>`, behind `Tessera::writeXdmf( stem, frame )`
  ([:222](../src/Tessera_Xdmf.hpp#L222)) and
  `writeXdmf( stem, frame, time )` ([:229](../src/Tessera_Xdmf.hpp#L229)). It
  **returns silently if `fopen` fails**
  ([:206-208](../src/Tessera_Xdmf.hpp#L206-L208)); that is a pre-existing
  quiet-degradation path this design does not fix, and the new emitter must not
  copy it.
- `detail::writeMeshH5( mesh, stem ) -> XdmfFrame`
  ([src/Tessera_HDF5Writer.hpp:85](../src/Tessera_HDF5Writer.hpp#L85)) does the
  collective HDF5 write. The two public `writeMesh()` overloads
  ([:430](../src/Tessera_HDF5Writer.hpp#L430),
  [:446](../src/Tessera_HDF5Writer.hpp#L446)) each call it, `MPI_Barrier`, emit
  the matching sidecar on rank 0, and return the frame.
  `TESSERA_SCOPED_TIMER( TIMER_WRITE_MESH )` sits in those two overloads and
  **not** in `writeMeshH5()`, so `MeshSeries` must go through `writeMesh()` to
  keep the frame timed.
- `Tessera::XdmfTimeStep` and `Tessera::writeXdmfSeries( masterStem, steps )`
  ([src/Tessera_Xdmf.hpp](../src/Tessera_Xdmf.hpp)) — the stateless, MPI-free
  temporal-collection emitter, writing via `<masterStem>.xmf.tmp` + `rename` and
  throwing `std::runtime_error` on empty `steps` or any fopen/fclose/rename
  failure (it deliberately does **not** copy `writeXdmfFile()`'s silent return).
- `Tessera::MeshSeries` ([src/Tessera_XdmfSeries.hpp](../src/Tessera_XdmfSeries.hpp),
  included from `Tessera.hpp`) — the caller-held handle: `write( mesh, frameStem,
  time )` validates on every rank, calls the public timed `writeMesh()`, appends
  the step on every rank, and on rank 0 appends to `<masterStem>.xmfindex` and
  rewrites the master. Nothing reads a written frame's metadata back for a
  restart yet — that is T4, and `MeshSeries` keeps every frame in memory.
- `tests/test_xdmf_series.cpp`, `unit`/`SERIAL`, ranks `1;2;3` — the first
  in-repo assertions on emitted XDMF text, including the user-field attribute
  block.
- `examples/02_mesh_pipeline` writes N independent frames
  ([mesh_pipeline.cpp:155](../examples/02_mesh_pipeline/mesh_pipeline.cpp#L155),
  [mesh_pipeline.cpp:221](../examples/02_mesh_pipeline/mesh_pipeline.cpp#L221)) —
  it reproduces the reported symptom exactly and is the natural demonstration
  vehicle.
- `tests/test_io.cpp` exercises the round trip and deletes `<stem>.xmf`
  ([test_io.cpp:294-295](../tests/test_io.cpp#L294-L295),
  [test_io.cpp:523-524](../tests/test_io.cpp#L523-L524)) but asserts nothing
  about its contents; all XML coverage lives in `tests/test_xdmf_series.cpp`.

**Not read, deliberately:** nothing in Paraview's reader sources, and no Beantik
code (not in this repository). T4 is the first task that reads HDF5 root
attributes back from the series layer; earlier tasks must not open that path —
`MeshSeries` in T2 keeps every frame's metadata in memory and needs no reads.

## Progress log

`tasks/fix-file-grouping-io-progress-log.md`. Read it before implementing any
task, before changing a signature this design states, and before reopening a
question the design treats as settled — a completed task may have changed the
plan for a later one, and the `**Affects:**` line of each entry is the index to
that.

## Task sequence

### T1 — `XdmfFrame` record; `writeMesh()` returns it; byte-identical output — **DONE**

**Depends on:** none

**Fill in:**
- `src/Tessera_Xdmf.hpp` — move `XdmfField` from `Tessera::detail` to `Tessera`;
  add `struct XdmfFrame`; split the grid body out of `writeXdmf()` into
  `detail::writeXdmfGrid( fp, frame, indent )` and an overload
  `detail::writeXdmfGrid( fp, frame, indent, double time )`; replace the 7-arg
  `writeXdmf()` with `writeXdmf( const std::string& stem, const XdmfFrame& )`
  and `writeXdmf( const std::string& stem, const XdmfFrame&, double time )`.
- `src/Tessera_HDF5Writer.hpp` — extract the whole existing body except the
  final sidecar block into `detail::writeMeshH5( mesh, stem ) -> XdmfFrame`;
  public `writeMesh( mesh, stem )` and `writeMesh( mesh, stem, double time )`
  both call it, barrier, and emit the matching sidecar on rank 0, returning the
  frame. Update the `detail::XdmfField` use at line 197.

**Reference:** the exact bytes to preserve are
[src/Tessera_Xdmf.hpp:91-148](../src/Tessera_Xdmf.hpp#L91-L148); the metadata to
capture in `XdmfFrame` is the current argument list at
[src/Tessera_Xdmf.hpp:80-83](../src/Tessera_Xdmf.hpp#L80-L83) plus the
`h5name` basename computed at [src/Tessera_Xdmf.hpp:85](../src/Tessera_Xdmf.hpp#L85).

**Do:**
1. Define `XdmfFrame { std::string h5name; int dim; int scalarBytes; unsigned long long Nv, Nf; std::vector<XdmfField> vFields, fFields; }`. `h5name` is the **basename** of the HDF5 file (documented on the member: an `.xmf` can only reference `.h5` files in its own directory).
2. Factor the grid emitter with an `indent` parameter — the master nests one level deeper than a standalone file, and this is the only reason the parameter exists. Say so in a comment.
3. Keep the `dim != 2 && dim != 3` attribute-only branch ([src/Tessera_Xdmf.hpp:115-121](../src/Tessera_Xdmf.hpp#L115-L121)) verbatim inside the emitter.
4. Do not change any call site other than [src/Tessera_HDF5Writer.hpp:411-413](../src/Tessera_HDF5Writer.hpp#L411-L413) — `writeXdmf` has no other callers in `src`, `tests`, or `examples`, and the discarded `writeMesh` return value keeps `test_io.cpp:226`, `test_io.cpp:458`, `test_latlon_sphere.cpp:709`, `test_distributed_build.cpp:697`, `mesh_pipeline.cpp:155`, and `mesh_pipeline.cpp:221` compiling untouched.
5. Mirror the new `writeMesh` signature into [README.md:118](../README.md#L118) and [docs/design.md:1538-1545](../docs/design.md#L1538-L1545).

**Exit criterion:** save a copy of an `.xmf` produced before the change, then
`ctest -L regression -R "io_SERIAL"` passes at np1-5 and the newly produced
`.xmf` for the same case is **byte-identical** (`cmp` exits 0) to the saved copy.
Failure direction: with the timed overload called instead, `cmp` must report a
difference and the only difference must be an added `<Time Value=` line.

**Met.** `ctest -L regression -R "io_SERIAL"` passes at np1, np2, np3, np4 and
np5 (5/5, job `f3Ss7AUnAj8j`), and every target in the build tree compiles — which
is what proves the `void` -> `XdmfFrame` return change is source-compatible with
the six untouched call sites. Byte-identity was measured on
`examples/02_mesh_pipeline --iters 1` at 4 ranks: run-to-run determinism first
(two runs, `--out runA/mesh` vs `--out runB/mesh`, all 6 sidecars identical),
then the 6 post-change sidecars
`mesh_{Serial,Default,OpenMP}_conforming_np4_frame{0,1}.xmf` against their saved
pre-change copies -- `cmp` exit 0 on all 6. That example has empty field packs
and `Dim=3`, so the user-field attribute loop and the `dim != 2 && dim != 3`
branch were covered separately by a throwaway header-only probe diffing the pre-
and post-change headers over dim 3 / dim 2 / dim 4 / empty-field cases with
scalar and multi-extent user fields on both centerings: all identical (see the
log). Failure direction: with `writeMesh( ..., 0.0 )` temporarily substituted at
[mesh_pipeline.cpp:155](../examples/02_mesh_pipeline/mesh_pipeline.cpp#L155),
each `frame0.xmf` differed from its baseline at byte 102 by exactly one added
line, `    <Time Value="0"/>`, and nothing else, while every `frame1.xmf` (the
untouched call site) stayed identical. That edit is reverted. See
[the progress log](fix-file-grouping-io-progress-log.md#t1) for the two internal
departures from the **Do** steps and the `/tmp`-is-node-local job trap.

### T2 — `writeXdmfSeries()` + `MeshSeries`: a master `.xmf` Paraview opens as one timestepped dataset — **DONE**

**Depends on:** T1

**Fill in:**
- `src/Tessera_Xdmf.hpp` — `struct XdmfTimeStep { XdmfFrame frame; double time; }`
  and `void writeXdmfSeries( const std::string& masterStem, const std::vector<XdmfTimeStep>& steps )`.
- `src/Tessera_XdmfSeries.hpp` (new) — `class MeshSeries`.
- `src/Tessera.hpp` — add the include.
- `tests/test_xdmf_series.cpp` (new), `tests/CMakeLists.txt` — one `unit`,
  `SERIAL` registration written as `RANKS    "1;2;3"`. Copy the *shape* of the
  block at [tests/CMakeLists.txt:537-550](../tests/CMakeLists.txt#L537-L550),
  which is the `regression`-tier `io` block registered for two backends; T2
  registers one SERIAL block at `unit` tier. The literal rank list works because
  the macro iterates `foreach(_np IN LISTS TAT_RANKS)`, but T2's is the file's
  first literal multi-rank list — every existing block passes `"1"` or
  `${TESSERA_TEST_MPI_RANKS}`.

**Reference:** the master's required shape —
`<Grid Name="Tessera" GridType="Collection" CollectionType="Temporal">`
wrapping one full child `<Grid>` per step, each with its own
`<Time Value="…"/>`, `<Topology>`, `<Geometry>` and the identical attribute set.
Cite this shape and the "`<Time>` is mandatory or Paraview can crash" fact on
`writeXdmfSeries()` as provenance.

**Do:**
1. `writeXdmfSeries()`: throw on empty `steps`; open `masterStem + ".xmf.tmp"`, throw `std::runtime_error` naming the path if `fopen` fails (do **not** copy the silent-return at [src/Tessera_Xdmf.hpp:206-208](../src/Tessera_Xdmf.hpp#L206-L208)); emit the collection reusing `detail::writeXdmfGrid( fp, s.frame, indent, s.time )` per step; `fclose`; `std::rename` the temp over `masterStem + ".xmf"`, throwing if the rename fails.
2. `MeshSeries`: `explicit MeshSeries( std::string masterStem )`. Members: master stem and `std::vector<XdmfTimeStep>`. No separate last-time member — the previous frame's time is the vector's `back().time`. Method `template <class MeshT> void write( const MeshT& mesh, const std::string& frameStem, double time )` — collective on `mesh.comm()`, documented as such.
3. `write()` order: validate, then `XdmfFrame f = writeMesh( mesh, frameStem, time )` (collective; also writes the per-frame sidecar), then append the step to the vector on **every** rank, and on rank 0 only append one `"<stem> <time>\n"` line to `masterStem + ".xmfindex"` (flushed each frame — T4 consumes it; write it now so a run predating T4 is still restartable) and call `writeXdmfSeries()`. The accumulator is rank-uniform so that `numFrames()` and the monotonic-time check mean the same thing on every rank; a rank-dependent accessor is a footgun. The cost is one replicated vector of per-frame metadata — a handful of strings and integers per frame — on every rank.
4. Validation, on **every** rank before any I/O so the throw is symmetric and cannot deadlock: `time` strictly greater than the previous frame's (message naming both values); the directory component of `frameStem` equal to that of the master stem (message naming both, and stating why — an `.xmf` references `.h5` by basename).
5. Accessors: `std::size_t numFrames() const` (rank-uniform, per step 3), `const std::string& masterStem() const`.
6. Document in `README.md` (a `MeshSeries` snippet in the I/O part of Usage/API near [README.md:118](../README.md#L118)) and in the Parallel I/O section of `docs/design.md` ([docs/design.md:1528-1580](../docs/design.md#L1528-L1580)): what the master is, that it must sit beside the frames, and that the frame `.h5` layout is unchanged so `readMesh()` still reads any single frame.

**Exit criterion:** `ctest -L unit -R xdmf_series_SERIAL` passes at np1, 2 and 3.
The test builds a small icosphere, distributes it, writes three frames through
one `MeshSeries` at times `0.0, 0.5, 1.25` — stems per the Test output paths
convention above — and asserts on rank 0: the master
`.xmf` exists and `<master>.xmf.tmp` does not; it contains exactly one
`CollectionType="Temporal"`, exactly three `<Time Value=` occurrences whose
parsed values are `0.0, 0.5, 1.25` in that order, exactly three `<Topology`,
exactly three `<Geometry`, and one `Format="HDF"` reference per frame naming
that frame's basename; each named `.h5` exists; `python3 -c "import
xml.etree.ElementTree as E; E.parse(...)"` exits 0 on the master.
Failure direction, both asserted to throw `std::runtime_error` and to leave the
previously written master byte-unchanged: a fourth `write()` at time `1.25`
(non-increasing), and a `write()` whose frame stem is in a subdirectory of the
master's directory.

**Met.** `ctest -L unit -R xdmf_series_SERIAL` passes 3/3 — np1, np2 and np3 —
at commit `0645f88` + this change, job `f3T2sWHo5ZMy`, spack env
`~/spack_envs/tuolumne_trilinos/`. Every target in the build tree also compiles.
Assertions that ran, all of them on the master produced by three
`MeshSeries::write()` calls on a distributed icosphere(2) carrying one vertex and
one face user field: `<master>.xmf` exists and `<master>.xmf.tmp` does not;
exactly one `CollectionType="Temporal"`; exactly three `<Time Value=` parsing to
`0.0`, `0.5`, `1.25` in that order; exactly three `<Topology` and three
`<Geometry`; each child grid carries `Format="HDF"` references to its own frame's
`.h5` basename **and to no other** `.h5`, and each of those three files exists on
disk; the `<Attribute Name=` list is identical in the same order across all three
children (R1's diagnostic — `v_gid`, `f_level`, `vu0`, `fu0`); `python3 -c
"import xml.etree.ElementTree as E; E.parse(...)"` exits 0 on the master; and
`numFrames()`/`masterStem()` are checked on **every** rank, which is what pins
the accumulator as rank-uniform. Both failure directions throw
`std::runtime_error` and leave the master byte-identical to the string read
before the attempt: a fourth `write()` at `1.25` (also asserted to leave no
`_frame3.h5` behind, i.e. it threw before any I/O) and a `write()` into a `sub/`
subdirectory of the master's directory. The emitted master was additionally
inspected by hand once, with the test's cleanup temporarily suppressed (that edit
is reverted), confirming the collection shape and the user-field attribute block
rather than only the counts. See
[the progress log](fix-file-grouping-io-progress-log.md#t2).

### T3 — `mesh_pipeline` writes a series, so the reported symptom is gone from the shipped example — **NOT STARTED**

**Depends on:** T2

**Fill in:** `examples/02_mesh_pipeline/mesh_pipeline.cpp` (the `run()` frame
writes at [lines 155](../examples/02_mesh_pipeline/mesh_pipeline.cpp#L155) and
[221](../examples/02_mesh_pipeline/mesh_pipeline.cpp#L221), and the
`frameStem()` helper at [lines 107-114](../examples/02_mesh_pipeline/mesh_pipeline.cpp#L107-L114));
`README.md` ([line 572](../README.md#L572), the `mesh_pipeline` row).

**Do:**
1. Construct one `MeshSeries` per `run()` invocation, master stem
   `<out>_<tag>_np<size>` — i.e. `frameStem()` minus the `_frame<i>` suffix, so
   the master sits beside the frames as required and each
   backend/rank-count/refine-mode combination gets its own series.
2. Replace both `writeMesh()` calls with `series.write( mesh, frameStem(...), time )`.
   The example has no physical time: pass the frame index as a `double` and say
   so in a comment, since that is exactly the explicit-caller-choice the API
   requires.
3. Keep the existing per-frame `printf` lines; add one rank-0 line at the end of
   `run()` naming the master `.xmf` to open in Paraview.
4. Update the `mesh_pipeline` README row: it now writes one master `.xmf` plus
   per-frame `.h5`/`.xmf`, and the master is what to open.

**Exit criterion:** running the example (Tuolumne: `scripts/tuolumne/run_mesh_pipeline_example.flux`;
see [docs/compile-and-run.md](../docs/compile-and-run.md)) with `--iters 3` produces
exactly one `<out>_<tag>_np<size>.xmf` per execution-space tag, each containing 4
`<Time Value=` entries (frame 0 plus 3 iterations) with values `0,1,2,3`, and the
run exits 0. Failure direction: an `.xmf` naming an `.h5` that does not exist on
disk must be absent — assert every `Format="HDF"` reference in the master
resolves to an existing file in the master's directory.

### T4 — Restart: reopen an existing series and keep appending — **NOT STARTED**

**Depends on:** T2. Independent of T3. Beantik needs it: its restarts resume
mid-series.

**Fill in:** `src/Tessera_XdmfSeries.hpp`, `tests/test_xdmf_series.cpp`.

**Do:**
1. Add a reopen constructor distinguished by a tag type, not a bool:
   `struct ReopenSeries {}; MeshSeries( std::string masterStem, ReopenSeries )`.
2. Read `<masterStem>.xmfindex` line by line for `(frameStem, time)`. Throw if
   the index is missing, malformed, or its times are not strictly increasing —
   a silently truncated series is exactly the quiet-degradation this project
   forbids.
3. Rebuild each `XdmfFrame` by reopening `<frameStem>.h5` **serially on rank 0**
   (`H5Fopen`, `H5P_DEFAULT` — not MPI-IO) and reading the root attributes with
   the existing helpers `detail::readIntAttr` / `detail::readU64Attr`
   ([src/Tessera_IoCommon.hpp:308-323](../src/Tessera_IoCommon.hpp#L308-L323)):
   `dim`, `scalar_bytes`, `Nv`, `Nf`, `n_user_v_fields` / `n_user_f_fields`, and
   `uv_ext_<j>` / `uf_ext_<j>` for the extents. Reconstruct the field display
   names by the same rule the writer uses — `"v" + "u<j>"` and `"f" + "u<j>"`
   ([src/Tessera_HDF5Writer.hpp:250](../src/Tessera_HDF5Writer.hpp#L250),
   [src/Tessera_HDF5Writer.hpp:361](../src/Tessera_HDF5Writer.hpp#L361)). Do not
   validate against the mesh template here: this constructor takes no mesh.
4. Rewrite the master immediately from the reconstructed steps, so a reopen that
   succeeds is visible on disk before any new frame is written.

**Exit criterion:** `ctest -L unit -R xdmf_series_SERIAL` passes at np1-3 with an
added case that writes two frames through one `MeshSeries`, destroys it,
constructs a second with `ReopenSeries`, writes a third frame, and asserts the
master then holds three `<Time Value=` entries in increasing order and that the
first two child grids are byte-identical to those the first object wrote.
Failure direction: reopening with the `.xmfindex` deleted, and with its two time
lines swapped into decreasing order, each throws `std::runtime_error` naming the
index path.

### V1 — Confirm in Paraview, in both reader families — **NOT STARTED**

**Depends on:** T2 (T3 gives a larger, more convincing dataset). Isolated
because it may block: neither `paraview` nor `pvpython` is installed on this
system, so this needs a machine with a Paraview install and cannot run in CI.

**Do:** open a T2- or T3-produced master `.xmf` in the Paraview GUI. Where the
open dialog offers a reader choice, verify **both** the XDMF3 temporal reader
(`Xdmf3ReaderT` — the variant that walks a temporal collection; `Xdmf3ReaderS`
is the single-file/spatial one) and the legacy `XDMFReader` (Xdmf2). For each:
confirm the Information tab lists N timesteps, the time slider animates through
all N, the surface renders at every step, and `v_gid` / `f_level` / `vu0` /
`fu0` are all selectable and change with the step. Record which readers worked,
with versions, in the progress log.

**Exit criterion:** a progress-log entry naming the Paraview version, each
reader tried, and for each the observed timestep count (must equal the number of
frames written) and whether all attributes coloured correctly. Failure
direction: if a reader shows 1 timestep, or 0, or errors, the entry records
which, and R2 or R3 becomes an open task rather than V1 being marked DONE.

## Known risks

**R1 — Attribute set drifts between frames.** Paraview flags attributes missing
at some timesteps as "partial", and a reported failure mode is the *first*
scalar in the file returning garbled values pulled from other arrays. Tessera's
attribute names come from the compile-time field pack, so within one run they
are constant — but two runs with different mesh template parameters must never
share a master. *Presents as:* correct geometry, wrong or flickering field
values, often only for one array. *Do:* `MeshSeries` never mixes frames from
different mesh types by construction (one object per `run()` in T3); if this is
ever seen, first diff the `<Attribute Name=` sets across child grids in the
master — they must be identical, in the same order.

**R2 — The wrong Paraview reader is used.** Paraview offers up to three XDMF
readers, and only the temporal XDMF3 reader walks a temporal collection. *
Presents as:* the master opens showing exactly **one** timestep — visually very
close to the symptom being fixed. *Distinguishing measurement:* count
`<Time Value=` in the master with `grep -c`. If the file has N and Paraview
shows 1, this is R2 (a reader-selection issue, documented in the README, not a
code bug); if the file itself has fewer than N, the bug is in
`writeXdmfSeries()` or in `MeshSeries`'s step accumulation.

**R3 — The `Version="3.0"` declaration with XDMF2 element spellings breaks the
XDMF3 temporal reader**, even though it is tolerated for a single uniform grid
today. *Presents as:* the master fails to parse or renders nothing under
`Xdmf3ReaderT` while the individual per-frame sidecars still open fine.
*Distinguishing from R2:* R2 shows one timestep and renders; R3 errors or
renders nothing. *Do:* do **not** pre-emptively rewrite the spellings — V1 is
what decides. If R3 fires, the minimal fix is the child-grid element names in
`detail::writeXdmfGrid()`, which is one function precisely because T1 factored
it.

**R4 — Rewrite cost on very long runs.** The master is rewritten every frame, so
total text written is O(frames²). At 10 000 frames that is ~10⁸ characters
cumulative on rank 0 — still small against 10 000 collective HDF5 writes, but
not free. *Presents as:* rank-0-only wall-clock creep late in a long run, with
`TIMER_WRITE_MESH` flat (the rewrite is outside it). *Do:* if measured as
material, add an explicit `flush()`-style cadence control as a new task; do not
silently switch to finalize-only writing, which reintroduces the
crashed-run-has-no-master failure this design rejected.

**R5 — A frame stem in a different directory from the master.** `.xmf` files
reference `.h5` by basename only, so a master and its frames in different
directories yield a master full of unresolvable references — which Paraview
reports as a missing-array or empty-dataset error, not as a path error.
*Presents as:* the master opens with the right timestep count but nothing
renders. *Do:* T2 step 4 makes this throw at `write()` time with both paths
named. If this is ever seen at runtime, that check is missing or was bypassed.
