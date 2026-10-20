# Tessera — Project Control Document

**This is the framework control document.** Read it first in every session; follow
its rules to keep the framework self-consistent across sessions. It is deliberately
short — task-specific detail lives in the reference docs below, read on demand.

| Read this doc | Before |
|---|---|
| [docs/compile-and-run.md](docs/compile-and-run.md) | Building, running an example, submitting a job, or setting up a fresh checkout |
| [docs/testing.md](docs/testing.md) | Running tests or changing what the ship gate covers |
| [docs/framework-maintenance.md](docs/framework-maintenance.md) | Adding a system, changing build/run plumbing, or planning a large multi-commit change |
| [docs/design.md](docs/design.md) | Changing algorithms or data structures (data model, halo, refinement, load balancing, I/O) |

---

## Repository layout

| Directory | Holds |
|---|---|
| `systems/<system>/` | Per-system **machine** instructions (`claude.md`) and env snapshots (e.g. `spack.yaml`). Hostname-keyed build/run facts live here and nowhere else. |
| `docs/` | Project documentation and the on-demand reference docs above. |

`README.md` links to `docs/design.md` rather than duplicating it — when the design
changes, update `docs/design.md` and keep the README's API/build sections in sync.

---

## System detection

Before building or running, run `hostname` and match the prefix:

| Hostname prefix | System token | Per-system instructions |
|---|---|---|
| `tuolumne*` | `tuolumne` | [systems/tuolumne/claude.md](systems/tuolumne/claude.md) |
| *(any other)* | `local` | [systems/local/claude.md](systems/local/claude.md) |

The resolver (`scripts/lib/tessera_env.sh`) performs this match automatically.
Override with `export TESSERA_SYSTEM=<token>` before sourcing.

**Fallback rule:** unmatched hostname or a `systems/<system>/claude.md` missing a
required section → stop and ask the user before proceeding.

---

## General guidelines

- **The full regression gate is a ~30-minute job — don't run it by reflex.** Run
  only the tests covering the code you touched unless the change is broad enough
  to require the whole gate. When it is required, submit it from the repo root,
  save progress, commit and push, and hand off with a prompt that tells a fresh
  session which output file to read. See
  [docs/testing.md](docs/testing.md#when-to-run-the-full-gate).
- **Never run the formatter.** The user formats by hand, or asks for the
  clang-format target explicitly. Do not add it to any workflow or plan.
- **License/header convention:** every new source file (`.cpp`, `.hpp`, `.h`,
  `.cu`, `.sh`, `CMakeLists.txt`, `.cmake`) carries the project's BSD 3-Clause
  header. For cmake/shell files, use the `#`-box style in
  [cmake/FindCLANG_FORMAT.cmake](cmake/FindCLANG_FORMAT.cmake). For C++ files,
  use an equivalent `/* */` block with `SPDX-License-Identifier: BSD-3-Clause`.
- **README.md in sync:** update when a public API or an example's accepted
  arguments change.
- **Future Optimizations:** track in README; ask before adding an entry.
- **Known Issues:** record in README what fails, how it reproduces, and whether
  it predates current work.
- **Describe the code as it is now, never how it used to be.** Comments,
  docstrings, error messages, the README and `docs/` say what the code does at
  present. Do not narrate history: no "X CHANGE" or "X CORRECTION" notes, and
  no "used to", "previously", "is gone", "no longer", "has since landed" or
  "the old version". When you edit a passage that already does this, rewrite it
  in the present tense. Task IDs are fine where they name pending work ("task
  T4d, not yet implemented"). History belongs in git and in the
  `tasks/*-progress-log.md` files, which are chronological by design and exempt.

---

## Invariants — never violate

- **No inline runtime-env in batch scripts** — source the resolver instead.
- **Gate definition is single-sourced:** label + backends + ranks must match
  identically across [docs/testing.md](docs/testing.md), `tests/CMakeLists.txt`,
  every `scripts/<system>/run_regression_minset.*`, and
  `.github/workflows/ci.yml`. Changing the gate is deliberate and requires
  updating all four locations.
- **`profile.local.sh` is never committed** (enforced by `.gitignore`).
- **`profile.defaults.sh` is always committed.**
- **Env snapshots stay in sync** with the live environment in the same change.
- **Example argument changes mirror into README.**
- **New files carry the license/SPDX header.**
- **A failing test is never silently excluded from the gate** — label it `unit`,
  record it in README Known Issues, and report it.

## Math in markdown
Write math with KaTeX delimiters, not Doxygen ones:
- Inline: `$ ... $`  — NOT `\f$ ... \f$`
- Display: `$$ ... $$` on their own lines, blank line above and below — NOT `\f[ ... \f]`

`\f[`/`\f$` are Doxygen-only. In a plain markdown reader (VSCode preview,
GitHub, mdBook) they don't open a math region, so the body is parsed as prose
and CommonMark strips the backslash from every escaped punctuation character —
`\;` becomes `;`, `\,` becomes `,`, `\_` disappears — leaving unreadable output.

KaTeX delimiter rules worth respecting:
- No space just inside the delimiters: `$x + y$`, not `$ x + y $`.
- Don't put a digit immediately after a closing `$`.
- Keep inline math on one line.
- For a literal dollar sign in prose, escape it: `\$`.
