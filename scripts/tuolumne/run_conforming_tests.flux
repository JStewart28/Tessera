#!/usr/bin/env bash
############################################################################
# Copyright (c) 2024, JStewart28                                           #
# All rights reserved.                                                     #
#                                                                          #
# This file is part of the Tessera library. Tessera is distributed under a #
# BSD 3-Clause license. For the licensing terms see the LICENSE file in   #
# the top-level directory.                                                 #
#                                                                          #
# SPDX-License-Identifier: BSD-3-Clause                                    #
############################################################################
#
# Conforming-refinement verification sweep (Task 8 of
# tasks/completed/conforming-refinement.md).
#
# Submit: flux batch scripts/tuolumne/run_conforming_tests.flux
#
# THIS IS NOT THE SHIP GATE. The gate is
# scripts/tuolumne/run_regression_minset.flux (label regression x
# {SERIAL,HIP} x ranks 1-5) and its definition is unchanged by this file.
#
# Scope: every test touched by the conforming-refinement work, which after
# Task 7 is the WHOLE suite -- Mesh gained a template parameter, its default
# flipped to Conforming, distribute()/migrate()/the HDF5 reader all changed,
# and every mode-insensitive test now runs in Conforming mode. Rather than
# enumerate (and drift), this runs both tiers (`regression` and `unit`) over
# both gated backends.
#
# Ranks: single-rank tests are registered at np1 only and multi-rank tests at
# np1..np5, so excluding `_np5` yields exactly "non-MPI tests at 1 process,
# MPI tests at 1, 2, 3, 4" as requested for this triage pass. The gate's np5
# runs come later, from run_regression_minset.flux.
#
# Per-test timeout is bounded so one hang cannot consume the 20-minute
# allocation; a TIMEOUT verdict is a finding, not an infrastructure problem.

#FLUX: --job-name=tessera-conforming-tests
#FLUX: --nodes=1
#FLUX: --exclusive
#FLUX: --queue=pdebug
#FLUX: --time-limit=20m
#FLUX: --output=tessera-conforming-tests.{{id}}.out

set -uo pipefail

# Source the resolver: activates spack env + runtime_env.sh (MPICH_GPU_*,
# HSA_XNACK, OMP_*, etc.) so they reach every flux-run sub-task.
source "${TESSERA_REPO}/scripts/lib/tessera_env.sh"

echo "=== Tessera conforming-refinement test sweep (Task 8 triage) ==="
echo "  System:    ${TESSERA_SYSTEM}"
echo "  Build dir: ${TESSERA_BUILD_DIR}"
echo "  Tiers:     regression + unit"
echo "  Backends:  SERIAL HIP"
echo "  Ranks:     1 2 3 4  (np5 excluded)"
echo "  Started:   $(date)"
echo ""

cd "${TESSERA_BUILD_DIR}"

ctest -R "SERIAL|HIP" -E "_np5$" \
      --timeout 100 \
      --output-on-failure \
      --no-tests=error
status=$?

echo ""
echo "=== Sweep complete (ctest exit ${status}): $(date) ==="
exit ${status}
