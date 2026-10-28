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
# DEV example-runner (not a release artifact — mesh_pipeline is a demo, not a
# ctest gate entry). Submits a batch job to the pdebug queue that runs the
# 02_mesh_pipeline example at 4 MPI ranks. Kokkos::DefaultExecutionSpace is
# HIP on Tuolumne, so the example's per-exec-space loop exercises the HIP
# backend (in addition to its always-run Serial pass) and, for each, writes its
# parallel HDF5+XDMF frames through one MeshSeries.
#
# TESSERA_REPO must be exported in the submitting shell (flux copies this
# script to a temp dir, so it cannot locate the repo from its own path).
# Submit with:
#   export TESSERA_REPO=$(pwd)
#   flux batch scripts/tuolumne/run_mesh_pipeline_example.flux [MESH_PIPELINE_ARGS...]
# Examples:
#   flux batch scripts/tuolumne/run_mesh_pipeline_example.flux
#   flux batch scripts/tuolumne/run_mesh_pipeline_example.flux --subdiv 3 --iters 5 --frac 0.15 --balance
#
# Output lands in the job's working directory (the directory `flux batch` was
# submitted from) unless --out names another one. Each execution space's series
# writes per-frame .h5/.xmf files plus one master <out>_<tag>_np<size>.xmf (and
# its .xmfindex); open the master in Paraview to step through the sequence as
# one timestepped dataset.

#FLUX: --job-name=tessera-mesh-pipeline-example
#FLUX: --queue=pdebug
#FLUX: --nodes=1
#FLUX: --exclusive
#FLUX: -t 20m
#FLUX: --output=tessera-mesh-pipeline-example.{{id}}.out

set -euo pipefail

# TESSERA_REPO comes from the submitting environment (flux captures it at
# submit time). Source the resolver (activates the spack env + runtime_env.sh
# so MPICH_GPU_*, HSA_XNACK, etc. reach the flux-run sub-task).
: "${TESSERA_REPO:?export TESSERA_REPO=<repo root> before submitting this job}"
source "${TESSERA_REPO}/scripts/lib/tessera_env.sh"

exe=$(tessera_exe examples/02_mesh_pipeline/mesh_pipeline)

echo "=== Tessera mesh_pipeline example (pdebug, 4 ranks, HIP) ==="
echo "  System:    ${TESSERA_SYSTEM}"
echo "  Build dir: ${TESSERA_BUILD_DIR}"
echo "  Exe:       ${exe}"
echo "  Args:      $*"
echo "  Commit:    $(git -C "${TESSERA_REPO}" rev-parse HEAD)"
echo "  Spack env: $(spack env status || true)"
echo "  Started:   $(date)"
echo ""

flux run --ntasks 4 --nodes=1 --exclusive --cores-per-task=16 \
    --env=GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8388608 \
    "${exe}" "$@"
_rc=$?

echo ""
echo "=== mesh_pipeline example complete (rc=${_rc}): $(date) ==="
exit ${_rc}
