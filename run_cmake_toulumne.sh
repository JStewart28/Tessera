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
# Configure Tessera for Tuolumne (AMD MI300A APU — Serial + OpenMP + HIP).
#
# Prerequisites:
#   spack env activate ~/spack_envs/tuolumne_trilinos/
#
# Usage (run from a build directory OR from the repo root):
#   mkdir build-tuolumne && cd build-tuolumne
#   bash ../run_cmake_toulumne.sh [extra cmake args]
#
# The MPIEXEC_* overrides make `ctest` the single launch entry point:
#   ctest invokes each test as:
#     flux run --ntasks N --nodes=1 --exclusive --cores-per-task=1 <exe>
#   Without them, FindMPI auto-detects `srun`, which deadlocks at >=3 ranks
#   because unbound ranks contend on the MI300A APU during Kokkos::initialize.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parallel HDF5 (Step 8). The Cray parallel HDF5 is a spack EXTERNAL, so
# `spack env activate` does not view-link its prefix onto CMAKE_PREFIX_PATH;
# a bare find_package(HDF5) would silently resolve the OS serial
# /usr/lib64 build instead (see tasks/completed/milestone1_mesh.md, Step-0 entry).
: "${HDF5_ROOT:=$(spack location -i hdf5 2>/dev/null || \
    echo /opt/cray/pe/hdf5-parallel/1.14.3.7/crayclang/20.0)}"

cmake \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER=CC \
    -DCMAKE_C_COMPILER=cc \
    -DCMAKE_HIP_COMPILER=amdclang++ \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DTessera_ENABLE_TESTING=ON \
    -DTessera_ENABLE_EXAMPLES=ON \
    -DTessera_ENABLE_PROFILING=ON \
    -DTessera_PROFILING_LEVEL=0 \
    -DMPIEXEC_EXECUTABLE="$(which flux)" \
    "-DMPIEXEC_NUMPROC_FLAG=run;--ntasks" \
    "-DMPIEXEC_PREFLAGS=--nodes=1;--exclusive;--cores-per-task=1;--env=GLIBC_TUNABLES=glibc.rtld.optional_static_tls=8388608" \
    -DHDF5_ROOT="${HDF5_ROOT}" \
    "$@" \
    "${SCRIPT_DIR}"
