# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone builder for the optional ``vllm.geminifs_ops`` extension.

This compiles ONLY the GeminiFS pybind module (``csrc/geminifs/pybind.cu``) and
drops the resulting ``vllm/geminifs_ops.*.so`` next to the rest of the vLLM
package. It is meant for setups where vLLM itself is installed from a
precompiled wheel (``VLLM_USE_PRECOMPILED=1``), so the main CMake build never
runs. GeminiFS sources are NEVER compiled here; we only link the prebuilt
``libgeminifs.so`` / ``libnvm.so`` from an external GeminiFS checkout.

Usage:
    export GEMINIFS_ROOT=/path/to/Geminifs   # prebuilt checkout
    python build_geminifs.py                 # builds vllm/geminifs_ops.*.so

After building:
    from vllm import geminifs_ops
    geminifs_ops.GeminiFS(...)
"""

import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = Path(__file__).parent.resolve()

# Default to the known local checkout but allow override via env.
DEFAULT_GEMINIFS_ROOT = "/home/zfw/nvme-over-ub/Geminifs"


def _resolve_geminifs_root() -> Path:
    raw_root = (
        os.environ.get("GEMINIFS_ROOT")
        or os.environ.get("LMCACHE_GEMINIFS_ROOT")
        or DEFAULT_GEMINIFS_ROOT
    )
    root = Path(raw_root).expanduser().resolve()
    if not root.exists():
        raise RuntimeError(
            f"GeminiFS root does not exist: {root}. Set GEMINIFS_ROOT to your "
            "prebuilt GeminiFS checkout."
        )
    return root


def _require(path: Path, description: str) -> None:
    if not path.exists():
        raise RuntimeError(
            f"Missing GeminiFS {description}: {path}. Build GeminiFS first; "
            "this script will NOT compile GeminiFS sources."
        )


def _build_extension() -> CUDAExtension:
    geminifs_root = _resolve_geminifs_root()
    inc_geminifs = geminifs_root / "libgeminifs" / "include"
    inc_nvm = geminifs_root / "libnvm" / "include"
    lib_dir = geminifs_root / "build" / "lib"

    # Validate the prebuilt artifacts we link against.
    _require(inc_geminifs / "geminifs.cuh", "header")
    _require(inc_geminifs / "io_deamon.cuh", "header")
    _require(inc_nvm, "libnvm include directory")
    _require(lib_dir / "libgeminifs.so", "shared library")
    _require(lib_dir / "libnvm.so", "shared library")

    source = ROOT_DIR / "csrc" / "geminifs" / "pybind.cu"
    _require(source, "pybind source")

    print(f"Building vllm.geminifs_ops against GeminiFS root: {geminifs_root}")

    return CUDAExtension(
        # Installed as vllm/geminifs_ops.*.so so it imports as vllm.geminifs_ops.
        name="vllm.geminifs_ops",
        sources=[str(source)],
        include_dirs=[
            str(ROOT_DIR / "csrc" / "geminifs"),
            str(inc_geminifs),
            str(inc_nvm),
        ],
        library_dirs=[str(lib_dir)],
        libraries=["geminifs", "nvm"],
        # Bake an RPATH so the module finds libgeminifs.so / libnvm.so at runtime.
        runtime_library_dirs=[str(lib_dir)],
        # torch's cpp_extension injects the matching -D_GLIBCXX_USE_CXX11_ABI
        # flag automatically; GeminiFS headers require C++20.
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20"],
            "nvcc": ["-O3", "-std=c++20"],
        },
    )


if __name__ == "__main__":
    # Build in place (writes vllm/geminifs_ops.*.so) unless the caller passes
    # explicit setuptools commands.
    if len(sys.argv) == 1:
        sys.argv += ["build_ext", "--inplace"]

    setup(
        name="vllm-geminifs",
        ext_modules=[_build_extension()],
        cmdclass={"build_ext": BuildExtension},
    )
