# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin Python wrapper exposing the GeminiFS backend to vLLM.

The native module ``vllm.geminifs_ops`` is an optional, opt-in extension built
only when vLLM is compiled with ``VLLM_BUILD_GEMINIFS=1`` against a prebuilt
GeminiFS checkout (``GEMINIFS_ROOT``). This module re-exports its public symbols
and raises a clear error when the extension is unavailable.
"""

from typing import TYPE_CHECKING

__all__ = ["GeminiFS", "launch_remote_io_xfer_batch"]

_IMPORT_ERROR_HINT = (
    "The GeminiFS extension (vllm.geminifs_ops) is not available. It is an "
    "opt-in component: rebuild vLLM with VLLM_BUILD_GEMINIFS=1 and GEMINIFS_ROOT "
    "pointing at a prebuilt GeminiFS checkout (containing libgeminifs.so / "
    "libnvm.so) to enable it."
)

if TYPE_CHECKING:
    from vllm.geminifs_ops import GeminiFS, launch_remote_io_xfer_batch
else:
    try:
        from vllm.geminifs_ops import GeminiFS, launch_remote_io_xfer_batch
    except ImportError as exc:  # pragma: no cover - depends on build config
        raise ImportError(_IMPORT_ERROR_HINT) from exc
