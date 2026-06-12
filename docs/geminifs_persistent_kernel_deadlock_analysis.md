# GeminiFS Persistent IO-Daemon vs. vLLM GEMM: Scheduling Deadlock — Verified Analysis

Scope: why the GeminiFS persistent IO-daemon kernels deadlock vLLM's model forward,
*exactly which* operation conflicts, and why the on-demand-daemon commit
(`1c0e4fb`) avoids it. All claims below are reproduced in isolation on this box
(2× H100 PCIe, 114 SMs, CUDA 12.8); repro sources in `/tmp/gemini_sched/`.

---

## 1. The persistent kernel (the SM-resident side of the conflict)

`launch_io_deamon_kernels()` → `DeamonManager::launch_deamons_server_kernels()`
→ per-GPU `DeamonState::launch_server_kernels()`
(`Geminifs/libgeminifs/io_deamon.cu:262`). For the **compute GPU** it launches,
once per client GPU (`gpu_num_ = cudaGetDeviceCount() = 2`):

| Kernel | Launch | Resident blocks on compute GPU |
|---|---|---|
| `io_deamon_server_kernel_request_scheduler` (`io_request_scheduler.cu:274`) | `<<<3, 736>>>` ×2 | **6 blocks × 736 threads** |
| `nvl_start_server_kernel` | `<<<1,1>>>` ×2 | 2 blocks × 1 thread |

The scheduler kernel is an unconditional `while (true) { … __nanosleep(20); }`
spin (`io_request_scheduler.cu:295`); it only exits when the host flips
`ctrl->stop_flag` to `STOP_DRAINED`. So **once launched, these blocks stay
resident on the compute GPU forever** and the GPU is never idle.

`PIPELINE_TOTAL_THREADS = 32 (ctrl) + 192 (nvlink) + 512 (pcie) = 736`,
`REQUEST_SCHEDULER_GRID_BLOCKS = 1 scheduler + 2 workers = 3`
(`include/io_deamon_defines.cuh`).

## 2. The conflicting operation (the load side of the conflict)

**It is not SM contention, and not a cooperative/stream-K GEMM needing all SMs.**
The conflict is **CUDA lazy *kernel loading*** — the first-ever launch of a
kernel whose code has not yet been instantiated into the CUDA context. That
first launch performs a one-time device-side load that requires a **device-wide
quiescent point**, which the never-exiting daemon spin denies. The load (and
therefore the kernel) blocks forever.

In vLLM the kernel that hits this is a **cuBLAS GEMM (`extern_kernels.mm` /
`cublasLtMatmul`) launched eagerly (cudagraph-bypassed) for a shape that was not
warmed before the daemon went resident** — concretely, an intermediate
**chunked-prefill** chunk. py-spy on the hung EngineCore froze at
`compilation/cuda_graph.py:223 → piecewise_backend.py:343 → inductor run →
extern_kernels.mm`, i.e. an eager cuBLAS GEMM, inside `cublasLtMatmul → libcuda`.

## 3. Verification (isolated repro, no NVMe → no panic hazard)

The daemon is modeled by a stoppable spin kernel (`/tmp/gemini_sched/repro.cu`,
`probe.cu`): N resident blocks polling a host flag. A GEMM / kernel is then
issued under a watchdog; on timeout the spinner is released and we observe
whether the victim then completes.

| # | Setup | Result |
|---|---|---|
| 1 | 6×736 spinner resident, cold BF16 4096³ `cublasLtMatmul` | **HANG** inside the matmul call; completes 0.02 s after spinner release |
| 2 | + `CUDA_MODULE_LOADING=EAGER` | **still HANG** → not covered by the env var |
| 3 | Warm the **identical** 4096³ GEMM *before* spinner, rerun after | **no hang** (0.05 s) → cost is first-use only |
| 4 | Warm a **different** shape (512³), then 4096³ after spinner | **HANG** → it is the *specific* kernel's first load |
| 5 | EAGER + `CUBLAS_WORKSPACE_CONFIG=:0:0` + `CUBLASLT_WORKSPACE_SIZE=0` + ws=0 | **still HANG** → not the workspace `cudaMalloc` |
| 6 | **1×1** spinner (one thread, 1 of 114 SMs) vs cold GEMM / cold kernel | **HANG** → not SM availability |
| 7 | Generic trivial custom kernel, cold first launch, 6×736 spinner | **HANG** at execution; 0.02 s after release |
| 8 | Generic kernel + `CUDA_MODULE_LOADING=EAGER` | **no hang** (preloaded) |
| 9 | Generic kernel warmed before spinner | **no hang** |

What each row rules in/out:
- **#6**: one resident thread (free SMs abundant) still deadlocks ⇒ refutes
  "stream-K / cooperative kernel needs all SMs co-resident". The blocker is a
  device-wide quiescence requirement, not occupancy.
- **#5**: workspace = 0 still deadlocks ⇒ refutes "cuBLAS workspace `cudaMalloc`
  blocks on the busy device". (`cudaMalloc` may also block, but it is not the
  cause of *this* hang.)
- **#3 vs #4**: warming the exact shape fixes it, a different shape does not ⇒
  the cost is the **first-use load of that one kernel**.
- **#7/#8/#9 vs #1/#2**: a plain custom kernel deadlocks cold but is fixed by
  EAGER *or* warming; a cuBLAS kernel is fixed by warming but **not** by EAGER.
  ⇒ `CUDA_MODULE_LOADING=EAGER` eagerly loads the application's own modules but
  **not cuBLAS's kernels**, which cuBLAS loads through its own deferred
  (algo-selection-time) path. Only actually executing that GEMM shape preloads
  it.

Mechanism (behaviorally proven; driver internal inferred): lazy loading a kernel
must install code into the device and allocate device memory for it, an
operation the driver serializes behind all resident grids / a device-idle point.
The daemon's infinite spin guarantees the device is never idle, so the load — and
thus the victim kernel — hangs indefinitely. Releasing a single resident block
lets the device quiesce and the load completes in ~20 ms.

## 4. Why this explains every vLLM observation

- **Persistent mode works with CUDA graphs + deferred daemon launch:** the daemon
  is launched only at the first KV transfer (`geminifs.py: ensure_daemon_launched`,
  deferred past warmup). vLLM's warmup + CUDA-graph capture executes the full
  decode size range (1…512) and the single-forward prefill *before* the daemon is
  resident, **preloading all those cuBLAS kernels**. Graph *replay* loads nothing
  new ⇒ no lazy load under the daemon ⇒ no deadlock.
- **Chunked prefill is the trigger:** an intermediate prefill chunk runs an
  **eager** GEMM of a shape/kernel that capture never warmed. Its first load
  happens while the daemon is resident ⇒ deadlock (rows #1/#4). This is exactly
  the py-spy frozen frame.
- **`max_num_batched_tokens ≥ max_model_len` is a fragile patch:** it forces the
  whole prefill into one forward whose kernels match what warmup already loaded;
  a longer prompt re-chunks and re-deadlocks.
- **The cuBLAS env vars don't reliably help:** they steer kernel/workspace
  selection but the deadlock is lazy load, not workspace or stream-K (rows #2/#5).

## 5. Why the on-demand daemon (`1c0e4fb`) is the correct fix

`_DaemonController` brings the daemon up only for the duration of a KV transfer
and tears it down (`stop_io_deamon_kernels`) before control returns to the
engine's forward (`worker/geminifs.py:104`, `transfer_async` synchronous path).
The daemon is therefore **never resident while any model GEMM runs**, so no
kernel ever has to lazy-load against a spinning daemon — by construction,
independent of chunking, `enforce_eager`, or whether a shape was warmed. Cost:
loss of transfer/compute overlap (each transfer blocks its step), which is the
documented trade-off.

## 6. One-line takeaway

The persistent IO-daemon spin keeps the compute GPU permanently non-idle; the
first-use **lazy load of an un-warmed cuBLAS GEMM kernel** (an eager
chunked-prefill `extern_kernels.mm`) needs a device-idle point to install its
code and therefore hangs forever. It is a *kernel-loading*-vs-*persistent-spin*
deadlock, not an SM-occupancy or workspace one — proven by a single 1-thread
resident block deadlocking a cold GEMM while warming or preloading that exact
kernel removes the hang.

## 7. MPS (`nvidia-cuda-mps-control`) evaluated and ruled out

Hypothesis: would running under NVIDIA MPS (Multi-Process Service) let the daemon
spin and the model GEMM coexist? Tested directly on this box (2× H100 PCIe, CUDA
12.8); repro is a two-process variant of the spinner-vs-cold-GEMM harness
(`/tmp/gemini_sched/mps_test.cu`): a persistent 6×736 spinner held resident in
one process, a cold BF16 4096³ `cublasLtMatmul` issued from a second process.

| # | Architecture | MPS | Result |
|---|---|---|---|
| 3 | spinner + GEMM, **same process / same context** | **on** | **HANG** — GEMM never completes, GPU pinned 100% |
| 1 | daemon ∥ victim, **two processes** | off | **No hang** — cold GEMM 0.150 s |
| 2 | daemon ∥ victim, **two processes** | on | **No hang** — cold GEMM 0.150 s |

(Test numbering continues §3's table. Row 3 here = the §3 single-process baseline
with an MPS server interposed; it still hangs.)

**Conclusion: MPS does not solve this deadlock.**

- **MPS is a multi-*process* service; the deadlock is single-*context*.** In
  GeminiFS the IO-daemon kernels and the model GEMM run in the *same process /
  same CUDA context*. A lone process connected to MPS is still one client context
  holding both the spinner and the un-warmed GEMM, so the lazy-load-vs-spin
  deadlock is unchanged (row 3 still hangs). "Just enable MPS" changes nothing.
- **Process/context separation is what breaks the deadlock — and it does not need
  MPS** (row 1). With two contexts the GPU time-slices: switching to the victim's
  context preempts the daemon grid, giving the cold cuBLAS kernel the device-idle
  window its lazy load requires. Row 2 (MPS on) behaves identically to row 1 (MPS
  off), so MPS is neither the mechanism nor a differentiator.
- **The separate-process route is a real re-architecture MPS does not provide.**
  The daemon needs direct access to vLLM's KV-cache GPU memory, which lives in a
  different address space — requiring CUDA IPC handles / peer mapping. Volta+ MPS
  clients have *separate* address spaces, so MPS gives concurrent scheduling, not
  shared KV pointers.
- **Where MPS *could* add value (but not as a deadlock fix):** if the daemon were
  moved to its own process, plain multi-process only time-slices (no real
  overlap). MPS would then let the daemon's transfer kernels and the model kernels
  run *concurrently*, restoring the compute/transfer overlap the on-demand daemon
  (`1c0e4fb`) gives up. That is a *performance* lever available only **after**
  building cross-process KV IPC — it does not address the deadlock itself.

Bottom line: the on-demand daemon (`1c0e4fb`) remains the correct fix. MPS is
irrelevant to the single-process deadlock, and the multi-process route that would
avoid it works without MPS anyway.

## 8. Explicit kernel warm-up evaluated — a real fix, and a cheap one

Hypothesis: instead of tearing the daemon down per transfer, **pre-execute every
cuBLAS GEMM kernel chunked prefill could hit, before the daemon goes resident**,
so no kernel ever has to lazy-load against the spin. Tested directly on this box
(2× H100 PCIe, CUDA 12.8). Repro: `/tmp/gemini_sched/warm_all.cu` (warms an
arbitrary set of token-counts M, then issues ONE held-out M under the resident
6×736 spinner; **all buffers/layouts/algos are built before the spinner so only
`cublasLtMatmul` runs under it** — isolating kernel lazy-load from any
`cudaMalloc` stall) plus `algo_scan.cu`/`algo_dump.cu` (hash the heuristic's
chosen algo bytes per M to count distinct kernels). Model GEMM is BF16, N=K=4096
unless noted; M = chunk token count.

| # | Setup | Result |
|---|---|---|
| 1 | No warm, test M=768 | **HANG**; completes 0.02 s after spinner release (baseline reproduces the deadlock) |
| 2 | Warm **exact** M=768, test 768 | **PASS** — confirms §3 row #3 |
| 3 | Warm coarse grid {256,512,1024,2048}, test 768 | **HANG** — an in-between M is not covered |
| 4 | Warm 768, test **769** | **HANG** — off-by-one is a different kernel |
| 5 | Warm 769, test 771 / 773 (**same algo-config**) | **PASS** |
| 6 | Warm 776, test 800 (**same algo-config**) | **PASS** |
| 7 | Warm 769, test 770 (**different algo-config**) | **HANG** |

What this proves:
- **Lazy-load is keyed on the cuBLAS algo-config (the cubin), not the raw
  token-count M.** Rows #5/#6 vs #7: two M-values that the heuristic maps to the
  *same* algo cross-cover (warming one preloads the other); two that map to
  *different* algos do not. Row #4 shows even M and M+1 can map to different
  algos, which is why a coarse grid (row #3) misses the gaps.
- **The distinct-kernel set is small.** Across M=1..2048 the heuristic retunes
  tile/split-K at nearly every adjacent M, but those collapse to only a few dozen
  distinct algo-configs **per (N,K) GEMM shape**:

  | GEMM (Llama-3-8B dims) | N, K | distinct configs over M∈[1,2048] |
  |---|---|---|
  | qkv_proj | 6144, 4096 | 64 |
  | o_proj | 4096, 4096 | 60 |
  | gate_up_proj | 28672, 4096 | 27 |
  | down_proj | 4096, 14336 | 62 |
  | lm_head | 128256, 4096 | 26 |

  So the whole model is on the order of a few hundred distinct GEMM kernels.
  Warming 64 GEMMs measured at **0.05 s** ⇒ warming every shape is sub-second and
  negligible against startup.

**Harness caveat (a real mismeasurement trap).** A first version allocated the
GEMM's buffers *inside* the under-spinner path; exact-warm then falsely "hung"
because `cudaMalloc` — not lazy-load — was the blocker on the busy device. All
buffers/layouts/algos must be built *before* the spinner (as `repro.cu` does), so
that the only operation issued under the daemon is `cublasLtMatmul`. With that
fixed, exact-warm passes (row #2), consistent with §3.

**Why this beats `CUDA_MODULE_LOADING=EAGER`.** EAGER does not preload cuBLAS
kernels (§3 rows #2/#8 — cuBLAS loads through its own deferred, algo-selection
path). Warm-up *executes* each shape, which is the only thing that preloads a
cuBLAS cubin. So warm-up succeeds exactly where EAGER fails.

**Practical recipe (not yet implemented in vLLM).** Before the daemon goes
resident in persistent mode, run a dummy forward at **every token-count M in
`[1, max_num_batched_tokens]`**. Any single forward's M is ≤ that budget, so this
preloads the kernels for all full prefill chunks, every possible last-chunk
remainder, and all decode batch sizes in one pass — and because it runs the real
forward it also preloads custom/Triton kernels, not just cuBLAS. (An optimized
variant enumerates the distinct algo-configs per shape via the heuristic scan and
warms one M each, ~300 GEMMs.)

**Trade-off vs the on-demand daemon (`1c0e4fb`).** Warm-up is the one option that
keeps the daemon *persistent* — so it **retains transfer/compute overlap**, which
the on-demand daemon gives up. Cost is a sub-second one-time warm-up plus the
requirement that the warm-up be exhaustive over `[1, max_num_batched_tokens]` for
every GEMM shape. Remaining risk to weigh before shipping: any eager GEMM whose
M, N or K escapes the warmed envelope (e.g. an unusually long single forward, or a
new shape introduced by a config change) re-deadlocks — the same fragility that
makes `max_num_batched_tokens ≥ max_model_len` a patch (§4), now pushed out to the
full token range rather than eliminated by construction. The on-demand daemon
remains deadlock-proof *independent* of shape; warm-up trades that guarantee for
overlap.
