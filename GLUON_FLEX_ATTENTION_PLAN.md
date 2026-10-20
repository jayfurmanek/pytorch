# Gluon backend for FlexAttention's Inductor lowering: plan

Status: draft. Based on `main` at `79f81f9e6d3`, Triton 3.8.0, and an MI350X (gfx950).
This plan applies Phase 3 of `GLUON_INDUCTOR_PLAN.md` (Inductor-generated Gluon
templates) to FlexAttention. It needs only a small subset of that plan's Phases 1
and 2 (milestone 1 below); the full import-table refactor is not a prerequisite.

## Goal

Add a `BACKEND="GLUON"` option to `flex_attention`. It uses a hand-written,
architecture-specific Gluon attention template (MFMA layouts, explicit shared-memory
staging, software pipelining). As with the Triton template, the user's `score_mod`
and `mask_mod` are inlined into it. Initial target: CDNA4 (gfx950), then CDNA3
(gfx942). NVIDIA is out of scope at first.

Because Gluon kernels are architecture-specific, templates are registered per
arch. On any arch without a Gluon template (gfx942 until it is added, other AMD
archs, NVIDIA), `BACKEND="GLUON"` falls back to the Triton forward template with
a one-time warning. See "Arch dispatch and fallback" below.

Non-goals for the first iteration:
- Gluon backward.
- Gluon decode. Short queries take the Triton decode kernel (see "Arch dispatch
  and fallback").
- Epilogue or prologue fusion into the Gluon template.
- Auto-selection under `BACKEND="AUTO"`.

## Findings that shape the design

### 1. FLASH (CuteDSL) is the routing precedent, not the implementation precedent

FLASH gets its own branch in the lowering (`kernel/flex/flex_attention.py:344`),
its own template class (`CuteDSLTemplate`), its own op overrides
(`codegen/cutedsl/cutedsl_op_overrides.py`, about 70 ops), its own scheduling
and its own benchmark request. Gluon does not need most of that:
- Gluon still compiles through `triton.compile`, so `TritonTemplateCaller`,
  `TritonBenchmarkRequest`, `@triton_heuristics.template` / `CachingAutotuner`,
  `autotune_select_algorithm` and Triton scheduling all carry over.
- Gluon keeps the Triton output contract: three outputs (out, LSE, MAX) with LSE
  in log2. FLASH returns only two outputs and natural-log LSE, which is why
  `stats_are_log2 = BACKEND != "FLASH"` exists
  (`nn/attention/flex_attention.py:2617`). A Gluon forward is therefore
  interchangeable with the Triton forward, so **Gluon forward + Triton backward
  works with no extra glue**.

Decision: subclass `TritonTemplate` / `TritonTemplateKernel` rather than build a
separate CuteDSL-style stack.

### 2. Inductor's inlined `score_mod`/`mask_mod` code is almost valid Gluon already

I tested this on gfx950 with `agent_space/gluon_probe{,2,3}.py`. Each `@gluon.jit`
kernel gets index tensors with a concrete `BlockedLayout`, exactly as the
template will pass `m`/`n` in the accumulator's layout.

| Construct (as Inductor emits it today) | In `@gluon.jit` |
|---|---|
| `tl.where`, arithmetic, comparisons, `.to(...)`, `tl.sigmoid`, `tl.abs`, `tl.math.exp2` | OK |
| `tl.load(ptr + idx, mask, other=...)` (captured-buffer gathers) | OK |
| `libdevice.*` (e.g. `tanh`) | OK |
| `tl.broadcast_to`, `tl.reshape` | OK |
| `tl.full(...)` (constants, masked `other`, `_broadcast_index`) | **FAIL**: needs a `layout` |
| `tl.arange(...)` | **FAIL**: needs a `layout` |
| `gl.full(shape, v, dtype)`: default AutoLayout, incl. `[]`, `[1,1]`, broadcast against a concrete layout, as a `where` condition | OK |
| Calling a `@triton.jit` helper | OK, but it compiles under Gluon semantics, so any `tl.full`/`tl.arange` inside it fails |

So `score_mod`/`mask_mod` lowering only needs the tensor-creating ops to
switch to `gl.full` / `gl.zeros_like` with the default AutoLayout. Layout
inference resolves them against the template's concrete layouts. No
layout-propagation logic is needed in Inductor.

### 3. Sites that emit `tl.full`/`tl.arange` on the modification path

- `TritonOverrides._shaped_constant` (`codegen/triton.py:1460`) and
  `TritonKernelOverrides.constant` (`:2532`).
- `TritonKernelOverrides.masked` (`:2605`): emits `tl.full(result.shape, other, ...)`.
- `TritonPrinter`: `tl.full([], ...)` for floats/pow, and `tl.zeros_like`.
- `ModificationWrapper._broadcast_index` (`select_algorithm.py:516`).
- `TritonTemplateKernel.size()` under int64 indexing (`tl.full([], ..., INDEX_DTYPE)`).
- `index_expr` → `TritonKernel.indexing` (`tl.full(expand, ...)` and
  `tl.full(..., True, tl.int1)` masks).

Range-tree `tl.arange` only matters for epilogue fusion and the TMA
`store_output` path. Both are out of scope at first.

### 4. Compile path

- `CachingAutotuner._precompile_config` (`runtime/triton_heuristics.py:1446`)
  hard-codes `ASTSource`. It must use `GluonASTSource` when `self.fn.is_gluon()`.
- `GluonASTSource.make_ir` sets `ttg.num-warps` from options, so the template's
  layouts must be derived from `num_warps`. Use `gl.num_warps()` / a
  `constexpr_function` inside the template so that every autotune config stays
  self-consistent.
- Other `ASTSource`/`.src.fn` users that need auditing: the static launcher
  (`runtime/static_triton_launcher.py`) and kernel pickling to compile workers.

## Design

### New pieces

1. **`GluonTemplate(TritonTemplate)` / `GluonTemplateKernel(TritonTemplateKernel)`**
   in `torch/_inductor/codegen/gluon_template.py` (done, milestone 2):
   - `jit_decorator = "@gluon.jit"`, a new class attribute on
     `TritonTemplateKernel` that `jit_lines` uses.
   - Imports: `gen_common_triton_imports` override adds
     `from triton.experimental import gluon` and
     `from triton.experimental.gluon import language as gl`.
   - `generate_and_load` constexpr defines stay `NAME : tl.constexpr = ...`;
     Gluon accepts them as-is.
   - Everything Inductor generates into the template (`modification()`,
     `store_output` / epilogue hooks via `_make_codegen_hook`, `size()`) passes
     through `triton_to_gluon`, which rewrites `tl.full` / `tl.zeros_like` /
     `tl.zeros` / `tl.arange` to their `gl.*` counterparts. With no `layout`
     argument these default to AutoLayout, which Gluon resolves against the
     template's concrete layouts (verified on gfx950 for all four).
   - Epilogue fusion is off via a new `TritonTemplate.allow_epilogue_fusion`
     class attribute, threaded through `TritonTemplateCaller` to the existing
     `TritonTemplateBuffer.allow_epilogue_fusion` flag. Templates that disallow
     it also skip deferred (multi-template) selection, which benchmarks fused
     epilogues. Prologue fusion is already off because the template never calls
     `load_input`.

2. **`GluonKernelOverrides(TritonKernelOverrides)`** overrides only `constant`,
   to emit rank-0 constants (`TritonOverrides.constant`).
   `TritonKernelOverrides.constant` pads constants to the kernel's range-tree
   rank, which is 1 for templates; Triton broadcasts `[1]` against 2D tiles but
   Gluon requires equal ranks.

   Why a text rewrite instead of per-op overrides plus a `GluonPrinter`: the
   tensor-creating ops are emitted from at least six places (printer,
   `_shaped_constant`, `masked`, `TritonKernel.indexing`,
   `ModificationWrapper._broadcast_index`, `size()`), several inside large
   methods that would have to be copied. Each `tl.<op>(` maps one-to-one to
   `gl.<op>(`, so one rewrite covers all of them, including emitters added later.
   A render-time guard is unnecessary because no `tl.<op>(` survives the
   rewrite. An op Gluon can't compile surfaces as a compile error of that choice.

3. **Runtime**: `_precompile_config` picks `GluonASTSource` (exposed via
   `runtime/triton_compat.py`, `None` when unavailable) for Gluon functions.

4. **Template** `kernel/flex/templates/gluon_flex_attention_cdna.py.jinja` (forward only):
   - **Uses the same hooks as the Triton template**: `def_kernel` (same nine
     inputs: Q, K, V, LSE, MAX, KV_NUM_BLKS, KV_IDX, FULL_KV_NUM_BLKS,
     FULL_KV_IDX), `size`, `stride`, `modification`, `store_output`,
     `gen_argdefs` / `gen_defines`. Keeping the signature identical means the
     lowering can pass the same `input_nodes`, `mutated_inputs`,
     `input_gen_fns` and grid.
   - **Layouts**:
     - `AMDMFMALayout(version=4, instr_shape=[32,32,16] or [16,16,32], transposed=True, warps_per_cta=[num_warps,1])` for the S=QK^T and O accumulators;
     - `DotOperandLayout`s for Q/K/V;
     - `BlockedLayout`s for global loads;
     - a padded/swizzled `SharedLayout` for K/V staging. Use `compute_efficient_padded_shared_layout` from `amd.cdna4`.
   - **Mod inlining**:
     - Build `m = gl.arange(..., layout=gl.SliceLayout(1, MFMA))[:, None]` and `n = gl.arange(..., layout=gl.SliceLayout(0, MFMA))[None, :]`, so `score_mod`/`mask_mod` run directly on the accumulator layout with no `convert_layout`.
     - Bind `b`/`h` as scalars.
     - The `modification(...)` call sites keep the same `input_shapes`/`input_dtypes` as `common.py.jinja:57,84`.
   - **Block-sparse structure**: copy the Triton template's two-pass loop
     unchanged: partial blocks with mask_mod, then full blocks without it.
     Sparse-index iteration (`get_offset_for_next_block`) is scalar loads and
     ports almost verbatim. Write the Gluon helpers with `@gluon.jit`; do not
     reuse `utilities.py.jinja` as-is.
   - **Kernel body**:
     - `cdna4.mfma` for QK^T and PV;
     - `amd.cdna4.buffer_load` / `async_copy` into shared memory for K/V with double buffering;
     - online softmax in exp2 with `RCP_LN2` folding;
     - `PRESCALE_QK`, `ROWS_GUARANTEED_SAFE`, GQA (`GQA_SHARED_HEADS`), `IS_DIVISIBLE` / boundary masking;
     - LSE in log2 and MAX written with `gl.store`.
   - Later: `warp_pipeline` stages (ping-pong) once correctness lands.

5. **Configs**:
   - Add a `get_flex_attention_gluon_fwd_configs` method on the ROCm config
     heuristic (`heuristics/template/triton.py`), returning
     `(BLOCK_M, BLOCK_N, num_warps, MFMA instr shape, num_buffers)`.
   - Keep the space small (4-8 configs).
   - Don't pass Triton-only kernargs (`kpack`, `matrix_instr_nonkdim`,
     `waves_per_eu`) unless Gluon's HIP options accept them. Verify each one.

### Lowering changes (`kernel/flex/flex_attention.py`)

- `nn/attention/flex_attention.py`:
  - add `"GLUON"` to `_Backend` and the `FlexKernelOptions` docstring;
  - `stats_are_log2` stays True for GLUON (the `!= "FLASH"` test already gives this);
  - reject `OUTPUT_MAX` only if the template can't produce it (it should be able to).
- Forward lowering: after the decode routing and before the FLASH gate, add a
  branch `if backend == "GLUON": return create_flex_gluon_attention_kernel(...)`.
  - Put it in a new `kernel/flex/flex_gluon.py`, which holds:
    - the arch-keyed template registry and the eligibility check (see
      "Arch dispatch and fallback");
    - the `GluonTemplate` instances;
    - the choice construction.
  - When the check fails, fall through to the Triton forward path instead of
    raising. This differs from FLASH (`_use_flex_flash_attention` raises): FLASH
    has a different output contract, but a Gluon forward is a drop-in for the
    Triton forward (finding 1), so falling back changes nothing downstream.
  - Share the setup now done inline (output layout, LSE/MAX buffers,
    `kernel_options` defaults, `IS_DIVISIBLE`) by factoring it out of the Triton
    path into a helper, rather than duplicating it.
- Backward: no change. The `flex_attention_backward` HOP receives Gluon's
  out/LSE and runs the Triton backward template.
- Later, opt-in auto-selection: register Gluon choices through
  `InductorChoices.append_flex_attention_choices` (`choices.py:189`) so they
  compete with Triton under `AUTO` + max-autotune, behind a config flag.

### Arch dispatch and fallback

Gluon exposes the hardware directly (`AMDMFMALayout(version=4)`, `cdna4.mfma`,
`amd.cdna4.buffer_load`), so a gfx950 template does not compile on gfx942, and
neither runs on NVIDIA. Each arch family needs its own template, and every other
arch needs a fallback.

- **Registry.** `flex_gluon.py` keeps a dict from arch to forward template, e.g.
  `{"gfx950": gluon_flex_attention_cdna4}`. Adding gfx942 later means adding a
  `cdna3` template and one entry; the lowering doesn't change. Key on the base
  arch: strip feature suffixes from
  `torch.cuda.get_device_properties(device).gcnArchName`
  (`"gfx950:sramecc+:xnack-"` -> `"gfx950"`). Per-arch config lists live next to
  each template, or are keyed the same way in the ROCm config heuristic.
- **Eligibility.** `_flex_gluon_template(query, key, value, kernel_options)`
  returns the template spec, or `None` plus a reason string. The mechanism is
  shared; only the constraint values come from the template. Checks run in
  three layers:
  1. *Shared, before lowering* (in `flex_gluon.py`, the same for every arch):
     ROCm and `has_gluon()`; a registry entry for the arch; no TMA/TDM
     descriptor options; static QK and V head dims (the template specializes on
     them; sequence lengths may stay dynamic); Q/K/V of the same dtype.
  2. *Per template, before lowering.* Each registry value is a small spec
     rather than a bare template:
     ```python
     @dataclasses.dataclass(frozen=True)
     class GluonFlexTemplateSpec:
         template: GluonTemplate
         qk_head_dims: frozenset[int]
         v_head_dims: frozenset[int]
         dtypes: frozenset[torch.dtype]
         get_configs: Callable[..., list[FlexConfig]]
         # Arch-specific extra constraints; returns a reason or None.
         check: Callable[[TensorBox, TensorBox, TensorBox, dict], str | None] | None = None
     ```
     The shared code tests head dims and dtypes against the spec, then calls
     `check` for anything else that's specific to the template (e.g. a GQA
     ratio or a minimum block size). If the user forces `BLOCK_M`/`BLOCK_N`
     in `kernel_options`, they must match one of `get_configs`'s block shapes;
     otherwise it's a fallback reason too.
  3. *At compile time.* A `score_mod`/`mask_mod` op that Gluon can't compile
     only shows up when the rendered kernel is compiled; `triton_to_gluon`
     leaves no layout-less `tl.*` op behind, so there is nothing to catch at
     render time. Render errors (`NotImplementedError`) are still caught by
     `maybe_append_choice` (`select_algorithm.py:2861`). For compile errors,
     the lowering runs `autotune_select_algorithm` on the Gluon choices alone
     (`select_gluon_flex_choice`, following `tuned_scaled_mm`'s NV Universal
     GEMM fallback) and returns early on success. If it raises
     `NoValidChoicesError` because every choice failed, the lowering falls
     through to the Triton choices. A lone choice (e.g. a forced `BLOCK_N`) or
     `config.deterministic` is picked without compiling, so that choice is
     precompiled first and any error from it counts as a failure.
- **Fallback.** On `None`, the lowering continues down the existing Triton
  forward path with the same `kernel_options` and warns once per reason, e.g.
  `BACKEND="GLUON" has no template for gfx942; using the Triton template`. A
  Gluon-only option that Triton doesn't understand is dropped before
  fallback, so that it can't reach the Triton template.
- **Backward** is unaffected; it already uses Triton.
- **Decode.** Decode routing runs before the Gluon branch. Queries that
  `_use_flex_decoding` accepts (query length < 128, static batch and heads)
  take the Triton decode kernel under `BACKEND="GLUON"` as under `AUTO`, with a
  one-time warning. Otherwise a 1-row query would pad to a 128-row Gluon tile
  with no split over KV.
- **AUTO, later.** `append_flex_attention_choices` simply adds no Gluon choices
  when the lookup returns `None`. No warning in that case, since the user didn't
  ask for Gluon.
- **Caching.** The choice is made at compile time from the compile device's
  arch. Check that the FX graph cache and autotune cache keys include the arch
  (they should via the device properties in the system key), so that an
  artifact compiled on gfx950 is never reused on gfx942. AOTI packages built
  for one arch are already arch-specific.

### Fusion

The flex output feeds user code, so Inductor will try to fuse epilogues into
`store_output`. Epilogue codegen uses `TritonKernel` range trees (`tl.arange`)
and pointwise overrides, so Gluon choices are marked non-fusible for v1.
Observed in milestone 2: with the flag lifted, a pointwise `relu` epilogue
fuses into the toy Gluon template and is numerically correct, because
`triton_to_gluon` also covers the epilogue hook. v2 needs broader validation
(reductions, multi-output, captured-buffer epilogues), not new machinery.

## Milestones and PRs

1. **Prereqs** (a subset of the general plan's Phases 1 and 2):
   - `has_gluon()` (Phase 1);
   - `GluonASTSource` in `triton_compat` (Phase 1);
   - the `_precompile_config` branch (Phase 2's runtime bullet);
   - no static-launcher change: verified on gfx950 that a Gluon kernel compiles
     in async compile workers and runs correctly with the static launcher both
     on and off.

   No output change. The Phase 1 import-extension table is not needed:
   `GluonTemplateKernel` overrides the `gen_common_triton_imports` classmethod
   to add the Gluon imports, so they appear only in Gluon template kernels.
2. **`GluonTemplate` infrastructure + `GluonKernelOverrides`**. Test with a toy
   Gluon template (e.g. elementwise add with a `{{modification}}` hook) in
   `test/inductor/test_gluon_template.py`. Cover a set of score_mods: constants,
   relative bias via captured buffer, soft-cap (`tanh`), `masked`, int64
   indexing, dynamic shapes.
3. **Gluon flex forward template + lowering branch** for gfx950, behind
   `BACKEND="GLUON"`, including the arch registry and the Triton fallback for
   every other arch.

   Done: `kernel/flex/templates/gluon_flex_attention_gfx950.py.jinja` and
   `kernel/flex/flex_gluon.py`. The template takes tokenspeed's MFMA v4
   `[32, 32, 16]` and dot-operand layouts, padded shared layouts and async
   K/V staging, and keeps flex's semantics and two-pass block-sparse loop. It
   is non-persistent, with flex's grid. The Gluon branch sits just before the
   Triton config loop, so it reuses all of the lowering's setup. Fallback
   layers 1 and 2 are implemented. Layer 3 (compile failure) followed in
   milestone 4's cleanup; it did not need the Triton choice loop factored
   out, since the Gluon branch sits before it and can fall through.

   Measured on MI350X, bf16 forward, both max-autotune, B*S = 16k, H = 16:
   Gluon is 1.04-1.13x the Triton template on causal and sliding-window
   (1024) masks for D = 64 and 128, and 0.95-0.98x on causal at 16k.
   tokenspeed's 1.6-2.1x at D = 64 comes from what v1 leaves out: the
   persistent, swizzled scheduler and the `main_end` split that skips
   masking on non-diagonal tiles.
4. **Performance**: pipelining/ping-pong and config tuning; benchmark against
   the Triton template with
   `benchmarks/transformer/score_mod.py` (or the attention-gym
   variants: causal, sliding window, document mask, ALiBi, soft-cap).

   Target: at least 1.4x the Triton template on causal D = 64 (tokenspeed's
   lead was 1.3-2.0x). Met. MI350X, bf16 forward, both max-autotune, static
   shapes, H = 32 / Hkv = 8 on tokenspeed's layout, causal:

   | D | S | tokenspeed | Triton | Gluon | Gluon / Triton |
   |---|---|---|---|---|---|
   | 64 | 1k-16k | 376-675 | 282-468 | 403-689 | 1.43-1.72 |
   | 128 | 1k-16k | 321-514 | 365-597 | 402-680 | 1.10-1.26 |

   With H = 16 (no GQA): causal D = 64 1.44-1.74x, D = 128 1.12-1.27x;
   sliding window (1024) D = 64 1.20-1.49x, D = 128 1.08-1.28x. Gluon now
   matches or beats tokenspeed on causal D = 64 and beats it on D = 128.

   What mattered, in order:
   - **Persistent swizzled scheduler** (about +25-40% at D = 64): an
     XCD-aware remap plus alternating q-slot direction across q cycles, as in
     tokenspeed. Turning tokenspeed's own swizzle off dropped it below Gluon
     v1, which is how this was identified. Slot counts are computed at run
     time because flex's batch and head counts can be dynamic;
     `NUM_PROGRAMS` is 4 per CU. Autotuning 2, 3 and 4 per CU over 30 MI350X
     shapes (causal and sliding window, D = 64 and 128, S = 1k-16k, H = 32 /
     Hkv = 8 and H = 16) found 4 per CU best or within 2% everywhere, and 3 per
     CU 8-54% slower: the slot counts round down, so with 768 programs and,
     e.g., B = 16, H = 32 only 512 get work. Only `BLOCK_N` (64 or 32) is
     autotuned now; 32 wins causal D = 128 by 8-17%, 64 wins everything else.
   - **One scale multiply per score** (about +10-17% at long S): the scale is
     folded into a single constexpr `SCORE_SCALE` applied inside the exp2
     argument (fusing into an FMA) and to the row max rather than to every
     score, which is exact for a positive scale. With no score_mod
     (`SCORE_MOD_IS_IDENTITY`, from `is_trivial_score_graph`) and a positive
     `SM_SCALE`, this also absorbs `SM_SCALE`, so it equals the cost of
     `PRESCALE_QK` without `PRESCALE_QK`'s bf16 rounding of Q. A negative
     scale keeps the separate `qk *= SM_SCALE`, since scaling the row max is
     only exact for a positive factor (`test_negative_scale`).
   - **Block-index loads** (about +5%, at long S): the index load is a
     vector load that shares vmcnt with the async K/V copies, and LLVM had
     turned the masked load into a branch with a wait at the end of each
     iteration. It is now prefetched one tile ahead, unmasked with a clamped
     index, and added to the vector offsets rather than the scalar base
     pointer.

   Not needed so far: double-buffered K/V and ping-pong. Both are still
   options if a later target needs more.

   Small problems. Launching all `NUM_PROGRAMS` programs costs nothing
   measurable: trimming the launch to the work-item count changed nothing
   (within 0.2 us) even with 1023 of 1024 programs idle. Gluon is, however,
   5-55% slower than the Triton template below about 4 work items per CU
   (batch x heads x 128-row query blocks < ~1024 on MI350X), e.g. 60 vs 39 us
   for B = 2, H = 16, S = 1k, D = 128, causal. The likely cause (unverified) is
   the fixed `BLOCK_M = 128`: each work item is one 4-warp workgroup walking
   its whole KV row, while Triton can pick smaller blocks. See the split-KV
   item in milestone 5.

   Benchmarking pitfall: `torch_key()` (the FX graph cache key) hashes only
   `.py` files, so after editing a `.jinja` template a dev checkout silently
   reuses stale kernels. Always benchmark template edits with a fresh
   `TORCHINDUCTOR_CACHE_DIR`. This is a pre-existing upstream issue that
   affects every Inductor Jinja template.
5. **Optional**: gfx942 (a `cdna3` template plus a registry entry); AUTO selection via the choices hook; Gluon
   backward; epilogue fusion; a Gluon split-KV template. Splitting KV across
   programs when (batch, head, query block) work is scarce would serve both
   decode (now routed to the Triton decode kernel) and small prefills, where
   the persistent template trails Triton (see milestone 4).

## Testing

New file `test/inductor/test_flex_gluon.py`, modelled on `test_flex_flash.py`'s
`flash_vs_triton`. It compares fp32 eager reference vs `GLUON` vs `TRITON`
using `instantiate_device_type_tests(..., only_for="cuda")`. Skip unless
`torch.version.hip` and the arch is supported and `has_gluon()`.

Parametrize over:
- `score_mod`: identity, causal, ALiBi, relative bias via captured tensor, soft-cap;
- `mask_mod`: none, causal, sliding window, document-ID gather;
- dtype: fp16/bf16;
- head dim;
- GQA;
- non-divisible sequence lengths;
- `return_lse` / `return_aux=AuxRequest(max_scores=True)`;
- dynamic shapes;
- backward through the Triton bwd (grad check vs TRITON).

Fallback tests run on any GPU, without the arch skip:
- With the registry patched to be empty (or on an arch with no entry),
  `BACKEND="GLUON"` matches `TRITON` output, emits the warning once, and the
  generated code contains no `@gluon.jit`.
- Each layer falls back the same way on gfx950: a dynamic head dim (layer 1),
  an unsupported head dim or a forced `BLOCK_M` outside the config space
  (layer 2), and a `score_mod` whose Gluon kernel fails to compile (layer 3;
  patch `triton_to_gluon` to append an undefined name to the score_mod code),
  both with two configs and with a forced `BLOCK_N` that leaves one.
- A query shorter than 128 takes the Triton decode kernel (`SPLIT_KV` in the
  generated code, no `@gluon.jit`) and warns.
- On gfx950 with supported inputs, the generated code does contain
  `@gluon.jit`, so that a silent fallback can't pass the numerics tests.

Run:

```
python test/inductor/test_flex_gluon.py
python test/inductor/test_flex_attention.py -k "rocm or TRITON"   # regression check: no output change
```

## Risks and open questions

- **Gluon API stability.** `triton.experimental.gluon` is experimental. Pin to
  the Triton version in `.ci/docker/triton_version.txt` and gate on
  `has_gluon()`. Expect churn in AMD layout constructors between Triton bumps.
- **AutoLayout resolution.** The probes cover simple cases. Captured-buffer
  gathers with AutoLayout constants inside `tl.where` chains need more coverage
  in milestone 2, especially when a pure-constant `mask_mod` never meets a
  concrete layout before the template's `where`.
- **Template fragility.** Arch-specific layouts mean one template per arch
  family. The registry above is a local version of the arch-keyed template
  registration proposed in `GLUON_INDUCTOR_PLAN.md`; move it there once a
  second Gluon template exists.
- **Fallback vs. error.** Falling back means a user who asks for Gluon on an
  unsupported setup gets Triton performance with only a warning. If that turns
  out to hide problems (e.g. in benchmarking), add a strict option that raises
  instead. Leave it out of v1.
- **Mods that cost bandwidth.** Mods that gather from captured buffers produce
  `gl.load` on MFMA-layout pointer tensors, which coalesce poorly. The Triton
  template has the same limitation; it is acceptable for v1.
- **Is the speedup worth it?** Measured on 2026-10-15 on MI350X with torch
  2.11.0+rocm7.14.1 and Triton 3.8.0. The comparison point is tokenspeed's gfx950
  Gluon MHA prefill (`lightseekorg/tokenspeed@e031b9d`, run on upstream Triton
  with a `gl.any = gl.reduce_or` shim, outputs matched flex within 1 bf16 ulp).
  Script: `agent_space/bench_flex_vs_tokenspeed.py`. bf16 forward TFLOPS:
  - Causal, D=128: tokenspeed 350-520, flex max-autotune 300-530 (it is
    1.06-1.17x faster than flex up to 8k tokens and at parity at 16k), SDPA 190-545.
    Flex with default compile settings is 2-3x slower than either.
  - Causal, D=64: tokenspeed is **1.6-2.1x faster than flex max-autotune**
    (450-690 vs 240-430 TFLOPS). This is the largest gap.
  - Sliding window 1024: flex max-autotune is **1.4-2x faster** than tokenspeed.
    Flex's block sparsity skips more work than tokenspeed's non-persistent
    sliding kernel.

  Takeaway: a Gluon template is worth most for small head dims and short to
  medium causal sequences. Its gain comes from the persistent scheduler,
  pipelining and the MFMA layout, not from masking. A Gluon flex template must
  keep flex's block-sparse iteration to avoid losing on sparse masks. The first
  target should be a persistent causal-friendly schedule with D=64 and D=128.
- **Upstream ownership.** This adds a third flex GPU backend. Socialize it with
  the flex/Inductor maintainers (an RFC issue) before landing milestone 3.
