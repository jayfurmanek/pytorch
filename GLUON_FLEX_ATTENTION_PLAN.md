# Gluon backend for FlexAttention's Inductor lowering: plan

Status: draft. Based on `main` at `79f81f9e6d3`, Triton 3.8.0, and an MI350X (gfx950).
This plan applies Phase 3 of `GLUON_INDUCTOR_PLAN.md` (Inductor-generated Gluon
templates) to FlexAttention. Phase 1 of that plan (kernel-language imports) is a
prerequisite.

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
- Gluon decode.
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
   in `torch/_inductor/codegen/gluon_template.py` (or alongside `select_algorithm.py`):
   - `jit_lines`: `@gluon.jit` instead of `@triton.jit` under
     `@triton_heuristics.template(...)`.
   - Imports: the Gluon extension entry from Phase 1 of the general plan
     (`from triton.experimental import gluon`,
     `from triton.experimental.gluon import language as gl`).
   - `generate_and_load` constexpr defines: `NAME : gl.constexpr = ...` (Gluon
     accepts `tl.constexpr` too, but use `gl` for clarity). Verify this before
     changing it.
   - `size()` int64 path: `gl.full([], ...)`.
   - `overrides = GluonKernelOverrides`.
   - `modification()`: reuse the Triton implementation. Swap
     `ModificationWrapper` for a subclass whose `_broadcast_index` emits `gl.full`.
   - `store_output`: allow only the non-TMA pointer path, and reject epilogue
     nodes (see "Fusion" below).
   - Mark choices from this template as not fusible, for both prologue and
     epilogue. Find the existing knob on `TritonTemplateCaller` /
     `TritonTemplateBuffer` rather than adding a new one.

2. **`GluonKernelOverrides(TritonKernelOverrides)`**: override only `constant`,
   `_shaped_constant`, `masked`, and any op whose Triton text uses
   `tl.full`/`tl.zeros*`/`tl.arange`. Everything else inherits. A
   `GluonPrinter(TritonPrinter)` handles the `tl.full([], …)` / `tl.zeros_like`
   cases in the same way.
   - Add a guard: after `modification()` renders, assert that no `tl.full(` /
     `tl.arange(` / `tl.zeros(` substrings remain, and raise
     `NotImplementedError("unsupported op in score_mod for GLUON backend: ...")`
     otherwise. This keeps new emitters from silently breaking Gluon, and the
     lowering turns that error into a Triton fallback (see "Arch dispatch and
     fallback").

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
  3. *At render time.* Some `score_mod`/`mask_mod` ops can't be expressed in
     Gluon, and this only shows up when `modification()` renders (see the
     guard in "New pieces"). The guard raises `NotImplementedError`, which
     `maybe_append_choice` (`codegen/common.py:2879`) already catches and
     returns. If every Gluon config fails to append, the lowering falls back
     using the returned error as the reason.
- **Fallback.** On `None`, the lowering continues down the existing Triton
  forward path with the same `kernel_options` and warns once per reason, e.g.
  `BACKEND="GLUON" has no template for gfx942; using the Triton template`. A
  Gluon-only option that Triton doesn't understand is dropped before
  fallback, so that it can't reach the Triton template.
- **Backward and decode** are unaffected; they already use Triton.
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
and pointwise overrides, so Gluon choices must be marked non-fusible for v1.
v2 can route epilogue nodes through `GluonKernelOverrides` and emit index
tensors in the store layout.

## Milestones and PRs

1. **Prereqs** (general plan, Phase 1):
   - `has_gluon()`;
   - the Gluon import extension;
   - `GluonASTSource` in `triton_compat`;
   - `_precompile_config` branch.

   No output change.
2. **`GluonTemplate` infrastructure + `GluonKernelOverrides`**. Test with a toy
   Gluon template (e.g. elementwise add with a `{{modification}}` hook) in
   `test/inductor/test_gluon_template.py`. Cover a set of score_mods: constants,
   relative bias via captured buffer, soft-cap (`tanh`), `masked`, int64
   indexing, dynamic shapes.
3. **Gluon flex forward template + lowering branch** for gfx950, behind
   `BACKEND="GLUON"`, including the arch registry and the Triton fallback for
   every other arch.
4. **Performance**: pipelining/ping-pong and config tuning; benchmark against
   the Triton template with
   `benchmarks/transformer/score_mod.py` (or the attention-gym
   variants: causal, sliding window, document mask, ALiBi, soft-cap).
5. **Optional**: gfx942 (a `cdna3` template plus a registry entry); AUTO selection via the choices hook; Gluon
   backward; epilogue fusion.

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
  (layer 2), and a `score_mod` that trips the render guard (layer 3; patch an
  override to emit `tl.full` to force it).
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
