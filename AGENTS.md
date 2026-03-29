# AGENTS.md

## Scope

These instructions apply to the entire repository.

## Project Summary

This repository contains SPADA, a Python compiler toolchain for spatial dataflow architectures such as Cerebras WSE.

General compilation flow:

1. Spatial IR canonicalization and optimization
2. Spatial IR -> Cerebras CSL
3. Runtime execution through generated metadata and launch tooling

Stencil compilation flow:

1. GT4Py input -> Stencil IR
2. Stencil IR -> Spatial IR
3. Spatial IR canonicalization and optimization
4. Spatial IR -> Cerebras CSL
5. Runtime execution through generated metadata and launch tooling

## Environment And Commands

Use the workspace virtual environment if it is present.

Common setup commands:

```bash
pip install -e .
pip install -e ".[dev]"
pip install -e ".[docs]"
```

Validation commands:

```bash
pytest
pytest tests/spatial_ir/
pytest tests/stencil_ir/
pytest tests/placement/
black spatialstencil tests
isort spatialstencil tests
flake8 spatialstencil tests
```

Documentation command:

```bash
cd irspec && mkdocs serve
```

Compiler entry points:

```bash
sptlc input.sptl output/ --param I=128 --param J=128
python -m spatialstencil.cli.gt4py_to_spatial samples/stencils.py 128,128,80 output/ --function-name laplacian
```

## Repository Map

- `spatialstencil/syntax/stencil_ir/`: Stencil IR nodes, parser, analysis, canonicalization, type/domain inference
- `spatialstencil/syntax/spatial_ir/`: Spatial IR nodes, parser, canonicalization, analysis, optimization passes
- `spatialstencil/lowering/`: lowering pipeline from GT4Py and Stencil IR into Spatial IR and CSL
- `spatialstencil/syntax/csl/`: CSL syntax model, tasks, statements, DSD operations
- `spatialstencil/placement/`: placement graph model and optimization
- `spatialstencil/runtime/`: runtime integration and metadata-driven execution
- `spatialstencil/cli/`: command-line entry points
- `tests/`: subsystem-oriented tests
- `irspec/docs/`: language and IR specification docs; link here rather than duplicating specification text

## Working Rules

Prefer focused changes inside the existing compiler stage rather than adding parallel logic elsewhere.

When changing compiler behavior:

1. Preserve the existing pass boundaries unless the task explicitly requires pipeline changes.
2. Maintain current visitor and transformer patterns used by the IR layers.
3. Update or add tests in the matching subsystem directory.
4. Validate the smallest relevant test slice first, then widen if needed.

When changing CLI or runtime behavior:

1. Keep public command names and argument shapes stable unless requested.
2. Prefer extending metadata-driven behavior over introducing hard-coded layout assumptions.

## Spatial IR And Lowering Constraints

These constraints are easy to break and should be treated as guardrails:

- Pass ordering matters. Canonicalization, parameter concretization, communication lowering, argument lowering, and optimization stages rely on earlier structural invariants.
- Many passes assume parameters are concrete before lowering decisions are made.
- Rectangle and block equivalence-class structure matters for downstream code generation. Preserve the relationship between place, dataflow, and compute regions.

## Testing Guidance

Choose tests based on the area you changed:

- Spatial IR optimization or analysis: `pytest tests/spatial_ir/`
- Stencil IR or lowering changes: `pytest tests/stencil_ir/`
- Placement changes: `pytest tests/placement/`
- Runtime changes: `pytest tests/csl_runtime/`

If you change copy elimination, lowering, or canonicalization, run the targeted tests for that subsystem before broader validation.

## Key References

Start with these files when working in the corresponding area:

- `README.md`
- `spatialstencil/syntax/spatial_ir/copy_elimination.py`
- `spatialstencil/syntax/spatial_ir/canonicalization.py`
- `spatialstencil/syntax/spatial_ir/analysis.py`
- `spatialstencil/syntax/spatial_ir/passes.py`
- `spatialstencil/lowering/stencil_to_spatial.py`
- `spatialstencil/lowering/spatial_ir_to_csl.py`
- `spatialstencil/lowering/stencil_to_spatial_routing.py`
- `spatialstencil/placement/README.md`
- `irspec/docs/spatial/spatial.md`
- `irspec/docs/stencil/lowering.md`

## Change Hygiene

- Do not rewrite large generated or sample output directories unless the task explicitly targets them.
- Avoid broad formatting-only edits in compiler modules.
- Preserve existing naming and data-model conventions in IR node definitions.
- Prefer adding narrowly scoped regression tests for compiler bugs.