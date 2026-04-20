# SPADA — Multi-Level Spatial IR Specification

SPADA is a programming language and compiler for spatial dataflow architectures such as the [Cerebras Wafer-Scale Engine](https://www.cerebras.net/). It provides precise control over data placement, communication streams, and asynchronous execution while abstracting architecture-specific routing details.

This site documents the three intermediate representations (IRs) used in the SPADA compilation pipeline:

| IR | Input | Output |
|---|---|---|
| **Stencil IR** | GT4Py stencil definitions | Spatial IR |
| **Spatial IR** | Stencil IR / hand-written SPADA kernels | Dataflow Task IR |
| **Dataflow Task IR** | Spatial IR | Cerebras CSL |

For full details on the SPADA language, compiler, and hardware results, see:

> Lukas Gianinazzi, Tal Ben-Nun, Torsten Hoefler. *SPADA: A Spatial Dataflow Architecture Programming Language.* arXiv:2511.09447, 2025.


