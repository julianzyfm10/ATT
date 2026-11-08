# Core Local 30 Case Catalog

| # | Case | Servers | Level | Primary challenge |
|---:|---|---|---|---|
| 1 | Plural length units | Unit Converter | L1 | plural to canonical enum |
| 2 | Temperature symbol normalization | Unit Converter | L1 | symbols to canonical enum |
| 3 | Mass enum normalization | Unit Converter | L1 | plural mass schema |
| 4 | Strict 24-hour time | Time MCP | L1 | `HH:MM` validation |
| 5 | Case-sensitive multiplication schema | Math MCP | L1 | `SecondNumber` casing |
| 6 | Exact division tool name | Math MCP | L1 | tool-name selection |
| 7 | Tensor shape and flat-value schema | Scientific Computing | L1 | shape/value representation |
| 8 | Boolean sex parameter | Medical Calculator | L1 | boolean schema |
| 9 | Inspect units before uncommon conversion | Unit Converter | L2 | list-before-convert |
| 10 | Create then compute determinant | Scientific Computing | L2 | stored tensor dependency |
| 11 | Create then invert matrix | Scientific Computing | L2 | stored tensor dependency |
| 12 | Two-state matrix multiplication | Scientific Computing | L2 | two preconditions |
| 13 | Create then eigendecompose | Scientific Computing | L2 | stored tensor dependency |
| 14 | Mean through explicit arithmetic chain | Math MCP | L2 | output-to-input dataflow |
| 15 | Conditional adjusted body weight | Medical Calculator | L2 | threshold branch |
| 16 | Conditional cystatin-C refinement | Medical Calculator | L2 | threshold branch |
| 17 | Convert then scale distance | Unit + Math | L3 | cross-server handoff |
| 18 | Convert temperatures then average | Unit + Math | L3 | two-to-one fan-in |
| 19 | Convert height then calculate BMI | Unit + Medical | L3 | cross-server units |
| 20 | Determinant then arithmetic scaling | Scientific + Math | L3 | stateful handoff |
| 21 | Correct calcium then scale | Medical + Math | L3 | result reuse |
| 22 | Conversions populate a tensor | Unit + Scientific | L3 | cross-server state creation |
| 23 | Recover from missing tensor | Scientific Computing | L4 | expected missing-state error |
| 24 | Recover from singular matrix | Scientific Computing | L4 | replanning after tool error |
| 25 | Recover from speed alias mismatch | Unit + Math | L4 | schema inspection and retry |
| 26 | Conditional renal refinement and rounding | Medical + Math | L4 | branch plus handoff |
| 27 | Normalize imperial units across three servers | Unit + Medical + Math | L4 | schema recovery chain |
| 28 | Imperial-to-clinical calculation pipeline | Unit + Medical + Math | L4 | eight-step fan-out pipeline |
| 29 | Conversions build a scientific matrix | Unit + Scientific + Math | L4 | stateful three-server pipeline |
| 30 | Timezone and duration meeting pipeline | Time + Unit + Math | L4 | multi-domain convergence |

All prompts are backend-blind. Exact prompts, oracle calls, expected values,
tolerances, dependency edges, and runtime constraints live in the JSONL file.

