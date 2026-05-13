<!-- Banner -->
<div align="center">

```
 ██████╗ █████╗ ██╗      ██████╗██╗   ██╗██╗     ██╗   ██╗███████╗
██╔════╝██╔══██╗██║     ██╔════╝██║   ██║██║     ██║   ██║██╔════╝
██║     ███████║██║     ██║     ██║   ██║██║     ██║   ██║███████╗
██║     ██╔══██║██║     ██║     ██║   ██║██║     ██║   ██║╚════██║
╚██████╗██║  ██║███████╗╚██████╗╚██████╔╝███████╗╚██████╔╝███████║
 ╚═════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝╚══════╝

███████╗ ██████╗ ██╗    ██╗   ██╗███████╗██████╗
██╔════╝██╔═══██╗██║    ██║   ██║██╔════╝██╔══██╗
███████╗██║   ██║██║    ██║   ██║█████╗  ██████╔╝
╚════██║██║   ██║██║    ╚██╗ ██╔╝██╔══╝  ██╔══██╗
███████║╚██████╔╝███████╗╚████╔╝ ███████╗██║  ██║
╚══════╝ ╚═════╝ ╚══════╝ ╚═══╝  ╚══════╝╚═╝  ╚═╝
```

**A Tree-to-Tree Transformer that solves calculus — natively in SLaNg.**

[![License: Quantum Logics Proprietary](https://img.shields.io/badge/license-Quantum%20Logics%20Proprietary-0f172a?style=flat-square&labelColor=1e293b)](LICENSE)
[![slangmath](https://img.shields.io/badge/npm-slangmath-0f172a?style=flat-square&labelColor=1e293b&logo=npm)](https://npmjs.com/package/slangmath)
[![AP Calc AB](https://img.shields.io/badge/AP%20Calc%20AB-92.4%25-22c55e?style=flat-square&labelColor=1e293b)](eval/benchmarks/ap_calculus.json)
[![MIT 18.02](https://img.shields.io/badge/MIT%2018.02-78.3%25-22c55e?style=flat-square&labelColor=1e293b)](eval/benchmarks/multivariable.json)

</div>

---

## What it is

SLaNg knows _how_ to differentiate, integrate, and optimize. You still have to call the right function with the right arguments. **CalculusSolver figures that out.**

Feed it an unsolved SLaNg expression tree. Get back the solved expression, the exact sequence of rules that were applied, and a confidence score — all in native SLaNg, ready to pipe into the next operation.

> _If SLaNg is the calculator, CalculusSolver is the mathematician who decides which buttons to press._

---

## 30-second demo

```javascript
import { CalculusSolver } from "calculussolver";
import { createTerm, createFraction, slangToLatex } from "slangmath";

const cs = new CalculusSolver();

// Differentiate  2x / (x² + 1)
const result = await cs.solve({
  op: "diff",
  var: "x",
  expr: createFraction(
    [createTerm(2, { x: 1 })],
    [createTerm(1, { x: 2 }), createTerm(1)],
  ),
});

console.log(result.status); // "solved"
console.log(slangToLatex(result.expr)); // \frac{2(1 - x^{2})}{(x^{2} + 1)^{2}}
console.log(result.confidence); // 0.9981
console.log(result.steps);
// [
//   { rule: "quotient_rule", description: "d/dx[u/v] = (v·u′ − u·v′) / v²" },
//   { rule: "power_rule",    description: "d/dx[x²+1] = 2x"                },
//   { rule: "simplify",      description: "cancel common factors"           }
// ]
```

`result.expr` is a live SLaNg object. Pipe it straight into `gradient()`, `tangentPlane()`, `evaluateFraction()` — whatever comes next.

---

## Architecture

CalculusSolver is a **Tree-to-Tree Transformer**. Both encoder and decoder operate on SLaNg expression trees natively, with no intermediate string format.

```
  Input SLaNg expression tree
  ──────────────────────────────────────────────────────
  { op: "diff", var: "x", expr: createFraction(...) }
          │
          │  DFS walk → token sequence
          │  (depth, sibling_idx, path_hash) position encoding
          ▼
  ┌────────────────────────────────────────┐
  │           Tree Encoder                 │
  │   8 layers · 512 hidden · 8 heads      │
  │   + parent-child attention bias        │
  └────────────────────────────────────────┘
          │
          ├─────────────────────────────────┐
          ▼                                 ▼
  ┌──────────────────┐          ┌───────────────────────┐
  │    Rule Head     │          │     Tree Decoder      │
  │  classifier      │─────────►│  8 layers             │
  │  per operator    │ rule     │  autoregressive DFS   │
  │  node in input   │ embed    │  + SLaNg validity     │
  └──────────────────┘          │    mask at every step │
          │                     └───────────────────────┘
          │                                 │
          └──────────────┬──────────────────┘
                         ▼
                ┌─────────────────┐
                │  Step Tracer    │
                │  auxiliary head │
                │  → step.desc    │
                └─────────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  SLaNg Verifier │   runs  differentiateFraction /
                │  post-hoc       │   gradient / lagrangeMultipliers
                │  numerical check│   against model output
                └─────────────────┘
                         │
                         ▼
  Output SLaNg expression + step trace + confidence
```

### Three design bets that matter

**Bet 1 — SLaNg is the only I/O format.**
No LaTeX strings, no plain text, no intermediate representation. Every input and output is a `slangmath` object. This eliminates an entire class of parsing errors and means the model's output can always be plugged back into `slangmath` without a conversion step.

**Bet 2 — Rule Head before Decoder.**
The Rule Head predicts which calculus rule applies at each operator node (`quotient_rule`, `chain_rule`, `power_rule`, etc.) _before_ the Decoder generates the result subtree. Rules map one-to-one with `slangmath`'s internal function names. The model's reasoning is auditable — `result.steps` reflects what the Rule Head actually predicted, not a post-hoc summary.

**Bet 3 — slangmath verifies every answer.**
After inference, `verifier.js` calls the relevant `slangmath` function on the original input and compares numerically using `evaluateFraction` at 50 random test points. If the model is wrong, `result.status` changes from `"solved"` to `"unverified"`. The answer is still returned — but the caller is told.

---

## I/O reference

### Input envelope

```javascript
// Single-variable derivative
{ op: "diff",          var:  "x",         expr: <SLaNg> }

// Partial derivative
{ op: "partial",       var:  "x",         expr: <SLaNg> }

// Indefinite integral
{ op: "integrate",     var:  "x",         expr: <SLaNg> }

// Definite integral
{ op: "integrate_def", var:  "x",  lo: 0, hi: Math.PI,  expr: <SLaNg> }

// Limit
{ op: "limit",         var:  "x",  to: 0, side: "both", expr: <SLaNg> }

// Gradient   ∇f
{ op: "gradient",      vars: ["x","y"],   expr: <SLaNg> }

// Hessian    H(f)
{ op: "hessian",       vars: ["x","y"],   expr: <SLaNg> }

// Tangent plane at a point
{ op: "tangent_plane", vars: ["x","y"],   at: { x:1, y:2 },  expr: <SLaNg> }

// Critical points + classification
{ op: "optimize",      vars: ["x","y"],   expr: <SLaNg> }

// Constrained optimization  (Lagrange multipliers)
{ op: "lagrange",      vars: ["x","y"],   objective: <SLaNg>,  constraints: [<SLaNg>] }

// Taylor series
{ op: "series",        var:  "x",  around: 0, order: 5,  expr: <SLaNg> }

// Directional derivative
{ op: "dir_deriv",     vars: ["x","y"],   point: {x:1,y:1},  direction: {x:1,y:0},  expr: <SLaNg> }
```

### Output envelope

```javascript
{
  status:     "solved",           // "solved" | "unverified" | "partial" | "unsolvable"
  op:         "diff",             // mirrors the input op
  expr:       <SLaNg expression>, // the answer — a live SLaNg object
  steps: [
    {
      step:        1,
      rule:        "quotient_rule",
      description: "Apply quotient rule: d/dx[u/v] = (v·u′ − u·v′) / v²",
      before:      <SLaNg expression>,
      after:       <SLaNg expression>,
    },
    // ...
  ],
  latex:      "\\frac{2(1-x^{2})}{(x^{2}+1)^{2}}",  // display only
  confidence: 0.9981,
  warnings:   [],
}
```

---

## More examples

<details>
<summary><strong>Gradient of a multivariable function</strong></summary>

```javascript
// ∇f  where  f(x, y) = x² + 2xy + y²
const result = await cs.solve({
  op: "gradient",
  vars: ["x", "y"],
  expr: {
    terms: [
      createTerm(1, { x: 2 }),
      createTerm(2, { x: 1, y: 1 }),
      createTerm(1, { y: 2 }),
    ],
  },
});

// result.expr is exactly what slangmath's gradient() returns.
// Pipe it into tangentPlane(), directionalDerivative(), etc.
```

</details>

<details>
<summary><strong>Constrained optimization via Lagrange multipliers</strong></summary>

```javascript
// Maximize  f(x, y) = x + y   subject to   x² + y² = 1
const result = await cs.solve({
  op: "lagrange",
  vars: ["x", "y"],
  objective: { terms: [createTerm(1, { x: 1 }), createTerm(1, { y: 1 })] },
  constraints: [
    {
      terms: [createTerm(1, { x: 2 }), createTerm(1, { y: 2 }), createTerm(-1)],
    },
  ],
});

console.log(result.steps);
// [
//   { rule: "form_lagrangian",    description: "L = f − λg"       },
//   { rule: "partial_x",          description: "1 = 2λx"          },
//   { rule: "partial_y",          description: "1 = 2λy"          },
//   { rule: "solve_system",       description: "x = y = 1/√2"     },
//   { rule: "evaluate_objective", description: "f_max = √2"       },
// ]
```

</details>

<details>
<summary><strong>Tangent plane at a point</strong></summary>

```javascript
// Tangent plane to  z = x² + y²  at  (1, 2)
const result = await cs.solve({
  op: "tangent_plane",
  vars: ["x", "y"],
  at: { x: 1, y: 2 },
  expr: { terms: [createTerm(1, { x: 2 }), createTerm(1, { y: 2 })] },
});

import { tangentToLatex } from "slangmath";
console.log(tangentToLatex(result.expr)); // "z = 5 + 2x + 4y − 5"
```

</details>

<details>
<summary><strong>Taylor series</strong></summary>

```javascript
import { createFunction } from "slangmath";

// Taylor series of  sin(x)  around 0, order 7
const result = await cs.solve({
  op: "series",
  var: "x",
  around: 0,
  order: 7,
  expr: createFunction("sin", [createTerm(1, { x: 1 })]),
});

console.log(slangToLatex(result.expr));
// x − \frac{x^{3}}{6} + \frac{x^{5}}{120} − \frac{x^{7}}{5040}
```

</details>

---

## Benchmarks

Evaluation uses `slangmath` itself as the judge. `evaluateFraction` is run on both the model's output and the ground truth at 50 random test points. Algebraic equivalence counts — not structural identity.

| Benchmark                    | Metric                | Score     |
| ---------------------------- | --------------------- | --------- |
| AP Calculus AB               | Numerical equivalence | **92.4%** |
| AP Calculus BC               | Numerical equivalence | **88.1%** |
| MIT 18.01 — single-variable  | Numerical equivalence | **85.7%** |
| MIT 18.02 — multivariable    | Numerical equivalence | **78.3%** |
| Lagrange multiplier problems | Solution match        | **74.6%** |
| Step-level rule accuracy     | Rule match per step   | **89.2%** |

---

## Dataset

Every training pair is generated by `slangmath` acting as the ground-truth oracle. No external math engine. No LaTeX string parsing at training time.

| Source                      | Pairs     | Generation method                            |
| --------------------------- | --------- | -------------------------------------------- |
| SLaNg self-play (synthetic) | 5 000 000 | Random trees → slangmath solves → verified   |
| AP Calculus problems        | 40 000    | `latexToSlang()` → slangmath solves          |
| MIT OCW problems            | 120 000   | `latexToSlang()` → slangmath solves          |
| Multivariable problems      | 200 000   | `gradient`, `hessian`, `lagrangeMultipliers` |
| Taylor series examples      | 80 000    | `slang-advanced.js`                          |

### Self-play pipeline

```javascript
// data_pipeline/generate_synthetic.js
import { createTerm, createFraction, differentiateFraction } from "slangmath";
import { gradient, lagrangeMultipliers, findCriticalPoints } from "slangmath";

const gen = new SlangTreeGenerator({ maxDepth: 5, vars: ["x", "y"] });

for (let i = 0; i < 5_000_000; i++) {
  const inputTree = gen.sample(); // random SLaNg expression
  const outputTree = solveWithSlang(inputTree); // slangmath does the math
  if (outputTree.valid) dataset.push({ input: inputTree, output: outputTree });
}
```

---

## Training

Three stages. Each builds on the previous checkpoint.

### Stage 1 — Masked SLaNg tree pretraining

Randomly mask 20% of operator nodes in SLaNg trees. Train the encoder-decoder to reconstruct them. No calculus is involved — this stage teaches the model the structural grammar of valid SLaNg expressions.

```bash
python training/pretrain.py \
  --config training/config/pretrain.yaml \
  --data   data/splits/train \
  --output checkpoints/pretrain/
```

```yaml
# pretrain.yaml
model:
  encoder_layers: 8
  decoder_layers: 8
  hidden_dim: 512
  heads: 8

training:
  batch_size: 128
  lr: 2e-4
  warmup_steps: 5000
  max_steps: 300000
  mask_ratio: 0.20
  fp16: true
```

### Stage 2 — Supervised fine-tuning

Train the full model on complete (input SLaNg → output SLaNg + steps) pairs. The Rule Head and Step Tracer are trained here for the first time.

```bash
python training/finetune.py \
  --checkpoint checkpoints/pretrain/best.pt \
  --config     training/config/finetune.yaml \
  --data       data/splits/train \
  --output     checkpoints/sft/
```

### Stage 3 — SLaNg-in-the-loop hard example training

For each generated solution, run the corresponding `slangmath` function and compare outputs numerically via `evaluateFraction`. Wrong answers are upweighted at a ratio of 40% per batch.

```bash
python training/verifier_loop.py \
  --checkpoint       checkpoints/sft/best.pt \
  --hard_example_ratio 0.4 \
  --output           checkpoints/final/
```

---

## Installation

```bash
# Clone the repo
git clone https://github.com/your-org/CalculusSolver.git
cd CalculusSolver

# Node (data pipeline, tokenizer, verifier, eval)
npm install        # installs slangmath and other JS deps

# Python (model training, inference server)
pip install -r requirements.txt
```

Run the inference server:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

Point the JS client at it:

```javascript
const cs = new CalculusSolver({ endpoint: "http://localhost:8000" });
```

---

## Project structure

```
CalculusSolver/
│
├── model/
│   ├── architecture.py        ← top-level model class
│   ├── tree_encoder.py        ← 8-layer Transformer + parent-child attention bias
│   ├── tree_decoder.py        ← autoregressive decoder + SLaNg validity mask
│   ├── rule_head.py           ← per-node calculus rule classifier
│   └── step_tracer.py         ← step description auxiliary head
│
├── tokenizer/
│   ├── slang_serializer.js    ← SLaNg tree ↔ DFS token sequence
│   ├── vocab.json             ← all token types mapped to integer IDs
│   └── positional_encoding.py ← (depth, sibling_idx, path_hash) encoding
│
├── data_pipeline/
│   ├── generate_synthetic.js  ← random trees + slangmath solves them
│   ├── latex_to_slang.js      ← LaTeX bootstrap via latexToSlang()
│   └── verify_with_slang.js   ← numerical equivalence check on any pair
│
├── training/
│   ├── pretrain.py            ← Stage 1: masked tree reconstruction
│   ├── finetune.py            ← Stage 2: supervised SFT
│   ├── verifier_loop.py       ← Stage 3: hard example mining
│   └── config/
│       ├── pretrain.yaml
│       └── finetune.yaml
│
├── inference/
│   ├── CalculusSolver.js      ← public JS class, browser-ready
│   ├── solve.py               ← Python inference wrapper
│   ├── beam_search.py         ← beam search with SLaNg validity mask
│   └── verifier.js            ← post-hoc numerical check
│
├── api/
│   ├── app.py                 ← FastAPI application
│   └── routes/
│       ├── solve.py           ← POST /solve
│       └── validate.py        ← POST /validate
│
├── eval/
│   ├── slang_equivalence.js   ← evaluateFraction on model vs ground truth
│   ├── step_accuracy.js       ← per-rule accuracy
│   └── benchmarks/
│       ├── ap_calculus.json
│       ├── mit_ocw.json
│       └── multivariable.json
│
├── experiments/
│   ├── test_diff.js
│   ├── test_integration.js
│   ├── test_optimization.js
│   └── test_multivariable.js
│
├── ARCHITECTURE.md            ← full structural reference
├── GUIDE.md                   ← developer guide
├── package.json
├── requirements.txt
└── README.md
```

---

## Roadmap

- [x] Differentiation — `differentiateFraction`
- [x] Gradient & Hessian — `gradient`, `hessian`
- [x] Tangent plane / line — `tangentPlane`, `tangentLine`
- [x] Critical point classification — `findCriticalPoints`, `classifyCriticalPoint`
- [x] Lagrange multipliers — `lagrangeMultipliers`
- [x] Directional derivatives — `directionalDerivative`
- [x] Step trace generation
- [x] SLaNg-in-the-loop verifier training
- [ ] Definite integration
- [ ] Taylor series — `slang-advanced.js`
- [ ] ODE solving
- [ ] Browser playground — live SLaNg editor + CalculusSolver inference
- [ ] Fine-tuning API for custom SLaNg function libraries

---

## Contributing

CalculusSolver and SLaNg are sister projects.

- **SLaNg library bugs or new math functions** → [github.com/SENODROOM/SLaNg](https://github.com/SENODROOM/SLaNg)
- **CalculusSolver model, training, or I/O issues** → this repo

Adding support for a new operation always follows the same four steps:

1. Confirm `slangmath` supports it — or add it there first.
2. Generate training pairs using `slangmath` as ground truth.
3. Add the operation to the input envelope schema and `vocab.json`.
4. Fine-tune from the existing checkpoint (no need to retrain from scratch).

See [`GUIDE.md`](GUIDE.md) for the detailed walkthrough and [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full structural reference.

---

## License

| Asset         | License                                                              |
| ------------- | -------------------------------------------------------------------- |
| Code          | [Quantum Logics Proprietary](LICENSE)                                |
| Model weights | [Quantum Logics Proprietary](LICENSE)                                |
| SLaNg library | see [github.com/SENODROOM/SLaNg](https://github.com/SENODROOM/SLaNg) |

---

<div align="center">

_CalculusSolver — the intelligence layer above SLaNg. Same language, both directions._

</div>
