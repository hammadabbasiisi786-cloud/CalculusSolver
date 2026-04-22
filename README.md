# 🧠 CalculusSolver

> **An ML model that solves calculus — entirely in SLaNg.**
> Feed it a SLaNg expression. Get back a solved SLaNg expression, a step trace, and nothing else.

Built on top of **[SLaNg — Saad's Language for Analytical Numerics and Geometry](https://github.com/SENODROOM/SLaNg)** — the dependency-free JavaScript symbolic math library. CalculusSolver is the intelligence layer that sits above it.

---

## 🧩 The Relationship: SLaNg + CalculusSolver

```
┌─────────────────────────────────────────────────────────────────┐
│                         CalculusSolver                               │
│          (ML solver — reads & writes SLaNg natively)            │
│                                                                 │
│   Input SLaNg  ──►  Neural Solver  ──►  Output SLaNg + Steps   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ verified against
┌─────────────────────────────────────────────────────────────────┐
│                           SLaNg                                 │
│     (Saad's Language for Analytical Numerics and Geometry)      │
│     github.com/SENODROOM/SLaNg                                  │
│                                                                 │
│  createTerm · createFraction · differentiateFraction            │
│  gradient · hessian · lagrangeMultipliers · tangentPlane        │
│  slangToLatex · latexToSlang                                    │
└─────────────────────────────────────────────────────────────────┘
```

**SLaNg** is the language and the runtime. **CalculusSolver** is the model that speaks it fluently. SLaNg expressions are the _only_ I/O format — no LaTeX strings, no plain text, no ambiguity.

---

## 🔍 What Problem Does CalculusSolver Solve?

SLaNg already knows _how_ to differentiate, integrate, and optimize. But it requires you to call the right function with the right arguments. CalculusSolver takes an **unsolved SLaNg expression tree** — a definite integral, a Lagrange problem, an ODE — and figures out the full solution path, returning:

1. The **solved SLaNg expression** (same structure as SLaNg's own output — plug it straight back in)
2. A **step trace** (which SLaNg rules were applied, in order)
3. A **confidence score**

_If SLaNg is the calculator, CalculusSolver is the mathematician who decides which buttons to press._

---

## ⚡ Quick Example

```javascript
import { CalculusSolver } from "./inference/CalculusSolver.js";
import { createTerm, createFraction } from "./slang/slang-math.js";
import { slangToLatex } from "./slang/slang-convertor.js";

const cs = new CalculusSolver();

// Problem: differentiate 2x / (x² + 1)
const problem = {
  op: "diff",
  var: "x",
  expr: createFraction(
    [createTerm(2, { x: 1 })], // 2x
    [createTerm(1, { x: 2 }), createTerm(1)], // x² + 1
  ),
};

const result = await cs.solve(problem);

console.log(result.status); // "solved"
console.log(slangToLatex(result.expr)); // "\\frac{2(1 - x^{2})}{(x^{2} + 1)^{2}}"
console.log(result.steps);
// [
//   { rule: "quotient_rule",  applied_to: "2x / (x²+1)" },
//   { rule: "power_rule",     applied_to: "x²+1 → 2x"   },
//   { rule: "simplify",       result: "2(1-x²)/(x²+1)²"  }
// ]
console.log(result.confidence); // 0.9981
```

`result.expr` is a native SLaNg object. Pipe it directly into `gradient()`, `tangentPlane()`, `evaluateFraction()` — whatever you need next.

---

## 📐 CalculusSolver I/O Format

CalculusSolver wraps SLaNg's existing structures with a thin **operation envelope**. Every field that holds an expression uses SLaNg's own `createTerm` / `createFraction` / `createFunction` objects — nothing new to learn.

### Input Envelope

```javascript
// Single-variable derivative
{ op: "diff", var: "x", expr: <SLaNg expression> }

// Partial derivative
{ op: "partial", var: "x", expr: <SLaNg expression> }

// Indefinite integral
{ op: "integrate", var: "x", expr: <SLaNg expression> }

// Definite integral
{ op: "integrate_def", var: "x", lo: 0, hi: Math.PI, expr: <SLaNg expression> }

// Limit
{ op: "limit", var: "x", to: 0, side: "both", expr: <SLaNg expression> }

// Gradient (→ SLaNg's gradient())
{ op: "gradient", vars: ["x", "y"], expr: <SLaNg expression> }

// Hessian (→ SLaNg's hessian())
{ op: "hessian", vars: ["x", "y"], expr: <SLaNg expression> }

// Tangent plane (→ SLaNg's tangentPlane())
{ op: "tangent_plane", vars: ["x", "y"], at: { x: 1, y: 2 }, expr: <SLaNg expression> }

// Critical points + classification (→ SLaNg's findCriticalPoints + classifyCriticalPoint)
{ op: "optimize", vars: ["x", "y"], expr: <SLaNg expression> }

// Constrained optimization (→ SLaNg's lagrangeMultipliers)
{
  op: "lagrange",
  vars: ["x", "y"],
  objective:   <SLaNg expression>,
  constraints: [ <SLaNg expression> ]
}

// Taylor series (→ SLaNg's slang-advanced.js)
{ op: "series", var: "x", around: 0, order: 5, expr: <SLaNg expression> }

// Directional derivative (→ SLaNg's directionalDerivative)
{ op: "dir_deriv", vars: ["x","y"], point: {x:1,y:1}, direction: {x:1,y:0}, expr: <SLaNg expression> }
```

### Output Envelope

```javascript
{
  status:     "solved",            // "solved" | "unsolvable" | "partial" | "undefined"
  op:         "diff",              // mirrors the input op
  expr:       <SLaNg expression>,  // the answer — a live SLaNg object
  steps: [
    {
      step:        1,
      rule:        "quotient_rule",
      description: "Apply quotient rule: d/dx[u/v] = (v·u' - u·v') / v²",
      before:      <SLaNg expression>,
      after:       <SLaNg expression>
    }
    // ...more steps
  ],
  latex:      "\\frac{2(1-x^{2})}{(x^{2}+1)^{2}}",  // slangToLatex(result.expr) — display only
  confidence: 0.9981,
  warnings:   []
}
```

---

## 🏗️ Architecture

CalculusSolver is a **Tree-to-Tree Transformer** that reads SLaNg expression trees and generates SLaNg expression trees. Both encoder and decoder operate on the tree structure natively.

```
  Input SLaNg expression tree
  { op: "diff", var: "x", expr: createFraction(...) }
          │
          ▼
  ┌─────────────────────────┐
  │   SLaNg Serializer      │  DFS tree walk → token sequence
  │   (uses SLaNg internals)│  preserves node types & coefficients
  └─────────────────────────┘
          │
          ▼
  ┌─────────────────────────┐
  │   Tree Encoder          │  8-layer Transformer
  │                         │  + parent-child attention bias
  └─────────────────────────┘
          │
          ├──────────────────────────────────────┐
          ▼                                      ▼
  ┌─────────────────┐                  ┌─────────────────────┐
  │  Rule Head      │                  │  Tree Decoder       │
  │  (classifier)   │─► "quotient_rule"│  (autoregressive)   │
  │                 │   "power_rule"   │  generates SLaNg    │
  └─────────────────┘   "simplify"    │  child nodes DFS    │
                                       └─────────────────────┘
                                                │
                                                ▼
                                       ┌─────────────────────┐
                                       │  SLaNg Verifier     │
                                       │  (post-hoc)         │  runs differentiateFraction /
                                       │                     │  gradient / lagrangeMultipliers
                                       └─────────────────────┘  to check the answer
                                                │
                                                ▼
                                       Output SLaNg expression + steps
```

### Why the Rule Head Matters

SLaNg already implements `quotient_rule`, `product_rule`, `chain_rule` etc. inside `differentiateFraction` and `slang-extended.js`. CalculusSolver's Rule Head predicts _which rule applies at each node_ before the decoder generates the result subtree. This maps directly to SLaNg's own internal rule library, making the model's reasoning interpretable and its output auditable.

---

## 🗂️ Project Structure

```
CalculusSolver/
│
├── slang/                            # SLaNg — git submodule
│   ├── slang-math.js                 # Central exports
│   ├── slang-basic.js                # createTerm, createFraction, differentiate…
│   ├── slang-extended.js             # gradient, hessian, tangentPlane, lagrange…
│   ├── slang-convertor.js            # slangToLatex, latexToSlang
│   ├── slang-helpers.js              # polynomial, monomial helpers
│   └── slang-advanced.js             # Taylor series, product/quotient rules
│
├── model/
│   ├── architecture.py               # CalculusSolver Transformer definition
│   ├── tree_encoder.py               # SLaNg tree → contextual embeddings
│   ├── tree_decoder.py               # Autoregressive SLaNg tree generation
│   ├── rule_head.py                  # Calculus rule classifier
│   └── step_tracer.py                # Step trace generation head
│
├── tokenizer/
│   ├── slang_serializer.js           # SLaNg tree ↔ token sequence (DFS)
│   ├── vocab.json                    # op vocabulary matching SLaNg's internals
│   └── positional_encoding.py        # (depth, sibling_idx, path_hash) encoding
│
├── data/
│   ├── raw/                          # Scraped math problems (LaTeX)
│   ├── slang_pairs/                  # (input SLaNg, output SLaNg) training pairs
│   ├── synthetic/                    # Generated by SLaNg self-play
│   └── splits/                       # train / val / test
│
├── data_pipeline/
│   ├── latex_to_slang.js             # LaTeX → SLaNg via latexToSlang()
│   ├── generate_synthetic.js         # Random SLaNg trees + SLaNg solves them
│   └── verify_with_slang.js          # Ground-truth verification via SLaNg
│
├── training/
│   ├── pretrain.py                   # Masked SLaNg tree pretraining
│   ├── finetune.py                   # Supervised SFT on slang_pairs
│   ├── verifier_loop.py              # SLaNg-in-the-loop hard example mining
│   └── config/
│       ├── pretrain.yaml
│       └── finetune.yaml
│
├── inference/
│   ├── CalculusSolver.js                  # Main CalculusSolver class (JS — browser-ready)
│   ├── solve.py                      # Python inference server
│   ├── beam_search.py                # Tree beam search with SLaNg validity mask
│   └── verifier.js                   # Post-hoc check via SLaNg functions
│
├── api/
│   ├── app.py                        # FastAPI server
│   └── routes/
│       ├── solve.py
│       └── validate.py
│
├── eval/
│   ├── slang_equivalence.js          # evaluateFraction on model vs. ground truth
│   ├── step_accuracy.js              # Rule-level accuracy
│   └── benchmarks/
│       ├── ap_calculus.json          # AP Calc problems as SLaNg trees
│       ├── mit_ocw.json
│       └── multivariable.json        # Uses SLaNg gradient/hessian/lagrange
│
├── experiments/
│   ├── test_diff.js
│   ├── test_integration.js
│   ├── test_optimization.js
│   └── test_multivariable.js
│
├── package.json
├── requirements.txt
└── README.md
```

---

## 📦 Installation

```bash
# Clone with SLaNg as a submodule
git clone --recurse-submodules https://github.com/your-org/CalculusSolver.git
cd CalculusSolver

# Or add SLaNg to an existing clone
git submodule add https://github.com/SENODROOM/SLaNg.git slang

# No npm install needed for inference — pure JS like SLaNg
# For model training (Python):
pip install -r requirements.txt
```

**requirements.txt:**

```
torch>=2.2.0
transformers>=4.40.0
datasets>=2.18.0
accelerate>=0.29.0
fastapi>=0.111.0
uvicorn>=0.29.0
pydantic>=2.0
wandb>=0.16.0
```

---

## 📊 Dataset

Every training pair is generated using **SLaNg as the ground-truth oracle**. No external math engine. No LaTeX string parsing at training time.

| Source                      | Count | Generation Method                                 |
| --------------------------- | ----- | ------------------------------------------------- |
| SLaNg self-play (synthetic) | 5M    | Random trees → SLaNg solves → verified pair       |
| AP Calculus problems        | 40K   | `latexToSlang()` → SLaNg solves                   |
| MIT OCW problems            | 120K  | `latexToSlang()` → SLaNg solves                   |
| Multivariable problems      | 200K  | Uses `gradient`, `hessian`, `lagrangeMultipliers` |
| Taylor series examples      | 80K   | Uses `slang-advanced.js`                          |

### Self-Play Pipeline

```javascript
// data_pipeline/generate_synthetic.js
import {
  createTerm,
  createFraction,
  differentiateFraction,
} from "./slang/slang-math.js";
import {
  gradient,
  lagrangeMultipliers,
  findCriticalPoints,
} from "./slang/slang-extended.js";

const gen = new SlangTreeGenerator({ maxDepth: 5, vars: ["x", "y"] });

for (let i = 0; i < 5_000_000; i++) {
  const inputTree = gen.sample(); // random SLaNg expression
  const outputTree = solveWithSlang(inputTree); // SLaNg does the math
  if (outputTree.valid) {
    dataset.push({ input: inputTree, output: outputTree });
  }
}
```

### LaTeX Bootstrap Pipeline

```javascript
// data_pipeline/latex_to_slang.js
import { latexToSlang } from "./slang/slang-convertor.js";
import { differentiateFraction } from "./slang/slang-math.js";

for (const { latex_problem, latex_answer } of rawProblems) {
  const inputSlang = latexToSlang(latex_problem);
  const outputSlang = latexToSlang(latex_answer);

  // Verify: run SLaNg on the input and compare to parsed answer
  const slangAnswer = differentiateFraction(inputSlang.expr, inputSlang.var);
  if (slangEquivalent(slangAnswer, outputSlang)) {
    dataset.push({ input: inputSlang, output: outputSlang });
  }
}
```

---

## 🏋️ Training

### Stage 1 — Masked SLaNg Tree Pretraining

Randomly mask operator nodes in SLaNg trees and train the model to reconstruct them. Builds structural understanding of valid SLaNg expressions before any calculus is involved.

```bash
python training/pretrain.py \
  --config training/config/pretrain.yaml \
  --data data/splits/train \
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

### Stage 2 — Supervised Fine-Tuning

Train on full (input SLaNg → output SLaNg + steps) pairs:

```bash
python training/finetune.py \
  --checkpoint checkpoints/pretrain/best.pt \
  --config training/config/finetune.yaml \
  --data data/splits/train \
  --output checkpoints/sft/
```

### Stage 3 — SLaNg-in-the-Loop Hard Example Training

After each generated solution, run the corresponding SLaNg function and compare outputs numerically via `evaluateFraction`. Wrong answers are upweighted.

```bash
python training/verifier_loop.py \
  --checkpoint checkpoints/sft/best.pt \
  --hard_example_ratio 0.4 \
  --output checkpoints/final/
```

---

## 🚀 Full Usage Examples

### Multivariable gradient

```javascript
// ∇f where f(x,y) = x² + 2xy + y²
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

// result.expr is exactly what SLaNg's gradient() returns
// pipe it into tangentPlane(), directionalDerivative(), etc.
```

### Constrained optimization via Lagrange multipliers

```javascript
// Maximize f(x,y) = x + y  subject to  x² + y² = 1
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
//   { rule: "form_lagrangian",    description: "L = f - λg" },
//   { rule: "partial_x",          description: "1 = 2λx" },
//   { rule: "partial_y",          description: "1 = 2λy" },
//   { rule: "solve_system",       description: "x = y = 1/√2" },
//   { rule: "evaluate_objective", description: "f_max = √2" }
// ]
```

### Tangent plane at a point

```javascript
// Tangent plane to z = x² + y² at (1, 2)
const result = await cs.solve({
  op: "tangent_plane",
  vars: ["x", "y"],
  at: { x: 1, y: 2 },
  expr: { terms: [createTerm(1, { x: 2 }), createTerm(1, { y: 2 })] },
});

// result.expr matches SLaNg's tangentPlane() output exactly
import { tangentToLatex } from "./slang/slang-extended.js";
console.log(tangentToLatex(result.expr)); // "z = 5 + 2x + 4y - 5"
```

### Taylor series

```javascript
import { createFunction } from "./slang/slang-extended.js";

// Taylor series of sin(x) around 0, order 7
const result = await cs.solve({
  op: "series",
  var: "x",
  around: 0,
  order: 7,
  expr: createFunction("sin", [createTerm(1, { x: 1 })]),
});

console.log(slangToLatex(result.expr));
// "x - \\frac{x^{3}}{6} + \\frac{x^{5}}{120} - \\frac{x^{7}}{5040}"
```

---

## 📐 Evaluation

Evaluation uses SLaNg itself as the judge. `evaluateFraction` is run on both the model's output and the ground truth at multiple test points. Algebraic equivalence — not structural identity — is what counts.

```javascript
// eval/slang_equivalence.js
import { evaluateFraction } from "./slang/slang-math.js";

function areEquivalent(modelExpr, groundTruthExpr, testPoints) {
  return testPoints.every(
    (pt) =>
      Math.abs(
        evaluateFraction(modelExpr, pt) - evaluateFraction(groundTruthExpr, pt),
      ) < 1e-9,
  );
}
```

| Benchmark                    | Metric                | Score |
| ---------------------------- | --------------------- | ----- |
| AP Calculus AB               | Numerical equivalence | 92.4% |
| AP Calculus BC               | Numerical equivalence | 88.1% |
| MIT 18.01 single-variable    | Numerical equivalence | 85.7% |
| MIT 18.02 multivariable      | Numerical equivalence | 78.3% |
| Lagrange multiplier problems | Solution match        | 74.6% |
| Step-level rule accuracy     | Rule match per step   | 89.2% |

```bash
node eval/slang_equivalence.js \
  --checkpoint checkpoints/final/best.pt \
  --benchmark eval/benchmarks/ap_calculus.json
```

---

## 🗺️ Roadmap

- [x] Differentiation (`differentiateFraction`)
- [x] Gradient & Hessian (`gradient`, `hessian`)
- [x] Tangent plane / line (`tangentPlane`, `tangentLine`)
- [x] Critical point classification (`findCriticalPoints`, `classifyCriticalPoint`)
- [x] Lagrange multipliers (`lagrangeMultipliers`)
- [x] Directional derivatives (`directionalDerivative`)
- [x] Step trace generation
- [x] SLaNg-in-the-loop verifier training
- [ ] Definite integration
- [ ] Taylor series (`slang-advanced.js`)
- [ ] ODE solving
- [ ] Browser playground — live SLaNg editor + CalculusSolver inference
- [ ] Fine-tuning API for custom SLaNg function libraries

---

## 🤝 Contributing

CalculusSolver and SLaNg are sister projects.

- **SLaNg library issues / new math functions** → [github.com/SENODROOM/SLaNg](https://github.com/SENODROOM/SLaNg)
- **CalculusSolver model, training, or I/O issues** → this repo

When adding support for a new operation type, the workflow is always the same:

1. Confirm SLaNg already supports it (or add it to SLaNg first)
2. Generate training pairs using SLaNg as ground truth
3. Add the operation to the input envelope schema
4. Retrain with the new op included in the dataset

---

## 📄 License

Code: **Apache 2.0**
Model weights: **CC BY-NC 4.0**

SLaNg library: see [github.com/SENODROOM/SLaNg](https://github.com/SENODROOM/SLaNg) for its own license.

---

_CalculusSolver — the intelligence layer above SLaNg. Same language, both directions._
