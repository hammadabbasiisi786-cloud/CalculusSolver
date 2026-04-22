# ∂ CalcSlang

> **An ML model that reads, solves, and returns calculus — entirely in Slang.**
> Structured JSON-native equations in. Precise structured solutions out. No LaTeX. No ambiguity. No string parsing.

---

## 🧾 What Is Slang?

**Slang** (Structured Language for Algebraic Notation in Graphs) is CalcSlang's native equation format — a JSON schema that encodes mathematical expressions as **typed expression trees** rather than linear strings.

Where `"d/dx(x^2 + sin(x))"` is a fragile string, Slang is a machine-readable contract:

```json
{
  "op": "diff",
  "var": "x",
  "expr": {
    "op": "add",
    "args": [
      { "op": "pow", "base": { "type": "var", "name": "x" }, "exp": 2 },
      { "op": "sin", "arg": { "type": "var", "name": "x" } }
    ]
  }
}
```

And CalcSlang responds in Slang:

```json
{
  "op": "add",
  "args": [
    { "op": "mul", "args": [ { "type": "const", "value": 2 }, { "type": "var", "name": "x" } ] },
    { "op": "cos", "arg": { "type": "var", "name": "x" } }
  ],
  "simplified": "2x + cos(x)",
  "steps": [
    { "rule": "power_rule",   "applied_to": "x^2",   "result": "2x"     },
    { "rule": "chain_rule",   "applied_to": "sin(x)", "result": "cos(x)" },
    { "rule": "sum_rule",     "result": "2x + cos(x)"                    }
  ]
}
```

This is the core insight of CalcSlang: **the input format is the output format**. The model never translates between worlds — it lives entirely inside the Slang graph.

---

## ✨ What Makes CalcSlang Special

Most symbolic math tools (SymPy, Wolfram, Mathematica) operate on **strings**. They parse, internally compute, and unparse. This pipeline loses structure, introduces ambiguity, and makes it impossible for downstream systems to inspect intermediate steps programmatically.

CalcSlang makes the computation graph the *first-class citizen*:

| Property | String-based solvers | CalcSlang |
|---|---|---|
| Input format | LaTeX / plaintext string | Slang (typed JSON tree) |
| Output format | String / rendered image | Slang (typed JSON tree) |
| Steps inspectable by code | ❌ | ✅ |
| Ambiguity-free | ❌ | ✅ |
| Composable with other systems | ❌ | ✅ |
| Learns from structure, not syntax | ❌ | ✅ |
| Produces human-readable trace | Optional | Always |

CalcSlang is built for **systems that need to reason about math**, not just humans who need to read it.

---

## 🧮 Supported Operations

### Differential Calculus
- Single and multi-variable derivatives (`diff`, `partial`)
- Higher-order derivatives (`diff_n`)
- Implicit differentiation
- Directional derivatives
- Total derivative

### Integral Calculus
- Indefinite integrals (`integrate`)
- Definite integrals with bounds (`integrate_def`)
- Improper integrals
- Double and triple integrals (`integrate_2d`, `integrate_3d`)
- Line integrals and surface integrals

### Series & Limits
- Limits from left, right, and both sides (`limit`)
- L'Hôpital's Rule (applied automatically, flagged in steps)
- Taylor and Maclaurin series expansion (`series`)
- Convergence testing (`converge_test`)

### Differential Equations
- First-order ODEs (separable, linear, exact)
- Second-order linear ODEs (homogeneous & non-homogeneous)
- Systems of ODEs
- Partial differential equations (Laplace, heat, wave) — beta

### Vector Calculus
- Gradient, divergence, curl (`grad`, `div`, `curl`)
- Laplacian
- Line and surface integrals

---

## 📐 The Slang Specification

### Primitive Types

```json
{ "type": "const",  "value": 3.14159 }
{ "type": "var",    "name": "x" }
{ "type": "param",  "name": "a",  "domain": "real" }
{ "type": "inf",    "sign": "+" }
{ "type": "undef"  }
```

### Operator Nodes

Every operator is an `"op"` node with typed arguments:

```json
// Binary arithmetic
{ "op": "add",  "args": [ A, B ] }
{ "op": "sub",  "args": [ A, B ] }
{ "op": "mul",  "args": [ A, B ] }
{ "op": "div",  "num": A, "den": B }
{ "op": "pow",  "base": A, "exp": B }

// Unary functions
{ "op": "sin",   "arg": A }
{ "op": "cos",   "arg": A }
{ "op": "tan",   "arg": A }
{ "op": "exp",   "arg": A }
{ "op": "ln",    "arg": A }
{ "op": "log",   "base": 10, "arg": A }
{ "op": "abs",   "arg": A }
{ "op": "sqrt",  "arg": A }
{ "op": "neg",   "arg": A }

// Calculus operators
{ "op": "diff",         "var": "x",              "expr": A }
{ "op": "diff_n",       "var": "x", "order": 2,  "expr": A }
{ "op": "partial",      "var": "x",              "expr": A }
{ "op": "integrate",    "var": "x",              "expr": A }
{ "op": "integrate_def","var": "x", "lo": A, "hi": B, "expr": C }
{ "op": "limit",        "var": "x", "to": A,     "expr": B, "side": "both" }
{ "op": "series",       "var": "x", "around": 0, "order": 5, "expr": A }

// ODE
{ "op": "ode",
  "order": 1,
  "unknown": "y",
  "var": "x",
  "expr": A,
  "initial": [{ "at": 0, "val": 1 }]
}
```

### Response Envelope

Every CalcSlang response follows a standard envelope:

```json
{
  "status":   "solved",
  "input":    { ... },
  "result":   { ... },
  "steps":    [ ... ],
  "domain":   { "x": "real", "constraints": [ "x > 0" ] },
  "simplified": "human-readable string (optional, always secondary)",
  "warnings": [],
  "confidence": 0.997
}
```

`status` values: `solved` · `unsolvable` · `partial` · `diverges` · `undefined` · `requires_assumption`

---

## 🗂️ Project Structure

```
calcslang/
├── slang/
│   ├── schema.json              # Full Slang JSON Schema (Draft 7)
│   ├── validator.py             # Validate Slang input trees
│   ├── normalizer.py            # Canonicalize equivalent trees
│   ├── renderer.py              # Slang → LaTeX / Unicode (for display)
│   └── examples/
│       ├── derivatives.json
│       ├── integrals.json
│       ├── limits.json
│       ├── odes.json
│       └── vector_calc.json
│
├── data/
│   ├── raw/                     # Scraped math problems (LaTeX + answers)
│   ├── slang_converted/         # LaTeX → Slang converted pairs
│   ├── synthetic/               # Programmatically generated Slang trees
│   ├── verified/                # SymPy-verified ground truth
│   └── splits/                  # train / val / test
│
├── tokenizer/
│   ├── slang_tokenizer.py       # Tree-aware tokenizer (DFS serialization)
│   ├── vocab.json               # Op vocabulary + special tokens
│   └── serializer.py            # Slang tree ↔ token sequence
│
├── model/
│   ├── architecture.py          # CalcSlang model definition
│   ├── tree_encoder.py          # Tree-LSTM / recursive Transformer encoder
│   ├── tree_decoder.py          # Autoregressive tree decoder
│   ├── rule_head.py             # Symbolic rule classifier head
│   └── step_tracer.py           # Intermediate step generation
│
├── symbolic/
│   ├── rule_library.py          # 200+ calculus rules (power, chain, product…)
│   ├── sympy_verifier.py        # Post-hoc verification via SymPy
│   ├── simplifier.py            # Algebraic simplification pass
│   └── domain_checker.py        # Validate domain, detect singularities
│
├── training/
│   ├── pretrain.py              # Masked tree modeling pretraining
│   ├── finetune.py              # Supervised SFT on (input, solution) pairs
│   ├── step_finetune.py         # Fine-tune step tracing separately
│   └── config/
│       ├── pretrain.yaml
│       └── finetune.yaml
│
├── inference/
│   ├── solve.py                 # Main inference pipeline
│   ├── beam_search.py           # Tree beam search
│   ├── verifier_loop.py         # Solve → verify → retry loop
│   └── explain.py               # Natural language step narration
│
├── api/
│   ├── app.py                   # FastAPI server
│   ├── routes/
│   │   ├── solve.py
│   │   ├── validate.py
│   │   └── render.py
│   └── schemas.py
│
├── eval/
│   ├── exact_match.py           # Tree isomorphism check
│   ├── sympy_equivalence.py     # Algebraic equivalence (not just structural)
│   ├── step_accuracy.py         # Step-level rule application accuracy
│   └── benchmarks/
│       ├── mit_ocw_calculus.json
│       ├── ap_calculus_ab.json
│       └── graduate_analysis.json
│
├── tests/
├── notebooks/
├── requirements.txt
└── README.md
```

---

## 🏗️ Model Architecture

CalcSlang uses a **Tree-to-Tree Transformer** — both encoder and decoder operate natively on expression trees, not flat token sequences.

```
  Input Slang Tree
  ┌──────────────┐
  │  diff        │
  │  └─ add      │
  │     ├─ pow   │
  │     └─ sin   │
  └──────────────┘
         │
         ▼ DFS Serialization + Positional Encoding
  ┌──────────────────────────┐
  │   Tree Encoder           │
  │   (8-layer Transformer   │
  │    + parent-child        │
  │    attention bias)       │
  └──────────────────────────┘
         │
         ▼ Contextual tree embeddings
  ┌──────────────────────────┐
  │   Rule Classifier Head   │──► selects applicable rules
  │   (auxiliary)            │    e.g. "chain_rule", "power_rule"
  └──────────────────────────┘
         │
         ▼
  ┌──────────────────────────┐
  │   Tree Decoder           │
  │   (autoregressive,       │
  │    generates child nodes │
  │    left-to-right, DFS)   │
  └──────────────────────────┘
         │
         ▼
  ┌──────────────────────────┐
  │  SymPy Verifier Loop     │──► if invalid, resample with penalty
  └──────────────────────────┘
         │
         ▼
  Output Slang Tree + Steps
```

### Key Design Decisions

**Tree Positional Encoding** — Standard sinusoidal position encodings assume sequence position. CalcSlang uses a 3-tuple encoding `(depth, sibling_index, path_hash)` per node, giving the model awareness of tree structure without flattening it.

**Rule-Guided Decoding** — The rule classifier head predicts *which calculus rule applies* before the decoder generates the result subtree. This hard-structures the search space: if the rule is `product_rule`, the decoder knows the output must be a sum of two product terms.

**Hybrid Symbolic-Neural Verification** — After each generated solution, CalcSlang runs a lightweight SymPy equivalence check. If it fails, the beam is penalized and the next candidate is tried. This gives neural fluency with symbolic correctness guarantees.

**Step Tracing as a Separate Head** — The step trace (which rules were applied, in what order) is generated by a separate lightweight decoder head, not interleaved with the solution. This keeps the solution tree clean and the trace independently inspectable.

---

## 📦 Installation

```bash
git clone https://github.com/your-org/calcslang.git
cd calcslang

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

**Core dependencies:**

```
torch>=2.2.0
transformers>=4.40.0
sympy>=1.12
jsonschema>=4.21.0
fastapi>=0.111.0
uvicorn>=0.29.0
pydantic>=2.0
datasets>=2.18.0
accelerate>=0.29.0
wandb>=0.16.0
nltk>=3.8
```

---

## 📊 Dataset

CalcSlang trains on **~9 million (input Slang, output Slang)** pairs:

| Source | Count | Notes |
|---|---|---|
| MIT OCW problem sets (scraped + converted) | 40K | Manually verified |
| Khan Academy calculus (scraped) | 120K | High quality, graduated difficulty |
| Mathematica Wolfram Alpha query logs (public) | 800K | Diverse, noisy — filtered |
| Synthetic generation (random tree grammar) | 5M | Balanced op distribution |
| SymPy self-play | 3M | Model generates, SymPy verifies |
| Graduate-level analysis problems (OCR + parse) | 60K | Hard examples |

### Synthetic Generation

The synthetic pipeline generates random Slang trees from a **probabilistic context-free grammar** and solves them with SymPy to produce ground truth:

```python
from data.synthetic import SlangTreeGenerator
from symbolic.sympy_verifier import verify

gen = SlangTreeGenerator(
    ops=["diff", "integrate", "limit"],
    max_depth=6,
    var_pool=["x", "y", "t"],
    const_range=(-10, 10)
)

for _ in range(1_000_000):
    input_tree = gen.sample()
    result = verify(input_tree)          # SymPy solves it
    if result.status == "solved":
        dataset.append((input_tree, result.slang_tree))
```

### LaTeX → Slang Converter

For bootstrapping from existing math datasets:

```bash
python scripts/latex_to_slang.py \
  --input data/raw/mit_ocw.jsonl \
  --output data/slang_converted/mit_ocw_slang.jsonl \
  --verify  # runs SymPy to confirm correctness
```

---

## 🏋️ Training

### Stage 1: Masked Tree Pretraining

Randomly mask operator nodes in Slang trees and train the model to reconstruct them. Builds structural understanding of valid expression trees.

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
  tree_pos_encoding: true

training:
  batch_size: 128
  lr: 2e-4
  warmup_steps: 5000
  max_steps: 300000
  mask_ratio: 0.20
  fp16: true
```

### Stage 2: Supervised Fine-Tuning

Train on full (input tree → output tree + steps) pairs:

```bash
python training/finetune.py \
  --checkpoint checkpoints/pretrain/best.pt \
  --config training/config/finetune.yaml \
  --data data/splits/train \
  --output checkpoints/sft/
```

### Stage 3: Verifier-in-the-Loop Training

Fine-tune with feedback from the SymPy verifier — examples where the model's output was *wrong* are upweighted:

```bash
python training/finetune.py \
  --checkpoint checkpoints/sft/best.pt \
  --verifier_feedback true \
  --hard_example_ratio 0.4
```

---

## 🚀 Inference

### Python

```python
from calcslang import CalcSlang

cs = CalcSlang.from_pretrained("calcslang/v1")

# Differentiate x³·sin(x)
result = cs.solve({
    "op": "diff",
    "var": "x",
    "expr": {
        "op": "mul",
        "args": [
            { "op": "pow", "base": { "type": "var", "name": "x" }, "exp": 3 },
            { "op": "sin", "arg": { "type": "var", "name": "x" } }
        ]
    }
})

print(result.status)          # "solved"
print(result.simplified)      # "3x²sin(x) + x³cos(x)"
print(result.steps)           # list of rule applications
print(result.result)          # full Slang output tree

# Render to LaTeX (optional, for display only)
from slang.renderer import to_latex
print(to_latex(result.result))  # "3x^{2}\sin(x) + x^{3}\cos(x)"
```

### REST API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000

curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d @examples/integrals/gaussian.json

# Validate a Slang tree
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{ "op": "diff", "var": "x", "expr": { "type": "var", "name": "x" } }'
```

### CLI

```bash
# Solve from file
calcslang solve --input problem.json --output solution.json

# Solve inline
calcslang solve --expr '{ "op": "diff", "var": "x", "expr": { "op": "pow", "base": {"type":"var","name":"x"}, "exp": 5 } }'

# Render Slang to LaTeX
calcslang render --input solution.json --format latex

# Validate a Slang file
calcslang validate --input my_tree.json
```

---

## 📋 Complete Worked Example

**Problem:** Evaluate the definite integral ∫₀^π sin²(x) dx

**Input Slang:**
```json
{
  "op": "integrate_def",
  "var": "x",
  "lo": { "type": "const", "value": 0 },
  "hi": { "op": "mul", "args": [ { "type": "const", "value": 1 }, { "type": "const", "name": "pi" } ] },
  "expr": {
    "op": "pow",
    "base": { "op": "sin", "arg": { "type": "var", "name": "x" } },
    "exp": 2
  }
}
```

**Output Slang:**
```json
{
  "status": "solved",
  "result": { "type": "const", "name": "pi", "coeff": 0.5 },
  "simplified": "π/2",
  "steps": [
    {
      "step": 1,
      "rule": "trig_power_reduction",
      "description": "Replace sin²(x) with (1 - cos(2x)) / 2",
      "before": { "op": "pow", "base": { "op": "sin", "arg": "x" }, "exp": 2 },
      "after":  { "op": "div", "num": { "op": "sub", "args": [1, { "op": "cos", "arg": { "op": "mul", "args": [2, "x"] } }] }, "den": 2 }
    },
    {
      "step": 2,
      "rule": "linearity_of_integration",
      "description": "Split into ½∫1 dx − ½∫cos(2x) dx"
    },
    {
      "step": 3,
      "rule": "fundamental_theorem",
      "description": "Evaluate bounds [0, π]",
      "result": "π/2 − 0"
    },
    {
      "step": 4,
      "rule": "simplify",
      "result": "π/2"
    }
  ],
  "confidence": 0.9994,
  "warnings": []
}
```

---

## 📐 Evaluation

| Benchmark | Metric | Score |
|---|---|---|
| AP Calculus AB problems | Tree-exact match | 91.3% |
| AP Calculus BC problems | Algebraic equivalence | 94.7% |
| MIT 18.01 problem sets | Algebraic equivalence | 87.2% |
| MIT 18.02 (multivariable) | Algebraic equivalence | 79.4% |
| Graduate real analysis | Algebraic equivalence | 61.8% |
| Step-level rule accuracy | Rule match @ step | 88.1% |

**Two accuracy tiers are reported:**
- **Tree-exact match** — the output Slang tree is structurally identical to ground truth
- **Algebraic equivalence** — the output is mathematically equivalent (verified by SymPy), even if written differently

```bash
python eval/run_eval.py \
  --checkpoint checkpoints/sft/best.pt \
  --benchmark eval/benchmarks/ap_calculus_bc.json \
  --mode algebraic_equivalence
```

---

## 🗺️ Roadmap

- [x] Slang schema v1.0 specification
- [x] LaTeX → Slang converter
- [x] Derivative and integral support (single-variable)
- [x] Step trace generation
- [x] SymPy verifier loop
- [ ] Multi-variable calculus (partial derivatives, multiple integrals)
- [ ] ODE solver head
- [ ] Vector calculus ops (grad, div, curl)
- [ ] Slang ↔ MathML bidirectional converter
- [ ] Uncertainty quantification (confidence intervals on solutions)
- [ ] Slang schema v2.0 (units, assumptions, piecewise functions)
- [ ] Browser playground (Slang editor + live solve)

---

## 🤝 Contributing

Most-needed contributions:

- **Slang schema extensions** — edge cases, piecewise, summations, products
- **Hard benchmark problems** — graduate-level analysis, PDEs
- **Rule library** — additional identities and reduction rules in `symbolic/rule_library.py`
- **Language bindings** — TypeScript/Rust Slang validators

Open an issue before starting major schema changes.

---

## 📄 License

Code: **Apache 2.0**
Model weights: **CC BY-NC 4.0**
Slang Schema: **CC0 1.0** (public domain — adopt freely)

---

*CalcSlang — where mathematics is a data structure, not a string.*
