# Bryan — Independent ML Researcher

Building behavioral auditing tools for LLMs: using teacher-forced confidence (rho) to map where factual knowledge, bias, sycophancy, and reasoning live inside transformer layers — then using that map to control what compression preserves or removes.

**Core thesis:** Compression isn't just size reduction — it's behavioral surgery. Different behaviors are encoded in different layer regions. Once you know the map, you can selectively denoise, preserve, or steer.

---

### Current Work

**[knowledge-fidelity](https://github.com/SolomonB14D3/knowledge-fidelity)** (v0.4.0) — Compress an LLM while auditing what it still knows. Includes `rho-audit`, a standalone behavioral auditing CLI.

```bash
pip install knowledge-fidelity
rho-audit Qwen/Qwen2.5-7B-Instruct --behaviors all
```

**Key results:**

- SVD compression at 70% rank *improves* Mandela rho from 0.829 to 0.943 on Qwen-7B — compression as denoising
- Behavioral localization via freeze-ratio sweep (Qwen2.5-7B, SVD 70%, LoRA recovery):

| Behavior | Baseline | Best delta | Best freeze | Where it lives |
|----------|:--------:|:----------:|:-----------:|----------------|
| Factual  | 0.474 | **+0.072** | 75% | Early layers |
| Bias     | 0.773 | **+0.093** | 25% | Late layers |
| Sycophancy | 0.120 | **+0.027** | 50% | Early layers |
| Reasoning | 0.010 | **+0.040** | 50% | Late layers |
| Toxicity | 0.521 | -0.005 | — | Immovable |

- Merge method behavioral audit — SLERP/TIES/DARE-TIES on Qwen2.5-7B-Instruct + Coder:

| Method | Factual ρ | Bias ρ | Sycophancy ρ |
|--------|:---------:|:------:|:------------:|
| Baseline | 0.474 | **0.773** | 0.120 |
| SLERP | 0.517 | 0.613 | 0.140 |
| TIES | 0.546 | 0.363 | **0.280** |
| DARE-TIES | **0.612** | 0.203 | 0.007 |

DARE-TIES gains +0.138 factual but destroys bias detection (-0.570) and sycophancy. Standard benchmarks won't catch this — `rho-audit` does.

---

### Published Repos

| Repo | What it does |
|------|-------------|
| [knowledge-fidelity](https://github.com/SolomonB14D3/knowledge-fidelity) | Compress + audit LLMs with shared factual probes. `rho-audit` CLI for behavioral profiling. |
| [confidence-cartography](https://github.com/SolomonB14D3/confidence-cartography) | Teacher-forced confidence as a false-belief sensor. Human false-belief correlation rho=0.652 across Pythia 160M-12B. |
| [intelligent-svd](https://github.com/SolomonB14D3/intelligent-svd) | Knowledge-preserving SVD compression. CF90 method: TruthfulQA +5%, 75% fact retention. |
| [Awesome-LLM-Compression](https://github.com/SolomonB14D3/Awesome-LLM-Compression) | Curated list of LLM compression research. |

---

### What's Next

1. **Rho leaderboard** — Expand merge audit to Mistral-7B, Llama-3.1-8B, and more merge recipes; publish public behavioral leaderboard
2. **Behaviorally-aware merging** — Use per-behavior rho scores as optimization targets for layer-wise merge weights
3. **Steering vectors** — Extract activation directions from rho probes for runtime behavioral control
4. **Paper** — Layer localization + merge audit findings are novel; targeting a workshop or findings track submission

---

*All experiments on Apple Silicon (M3 Ultra, 192GB). No cloud compute.*
