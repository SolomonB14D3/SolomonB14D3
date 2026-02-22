# Bryan — Independent ML Researcher

Building subspace sculpting tools for LLMs: using SVD compression + teacher-forced confidence to carve separable behavioral traits and turn compressed models into tunable agents.

**Core thesis:** Compression isn't just size reduction — it's behavioral control. Denoise factual subspaces, recover bias signals, reduce sycophancy, then switch or merge specialized checkpoints into hybrid agents.

---

### Current Work

**[knowledge-fidelity](https://github.com/SolomonB14D3/knowledge-fidelity)** — Flagship toolkit. Compress an LLM while auditing whether it still knows truth vs popular myths. One call to `compress_and_audit()`:
- SVD compression of Q/K/O attention projections (CF90 method)
- Teacher-forced confidence as a false-belief sensor (rho metric)
- Mandela effect calibration, medical claims, cross-behavioral probes
- **Key finding:** SVD compression at 70% rank *improves* Mandela rho from 0.829 → 0.943 on Qwen-7B — compression as denoising

**Cross-behavioral generalization** (running now) — Testing whether the CF90 denoising effect extends beyond factual probes to toxicity (ToxiGen), bias (BBQ), sycophancy (Anthropic evals), and reasoning (GSM8K). Early results on Qwen2.5-7B:

| Ratio | Factual | Toxicity | Bias | Sycophancy | Reasoning |
|:-----:|:-------:|:--------:|:----:|:----------:|:---------:|
| baseline | 0.474 | 0.522 | 0.773 | 0.120 | 0.010 |
| 50% | +0.008 | -0.006 | -0.230 | +0.053 | +0.010 |
| 60% | -0.022 | -0.008 | -0.013 | +0.040 | +0.020 |

Bias recovers dramatically between 50% and 60%. Sycophancy and reasoning consistently improve. Full sweep (50-90%) in progress.

---

### Published Repos

| Repo | What it does |
|------|-------------|
| [knowledge-fidelity](https://github.com/SolomonB14D3/knowledge-fidelity) | Compress + audit LLMs with shared factual probes. SVD + confidence cartography unified. |
| [confidence-cartography](https://github.com/SolomonB14D3/confidence-cartography) | Teacher-forced confidence as a false-belief sensor. Mandela effect calibration. Human false-belief correlation rho=0.652 across Pythia 160M-12B. |
| [intelligent-svd](https://github.com/SolomonB14D3/intelligent-svd) | Knowledge-preserving SVD compression. CF90 method: TruthfulQA +5%, 75% fact retention. |
| [Awesome-LLM-Compression](https://github.com/SolomonB14D3/Awesome-LLM-Compression) | Curated list of LLM compression research. |

---

### What's Next

1. **Freeze-ratio sweep** — Which layers carry which behavioral signals? Fix compression at 70%, sweep freeze ratios to map early-layer vs late-layer trait localization.
2. **Subspace merging** — Specialized checkpoints (factual-optimized, bias-denoised, sycophancy-reduced) merged via task arithmetic or TIES into a single hybrid model.
3. **Agentic evals** — Can a compressed hybrid model do tool-calling, multi-step planning, and long-context retrieval as well as uncompressed baselines?
4. **Scaling** — From 7B to 13B to 32B. Does behavioral separability survive scale?

---

*All experiments on Apple Silicon (M3 Ultra, 192GB). No cloud compute.*
