# Bryan — Independent ML Researcher

Building behavioral auditing and mechanistic interpretability tools for LLMs. Current focus: **general-purpose behavioral diagnostics** — measuring what models know, where they're biased, and when they're sycophantic, using teacher-forced confidence probes and activation steering.

**Core thesis:** Social compliance and social awareness share representational capacity at mid-depth transformer layers. Factual representations are architecturally universal; sycophancy suppression is not. Activation steering is architecture-contingent — each model family needs its own behavioral map.

---

### Current Work

**[rho-eval](https://pypi.org/project/rho-eval/)** (v2.1.0) — Drop-in behavioral audit for any LLM. 926 probes across 5 dimensions, no internet required. Plugin architecture for custom behaviors. **Now with Apple Silicon MLX acceleration.**

```bash
pip install rho-eval
rho-eval Qwen/Qwen2.5-7B-Instruct --behaviors all --format table
```

```python
# Works with PyTorch models — or MLX models with zero code changes
import mlx_lm
from rho_eval import audit

model, tokenizer = mlx_lm.load("mlx-community/Qwen2.5-7B-Instruct-4bit")
report = audit(model=model, tokenizer=tokenizer, behaviors="all")
# ~5x faster inference, ~10x faster training on Apple Silicon
```

**v2.1.0 — MLX Acceleration:**

- **Transparent MLX dispatch** — `audit()`, `analyze_confidence()`, `generate()`, and `get_mean_logprob()` auto-detect MLX models and use native Apple Silicon inference. No code changes needed.
- **MLX training backends** — `mlx_rho_guided_sft()` for alignment, `mlx_gentle_finetune()` for post-compression recovery. Avoids PyTorch MPS NaN gradient bugs entirely.
- **~10x training speedup** — Rho-guided SFT sweep: 22 hours (CPU PyTorch) → ~2 hours (MLX). Enables rapid iteration on alignment experiments.
- **Bigger models on less hardware** — MLX unified memory means 7B-4bit runs on 16GB MacBooks.
- **Plugin architecture** — `ABCBehavior` base class with `@register` decorator. Add custom behaviors in 30 lines.
- **Comparison system** — `rho_eval.compare(after, before)` with IMPROVED/DEGRADED delta tables. CI-friendly exit codes.

**Earlier findings (v1.x) — Comparative Anatomy of Behavioral Representations:**

| Property | Qwen2.5-7B | Mistral-7B |
|----------|:----------:|:----------:|
| Factual sweet spot | L24 (86%), +0.152 | L24 (75%), +0.117 |
| Sycophancy sweet spot | L17 (61%), +0.293 | *None* (+0.013 max) |
| Kill zone | L17 (bias: -0.437) | L14-L18 (bias: -0.460) |
| Factual transfer | Yes | Yes |
| Sycophancy transfer | -- | No |

---

### Published Repos

| Repo | What it does |
|------|-------------|
| [knowledge-fidelity](https://github.com/SolomonB14D3/knowledge-fidelity) | Behavioral auditing + SVD compression toolkit for LLMs. Now **rho-eval** on PyPI. 926 probes, MLX-accelerated inference/training, activation steering, cross-model validation. Featured in [Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression#tools). |
| [confidence-cartography](https://github.com/SolomonB14D3/confidence-cartography) | Teacher-forced confidence as a false-belief sensor. Human false-belief correlation rho=0.652 across Pythia 160M-12B. |
| [intelligent-svd](https://github.com/SolomonB14D3/intelligent-svd) | Knowledge-preserving SVD compression. CF90 method: TruthfulQA +5%, 75% fact retention. |

### Research Roadmap

1. ~~General-purpose behavioral diagnostic toolkit~~ — **Done** (rho-eval v2.0.0)
2. ~~Mechanistic interpretability of behavioral subspaces~~ — **Done** (SVD subspace analysis, Grassmann angles, Universal Kill Zone discovery)
3. **Rho-guided alignment / fine-tuning** — *in progress* (MLX + PyTorch backends, rho_weight sweep running)
4. Hybrid weight + activation control framework
5. Open behavioral benchmark suite (Fidelity-Bench 2.0)
