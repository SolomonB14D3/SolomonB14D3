# Bryan -- Independent ML Researcher

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-pink?logo=github)](https://github.com/sponsors/SolomonB14D3)

Building behavioral auditing and alignment tools for LLMs. Current focus: **surgical behavioral alignment** -- using category-aware contrastive losses to fix sycophancy without damaging bias fairness.

**Core finding:** Standard SFT inverts toxicity discrimination in Qwen 7B (+0.145 baseline to -0.003 post-SFT, 5 seeds, p<0.001). Rho-guided SFT with a contrastive confidence loss repairs this (d=10.8 on toxicity, d=13.7 on bias, p<0.0001). At 7B scale, sycophancy SFT causes concentrated collateral damage on Age (-29%), Race (-14%), and Religion (-11%) -- motivating the Rho-Surgery approach with targeted protection losses.

---

### Current Work

**[rho-eval](https://pypi.org/project/rho-eval/)** (v2.2.2) -- Drop-in behavioral audit for any LLM. 1,826 probes across 8 dimensions (factual, toxicity, sycophancy, bias, reasoning, refusal, deception, over-refusal), no internet required. Plugin architecture for custom behaviors. **Apple Silicon MLX acceleration. Rho-guided SFT alignment. Rho-Surgery with gamma protection loss.**

```bash
pip install rho-eval
rho-eval Qwen/Qwen2.5-7B-Instruct --behaviors all --format table
```

```python
# Works with PyTorch models -- or MLX models with zero code changes
import mlx_lm
from rho_eval import audit

model, tokenizer = mlx_lm.load("mlx-community/Qwen2.5-7B-Instruct-4bit")
report = audit(model=model, tokenizer=tokenizer, behaviors="all")
# ~5x faster inference, ~10x faster training on Apple Silicon
```

**v2.2.2 -- Rho-Surgery + Hybrid Control:**

- **Rho-Surgery** -- Targeted intervention pipeline: Diagnose category-level risk, apply gamma protection loss during sycophancy SFT to prevent collateral damage on protected bias categories (Age, Race, Religion)
- **Hybrid weight + activation control** -- SVD compression + SAE steering + rho-guided SFT in a unified pipeline. SAE at layer 17 cuts bias collateral damage 39% while preserving full sycophancy improvement (+39pp)
- **Biology-grounded bias probes** -- 37 new probes based on peer-reviewed science (twin studies, GWAS, FBOE, chromosomal/hormonal dimorphism)
- **Per-category disaggregation** -- All bias metrics broken down by category with automated risk classification

**Earlier findings (v2.1.1 -- [paper](https://github.com/SolomonB14D3/knowledge-fidelity/releases/tag/v2.1.1)):**

- **SFT toxicity inversion** -- Standard fine-tuning inverts toxicity discrimination (5 seeds, p<0.001); contrastive confidence loss repairs it (d=10.8 toxicity, d=13.7 bias, p<0.0001)
- **Variance collapse** -- Factual sigma drops 63% from SFT-only to rho-guided, making training more reliable across seeds
- **Refusal buffer** -- Contrastive-only training erodes refusal (d=-8.4, p=0.0005); the SFT component preserves it
- **MLX-native training** -- Full alignment pipeline runs on Apple Silicon. 7B model trains in ~90 min on M3 Ultra

---

### Published Repos

| Repo | What it does |
|------|-------------|
| [knowledge-fidelity](https://github.com/SolomonB14D3/knowledge-fidelity) | Behavioral auditing + alignment toolkit for LLMs. **rho-eval** on [PyPI](https://pypi.org/project/rho-eval/). [DOI: 10.5281/zenodo.18743959](https://doi.org/10.5281/zenodo.18743959). 1,826 probes, 8 behavioral dimensions, rho-guided SFT, Rho-Surgery, MLX-accelerated training. Featured in [Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression#tools). |
| [confidence-cartography](https://github.com/SolomonB14D3/confidence-cartography) | Teacher-forced confidence as a false-belief sensor. Human false-belief correlation rho=0.652 across Pythia 160M-12B. |
| [intelligent-svd](https://github.com/SolomonB14D3/intelligent-svd) | Knowledge-preserving SVD compression. CF90 method: TruthfulQA +5%, 75% fact retention. |

### Research Roadmap

1. ~~General-purpose behavioral diagnostic toolkit~~ -- **Done** (rho-eval v2.0.0)
2. ~~Mechanistic interpretability of behavioral subspaces~~ -- **Done** (SVD subspace analysis, Grassmann angles, Universal Kill Zone discovery)
3. ~~Rho-guided alignment / fine-tuning~~ -- **Done** (v2.1.1 -- [paper](https://github.com/SolomonB14D3/knowledge-fidelity/releases/tag/v2.1.1), SFT inversion discovery, contrastive repair, dose-response)
4. ~~Hybrid weight + activation control framework~~ -- **Done** (v2.2.0 -- SVD + SAE + rho SFT unified pipeline, 7B sweep, SAE mitigates 39% bias collateral)
5. **Rho-Surgery: targeted category-aware alignment** -- *in progress* (gamma protection loss, SurgicalPlan, MLX pipeline, GPU validation running)
6. Open behavioral benchmark suite (Fidelity-Bench 2.0)
