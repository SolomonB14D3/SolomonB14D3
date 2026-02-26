# Bryan -- Independent ML Researcher

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-pink?logo=github)](https://github.com/sponsors/SolomonB14D3)

Building behavioral auditing and alignment tools for LLMs. Current focus: **rho-guided alignment** -- using teacher-forced confidence probes to steer fine-tuning away from behavioral damage.

**Core finding:** Standard SFT inverts toxicity discrimination in Qwen 7B (+0.145 baseline to -0.003 post-SFT, 5 seeds, p<0.001). Adding a contrastive confidence loss during training repairs this (d=10.8 on toxicity, d=13.7 on bias, p<0.0001). Contrastive-only training erodes refusal (d=-8.4) but the full method preserves it. The margin gamma=0.1 is structurally necessary.

---

### Current Work

**[rho-eval](https://pypi.org/project/rho-eval/)** (v2.1.1) -- Drop-in behavioral audit for any LLM. 926 probes across 6 dimensions (factual, toxicity, sycophancy, bias, reasoning, refusal), no internet required. Plugin architecture for custom behaviors. **Apple Silicon MLX acceleration. Rho-guided SFT alignment.**

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

**v2.1.1 -- Rho-Guided SFT ([paper](https://github.com/SolomonB14D3/knowledge-fidelity/releases/tag/v2.1.1)):**

- **SFT toxicity inversion** -- Standard fine-tuning inverts toxicity discrimination (5 seeds, p<0.001); contrastive confidence loss repairs it (d=10.8 toxicity, d=13.7 bias, p<0.0001)
- **Variance collapse** -- Factual sigma drops 63% from SFT-only to rho-guided, making training more reliable across seeds
- **Refusal buffer** -- Contrastive-only training erodes refusal (d=-8.4, p=0.0005); the SFT component preserves it. Never train contrastive-only if refusal matters
- **Margin necessity** -- gamma=0 causes bias to go negative; gamma=0.1 prevents over-optimization past the natural separation boundary
- **OOD transfer** -- In-distribution contrastive training transfers to unseen clinical, social, and logic domains (+5pp accuracy)
- **MLX-native training** -- Full alignment pipeline runs on Apple Silicon. 7B model trains in ~10 min per condition on M3 Ultra

**Earlier findings -- Comparative Anatomy of Behavioral Representations:**

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
| [knowledge-fidelity](https://github.com/SolomonB14D3/knowledge-fidelity) | Behavioral auditing + alignment toolkit for LLMs. **rho-eval** on [PyPI](https://pypi.org/project/rho-eval/). [DOI: 10.5281/zenodo.18743959](https://doi.org/10.5281/zenodo.18743959). 926 probes, 6 behavioral dimensions, rho-guided SFT, MLX-accelerated training. Featured in [Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression#tools). |
| [confidence-cartography](https://github.com/SolomonB14D3/confidence-cartography) | Teacher-forced confidence as a false-belief sensor. Human false-belief correlation rho=0.652 across Pythia 160M-12B. |
| [intelligent-svd](https://github.com/SolomonB14D3/intelligent-svd) | Knowledge-preserving SVD compression. CF90 method: TruthfulQA +5%, 75% fact retention. |

### Research Roadmap

1. ~~General-purpose behavioral diagnostic toolkit~~ -- **Done** (rho-eval v2.0.0)
2. ~~Mechanistic interpretability of behavioral subspaces~~ -- **Done** (SVD subspace analysis, Grassmann angles, Universal Kill Zone discovery)
3. ~~Rho-guided alignment / fine-tuning~~ -- **Done** (v2.1.1 -- [paper](https://github.com/SolomonB14D3/knowledge-fidelity/releases/tag/v2.1.1), SFT inversion discovery, contrastive repair, dose-response across Qwen + Llama)
4. **Cross-architecture and scale validation** -- *in progress* (5-seed ablation complete, refusal buffer discovered, safety stress test done, 70B planned)
5. Hybrid weight + activation control framework
6. Open behavioral benchmark suite (Fidelity-Bench 2.0)
