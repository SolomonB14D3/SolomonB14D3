# Bryan — Independent ML Researcher

Building behavioral auditing and mechanistic interpretability tools for LLMs. Current focus: mapping the **comparative anatomy of behavioral representations** — where factual knowledge, sycophancy, and bias live inside transformer layers, how they interact, and whether steering vectors transfer across architectures.

**Core thesis:** Social compliance and social awareness share representational capacity at mid-depth transformer layers. Factual representations are architecturally universal; sycophancy suppression is not. Activation steering is architecture-contingent — each model family needs its own behavioral map.

---

### Current Work

**[knowledge-fidelity](https://github.com/SolomonB14D3/knowledge-fidelity)** (v1.1.0) — Compress LLMs while auditing what they still know. SVD compression + behavioral auditing + activation steering in one toolkit. Featured in [Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression#tools).

```bash
pip install knowledge-fidelity
rho-audit Qwen/Qwen2.5-7B-Instruct --behaviors all
```

**v1.1.0 highlights — Comparative Anatomy of Behavioral Representations:**

- **Behavioral decoupling at Layer 17**: Sycophancy resistance 3.4x (0.120 to 0.413) at the cost of bias detection (slope = -1.37). Social compliance and social awareness share representational capacity.
- **Sycophancy suppression is architecture-contingent**: The Qwen L17 sweet spot does not exist in Mistral — no layer at any depth achieves meaningful sycophancy improvement. "Alignment Kill Zone" at Mistral L14-L18 destroys bias without sycophancy benefit.
- **Factual steering transfers universally**: +0.152 on Qwen, +0.117 on Mistral, both at ~75% depth. This is an architectural universal, not training-specific.
- **SVD compression as denoising**: Mandela rho 0.829 to 0.943 on Qwen-7B at 70% rank.

**Comparative anatomy table (Qwen vs Mistral):**

| Property | Qwen2.5-7B | Mistral-7B |
|----------|:----------:|:----------:|
| Factual sweet spot | L24 (86%), +0.152 | L24 (75%), +0.117 |
| Sycophancy sweet spot | L17 (61%), +0.293 | *None* (+0.013 max) |
| Kill zone | L17 (bias: -0.437) | L14-L18 (bias: -0.460) |
| Factual transfer | Yes | Yes |
| Sycophancy transfer | — | No |

---

### Published Repos

| Repo | What it does |
|------|-------------|
| [knowledge-fidelity](https://github.com/SolomonB14D3/knowledge-fidelity) | Compress + audit LLMs with shared factual probes. Multi-vector steering, cross-model validation, `rho-audit` CLI. Featured in [Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression#tools). |
| [confidence-cartography](https://github.com/SolomonB14D3/confidence-cartography) | Teacher-forced confidence as a false-belief sensor. Human false-belief correlation rho=0.652 across Pythia 160M-12B. |
| [intelligent-svd](https://github.com/SolomonB14D3/intelligent-svd) | Knowledge-preserving SVD compression. CF90 method: TruthfulQA +5%, 75% fact retention. |

### What's Next

- Orthogonal steering methods to break the sycophancy-bias coupling
- Third architecture validation (Llama-3.1-8B)
- Training-time behavioral disentanglement
