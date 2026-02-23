# Bryan — Independent ML Researcher

Building behavioral auditing and mechanistic interpretability tools for LLMs. Current focus: **general-purpose behavioral diagnostics** — measuring what models know, where they're biased, and when they're sycophantic, using teacher-forced confidence probes and activation steering.

**Core thesis:** Social compliance and social awareness share representational capacity at mid-depth transformer layers. Factual representations are architecturally universal; sycophancy suppression is not. Activation steering is architecture-contingent — each model family needs its own behavioral map.

---

### Current Work

**[rho-eval](https://pypi.org/project/rho-eval/)** (v2.0.0) — Drop-in behavioral audit for any LLM. 806 probes across 5 dimensions, no internet required. Plugin architecture for custom behaviors.

```bash
pip install rho-eval
rho-eval Qwen/Qwen2.5-7B-Instruct --behaviors all --format table
```

```python
import rho_eval

report = rho_eval.audit("Qwen/Qwen2.5-7B-Instruct", behaviors="all")
print(report.overall_status)  # PASS / WARN / FAIL
```

**v2.0.0 highlights:**

- **Plugin architecture** — `ABCBehavior` base class with `@register` decorator. Add custom behaviors in 30 lines.
- **806 pre-sampled probes** — 5 dimensions (factual, toxicity, bias, sycophancy, reasoning), shipped as JSON, zero network dependencies.
- **Standardized output** — `AuditReport` with PASS/WARN/FAIL thresholds (rho >= 0.5 / >= 0.2 / < 0.2), JSON/Markdown/CSV/table export.
- **Comparison system** — `rho_eval.compare(after, before)` with IMPROVED/DEGRADED delta tables. CI-friendly exit codes.
- **Backward compatible** — all v1.x imports and `rho-audit` CLI still work.

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
| [knowledge-fidelity](https://github.com/SolomonB14D3/knowledge-fidelity) | Behavioral auditing + SVD compression toolkit for LLMs. Now **rho-eval** on PyPI. Plugin architecture, 806 probes, activation steering, cross-model validation. Featured in [Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression#tools). |
| [confidence-cartography](https://github.com/SolomonB14D3/confidence-cartography) | Teacher-forced confidence as a false-belief sensor. Human false-belief correlation rho=0.652 across Pythia 160M-12B. |
| [intelligent-svd](https://github.com/SolomonB14D3/intelligent-svd) | Knowledge-preserving SVD compression. CF90 method: TruthfulQA +5%, 75% fact retention. |

### Research Roadmap

1. ~~General-purpose behavioral diagnostic toolkit~~ — **Done** (rho-eval v2.0.0)
2. **Mechanistic interpretability of behavioral subspaces** — *next*
3. Rho-guided alignment / fine-tuning
4. Hybrid weight + activation control framework
5. Open behavioral benchmark suite (Fidelity-Bench 2.0)
