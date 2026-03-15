# Bryan — Independent ML Researcher

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-pink?logo=github)](https://github.com/sponsors/SolomonB14D3)
[![Paper: SFT](https://zenodo.org/badge/DOI/10.5281/zenodo.18854943.svg)](https://doi.org/10.5281/zenodo.18854943)
[![Paper: Grassmann](https://zenodo.org/badge/DOI/10.5281/zenodo.18865861.svg)](https://doi.org/10.5281/zenodo.18865861)
[![Paper: Phase Transitions](https://zenodo.org/badge/DOI/10.5281/zenodo.18865198.svg)](https://doi.org/10.5281/zenodo.18865198)
[![Paper: Confidence Cartography](https://zenodo.org/badge/DOI/10.5281/zenodo.18703505.svg)](https://doi.org/10.5281/zenodo.18703505)
[![Paper: Contrastive Pretraining](https://zenodo.org/badge/DOI/10.5281/zenodo.18870555.svg)](https://doi.org/10.5281/zenodo.18870555)
[![Paper: Expression Bottleneck](https://zenodo.org/badge/DOI/10.5281/zenodo.18895248.svg)](https://doi.org/10.5281/zenodo.18895248)
[![Paper: Snap-On](https://zenodo.org/badge/DOI/10.5281/zenodo.18902617.svg)](https://doi.org/10.5281/zenodo.18902617)
[![Paper: CF90](https://zenodo.org/badge/DOI/10.5281/zenodo.18718545.svg)](https://doi.org/10.5281/zenodo.18718545)
[![Paper: STEM Truth Oracle](https://zenodo.org/badge/DOI/10.5281/zenodo.19005729.svg)](https://doi.org/10.5281/zenodo.19005729)
[![Paper: Breaking Frozen Priors](https://zenodo.org/badge/DOI/10.5281/zenodo.19017290.svg)](https://doi.org/10.5281/zenodo.19017290)
[![Paper: NoetherSolve Toolkit](https://zenodo.org/badge/DOI/10.5281/zenodo.19029880.svg)](https://doi.org/10.5281/zenodo.19029880)

Building behavioral auditing and alignment tools for LLMs. [Try the demo →](https://huggingface.co/spaces/bsanch52/knowledge-fidelity-demo)

---

### Tools

**[rho-eval](https://pypi.org/project/rho-eval/)** — Drop-in behavioral audit for any LLM. Measures 8 dimensions, no internet required. Apple Silicon MLX + CUDA + CPU.

```bash
pip install rho-eval

# Audit any model
rho-eval Qwen/Qwen2.5-7B-Instruct --behaviors all

# One-command behavioral repair
rho-surgery Qwen/Qwen2.5-7B-Instruct -o ./repaired-7b/

# Diagnose expression gaps in base models
rho-unlock diagnose Qwen/Qwen2.5-1.5B --behaviors mmlu,arc,truthfulqa

# Train a modular adapter (zero knowledge damage)
snap-on train --model Qwen/Qwen2.5-1.5B --mode logit --save_dir ./adapter
```

**[noethersolve](https://pypi.org/project/noethersolve/)** — Conservation law monitoring, discovery, and scientific auditing. 20 tools across physics, genetics, and unsolved mathematics. No ML required at runtime.

```bash
pip install noethersolve

# Monitor vortex dynamics conservation laws
from noethersolve import VortexMonitor

# Audit drug interactions
from noethersolve import audit_drug_list

# Verify number theory conjectures
from noethersolve import verify_goldbach, verify_collatz
```

### Papers

1. **Rho-Guided SFT** — Post-training repair of calibration damage in LLMs. [DOI: 10.5281/zenodo.18854943](https://doi.org/10.5281/zenodo.18854943)
2. **Grassmann Geometry of Behavioral Entanglement** — Surgery compresses subspaces, doesn't rotate them. [DOI: 10.5281/zenodo.18865861](https://doi.org/10.5281/zenodo.18865861)
3. **Behavioral Phase Transitions** — Geometric scaffolding precedes behavioral emergence. [DOI: 10.5281/zenodo.18865198](https://doi.org/10.5281/zenodo.18865198)
4. **Confidence Cartography** — Teacher-forced probability as a false-belief sensor. [DOI: 10.5281/zenodo.18703505](https://doi.org/10.5281/zenodo.18703505)
5. **CF90** — Knowledge-preserving SVD compression for LLMs. [DOI: 10.5281/zenodo.18718545](https://doi.org/10.5281/zenodo.18718545)
6. **Contrastive Pretraining Teaches Format Generation, Not Behavioral Knowledge** — 5% injection breaks the behavioral wall at 7M. [DOI: 10.5281/zenodo.18870555](https://doi.org/10.5281/zenodo.18870555)
7. **Small Language Models Already Know More Than They Can Say** — The 41% universal constant and the generation bottleneck. [DOI: 10.5281/zenodo.18895248](https://doi.org/10.5281/zenodo.18895248)
8. **Snap-On Communication Modules** — Logit-space adapters that preserve base model knowledge. [DOI: 10.5281/zenodo.18902617](https://doi.org/10.5281/zenodo.18902617)
9. **STEM Truth Oracle** — Log-probability ranking reveals and corrects scale-invariant factual biases. [DOI: 10.5281/zenodo.19005729](https://doi.org/10.5281/zenodo.19005729)
10. **Breaking Frozen Priors** — Teaching LLMs to discover conservation laws from numerical simulation. Three-phase pipeline achieves Spearman rho = 0.932 physics ranking. [DOI: 10.5281/zenodo.19017290](https://doi.org/10.5281/zenodo.19017290)
11. **NoetherSolve Toolkit** — 20 conservation law monitoring, discovery, and scientific auditing tools across physics, genetics, and mathematics. 777 tests, 275 oracle-verified facts. [DOI: 10.5281/zenodo.19029880](https://doi.org/10.5281/zenodo.19029880)

### Repos

| Repo | What it does |
|------|-------------|
| [knowledge-fidelity](https://github.com/SolomonB14D3/knowledge-fidelity) | Behavioral auditing + alignment toolkit. [PyPI](https://pypi.org/project/rho-eval/). |
| [noethersolve](https://github.com/SolomonB14D3/noethersolve) | Autonomous scientific discovery: 20 tools across physics, genetics, and unsolved math. 275/275 facts taught to LLMs. [PyPI](https://pypi.org/project/noethersolve/). [Dashboard](https://solomonb14d3.github.io/noethersolve/). |
| [confidence-cartography](https://github.com/SolomonB14D3/confidence-cartography) | Teacher-forced confidence as a false-belief sensor. |
| [intelligent-svd](https://github.com/SolomonB14D3/intelligent-svd) | Knowledge-preserving SVD compression for LLMs. |
