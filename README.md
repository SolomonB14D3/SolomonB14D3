# Bryan — Independent ML Researcher

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-pink?logo=github)](https://github.com/sponsors/SolomonB14D3)
[![Paper: SFT](https://zenodo.org/badge/DOI/10.5281/zenodo.18854944.svg)](https://doi.org/10.5281/zenodo.18854944)
[![Paper: Grassmann](https://zenodo.org/badge/DOI/10.5281/zenodo.18865862.svg)](https://doi.org/10.5281/zenodo.18865862)
[![Paper: Phase Transitions](https://zenodo.org/badge/DOI/10.5281/zenodo.18865199.svg)](https://doi.org/10.5281/zenodo.18865199)

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
```

### Papers

1. **Rho-Guided SFT** — Post-training repair of calibration damage in LLMs. [DOI: 10.5281/zenodo.18854944](https://doi.org/10.5281/zenodo.18854944)
2. **Grassmann Geometry of Behavioral Entanglement** — Surgery compresses subspaces, doesn't rotate them. [DOI: 10.5281/zenodo.18865862](https://doi.org/10.5281/zenodo.18865862)
3. **Behavioral Phase Transitions** — Geometric scaffolding precedes behavioral emergence. [DOI: 10.5281/zenodo.18865199](https://doi.org/10.5281/zenodo.18865199)

### Repos

| Repo | What it does |
|------|-------------|
| [knowledge-fidelity](https://github.com/SolomonB14D3/knowledge-fidelity) | Behavioral auditing + alignment toolkit. [PyPI](https://pypi.org/project/rho-eval/). |
| [confidence-cartography](https://github.com/SolomonB14D3/confidence-cartography) | Teacher-forced confidence as a false-belief sensor. |
| [intelligent-svd](https://github.com/SolomonB14D3/intelligent-svd) | Knowledge-preserving SVD compression for LLMs. |
