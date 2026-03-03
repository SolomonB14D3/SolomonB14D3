# Bryan — Independent ML Researcher

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-pink?logo=github)](https://github.com/sponsors/SolomonB14D3)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18743959.svg)](https://doi.org/10.5281/zenodo.18743959)

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

### Repos

| Repo | What it does | Paper |
|------|-------------|-------|
| [knowledge-fidelity](https://github.com/SolomonB14D3/knowledge-fidelity) | Behavioral auditing + alignment toolkit. [PyPI](https://pypi.org/project/rho-eval/). | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18743959.svg)](https://doi.org/10.5281/zenodo.18743959) |
| [confidence-cartography](https://github.com/SolomonB14D3/confidence-cartography) | Teacher-forced confidence as a false-belief sensor. | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18718611.svg)](https://doi.org/10.5281/zenodo.18718611) |
| [intelligent-svd](https://github.com/SolomonB14D3/intelligent-svd) | Knowledge-preserving SVD compression for LLMs. | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18718545.svg)](https://doi.org/10.5281/zenodo.18718545) |
