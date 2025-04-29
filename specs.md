| Resource | Estimate |
|:---|:---|
| **Model Size** | ~1.8 trillion parameters (GPT-4 is roughly estimated here; exact size is undisclosed, but it's *bigger than GPT-3's 175B*) |
| **Training Dataset** | 13–23 trillion tokens (open datasets + filtered web data) |
| **VRAM Needed** | ~700–900 GB *active model memory* (just for forward/backward at typical 2048-4096 sequence lengths) |
| **Total Compute** | ~25,000,000 GPU-hours |
| **Cluster Size** | 10,000+ A100s (80 GB) **for months** |
| **Training Time** | 2–3 months continuous 24/7 training |
| **Cost** (if renting) | ~$50–$100 million USD (assuming $2-$3/hour per A100) |
| **Storage** | 1–2 PB (petabytes) for dataset storage, checkpoints, logs |
| **CPU RAM** | Cluster-wide memory in **petabytes** (you need fast memory access for parallel data loading) |
| **Interconnect** | **Infiniband HDR** or better (200 Gbps+ network fabric) |
