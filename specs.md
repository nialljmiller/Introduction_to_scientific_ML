| Resource | MLP Needs | MedBow Has | % of Need |
|:--|:--|:--|--:|
| **Total GPUs** | **0** (seriously, CPU is fine) | 32 A100s, 32 V100s, lots of CPUs | **3200%** overkill |
| **VRAM** | **0** (only needs CPU RAM) | 80–320 GB VRAM | **∞% excess** |
| **RAM** | ~100 MB | 192 GB per node | **200,000%** oversupply |
| **Disk** | ~10–100 MB for data + checkpoints | 100s of TB available | **1,000,000% excess** |
| **Network** | None required | Infiniband fabric (56–200 Gbps) | **Purely wasted** |
| **Training time** | Minutes to 1–2 hours (CPU) | MedBow can do it in seconds (GPU) | **99.9% faster than needed** |
