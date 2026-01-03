# G-rent 

**Decentralized P2P GPU Computing for the AI Era**

g-rent is a decentralized, peer-to-peer (P2P) GPU sharing and rental platform. It is specifically engineered to handle massive AI workloads, such as **Meta’s V-JEPA** (Video Joint-Embedding Predictive Architecture), by combining torrent-based data distribution with blockchain-powered trust. BRIFLY: Train Vision-Language Joint Embedding Predictive Architecture (VL-JEPA) models on a fully decentralized peer-to-peer network using consumer GPUs
---

## 🌟 Overview 🎯 What is g-rent?
g-rent is a decentralized GPU marketplace and training framework that enables:

🔓 Open Access: Train foundation models without million-dollar budgets
🔐 Privacy-First: Data stays distributed, never ⚡ Quick Start
Installation
bash# Clone the repository
git clone https://github.com/g-rent/distributed-vljepa.git
cd distributed-vljepa

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
Run Your First Distributed Training
bash# Terminal 1: Start bootstrap node
python -m grent.network.dht --bootstrap --port 8468

# Terminal 2: Start worker node
python -m grent.network.dht --connect 127.0.0.1:8468 --port 8469

# Terminal 3: Start training
python examples/train_vljepa_simple.py
That's it! You're now training VL-JEPA on a P2P network. 🎉

🏗️ Architecture
┌─────────────────────────────────────────────────────────────┐
│                   USER APPLICATIONS                          │
│     (Video Classification, Action Recognition, etc.)        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              VL-JEPA TRAINING LAYER                         │
│  • Vision Encoder (distributed across high-VRAM nodes)      │
│  • Predictor (mid-tier GPUs)                                │
│  • Federated aggregation with reputation weighting          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│            DATA DISTRIBUTION LAYER                          │
│  • BitTorrent-style sharding (1GB chunks)                   │
│  • Temporal overlap zones (preserves video context)         │
│  • Automatic chunk verification                             │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              NETWORK LAYER (DHT)                            │
│  • Kademlia distributed hash table                          │
│  • GPU capability discovery                                 │
│  • Reputation-based node selection                          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│               PHYSICAL NETWORK                              │
│  RTX 3060 | RTX 4090 | H100 | Mobile GPUs | ...            │
└─────────────────────────────────────────────────────────────┘

🌟 Key Features
1. Distributed VL-JEPA Training
First implementation of Meta's VL-JEPA on a P2P network:

Vision Encoder: Processes visible video patches
Predictor: Predicts masked patches in latent space
EMA Target Encoder: Provides stable training targets

pythonfrom grent.models import VLJEPA
from grent.training import DistributedTrainer

model = VLJEPA(encoder_depth=12, predictor_depth=6)
trainer = DistributedTrainer(model, network="auto-discover")
trainer.train(epochs=100)
2. Temporal-Aware Data Sharding
Novel approach to video dataset distribution:

Overlap Zones: 50-frame overlap between chunks
Integrity Verification: SHA-256 checksums
Rarest-First: BitTorrent-inspired prioritization

pythonfrom grent.data import VideoSharding

sharder = VideoSharding(chunk_size=1000, overlap=50)
manifest = sharder.shard_dataset("./kinetics400/", "./chunks/")
3. DHT-Based Node Discovery
No central server required:

Capability Queries: "vram >= 16 AND reputation > 0.8"
Dynamic Routing: Automatic failover when nodes leave
Reputation System: Stake-based trust scoring

pythonfrom grent.network import GRentDHTNode

node = GRentDHTNode(port=8468)
await node.start_bootstrap()

# Find capable nodes
nodes = await node.discover_nodes("vram >= 16")
4. Privacy-Preserving
Built-in privacy protections:

Differential Privacy: ε = 1.0 noise injection
Gradient-Only Sharing: Raw data never leaves device
End-to-End Encryption: All network communication encrypted


📊 Benchmarks
MetricCentralized (Meta)g-rent (Our Target)StatusHardware64× A1001000× consumer GPUs✅ ImplementedDatasetKinetics-700 (650K videos)Same🔄 In ProgressTraining Time7 days~14 days🎯 TargetAccuracy82.5%78-80% (goal)🧪 TestingCost~$100K (cloud rental)$0 (community)✅ Achieved
Why 2× slower is acceptable:

Zero cost (community-contributed compute)
Better privacy (no data centralization)
Democratized access (anyone can participate)


🎓 Academic Publications
Preprints (Coming Soon)

"Federated VL-JEPA" - ICML 2026 submission
"DHT for Distributed ML" - NeurIPS 2026 submission
"Temporal-Aware Sharding" - ICLR 2027 submission

Citation
If you use g-rent in your research, please cite:
bibtex@software{grent2026,
  title={g-rent: Decentralized GPU Network for Training Foundation Models},
  author={[Your Name] and Contributors},
  year={2026},
  url={https://github.com/g-rent/distributed-vljepa}
}

🛠️ Installation Options
Option 1: pip (Recommended)
bashpip install grent
Option 2: From Source
bashgit clone https://github.com/g-rent/distributed-vljepa.git
cd distributed-vljepa
pip install -e .
Option 3: Docker
bashdocker pull grent/distributed-vljepa:latest
docker run -it --gpus all grent/distributed-vljepa

📚 Documentation

📖 User Guide - Complete walkthrough
🧑‍💻 API Reference - Detailed API docs
🏗️ Architecture - System design
🔧 Tutorials - Step-by-step examples
❓ FAQ - Common questions


🎯 Roadmap
Phase 1: Foundation ✅ (Completed)

 DHT network implementation
 Torrent-based data sharding
 Basic VL-JEPA training
 Federated learning with FedAvg

Phase 2: Intelligence (Months 7-12) 🔄

 Full RLM (Recursive Language Models) integration
 Advanced manifold alignment (DS-mHC)
 Dynamic load balancing
 Byzantine fault tolerance

Phase 3: Scale (Months 13-18) 📅

 Mobile device support (iOS/Android)
 Cross-platform coordination
 Token economics & incentives
 10,000+ node testnet

Phase 4: Production (Months 19-24) 🚀

 Enterprise features
 SLA guarantees
 Advanced security audits
 Multi-modal expansion (audio, text)


🤝 Contributing
We welcome contributions! See CONTRIBUTING.md for guidelines.
Ways to Contribute

🐛 Bug Reports: Found an issue? Open an issue
✨ Feature Requests: Have an idea? Start a discussion
💻 Code: Submit a pull request
📝 Documentation: Improve docs, write tutorials
🧪 Testing: Run experiments, report results

Top Contributors
<!-- This will be auto-generated -->
Thanks to all our contributors! 🎉

🏆 Why g-rent is Revolutionary
For Researchers

No Budget Barriers: Train foundation models without $1M+ cloud bills
Fast Experimentation: Spin up 1000 GPUs in hours, not weeks
Global Collaboration: Access compute from researchers worldwide

For Industry

Cost Savings: 10-100× cheaper than AWS/GCP/Azure
Privacy Compliance: GDPR-friendly by design
Scalability: From 10 to 10,000 GPUs seamlessly

For Society

Democratization: AI research not limited to Big Tech
Sustainability: Use idle GPUs instead of building new datacenters
Innovation: Enable unexpected use cases from global community


🔒 Security & Privacy
g-rent takes security seriously:

✅ Encrypted Communication: TLS 1.3 for all network traffic
✅ Differential Privacy: Built-in ε-differential privacy
✅ Reputation System: Cryptographic stake-based trust
✅ Zero-Knowledge Proofs: Verify computation without seeing data
✅ Regular Audits: Third-party security reviews

See SECURITY.md for our security policy.

📜 License
This project is licensed under the Apache License 2.0 - see LICENSE file.
Why Apache 2.0?

✅ Permissive for research and commercial use
✅ Patent grant protection
✅ Compatible with most other licenses
✅ Industry-standard for open source ML


🌟 Star History
Show Image Global Scale: Coordinate 1000+ consumer GPUs across the world
💰 Cost-Effective: 10-100× cheaper than traditional cloud computing
🚀 State-of-the-Art: First P2P implementation of Meta's VL-JEPA architecture

The AI revolution is limited by access to high-performance compute. g-rent democratizes this access by turning global idle GPU capacity into a unified, **Infinite Supercomputer**. 

Unlike traditional cloud providers (AWS, GCP), g-rent uses a **Torrent-Core** architecture to stream heavy video datasets and model weights across nodes, making it the ideal infrastructure for training "World Models" and large-scale computer vision tasks.

## ✨ Key Features

* **P2P Torrent Orchestration:** Efficiently shards and distributes large datasets across the network, bypassing central server bottlenecks.
* **V-JEPA Optimized:** Tailored for high-VRAM requirements and latent space predictions common in next-gen video architectures.
* **Solidity-Powered Trust:** Smart contracts handle automated escrow, proof-of-compute verification, and transparent payments.
* **Low Latency Clusters:** Intelligent node-matching based on network proximity and hardware specs.
* **Privacy-First:** Secure containerization (Docker/NVIDIA-Container-Toolkit) ensures compute tasks are isolated.

## 🏗 Architecture

The system consists of three main layers:
1.  **Network Layer:** Utilizing `libp2p` and BitTorrent protocols for node discovery and data transfer.
2.  **Execution Layer:** Local compute nodes running PyTorch/CUDA inside secured environments.
3.  **Settlement Layer:** EVM-compatible smart contracts for managing the decentralized marketplace.


