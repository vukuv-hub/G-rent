# G-rent 

**Decentralized P2P GPU Computing for the AI Era**

g-rent is a decentralized, peer-to-peer (P2P) GPU sharing and rental platform. It is specifically engineered to handle massive AI workloads, such as **Meta’s V-JEPA** (Video Joint-Embedding Predictive Architecture), by combining torrent-based data distribution with blockchain-powered trust.

---

## 🌟 Overview

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

## 🚀 Quick Start

### Prerequisites
* NVIDIA GPU with CUDA support
* Docker & NVIDIA Container Toolkit
* Node.js / Python 3.9+

### Installation
```bash
# Clone the repository
git clone [https://github.com/vukvu-hub/g-rent.git](https://github.com/vukuv-hub/g-rent.git)

# Install dependencies
cd g-rent
pip install -r requirements.txt

# Start the g-rent node
python main.py --mode provider --wallet YOUR_WALLET_ADDRESS
g-rent/
├── .gitignore               # Prevents tracking of sensitive/unnecessary files
├── LICENSE                  # Legal licensing (e.g., MIT or GNU GPLv3)
├── README.md                # Project overview, installation, and usage guides
├── requirements.txt         # List of Python dependencies (PyTorch, Web3, etc.)
├── package.json             # Dependencies for blockchain tools (Hardhat/Truffle)
│
├── src/                     # Main source code directory
│   ├── network/             # P2P and Torrent-based communication logic
│   │   ├── peer_node.py     # P2P node discovery and connectivity
│   │   └── torrent_core.py  # Sharding and distributed data streaming
│   ├── compute/             # Hardware management and AI execution
│   │   ├── gpu_orchestrator.py # CUDA management and GPU resource allocation
│   │   └── v_jepa_engine.py    # V-JEPA model training and inference pipelines
│   └── api/                 # Interfaces for external communication or CLI
│
├── contracts/               # Solidity smart contracts
│   ├── GPUProviderRegistry.sol # Management of available GPU nodes
│   └── RentalEscrow.sol        # Trustless payment and rental agreement logic
│
├── scripts/                 # Automation and utility scripts
│   ├── deploy.py            # Blockchain contract deployment scripts
│   └── setup_env.sh         # One-click environment setup for new nodes
│
├── tests/                   # Test suites
│   ├── test_network/        # Connectivity and peer discovery tests
│   ├── test_compute/        # GPU performance and AI workload tests
│   └── test_contracts/      # Smart contract functional and security tests
│
└── docs/                    # Technical documentation and architecture diagrams
    ├── architecture.md      # Detailed P2P and Blockchain workflow
    └── v_jepa_integration.md # Guide for running V-JEPA on g-rent
