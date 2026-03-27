# Bond-Quantity-Aware-Transformer : Integrating Continuous Physical Knowledge for Chemical Retrosynthesis Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3.0](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C.svg)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-Chem%20Informatics-green.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[//]: # (A deep learning framework for molecular property prediction using graph neural networks and transformer architectures. This repository provides tools for training custom models on molecular datasets and performing inference with beam search decoding.)

## 📋 Table of Contents

[//]: # (- [Features]&#40;#features&#41;)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)

[//]: # (- [Usage]&#40;#usage&#41;)
[//]: # (  - [Training]&#40;#training&#41;)
[//]: # (  - [Inference]&#40;#inference&#41;)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)

[//]: # (- [Configuration]&#40;#configuration&#41;)
[//]: # (- [Results]&#40;#results&#41;)
[//]: # (- [Contributing]&#40;#contributing&#41;)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

[//]: # (## ✨ Features)

[//]: # ()
[//]: # (- **State-of-the-art Architectures**: Graph Neural Networks &#40;GNN&#41; and Transformer-based models for molecular representation learning)

[//]: # (- **Beam Search Decoding**: Advanced inference mechanism for generating multiple candidate predictions)

[//]: # (- **RDKit Integration**: Full compatibility with RDKit for molecular featurization and validation)

[//]: # (- **Flexible Configuration**: YAML-based configuration system for easy hyperparameter tuning)

[//]: # (- **Checkpoint Management**: Automatic saving and resuming of training checkpoints)

[//]: # (- **GPU Acceleration**: Optimized for CUDA-enabled devices with mixed precision training support)

## 🔧 Installation

### Prerequisites

- Python &gt;= 3.10
- CUDA 11.8 or higher (for GPU support)
- Git

### Step-by-Step Installation

1. **Clone the repository**

```bash
git clone https://github.com/maxwell-cheng123456/BQAT.git
cd BQAT
```

2. **Create a virtual environment (recommended)**
using conda

```bash
conda create -n BQAT python=3.10
conda activate BQAT
```
3. **Install PyTorch 2.3.0**
```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```
4. **Install RDKit**
```bash
pip install rdkit
```
5. **Verify Installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import rdkit; print(f'RDKit: {rdkit.__version__}')"
```

## 🚀 Quickstart
1. **Prepare Your Data**
Place your dataset in the data/ directory. Expected format:
Training data: data/src-train.txt,data/tgt-train.txt\
Validation data: data/src-val.txt, data/tgt-val.txt\
Test data: data/src-test.txt, data/tgt-test.txt

2. **Train a Model**
```bash
python train.py
```
3. **Run Inference**
```bash
python new_beam_search.py
```

## 📁 Project Structure
BQAT/
├── data/                      # Dataset directory\
├── experiment/               # Saved model checkpoints\
├── main/                       # Source code\
│   ├── calculate_topk_reverse.py/ # calculate the result of inference\
│   ├── decoder.py/            # decoder block of transformer\
│   ├── encoder.py/            # encoder block of transformer\
│   ├── locate_atom.py/        # bond energy and distance knowledge of BQAT\
│   ├── my_model.py/           # model class of BQAT\
│   ├── new_beam_search.py/    # Inference script with beam search\
│   ├── position.py/           # position encoding blocks\
│   ├── prior_encoder.py/      # encoder block fusion with prior knowledge\
│   ├── requirements.txt/      # python dependency\
│   ├── smiles_graph.py/       # smiles graph building\
│   ├── tokenizer.py/          # smiles tokenizer\
│   └── train.py/              # Training loops and evaluation\
├── vocab/                     # vocab Source\
│   ├── show_vocab.py/         # print vocabulary\
│   ├── vocab_with_error.pt/   # vocabulary file\
│   └── vocab_with_error.py/   # build vocabulary file\
└── README.md                  # This file


## 📊 Dataset and Pre-trained Models

1. **Dataset**
The dataset and pre-trained models are hosted on Hugging Face Datasets.\
Download Links:
Dataset: [Download here]()\
Pre-trained Models: [Download here]()


## 🏗️ Model Architecture
Our model combines:\
Encoder: BQAT encoder or Transformer encoder\
Attention Mechanism: Multi-head self-attention for capturing long-range dependencies\
Decoder: Beam search decoder for generating property predictions or molecular optimization suggestions\
Output Layer: Task-specific heads for regression or classification\
Key Hyperparameters:\
Embedding dimension: 512\
Hidden dimension: 2048\
Number of layers: 6\
Attention heads: 8


## 📚 Citation
If you use this code in your research, please cite:

@software{BQAT,
  author = {Xiaobo Cheng},
  title = {Bond-Quantity-Aware-Transformer : Integrating Continuous Physical Knowledge for Chemical Retrosynthesis Prediction},
  url = {https://github.com/maxwell-cheng123456/BQAT},
  year = {2024},
}

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact
Issues: Please use GitHub Issues\
Discussions: Join our GitHub Discussions\
Email: maxwell_cheng@163.com

