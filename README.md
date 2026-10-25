# 💻 SmolLM-Light: Highly Optimized Local GQA Transformer Engine

A modular, clean-architecture language model implemented from scratch in **PyTorch**, inspired by the Hugging Face **SmolLM** and Meta **Llama 3** architectures. This repository is specifically calibrated and optimized to train and run inference efficiently on consumer hardware (such as a laptop CPU with 8GB/16GB RAM) without requiring a dedicated NVIDIA GPU.

---

## 📁 Repository Directory Structure

The project strictly follows a **Clean Code Architecture Pattern**, enforcing the single-responsibility principle by decoupling data processing, tokenization, model layers, optimization loops, and front-end interface mechanics.

```text
smol-lm-light/
│
├── smol_lm/
│   ├── __init__.py
│   ├── tokenizer.py      # Production Byte-Pair Encoding (BPE) Subword Interface
│   ├── dataset.py        # Vectorized Vector-Chunking (Non-Overlapping Blocks)
│   ├── trainer.py        # Abstract Gradient Step Optimizer & Pipeline Validator
│   └── model/            # Unified Transformer Neural Network Package
│       ├── __init__.py     # Package API routing and encapsulation masking
│       ├── norm.py         # Root Mean Square Layer Normalization (RMSNorm)
│       ├── embedding.py    # Rotary Position Embeddings (RoPE) Cache Array
│       ├── attention.py    # Grouped-Query Attention (GQA) & Causal Masking
│       ├── feedforward.py  # Swish Gated Linear Unit (SwiGLU) Layer Block
│       └── transformer.py  # Layer Block Stacking & Autoregressive Decoder
│
├── input.txt             # Rich Python Syntax & Concept Technical Corpus
├── train.py              # Central Pipeline Execution & Checkpoint Orchestrator
├── generate.py           # Advanced Terminal Top-P & Repetition Penalty Inference
└── app.py                # Streamlit Web UI Dashboard Application Portal
```

---

## 🧠 Under the Hood: Mathematical Architectures

To bypass standard laptop memory bandwidth restrictions, this codebase utilizes the exact three structural math layers found in modern state-of-the-art models like Llama 3 and Mistral.

### 1. Root Mean Square Normalization (RMSNorm)

Instead of traditional LayerNorm which calculates both the mean ($\mu$) and variance ($\sigma^2$) across hidden activations, RMSNorm completely discards the mean-centering operation. It regulates inputs based solely on their root mean square magnitude, accelerating processing time on the CPU.

$$\text{RMSNorm}(x) = \left( \frac{x}{\text{RMS}(x)} \right) \odot g$$

$$\text{where } \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}$$

### 2. Grouped-Query Attention (GQA)

Standard Multi-Head Attention maintains an identical count of Query ($Q$), Key ($K$), and Value ($V$) heads ($N_q = N_{kv}$). GQA scales this down by grouping multiple Query heads together to share a single localized Key and Value head pair. This cuts down the memory size of the Key-Value Cache by a factor of $g$.

$$\text{Grouping Ratio Layout Factor: } g = \frac{N_{\text{queries}}}{N_{kv}}$$

$$\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d_{\text{head}}}} + \text{Mask} \right) V$$

### 3. Rotary Position Embeddings (RoPE) at Axis 1

Instead of adding absolute positional indices to token vectors, RoPE partitions the features into 2D planes and applies an index-dependent vector rotation. To guarantee broadcasting shape compliance and completely eliminate text loops during generation, the rotation is executed natively while the sequence timeline ($T$) resides explicitly at Axis 1.

$$R_m \begin{bmatrix} x_0 \\ x_1 \end{bmatrix} = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \begin{bmatrix} x_0 \\ x_1 \end{bmatrix} \quad \text{where } \theta_i = 10000^{-\frac{2(i-1)}{d}}$$

---

## ⚙️ Hyperparameter Configuration Scaling Profile

The updated specifications expand the contextual memory window by **4x**, extending historical sentence memory retention across full technical documentation entries.

| Hyperparameter | Value     | Description                                                               |
| :------------- | :-------- | :------------------------------------------------------------------------ |
| `MAX_SEQ_LEN`  | **128**   | Active attention context window timeline threshold capacity               |
| `D_MODEL`      | **192**   | Internal structural coordinate depth dimension size                       |
| `NUM_LAYERS`   | **4**     | Number of stacked transformer block layers                                |
| `NUM_HEADS`    | **6**     | Distinct Query attention head partitions ($N_q$)                          |
| `NUM_KV_HEADS` | **2**     | Key/Value head partitions ($N_{kv}$), enforcing a **3:1 GQA group ratio** |
| `D_FF`         | **512**   | SwiGLU internal intermediate matrix representation expansion size         |
| `BATCH_SIZE`   | **4**     | Mini-batch sequence row chunk bounds to preserve system RAM limits        |
| `EPOCHS`       | **200**   | Total training cycles to converge cross-entropy loss down to ~0.01        |
| `DEVICE`       | **"cpu"** | Targeted local system computing execution profile                         |

---

## 🚀 Execution Guide

### 1. Installation and Virtual Environment Setup

Ensure you have Python 3.10+ installed. Initialize your environment workspace and update your package manifests:

```bash
# Create and activate the virtual environment
python -m venv venv
venv\Scripts\activate

# Install core packages optimized for CPU execution
pip install torch --index-url https://pytorch.org
pip install transformers streamlit
```

### 2. Run the Training Engine Pipeline

Bake the expanded 128-token context framework parameters directly into your network weight matrix:

```bash
python train.py
```

_The optimized `TextDataset` groups text using non-overlapping blocks, forcing structural convergence in minutes directly on your laptop processor._

### 3. Launch the Advanced Generation CLI Tool

Execute an advanced inference sampling pass using Top-P (Nucleus) and Repetition Penalty from your command line terminal:

```bash
python generate.py
```

### 4. Fire Up the Interactive Web UI Dashboard App

Launch the production web user interface server workspace while passing headless flags to suppress internal vision model file-watcher warnings from the `transformers` library:

```bash
streamlit run app.py --server.fileWatcherType none --client.showErrorDetails false
```

Once started, open your web browser and navigate directly to:
👉 **`http://localhost:8501`**

---

## 🎯 Fine-Tuning Inference Controls for Coherent Sentences

When interacting with your model via the Streamlit interface dashboard, use the sidebar sliders to set the optimal parameters for deterministic, highly accurate factual output:

- **Temperature (Creativity Scale):** Set to **`0.30` or `0.40`**. Low values flatten out overfitted probability spikes, forcing the network to select only high-confidence sentences.
- **Repetition Penalty:** Set to **`1.15` or `1.20`**. This provides a light penalty that stops token stuttering without restricting essential bridging grammar (like `is`, `and`, `the`).
- **Top-P (Nucleus):** Set to **`0.90`** to cut off the rare token tail distribution safely.
