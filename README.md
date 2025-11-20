

## 📌 Introduction

This is the official PyTorch implementation of the paper **""** (NeurIPS 2025).

**Abstract**

## 🛠️ Installation

This repository uses **Conda** to manage dependencies and CUDA versions, ensuring exact reproducibility.

### 1\. Clone the repositorybash

git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
cd your-repo-name

````

### 2. Create the environment
We provide an `environment.yml` file that locks all dependencies (PyTorch, CUDA, Hydra, Lightning).

```bash
# Create the environment named 'paper-env'
conda env create -f environment.yml

# Activate the environment
conda activate paper-env
````

## 📂 Data Preparation

Due to license/size constraints, the dataset is not included in this repository.

1.  Download the raw data from.
2.  Extract the data into `data/raw/`.
3.  Run the preprocessing script to generate standard tensors:

<!-- end list -->

```bash
# This will clean, split, and normalize the data into data/processed/
python src/data/make_dataset.py
```

**Expected Directory Structure:**
data/
└── processed/
├── train/
│   ├── class\_a/
│   └── class\_b/
└── val/

## 🚀 Training

You can run experiments via the command line or using the pre-configured PyCharm integration.

### Option A: Command Line (Hydra)

We use [Hydra](https://hydra.cc/) for configuration management. You can override any parameter from the command line.

```bash
# Train with default configuration
python train.py

# Train with specific hyperparameters
python train.py model.lr=1e-4 data.batch_size=64

# Run a specific experiment configuration
python train.py experiment=baseline_resnet
```

### Option B: PyCharm (One-Click Run)

This repository includes shared **Run Configurations** (located in `.idea/runConfigurations`).

1.  Open this project in **PyCharm**.
2.  Wait for indexing to finish.
3.  In the top-right toolbar, select **"Run Training"** (Green Play Button).
4.  This will automatically run `train.py` with the correct environment settings.

## 📊 Results

| Model | Accuracy | F1 Score | Weights |
| :--- | :---: | :---: | :--- |
| Baseline (ResNet18) | 92.5% | 0.91 |(link) |
| **Ours (Proposed)** | **95.2%** | **0.94** |(link) |

## ⚖️ Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@inproceedings{YourName2025Paper,
  title={Your Paper Title},
  author={Your Name and Co-authors},
  booktitle={Proceedings of the Neural Information Processing Systems (NeurIPS)},
  year={2025},
  url={[https://arxiv.org/abs/20xx.xxxxx](https://arxiv.org/abs/20xx.xxxxx)}
}
```

## 📝 License

This project is licensed under the [Apache 2.0 License](https://www.google.com/search?q=LICENSE).

```

### Key Placeholders to Update:
1.  ****: Replace with your actual title.
2.  **ArXiv Link**: Update the `arxiv.1001.2222` link in the badges section.
3.  **GitHub URL**: Update the `git clone` URL.
4.  **Citation**: Replace the BibTeX block with your actual citation info.
```
