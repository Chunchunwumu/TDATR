
# TDATR: Improving End-to-End Table Recognition via Table Detail-Aware Learning and Cell-Level Visual Alignment

[](https://www.google.com/search?q=%23) [](https://www.google.com/search?q=%23)

Official PyTorch implementation and pre-trained weights for the paper **"TDATR: Improving End-to-End Table Recognition via Table Detail-Aware Learning and Cell-Level Visual Alignment"**.

## 📖 Introduction

Tables are pervasive in diverse documents, making table recognition (TR) a fundamental task in document analysis. To address the issues of suboptimal integration in modular pipelines and poor generalization in data-constrained end-to-end models, we propose **TDATR**. TDATR improves end-to-end TR through table detail-aware learning and cell-level visual alignment.

## ✨ Key Features

  * **"Perceive-then-Fuse" Strategy:** This approach reduces reliance on large-scale labeled TR data and simplifies the end-to-end sequence modeling of TR.
  * **Table Detail-Aware Learning:** This unifies structure understanding and content recognition through a set of pretraining tasks under a language modeling paradigm, enabling effective utilization of diverse document data to enhance model robustness.
  * **Structure-Guided Cell Localization:** This module refines cell boxes via structure priors and multi-level visual features, enhancing visual alignment and TR accuracy.
  * **Robust Generalization:** The model was evaluated on seven public benchmarks without dataset-specific fine-tuning, demonstrating strong performance and robustness across diverse table styles and scenarios.

## 🏆 Performance

TDATR achieves state-of-the-art or highly competitive performance on seven benchmarks without dataset-specific fine-tuning. These include diverse and challenging datasets such as TabRecSet, iFLYTAB-full, PubTabNet, and PubTables-1M.

## 📦 Model Zoo
📥 Click and download the model weights.

## 🚀 Quick Start

### 1. Requirements & Installation
Our testing environment is based on **CUDA 11.3** and **python 3.7**. We highly recommend using this version or a compatible setup to ensure reproducibility.

```bash
git clone https://github.com/yourusername/TDATR.git
cd TDATR
conda create -n tdatr python==3.7
pip install -r requirements.txt
```

### 2\. Inference

You can run inference on a single table image to generate HTML structure and cell coordinates.

The inference process consists of the following 4 stages:

**Step 1: Download Model Weights**

Download the pre-trained model weights and place them into your project directory.

**Step 2: Prepare the Image List**

Record the list of image paths you need to infer into a JSON file. 
```json
[
  "/path/to/image1.jpg",
  "/path/to/image2.jpg"
]
```
**Step 3: Configure infer.sh**

Configure your infer.sh script. This includes specifying parameters such as **ROOT_DIR**, **TEST_FILE**, and **DEVICE_ID**.

**Step 4: Run Inference.**

Activate the tdatr environment and run the infer.sh script:

```Bash
conda activate tdatr
bash infer.sh
```

## 📝 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{qin2026tdatr,
  title={TDATR: Improving End-to-End Table Recognition via Table Detail-Aware Learning and Cell-Level Visual Alignment},
  author={Qin, Chunxia and Liu, Chenyu and Xia, Pengcheng and Du, Jun and Yin, Baocai and Yin, Bing and Liu, Cong},
  booktitle={CVPR},
  year={2026}
}
```
