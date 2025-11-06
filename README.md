# Attention-Driven Pseudo-Label Self-Training for Weakly Supervised Video Anomaly Detection

This repository contains the official implementation of the paper:

**"Attention-Driven Pseudo-Label Self-Training for Weakly Supervised Video Anomaly Detection"**

---

## 📌 Overview

Recently, two-stage self-training methods based on generating pseudo-labels for weakly supervised video anomaly detection (WSVAD) have achieved notable progress. However, the generated pseudo-labels often suffer from incompleteness and noise, which hampers further performance gains.

To address these challenges, we propose a novel **dual-branch framework** that synchronizes pseudo-label generation and self-training via attention mechanisms:

- **Branch 1:** Incorporates a *Video Snippet Separation and Fusion (VSSF)* module using self-attention and cross-attention to enhance anomaly distinction, followed by an *Attention-Driven Pseudo-Label Generation (PLG)* module with denoising strategies.
- **Branch 2:** Implements a *Multi-Scale Temporal Feature Interaction Module* and leverages updated pseudo-labels from the first branch to improve snippet-level anomaly discrimination.

Extensive experiments on three benchmark datasets (UCF-Crime, XD-Violence, ShanghaiTech) demonstrate superior performance over state-of-the-art methods.

---

## 📁 Datasets

Please download the datasets from the following links:

I3D Features:
- **[UCF-Crime](https://github.com/Roc-Ng/DeepMIL)**
- **[XD-Violence](https://roc-ng.github.io/XD-Violence/)**
- **[ShanghaiTech](https://drive.google.com/file/d/1kIv502RxQnMer-8HB7zrU_GU7CNPNNDv/view)**

CLIP Features:
- **[UCF-Crime](https://github.com/nwpu-zxr/VadCLIP)**
- **[XD-Violence](https://github.com/nwpu-zxr/VadCLIP)**

---

## 📦 Models Weights

You can download our model weights for each dataset from the following links:

- **UCF-Crime:** [Google Drive](https://drive.google.com/drive/folders/1Vy65rw3wd5IFGnQ553I19dJIOz7JDq9G?usp=drive_link)
- **XD-Violence:** [Google Drive](https://drive.google.com/drive/folders/12u5p5yDsonvdAWEsWA5O5OF2xV08_E99?usp=drive_link)
- **ShanghaiTech:** [Google Drive](https://drive.google.com/drive/folders/1Bp4f-xpsSZd1amxkR9-daQdTRDluS2v4?usp=drive_link)

---

## 🚀 Training and Testing

Ensure dataset paths and configs are correctly set before running.

### ▶️ UCF-Crime

**Train:**
```bash
python ucf_main.py --thre_pesudo_a 0.25 --thre_var_a 0.4 --pl_his_num 3 --warm_up 1000
```

**Test:**
```bash
python ucf_test.py --model_file <path_to_trained_model>
```

---

### ▶️ XD-Violence

**Train:**
```bash
python xd_main.py --thre_pesudo_a 0.15 --thre_var_a 0.5 --pl_his_num 5 --warm_up 1500
```

**Test:**
```bash
python xd_test.py --model_file <path_to_trained_model>
```

---

### ▶️ ShanghaiTech

**Train:**
```bash
python sht_main.py --thre_pesudo_a 0.2 --thre_var_a 0.4 --pl_his_num 4 --warm_up 1000
```

**Test:**
```bash
python sht_test.py --model_file <path_to_trained_model>
```

---

## 🔗 Reference Repositories

Our implementation referenced the following codebases:

- [UR-DMU](https://github.com/henrryzh1/UR-DMU)
- [XDVioDet](https://github.com/Roc-Ng/XDVioDet)
- [VadCLIP](https://github.com/nwpu-zxr/VadCLIP)

---

