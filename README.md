# 🚀 **MSFMamba: Multi-Scale Feature Fusion State Space Model for Multi-Source Remote Sensing Image Classification**  

### *(Accepted at IEEE TGRS 2025)*  

[![IEEE TGRS](https://img.shields.io/badge/IEEE-TGRS-blue)](https://ieeexplore.ieee.org/document/10856240)  [![arXiv](https://img.shields.io/badge/arXiv-2408.14255-b31b1b)](https://arxiv.org/abs/2408.14255) 

---

## 📌 **Introduction**

This repository contains the official implementation of our paper:  
📄 *MSFMamba: Multi-Scale Feature Fusion State Space Model for Multi-Source Remote Sensing Image Classification* *(IEEE TGRS 2025)*  

**MSFMamba** is an advanced **multi-scale feature fusion model** specifically designed for **multi-source remote sensing image classification**.
 By leveraging **state-space modeling techniques**, MSFMamba effectively captures both **spatial** and **spectral dependencies**, ensuring **high accuracy and computational efficiency**.

### 🔍 **Key Features**

🔍 **Key Features:**  
✅ Multi-Scale Feature Extraction  
✅ Cross-Modal Data Fusion  
✅ State Space Model for Efficient Representation  
✅ Enhanced Fusion for Multi-Source Remote Sensing Data  

---

## 📂 **Dataset**  

The dataset used in our experiments can be accessed from the following link:  
📥 **[Download Dataset (Google Drive)](https://drive.google.com/file/d/1iZEIAVhlt2QJb_RECp0bHFVN7C8po8ag/view?usp=sharing)**  

---

## 🛠 **Installation & Dependencies**

Before running the code, make sure you have the following dependencies installed:

```bash
pip install causal-conv1d==1.1.1
pip install mamba-ssm==1.0.1
```

---

## 🏋️‍♂️ **Usage: Training MSFMamba**

To train **MSFMamba** on the **Berlin** dataset, use the following command:

```bash
python train.py --epoch 40 --lr 1e-4 --batchsize 128 --dataset Berlin
```

### 🔧 **Training Arguments**:

- `--epoch`: Number of training epochs
- `--lr`: Learning rate
- `--batchsize`: Batch size
- `--dataset`: Dataset name

---

## 📬 **Contact**

If you have any questions, feel free to reach out to us:  
✉️ **Email:**  
📧 [gaofeng@ouc.edu.cn](mailto:gaofeng@ouc.edu.cn)  
📧 [jinxuepeng@stu.ouc.edu.cn](mailto:jinxuepeng@stu.ouc.edu.cn)  

We hope **MSFMamba** helps your research! ⭐ If you find our work useful, please consider citing our paper. Thank you! 🎉