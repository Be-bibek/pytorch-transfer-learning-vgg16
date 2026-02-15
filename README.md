# 🍎 FruitFresh-AI: Automated Quality Assessment using Deep Learning

[![NVIDIA DLI](https://img.shields.io/badge/NVIDIA-Deep%20Learning%20Institute-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/training/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Status](https://img.shields.io/badge/Assessment-Passed-success)
![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-93%25-brightgreen)
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
## 🌟 Executive Summary
This repository contains a high-performance computer vision model designed to solve a critical problem in the food supply chain: **Automated Fruit Quality Classification**. Leveraging **Transfer Learning** with the **VGG16 architecture**, I developed a system that identifies whether fruits (Apples, Bananas, and Oranges) are "Fresh" or "Rotten" with a validation accuracy of **93.31%**.

---

## 🏗️ Technical Workflow & Architecture

To understand how the model "sees," we follow this high-level pipeline:


### 1. The Core Architecture (VGG16)
I utilized the **VGG16** model, a 16-layer deep Convolutional Neural Network (CNN) pretrained on over 1 million images from the ImageNet database.

**Why VGG16?**
* **Feature Hierarchy:** The early layers detect basic features like edges and textures, while deeper layers detect complex shapes—perfect for distinguishing the subtle browning of a rotten banana from the skin of a fresh one.
* **Robustness:** Pre-trained weights provide a massive "head start," allowing the model to achieve high accuracy with a relatively small custom dataset.


### 2. Implementation Strategy: Transfer Learning
Rather than training a massive network from scratch, I applied **Transfer Learning**:
* **Freezing:** I disabled gradient updates for the convolutional base (`vgg_model.requires_grad_(False)`) to preserve the ImageNet knowledge.
* **Custom Head:** I appended a new classifier consisting of Linear (Dense) layers and ReLU activations tailored for my 6 specific classes.

### 3. Reusable Convolutional Block
```bash
class MyConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.model(x)


```
---

## 📊 Detailed Training Logic (Step-by-Step)

### I. Data Loading & Categorization
The dataset is split into 6 distinct categories:
1. `freshapples` | 2. `freshbanana` | 3. `freshoranges`
4. `rottenapples` | 5. `rottenbanana` | 6. `rottenoranges`

### II. Advanced Data Augmentation
To prevent **Overfitting** (where the model memorizes images rather than learning features), I implemented a stochastic augmentation pipeline:
* **Horizontal Flips:** To handle fruits viewed from different angles.
* **Random Rotation (10°):** To account for non-perfect alignment in real-world scenarios.
* **Color Jitter:** To make the model resilient to different lighting conditions (e.g., supermarket fluorescent lights vs. natural warehouse light).

### III. Loss & Optimization
* **Loss Function:** `nn.CrossEntropyLoss()` — Ideal for multi-class classification as it applies Softmax internally to output probabilities.
* **Optimizer:** `Adam` — An adaptive learning rate optimization algorithm that speeds up convergence.

---

## 📈 Performance Analysis

My model achieved the NVIDIA assessment requirement of **>92%** within just 10 epochs.

### Training Progress Table
| Epoch | Train Accuracy | Validation Accuracy | Train Loss |
| :--- | :--- | :--- | :--- |
| 1 | 89.76% | 90.27% | 10.58 |
| 5 | 94.08% | 93.01% | 5.85 |
| **8 (Peak)** | **95.01%** | **93.31%** | **5.19** |


---

## 🧬 Fine-Tuning (Optional Advanced Step)
Once the custom head was trained, I performed "Fine-Tuning":
1.  **Unfreeze:** `vgg_model.requires_grad_(True)`
2.  **Low LR:** Set learning rate to `0.0001`.
This allows the pre-trained VGG weights to slightly adjust to the specific textures of fruit rot, squeezing out that final 1-2% of accuracy.

---

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.8+
* NVIDIA GPU (Recommended) with CUDA installed

### Setup
```bash
# Clone the project
git clone [https://github.com/yourusername/fruitfresh-ai.git](https://github.com/yourusername/fruitfresh-ai.git)
cd fruitfresh-ai

# Install required packages
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install pillow glob2

```
---

## 🔮 Future Scope: Edge AI Integration

As an Electronics and Communication Engineering student, my goal is to bridge the gap between high-level AI and embedded hardware. The next phase of this project involves moving the model from the cloud to the **Edge**.

### 1. Deployment on Raspberry Pi 4
I plan to deploy this trained PyTorch model onto a **Raspberry Pi** environment. To ensure "fluent" performance on a small form factor:
* **Model Quantization:** Converting weights from `float32` to `INT8` to reduce memory footprint without significant accuracy loss.
* **TorchScript:** Serializing the model for high-performance execution in a non-Python environment.



### 2. Autonomous Fruit-Quality Rover
Integrating this classification system with my **Raspberry Pi Rover** project:
* **Real-time Detection:** Using a PiCamera to scan fruit crates in a warehouse.
* **Smart Sorting:** Using a robotic arm or actuator to physically separate "Rotten" items from "Fresh" ones based on the model's inference.
* **Connectivity:** Utilizing high-range Wi-Fi adapters and Li-ion power management for long-range autonomous monitoring.



### 3. Mobile Application (Flutter/React Native)
Building a mobile frontend that uses the smartphone camera to perform on-device inference, providing instant "Freshness Scores" for consumers at grocery stores.


## 👨‍💻 Author
**Bibek Das** *Third-Year B.Tech Student | ECE*

---
*This project was completed as part of the NVIDIA Deep Learning Institute (DLI) Certification.*
