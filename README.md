# 🧠 MNIST Handwritten Digit Classification with PyTorch

This project demonstrates how to train a neural network to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) using PyTorch.


## 📊 Dataset

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) contains:

* **60,000 training images**

  * split into **50,000 for training**, **10,000 for validation**
* **10,000 test images**
  Each image is **28×28 grayscale** and represents digits **0–9**.

---

## 📁 Project Structure

```
├── MNIST.ipynb
├── models/
├── images/
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mnist-classifier.git
cd mnist-classifier
```

### 2. Install Dependencies

Make sure you have Python 3.7+ and install the required libraries:

```bash
pip install torch torchvision matplotlib scikit-learn tqdm
```

### 3. Run the Notebook

```bash
jupyter notebook MNIST.ipynb
```

---

## 📊 Training Results

| Metric        | 🧮 Linear Model | 🧠 CNN Model           |
| ------------- | --------------- | ---------------------- |
| Final Val Acc | 0.9678          | 0.9846                 |
| Final Val F1  | 0.9676          | 0.9846                 |
| Test Accuracy | 0.9691          | 0.9891                 |
| Test F1 Score | 0.9690          | 0.9891                 |

> ✅ Model checkpoints saved in the `models/` directory.
> 🖼️ Confusion matrix saved in the `images/` folder.

---

## 🧠 Model Architectures

The notebook includes the following architectures:

* 🔹 **Linear Model** using `nn.Linear`
* 🔹 **Convolutional Neural Network (CNN)** using `nn.Conv2d`

---

## ✨ Features

* MNIST digit classification using PyTorch
* Custom training and validation loops
* Live progress bar via `tqdm`
* Evaluation with F1 Score and Accuracy
* Confusion matrix visualization
* GPU acceleration support

---

## 📉 Confusion Matrix

<p align="center"> <img src="images/Linearmodel.png" alt="Confusion Matrix - Linear Model" width="45%"/> <img src="images/CNNmodel.png" alt="Confusion Matrix - CNN Model" width="45%"/> </p> <p align="center"> <b>Linear Model</b>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<b>CNN Model</b> </p>
