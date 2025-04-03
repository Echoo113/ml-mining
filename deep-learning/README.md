# 🧠 Fashion MNIST Neural Network Playground

Welcome to the Neural Network Playground! In this project, you will implement and train neural networks from scratch and with PyTorch to classify Fashion MNIST images.

---

## 📂 Project Structure

```bash
├── data/FashionMNIST/raw                 # Dataset files  
├── numpy_ann.py                          # ANN built from scratch using only Numpy  
├── pytorch_ann.py                        # ANN built using PyTorch  
├── pytorch_cnn.py                        # CNN built using PyTorch  
├── utils.py                              # Helper functions (no edits needed)  
├── environment.yml                       # Environment setup file  
├── pytorch_ann_train_val_metrics.png     # ANN performance plots  
├── pytorch_cnn_train_val_metrics.png     # CNN performance plots  
```

---

## 🔢 Mathematical Foundations

### 🟩 ReLU Activation (1.2)

```math
\text{ReLU}(x) = 
\begin{cases}
0 & \text{if } x < 0 \\
x & \text{otherwise}
\end{cases}
```

### 🟩 ReLU Derivative (1.3)

```math
\frac{d}{dx} \text{ReLU}(x) = 
\begin{cases}
0 & \text{if } x \leq 0 \\
1 & \text{otherwise}
\end{cases}
```

---

### 🧮 Cross Entropy Loss (1.4)

Given predicted probabilities (`y_{pred}`) and one-hot true labels (`y_{true}`):

```math
\text{Loss}(y_{\text{true}}, y_{\text{pred}}) = -\frac{1}{n} \sum_{i=1}^{n} y_{\text{true}, i} \log(y_{\text{pred}, i})
```

---

### 🔁 Forward Pass (1.5)

For one hidden layer:

```math
z_1 = X W_1 + b_1 \\
a_1 = \text{ReLU}(z_1) \\
z_2 = a_1 W_2 + b_2 \\
a_2 = \text{Softmax}(z_2)
```

---

### 🔄 Backward Propagation (1.6)

#### Output Layer:

```math
\frac{\partial L}{\partial z_2} = a_2 - y_{\text{true}}
```

```math
\frac{\partial L}{\partial W_2} = a_1^\top \frac{\partial L}{\partial z_2}
```

```math
\frac{\partial L}{\partial b_2} = \sum \frac{\partial L}{\partial z_2}
```

#### Hidden Layer:

```math
\frac{\partial L}{\partial z_1} = \left( \frac{\partial L}{\partial z_2} W_2^\top \right) \odot \text{ReLU}'(z_1)
```

```math
\frac{\partial L}{\partial W_1} = X^\top \frac{\partial L}{\partial z_1}
```

```math
\frac{\partial L}{\partial b_1} = \sum \frac{\partial L}{\partial z_1}
```

---

### 🧮 Gradient Descent Update Rule

```math
\theta = \theta - \alpha \frac{\partial L}{\partial \theta}
```

Where:

```math
\alpha \text{ is the learning rate}
```

---

## 🧪 How to Run

### ▶️ Numpy ANN

Train a neural network from scratch:

```bash
python numpy_ann.py
```

Expected accuracy: **≥ 77%**

### ▶️ PyTorch ANN

```bash
python pytorch_ann.py
```

Expected accuracy: **≥ 80%**  
Submit learning curves from: `pytorch_ann_train_val_metrics.png`

### ▶️ PyTorch CNN

```bash
python pytorch_cnn.py
```

Expected accuracy: **≥ 85%**  
Submit learning curves from: `pytorch_cnn_train_val_metrics.png`

---

## 📈 Visualizations

Your training and validation curves should help identify:

- **Underfitting**: Both losses high
- **Overfitting**: Training loss low, validation loss high
- **Generalizing**: Both low and similar

---

## ✅ Submission Checklist

- [x] `numpy_ann.py`
- [x] `pytorch_ann.py`
- [x] `pytorch_cnn.py`
- [x] `pytorch_ann_train_val_metrics.png`
- [x] `pytorch_cnn_train_val_metrics.png`

---

## 💡 Notes

- Use mini-batch gradient descent.
- Avoid external packages beyond the allowed list.
- Version check:
  
```bash
python -c "import torch; print(torch.__version__)"
```

---

Happy modeling! 🎯 Your neurons are ready to fire. 🚀
