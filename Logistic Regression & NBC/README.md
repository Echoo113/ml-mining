# 🚀 **Logistic Regression & Naive Bayes Classifier - HW2** 🎯

## 📌 **Overview**
This repository contains my implementation of a **Logistic Regression classifier** and a **Naive Bayes classifier** as part of my university coursework. The goal of this assignment is to implement these fundamental machine learning algorithms and apply them to a dataset.

## 📂 **Files & Implementation**
### 🔧 **Edited Files (To Be Submitted)**
- `lr.py` – Implements the **Logistic Regression classifier**.
- `nbc.py` – Implements the **Naive Bayes classifier**.

### 📁 **Provided Files (No Edits Needed)**
- `utils.py` – Contains helper functions for dataset loading and processing.
- `train.pkl` – The dataset used for training the models.

---

## 📊 **Logistic Regression Implementation**
Logistic Regression is a supervised learning algorithm used for binary classification. It predicts probabilities using the **sigmoid function**:
  
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

where \( z = Xw \) and \( w \) is the weight vector.

### 🛠 **Key Implementations**
1️⃣ **Constructor** – Initializes attributes `self.w`, `self.X`, and `self.y` as `None`.  
2️⃣ **Sigmoid Function** – Computes the probability for logistic regression using the formula above.  
3️⃣ **Initialize Weights** – Sets initial model weights as an array of ones.  
4️⃣ **Compute Gradient** – Uses the gradient of the loss function:  

$$
\nabla_w L(w) = \frac{1}{m} X^T (\sigma(Xw) - y)
$$

5️⃣ **Fit Function** – Trains the model using **Gradient Descent** with update rule:

$$
w_{i+1} = w_i - \alpha \nabla_w L(w_i)
$$

where \( \alpha \) is the **learning rate**.  
6️⃣ **Predict Function** – Predicts labels based on probabilities:

$$
\hat{y} =
\begin{cases} 
1, & P(y=1|x) = \sigma(Xw) \geq 0.5 \\
0, & \text{otherwise}
\end{cases}
$$

7️⃣ **Accuracy Function** – Computes model accuracy:

$$
\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total predictions}}
$$

✅ **Expected Performance**: Training accuracy **≥ 0.70**, Validation accuracy **≥ 0.65**.

---

## 📊 **Naive Bayes Classifier Implementation**
Naive Bayes is a **probabilistic classifier** based on Bayes' Theorem:

$$
P(y|X) = \frac{P(X|y) P(y)}{P(X)}
$$

where:  
- \( P(y) \) is the **prior probability**  
- \( P(X|y) \) is the **likelihood**  
- \( P(X) \) is the **evidence**  
- \( P(y|X) \) is the **posterior probability**

### 🛠 **Key Implementations**
1️⃣ **Constructor** – Initializes attributes including `self.alpha` for Laplace smoothing.  
2️⃣ **Prior Probability Computation** – Uses:

$$
P(y=c) = \frac{\text{Number of samples with label } c}{\text{Total samples}}
$$

3️⃣ **Feature Probability Computation** – Uses Laplace smoothing:

$$
P(x_j | y=c) = \frac{\text{Count}(x_j, y=c) + \alpha}{\text{Total count for } y=c + \alpha d}
$$

where \( d \) is the number of possible feature values.  
4️⃣ **Fit Function** – Stores dataset and computes prior & feature probabilities.  
5️⃣ **Predict Probabilities** – Computes class probabilities for test data.  
6️⃣ **Predict Function** – Assigns class label with **highest probability**.  
7️⃣ **Evaluation Function** – Computes losses:

$$
\text{Zero-One Loss} = \frac{\text{Number of incorrect predictions}}{\text{Total predictions}}
$$

$$
\text{Squared Loss} = \frac{1}{m} \sum_{i=1}^{m} (1 - p_i)^2
$$

✅ **Expected Performance**: Zero-one loss **≤ 0.3**, Squared loss **≤ 0.25**.

---

## 📦 **Required Packages**
The following packages are required:
- ✅ `numpy` (Required)
- 🆗 `tqdm` (Optional, for progress bar)

To install:

```bash
pip install numpy tqdm
