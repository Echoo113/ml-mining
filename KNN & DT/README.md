# 🚀 **KNN & Decision Tree Classifier ** 🎯

## 📌 **Overview**
This repository contains my implementation of a **K-Nearest Neighbors (KNN) classifier** and a **Decision Tree classifier** as part of my university coursework. The goal of this assignment is to implement these fundamental machine learning algorithms and apply them to a dataset.

## 📂 **Files & Implementation**
### 🔧 **Edited Files (To Be Submitted)**
- `knn.py` – Implements the **KNN classifier**.
- `scorer.py` – Implements scoring functions for decision trees.
- `dt.py` – Implements the **Decision Tree classifier**.

### 📁 **Provided Files (No Edits Needed)**
- `utils.py` – Contains helper functions.
- Other test files used for evaluation.

---

## 🤖 **K-Nearest Neighbors (KNN) Implementation**
KNN is a non-parametric algorithm that classifies a sample based on the majority label among its `k` nearest neighbors. I implemented it using **Minkowski distance (p=3)**.

### 🛠 **Key Implementations**
1️⃣ **Constructor** – Initializes `self.k`, `self.X_train`, and `self.y_train`.
2️⃣ **Fit Function** – Stores training data for reference.
3️⃣ **Minkowski Distance** – Computes distance between test and training samples:

```math
D(x, y) = \left(\sum |x_i - y_i|^p\right)^{\frac{1}{p}}
```

4️⃣ **Find Neighbors** – Identifies `k` closest training samples.
5️⃣ **Predict Function** – Assigns the most common label among `k` neighbors.

✅ **Expected Performance**: Accurate classification with increasing `k` improving stability.

---

## 🌳 **Decision Tree Implementation**
A Decision Tree splits data based on feature values to maximize information gain or minimize impurity.

### 🛠 **Key Implementations**
1️⃣ **Scoring Functions** – Computes **Information Gain**, **Gini Impurity**, and **Chi-square Gain** for best splits.

- **Entropy** (Information Score):
```math
H(S) = -\sum p_i \log_2 p_i
```

- **Gini Impurity**:
```math
G(S) = 1 - \sum p_i^2
```

- **Information Gain**:
```math
IG(S, A) = H(S) - \sum \frac{|S_v|}{|S|} H(S_v)
```

- **Chi-square Gain**:
```math
\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}
```

2️⃣ **Subset Splitting** – Divides data based on feature values.
3️⃣ **Build Tree Function** – Recursively splits dataset until stopping conditions are met.
4️⃣ **Predict Function** – Traverses the tree to classify new samples.

✅ **Expected Performance**: Well-balanced trees that generalize to unseen data.

---

## 📦 **Required Packages**
- ✅ `numpy`
- ✅ `pandas`
- ✅ `scipy` (for chi-square test)
- 🆗 `tqdm` (optional, for progress visualization)

To install:
```bash
pip install numpy pandas scipy tqdm
```

---

## 🏆 **Evaluation & Submission**
- Run local tests:
  ```bash
  python <filename>.py
  ```
- Submit `knn.py`, `scorer.py`, `dt.py` to **Gradescope**.
- Pass public tests to validate implementation.
- Hidden test cases will determine final grade.

🚀 **Let’s classify some data!** 🔥
