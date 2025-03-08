# 🧠 KNN & Decision Tree Classifier Implementation

Welcome to my implementation of **K-Nearest Neighbors (KNN) and Decision Tree classifiers**! 🚀
This repository contains the solutions for HW1, where I implemented these fundamental machine learning algorithms from scratch using Python. 🐍💡

## 📂 Project Structure
Here's what you'll find in this repository:

```
📦 hw1
 ┣ 📜 knn.py        # KNN classifier implementation 🏃‍♂️
 ┣ 📜 scorer.py     # Scoring functions for decision trees 🎯
 ┣ 📜 dt.py         # Decision tree classifier implementation 🌳
 ┣ 📜 utils.py      # Helper functions (no modifications needed) 🔧
 ┗ 📜 README.md     # You are here! 📖
```

## 📖 Overview
This assignment focuses on implementing:
✅ **K-Nearest Neighbors (KNN)** using Minkowski distance (p=3) 📏
✅ **Decision Tree Classifier** with various splitting criteria 🌳
✅ **Scoring Functions** (Entropy, Gini, and Chi-square gain) 📊

## 🚀 Implementation Details
### 🔹 KNN Classifier
- Implements a **non-parametric classifier** that predicts based on neighbors.
- Uses **Minkowski distance** (p=3) as the metric.
- Efficiently retrieves **top-K nearest neighbors** using `np.argsort`.
- Predicts labels using **majority voting** (`np.bincount` & `np.argmax`).

#### 📏 Minkowski Distance Formula
The Minkowski distance of order \( p \) between two points \( x \) and \( y \) in an \( n \)-dimensional space is given by:
$$
D(x, y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}}
$$
For this assignment, we use \( p = 3 \).

### 🔹 Scorer Functions
- **Entropy (Information Score)**: Measures uncertainty in class labels.

  $$
  H(S) = - \sum_{i=1}^{c} p_i \log_2 p_i
  $$
  where \( p_i \) is the probability of class \( i \).

- **Gini Score**: Measures impurity in a dataset.
  $$
  G(S) = 1 - \sum_{i=1}^{c} p_i^2
  $$

- **Chi-square Gain**: Uses contingency tables to evaluate feature splits.
  $$
  \chi^2 = \sum \frac{(O - E)^2}{E}
  $$
  where \( O \) is the observed frequency and \( E \) is the expected frequency.

- Implements **subset selection, information gain, and Gini gain calculations**.

### 🔹 Decision Tree Classifier
- Uses **recursive partitioning** to build a tree from training data.
- Splits nodes based on **maximum information gain**.
- Supports **class probability predictions** at leaf nodes.
- Implements **pruning using max depth** to prevent overfitting.

## 📦 Dependencies
Before running the code, install the required libraries:
```bash
pip install numpy pandas scipy
```

## 🎯 Usage
Run the classifiers using the provided datasets:
```python
# KNN Example 🏃‍♂️
k = KNearestNeighbor(k=5)
k.fit(X_train, y_train)
predictions = k.predict(X_test)
```

```python
# Decision Tree Example 🌳
dt = DecisionTree(max_depth=5)
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
```

## 🏆 Why This Matters
These classifiers are foundational in machine learning and serve as building blocks for more complex models! Understanding their implementation helps in **optimizing real-world decision-making**. 🧠💡

---
Made with ❤️ by **[Your Name]** | 🚀 Happy Coding!
