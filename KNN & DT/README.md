# 🚀 **Logistic Regression & Naive Bayes Classifier - HW2** 🎯

Hi, I'm Echo. In this project, I implemented two core machine learning algorithms as part of my coursework: a **Logistic Regression classifier** and a **Naive Bayes classifier**. This assignment allowed me to apply both probabilistic modeling and optimization techniques, showcasing my ability to blend theory with practical coding.

---

## 📌 **Overview**
This repository contains my implementations for:
- **Logistic Regression** – A method for binary classification using a sigmoid function to map predictions to probabilities.
- **Naive Bayes Classifier** – A probabilistic classifier based on Bayes’ theorem that uses feature likelihoods and class priors.

I developed these models from scratch, demonstrating my strong grasp of both the underlying mathematics and their practical applications.

---

## 📂 **Files & Implementation**

### 🔧 **Files to Submit**
- `lr.py` – Contains my implementation of the **Logistic Regression classifier**.
- `nbc.py` – Contains my implementation of the **Naive Bayes classifier**.

### 📁 **Provided Resources**
- `utils.py` – Includes helper functions for data loading and preprocessing.
- `train.pkl` – The training dataset for model development.

---

## 📊 **Logistic Regression Implementation**
Logistic Regression is used for binary classification by predicting probabilities using the **sigmoid function**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

where \( z = Xw \) and \( w \) is the weight vector.

### 🛠 **Key Components in My Implementation**
- **Initialization:** I set up the weight vector and stored the training data.
- **Sigmoid Function:** I implemented the function to compute probabilities using the formula above.
- **Weight Initialization:** I started with an initial weight vector (e.g., all ones) before training.
- **Gradient Computation:** I derived the gradient of the loss function:

  $$
  \nabla_w L(w) = \frac{1}{m} X^T \left(\sigma(Xw) - y\right)
  $$

- **Gradient Descent:** I used an iterative update rule to optimize the weights:

  $$
  w_{i+1} = w_i - \alpha \nabla_w L(w_i)
  $$

  where \( \alpha \) is the learning rate.
- **Prediction:** I classified new samples based on the output probability:

  $$
  \hat{y} =
  \begin{cases} 
  1, & \text{if } \sigma(Xw) \geq 0.5 \\
  0, & \text{otherwise}
  \end{cases}
  $$

- **Accuracy Measurement:** I computed accuracy by comparing predicted labels with true labels.

---

## 📊 **Naive Bayes Classifier Implementation**
Naive Bayes uses Bayes' Theorem to compute the posterior probability for each class:

$$
P(y|X) = \frac{P(X|y) \, P(y)}{P(X)}
$$

where:
- \( P(y) \) is the prior probability.
- \( P(X|y) \) is the likelihood.
- \( P(X) \) is the evidence.

### 🛠 **Key Components in My Implementation**
- **Initialization:** I set up model parameters including a Laplace smoothing factor \( \alpha \).
- **Prior Probability:** I computed the prior probability for each class:

  $$
  P(y=c) = \frac{\text{Count}(y=c)}{\text{Total number of samples}}
  $$

- **Likelihood Estimation:** For each feature, I computed the likelihood using Laplace smoothing:

  $$
  P(x_j|y=c) = \frac{\text{Count}(x_j \text{ in class } c) + \alpha}{\text{Total count for } y=c + \alpha d}
  $$

  where \( d \) is the number of distinct feature values.
- **Model Fitting:** I stored the training data and calculated both prior and feature probabilities.
- **Prediction:** I determined the class label for each sample by selecting the one with the highest posterior probability.
- **Evaluation:** I measured performance using metrics such as Zero-One Loss and Squared Loss:

  $$
  \text{Zero-One Loss} = \frac{\text{Incorrect predictions}}{\text{Total predictions}}
  $$

  $$
  \text{Squared Loss} = \frac{1}{m} \sum_{i=1}^{m} \left(1 - p_i\right)^2
  $$

---

## 📦 **Required Packages**
The project relies on the following packages:
- ✅ `numpy` – For numerical computations.
- ✅ `tqdm` – (Optional) For displaying progress bars during training.

To install these packages, run:
```bash
pip install numpy tqdm
