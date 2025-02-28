# Bagging and Boosting

## Bagging: Efficient Use of Data

Training separate models on independently sampled datasets is **very wasteful of data** because it requires multiple independent datasets, which is often impractical. Instead, **bootstrap aggregation (bagging)** provides a clever and efficient solution. Here's how it works:

---

### Why Not Train a Single Model on the Union of All Sampled Datasets?

If you combine all independently sampled datasets into one large dataset and train a single model on it:
1. **No Variance Reduction**: The single model would still have high variance, as it would be sensitive to the specific noise in the combined dataset.
2. **No Ensemble Effect**: You would lose the benefit of averaging predictions from multiple models, which is key to reducing variance.

Thus, combining datasets into one would not achieve the same effect as bagging.

---

### The Bagging Solution

Bagging addresses the problem of data wastage by using **bootstrap sampling** to create multiple datasets from a single training set \( D \). Here's how it works:

#### 1. Bootstrap Sampling
- Start with a single dataset \( D \) containing \( n \) examples.
- Generate \( m \) new datasets (called **bootstrap samples**) by sampling \( n \) examples from \( D \) **with replacement**.
  - Sampling with replacement means some examples may appear multiple times in a bootstrap sample, while others may not appear at all.
- Each bootstrap sample is approximately the same size as the original dataset \( D \).

#### 2. Train Models on Bootstrap Samples
- Train a separate model (e.g., a decision tree) on each of the \( m \) bootstrap samples.
- Since the bootstrap samples are slightly different due to random sampling, the trained models will also be slightly different.

#### 3. Average Predictions
- For a given input, compute the predictions from all \( m \) models.
- Average the predictions (for regression) or take a majority vote (for classification) to produce the final prediction.

---

### Why Bagging Works

1. **Efficient Use of Data**:
   - Instead of requiring multiple independent datasets, bagging generates multiple datasets from a single dataset \( D \) using bootstrap sampling.
   - This makes bagging much more practical and data-efficient.

2. **Variance Reduction**:
   - Each model is trained on a slightly different dataset, so their predictions will vary.
   - Averaging these predictions reduces the overall variance of the final prediction.

3. **Bias Preservation**:
   - The expectation of the averaged prediction remains the same as the expectation of a single model’s prediction, so the bias is not increased.

4. **Handling Overfitting**:
   - Bagging is particularly effective for high-variance models (e.g., decision trees), as it reduces their tendency to overfit the training data.

---

### Mathematical Intuition

- Let \( y_i \) be the prediction of the \( i \)-th model trained on the \( i \)-th bootstrap sample.
- The final prediction \( y \) is the average of the \( m \) predictions:
  ```
  y = (1/m) * sum(y_i) for i = 1 to m.
  ```
- The variance of the final prediction \( y \) is:
  ```
  Var(y) = σ^2 / m,
  ```
  where \( σ^2 \) is the variance of a single model’s prediction.
- This shows that the variance decreases as the number of models \( m \) increases.

---

### Advantages of Bagging

1. **Reduced Variance**:
   - Averaging predictions from multiple models reduces the overall variance, leading to better generalization.

2. **Improved Stability**:
   - Bagging makes the model more robust to noise and outliers in the training data.

3. **Parallelization**:
   - Since each model is trained independently, bagging can be easily parallelized, making it computationally efficient.

4. **Works Well with High-Variance Models**:
   - Bagging is particularly effective for models like decision trees, which have high variance and are prone to overfitting.

---

### Limitations of Bagging

1. **Increased Computational Cost**:
   - Training multiple models can be computationally expensive, especially for large datasets or complex models.

2. **Limited Bias Reduction**:
   - Bagging primarily reduces variance and does not address bias. If the base model is biased, the ensemble will also be biased.

3. **Less Interpretable**:
   - The final ensemble model is less interpretable than a single model, as it combines the predictions of multiple models.

---

### Example: Random Forests

A popular application of bagging is the **Random Forest** algorithm:
- It uses bagging to train multiple decision trees on bootstrap samples.
- Additionally, it introduces randomness in the feature selection process for each tree, further reducing variance and improving performance.
---
## Why All Models Have the Same Mean and Variance in Bagging

The assumption that **all models have the same variance and the same mean** is a simplification often used in theoretical analyses of ensemble methods like bagging. Let's break down why this assumption is reasonable and what it implies in practice.

---

### 1. Why All Models Have the Same Mean

The **mean** of a model's predictions refers to the expected value of the predictions over different training sets. If all models are trained on datasets drawn from the same underlying distribution, their predictions will have the same expectation (mean).

#### Reasoning:
- Each model \( y_i \) is trained on a dataset sampled from the same data-generating distribution \( p_data \).
- The true relationship between the features and the target is fixed (e.g., \( y = f(x) + epsilon \), where \( epsilon \) is noise).
- The expectation of the predictions \( E[y_i] \) is the same for all models because they are all trying to approximate the same true function \( f(x) \).

#### Mathematically:
```
E[y_i] = μ for all i,
```
where \( μ \) is the true mean of the target variable.

---

### 2. Why All Models Have the Same Variance

The **variance** of a model's predictions measures how much the predictions fluctuate around their mean. If all models are trained on datasets of the same size and from the same distribution, their variances will be the same.

#### Reasoning:
- Each model \( y_i \) is trained on a dataset of the same size, so the variability in their predictions due to sampling noise will be similar.
- The models are trained independently, so their predictions are independent random variables.
- The variance \( Var(y_i) \) is determined by the model's sensitivity to the training data, which is the same for all models if they use the same algorithm and hyperparameters.

#### Mathematically:
```
Var(y_i) = σ^2 for all i,
```
where \( σ^2 \) is the variance of a single model’s predictions.

---

### Why This Assumption is Reasonable

1. **Same Data Distribution**:
   - All models are trained on datasets sampled from the same underlying distribution \( p_data \). This ensures that the expectations and variances of their predictions are consistent.

2. **Same Model Architecture**:
   - All models use the same algorithm (e.g., decision trees) and hyperparameters, so their behavior is consistent across different training sets.

3. **Independent Training**:
   - Each model is trained independently on a different dataset, so their predictions are independent random variables with the same distribution.

4. **Central Limit Theorem (CLT)**:
   - When averaging predictions from multiple models, the CLT ensures that the variance of the averaged prediction decreases as \( σ^2 / m \), where \( m \) is the number of models.

---

### What Happens in Practice

In practice, the assumptions of **same mean** and **same variance** may not hold exactly, but they are good approximations for understanding the behavior of bagging:

1. **Mean**:
   - If the models are unbiased (or have the same bias), their predictions will cluster around the true mean \( μ \).

2. **Variance**:
   - Even if the variances are not exactly the same, averaging predictions from multiple models will still reduce the overall variance, as long as the models are not perfectly correlated.

---

### Key Takeaways

- The assumption that all models have the same mean and variance is a simplification for theoretical analysis.
- In practice, this assumption is reasonable if:
  - All models are trained on datasets from the same distribution.
  - All models use the same algorithm and hyperparameters.
  - The models are trained independently.
- Bagging reduces variance by averaging predictions from multiple models, which cancels out some of the noise in individual predictions.

---

#### Mathematical Summary

- **Mean**: \( E[y_i] = μ \) for all \( i \).
- **Variance**: \( Var(y_i) = σ^2 \) for all \( i \).
- **Averaged Prediction**: \( y = (1/m) * sum(y_i) for i = 1 to m \).
  - Expectation: \( E[y] = μ \) (bias unchanged).
  - Variance: \( Var(y) = σ^2 / m \) (variance reduced).

This is why bagging is effective at improving model performance by reducing variance without increasing bias.

---

## How Bagging Changes the Hypothesis Space and Inductive Bias

Bagging (Bootstrap Aggregating) is primarily known for reducing the **variance** of a model's predictions without significantly affecting its **bias**. However, it can also indirectly influence the **hypothesis space** and **inductive bias** of the learning algorithm. Let’s explore how this happens:

---

### **1. Hypothesis Space**
The **hypothesis space** refers to the set of all possible models (or functions) that a learning algorithm can represent. Bagging changes the effective hypothesis space in the following ways:

#### **a. Enlarging the Hypothesis Space**
- Bagging combines multiple models (e.g., decision trees) trained on different bootstrap samples. The ensemble of these models can represent a richer set of functions than a single model.
- For example, a single decision tree might have limited capacity to model complex relationships, but an ensemble of trees (like in a Random Forest) can approximate more complex functions.

#### **b. Smoothing the Hypothesis Space**
- By averaging the predictions of multiple models, bagging smooths out the predictions and reduces overfitting. This effectively changes the hypothesis space to favor smoother, more stable functions.

#### **c. Implicit Regularization**
- Bagging introduces an implicit form of regularization by averaging over multiple models. This reduces the risk of overfitting and biases the learning process toward simpler, more generalizable models.

---

### **2. Inductive Bias**
The **inductive bias** of a learning algorithm refers to the assumptions it makes to generalize beyond the training data. Bagging can influence the inductive bias in the following ways:

#### **a. Bias Toward Robustness**
- Bagging reduces the sensitivity of the model to small changes in the training data. This biases the learning process toward more robust models that are less likely to overfit.
- For example, in a Random Forest, each tree is trained on a different bootstrap sample, and the final prediction is an average of all trees. This reduces the influence of outliers and noisy data points.

#### **b. Bias Toward Simplicity**
- By averaging predictions, bagging discourages overly complex models that fit the noise in the training data. This biases the learning process toward simpler models that generalize better to unseen data.

#### **c. Bias Toward Diversity**
- Bagging encourages diversity among the individual models in the ensemble. Each model is trained on a slightly different dataset, which leads to different hypotheses. The ensemble combines these diverse hypotheses, resulting in a more balanced and generalizable model.

---

### **3. Practical Implications**
- **For High-Variance Models**: Bagging is particularly effective for high-variance models like decision trees. It reduces their tendency to overfit and biases them toward simpler, more stable predictions.
- **For Low-Bias Models**: Bagging does not significantly change the bias of low-bias models (e.g., linear models). However, it can still reduce their variance and improve generalization.

---

### **4. Example: Random Forests**
Random Forests are a classic example of how bagging changes the hypothesis space and inductive bias:
- **Hypothesis Space**: A single decision tree has a limited hypothesis space, but a Random Forest combines many trees, effectively enlarging the hypothesis space to include more complex functions.
- **Inductive Bias**: Random Forests are biased toward robust and diverse models. Each tree is trained on a different bootstrap sample, and the final prediction is an average of all trees, which reduces overfitting and improves generalization.

---

### **Summary**
- **Hypothesis Space**: Bagging enlarges and smooths the hypothesis space by combining multiple models, allowing the ensemble to represent more complex and stable functions.
- **Inductive Bias**: Bagging biases the learning process toward robustness, simplicity, and diversity, reducing overfitting and improving generalization.
- **Practical Impact**: Bagging is particularly effective for high-variance models like decision trees, as it reduces their variance and biases them toward simpler, more generalizable solutions.

By changing the hypothesis space and inductive bias, bagging improves the performance and stability of machine learning models, especially in scenarios with noisy or limited data.

---

