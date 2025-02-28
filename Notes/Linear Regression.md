# Linear Regression

## Feature Scaling

Feature scaling is a crucial preprocessing step in linear regression, particularly when the ranges of features differ significantly. It ensures that the model performs optimally and converges efficiently.

### Importance of Feature Scaling

1. **Gradient Descent Convergence**:
   - When features have different scales, the cost function becomes elongated, causing gradient descent to converge slowly.
   - Scaling features ensures that the cost function is more symmetrical, leading to faster convergence.

2. **Model Performance**:
   - Features with larger ranges can dominate the model, leading to biased results.
   - Scaling ensures that all features contribute equally to the model.

3. **Regularization**:
   - If regularization (e.g., L1 or L2) is applied, feature scaling ensures that the penalty is applied uniformly across all features.

### Common Methods for Feature Scaling

1. **Normalization (Min-Max Scaling)**:
   - Rescales features to a fixed range, typically [0, 1].
   - Formula: X_scaled = (X - X_min) / (X_max - X_min)```

2. **Standardization (Z-score Normalization)**:
   - Rescales features to have a mean of 0 and a standard deviation of 1.
   - Formula: ```X_scaled = (X - μ) / σ```
     - μ: Mean of the feature
     - σ: Standard deviation of the feature

3. **Mean Normalization**:
   - Rescales features to have a mean of 0 and a range typically between [-1, 1].
   - Formula: ```X_scaled = (X - μ) / (X_max - X_min)```

4. **Robust Scaling**:
   - Uses the median and interquartile range (IQR) to scale features, making it robust to outliers.
   - Formula: ```X_scaled = (X - median) / IQR```
     - IQR: Interquartile range

### When to Use Feature Scaling

- **Gradient Descent Optimization**: Always scale features when using gradient descent for optimization.
- **Distance-Based Algorithms**: Scaling is crucial for algorithms that rely on distances, such as k-nearest neighbors (k-NN) or support vector machines (SVM).
- **Regularization**: Scaling is important when using regularization to ensure fair penalization of all features.