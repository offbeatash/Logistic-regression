# Logistic Regression from Scratch (Gradient Descent)

This project implements **Logistic Regression from scratch using NumPy**, without using any machine learning libraries for the model.

The objective is to understand how **binary classification models work internally**, including probability estimation, optimization, and decision boundaries.

---

## Problem

Predict whether a passenger survived the Titanic disaster.

Target:

```
Survived:
0 → Did not survive
1 → Survived
```

---

## Dataset

**Titanic Dataset (Kaggle)**

Features used:

* `Pclass` → passenger class
* `Age` → age of passenger
* `Sex` → gender (encoded)
* `Fare` → ticket price

---

## Model

Logistic Regression models probability:

```
p = σ(Xw + b)
```

Where:

* `σ(z)` = sigmoid function
* `w` = weights
* `b` = bias

---

## Sigmoid Function

```
σ(z) = 1 / (1 + e^(-z))
```

Transforms output into probability range:

```
(-∞, +∞) → (0, 1)
```

---

## Loss Function (Log Loss)

```
Loss = - (1/n) * Σ [y log(p) + (1-y) log(1-p)]
```

* Penalizes incorrect predictions
* Used for optimization

---

## Gradient Descent

Parameters are updated iteratively:

```
dw = (1/n) * Xᵀ(p - y)
db = (1/n) * Σ(p - y)
```

Update rule:

```
w = w - lr * dw
b = b - lr * db
```

---

## Regularization (L2)

To prevent overfitting:

```
dw += (λ/n) * w
```

* Penalizes large weights
* Improves generalization

---

## Project Structure

```
logistic_regression/
│
├── notebooks/
│   └── logistic_regression.ipynb
│
├── src/
│   └── logistic_regression.py
│
├── data/
│   └── titanic/
│       └── train.csv
│
└── README.md
```

---

## Training Pipeline

1. Load dataset
2. Select relevant features
3. Encode categorical variables
4. Handle missing values
5. Normalize features
6. Train using gradient descent
7. Predict probabilities
8. Convert to class labels (threshold = 0.5)
9. Evaluate accuracy

---

## Results

* Accuracy:

```
≈ 0.80
```

* Model successfully separates survival classes.

---

## Visualization

* Decision boundary plotted for 2D feature space
* Shows how model divides classes

---

## Key Concepts Demonstrated

* Logistic Regression
* Sigmoid Function
* Log Loss
* Gradient Descent
* Feature Scaling
* Binary Classification
* Decision Boundary
* Regularization (L2)

---

## Important Note

* Logistic Regression is implemented entirely from scratch
* scikit-learn is used **only for train-test splitting**

---

## Why this Project Matters

This project demonstrates:

* Strong understanding of classification algorithms
* Ability to implement ML models from first principles
* Debugging and numerical stability handling
* Clean ML project structure

---

## Author

Ashish Pise