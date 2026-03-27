# 🤖 Machine Learning Tool

An interactive, menu-driven Python application that demonstrates core machine learning concepts across four problem types — regression, binary classification, multi-class classification, and clustering — using real-world datasets.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Datasets](#datasets)
- [ML Models Implemented](#ml-models-implemented)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Overview

This tool provides a hands-on exploration of machine learning workflows through an interactive CLI. Each menu option loads a dataset, trains multiple models, evaluates them using standard metrics, and visualizes the results — all from a single Python script.

---

## ✨ Features

-. **Dataset Explorer** — Load and inspect four datasets with statistics, quality checks, and visualizations
-  **Regression** — Predict continuous values (house prices) using three regression techniques
-  **Binary Classification** — Classify flower species using three classifiers with ROC curve analysis
-  **Multi-class Classification** — Predict wine quality across six rating levels
-  **Clustering Analysis** — Segment customers using K-Means and Hierarchical clustering
-  **Rich Visualizations** — Correlation heatmaps, scatter plots, confusion matrices, dendrograms, and more
-  **Formatted Tables** — Clean tabular output using `tabulate` with `rounded_grid` style

---

## 🗃️ Datasets

| Dataset | File | Type | Target |
|---|---|---|---|
| Boston Housing | `boston.csv` | Regression | House prices (MEDV) |
| Iris Flowers | `iris.csv` | Binary Classification | Species (Setosa vs. Non-Setosa) |
| Red Wine Quality | `redwine.csv` | Multi-class Classification | Quality rating (3–8) |
| Customer Segmentation | `customers.csv` | Clustering | No label (unsupervised) |

> ⚠️ These CSV files must be placed in the same directory as the script before running.

---

## 🧠 ML Models Implemented

### Regression (Boston Housing)
- **Linear Regression** — Baseline model with coefficient analysis
- **Polynomial Regression** (degree=2) — Captures non-linear relationships
- **Multiple Regression** — Uses `SelectKBest` to pick the top 8 features

### Binary Classification (Iris)
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors** (k=5)

### Multi-class Classification (Wine Quality)
- **Random Forest** — With feature importance ranking
- **Support Vector Machine (SVM)**
- **Gradient Boosting**

### Clustering (Customer Segmentation)
- **K-Means Clustering** (k=5, determined via Elbow Method)
- **Hierarchical Clustering** (Ward linkage, with dendrogram)
- **Silhouette Score Analysis** for cluster quality

---

## 📦 Requirements

- Python 3.10+
- The following Python packages:

```
pandas
scikit-learn
scipy
seaborn
matplotlib
tabulate
numpy
```

---

## ⚙️ Installation

1. **Clone or download** this repository.

2. **Install dependencies:**

```bash
pip install pandas scikit-learn scipy seaborn matplotlib tabulate numpy
```

3. **Place the dataset CSV files** in the same directory as `main.py`:
   - `boston.csv`
   - `iris.csv`
   - `redwine.csv`
   - `customers.csv`

---

## 🚀 Usage

Run the script from the terminal:

```bash
python main.py
```

You will see the main menu:

```
=== MACHINE LEARNING TOOL ===
1. Load and Explore Datasets
2. Regression Model (Boston Housing)
3. Binary Classification Model (Iris Flower)
4. Multi-class Classification Models (Wine Quality)
5. Clustering Analysis (Customer Segmentation)
8. Exit
```

Navigate using number inputs. Each section displays metrics, tables, and plots, and prompts you to go back to the menu when done.

---

## 📁 Project Structure

```
ml-tool/
│
├── main.py            # Main script with all ML logic and menus
├── boston.csv         # Boston Housing dataset
├── iris.csv           # Iris Flowers dataset
├── redwine.csv        # Red Wine Quality dataset
├── customers.csv      # Customer Segmentation dataset
└── README.md
```

---

## 📊 Evaluation Metrics Used

| Task | Metrics |
|---|---|
| Regression | R² Score, RMSE, MAE |
| Classification | Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC |
| Clustering | WCSS (Elbow Method), Silhouette Score |

---

## 🛠️ Built With

- [scikit-learn](https://scikit-learn.org/) — ML models and utilities
- [pandas](https://pandas.pydata.org/) — Data loading and manipulation
- [matplotlib](https://matplotlib.org/) & [seaborn](https://seaborn.pydata.org/) — Visualizations
- [scipy](https://scipy.org/) — Statistical functions and hierarchical clustering
- [tabulate](https://pypi.org/project/tabulate/) — Formatted console tables
