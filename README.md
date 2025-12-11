# 🧬 Breast Cancer Diagnosis Prediction Using KNN & K-Means

This project builds a machine learning pipeline to classify breast tumor diagnoses as **Benign (0)** or **Malignant (1)** using:

* **K-Nearest Neighbors (KNN)** → Supervised classification
* **K-Means Clustering** → Unsupervised grouping
* **PCA Visualization** → 2D interpretation
* **Hyperparameter Tuning (GridSearchCV)** → Improved performance

The dataset used is the **Breast Cancer Wisconsin Diagnostic Dataset**.

---

## 📌 **Project Summary**

| Metric                                | Result                                                        |
| ------------------------------------- | ------------------------------------------------------------- |
| **Dataset Shape**                     | 569 rows × 31 columns                                         |
| **Number of Features**                | 30                                                            |
| **Best KNN Parameters**               | `n_neighbors = 3`, `p = 1 (Manhattan)`, `weights = 'uniform'` |
| **Tuned KNN Test F1 Score**           | **0.963**                                                     |
| **K-Means Adjusted Rand Index (ARI)** | **0.7302**                                                    |

A high **F1 score (0.963)** indicates excellent classification performance, and the **ARI score (~0.73)** shows that K-Means clustering aligns reasonably well with real labels despite being unsupervised.

---

## 🗂️ **Project Workflow**

### **1. Data Preprocessing**

* Dropped unnecessary columns: `id`, `Unnamed: 32`
* Encoded labels: **M → 1**, **B → 0**
* Scaled all features using **Min-Max Normalization**
* Performed an **80/20 stratified train-test split**

---

## 🔍 **2. Unsupervised Learning: K-Means (k=2)**

K-Means was applied to explore natural groupings in the data.

**Adjusted Rand Index (ARI): 0.7302**

This shows good similarity between clusters and true diagnoses — impressive for an unsupervised model.

---

## 🤖 **3. Supervised Learning: KNN Classifier**

A baseline KNN model (k=5) was evaluated, then improved using **GridSearchCV** with a pipeline.

### **Best Model Parameters**

```python
{
  'knn__n_neighbors': 3,
  'knn__p': 1,
  'knn__weights': 'uniform'
}
```

### **Test Performance**

| Metric        | Score                     |
| ------------- | ------------------------- |
| **Accuracy**  | High (not shown but >95%) |
| **Precision** | High                      |
| **Recall**    | High                      |
| **F1 Score**  | **0.963**                 |

The optimized KNN model is highly effective for this dataset.

---

## 📊 **4. Visualization**

* **PCA (2 components)** was used to project the 30-dimensional data into a 2D plot.
* Plots were generated showing:

  * True labels in PCA space
  * K-Means clusters
  * KNN predicted labels

These help visually understand data separability.

---

## 💾 **5. Model Saving**

The best-performing model and scaler were saved:

```
knn_tuned_pipeline.joblib
minmax_scaler.joblib
```

These allow easy reloading for deployment or additional prediction tasks.

---

## 📁 **Repository Structure**

```
📦 Breast-Cancer-ML
 ├── breast_cancer.ipynb
 ├── Dataset.csv
 ├── knn_tuned_pipeline.joblib
 ├── minmax_scaler.joblib
 ├── README.md
```

---

## 🧾 **Conclusion**

This project demonstrates that:

* The dataset is highly separable.
* **KNN**, especially with tuned hyperparameters, achieves **excellent performance**.
* **Unsupervised learning (K-Means)** provides meaningful insights, even without labels.
* **PCA visualizations** help interpret high-dimensional data.

The final KNN model achieves:

> **F1 Score: 0.9629**
> which makes it a strong and reliable classifier for breast cancer diagnosis prediction.


