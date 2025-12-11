
# 🧬 Breast Cancer Diagnosis Prediction Using K-Nearest Neighbors (KNN) & K-Means Clustering

This project focuses on predicting breast cancer diagnoses as **Malignant (1)** or **Benign (0)** using two machine learning techniques:

* **K-Nearest Neighbors (KNN)** → *Supervised classification*
* **K-Means Clustering** → *Unsupervised grouping*

The dataset used is from the **Breast Cancer Wisconsin Diagnostic Dataset**.

---

## 📌 **Project Objectives**

* Encode diagnosis labels (M → 1, B → 0)
* Preprocess dataset by removing unnecessary columns
* Apply Min-Max Normalization
* Train/Test split of 80/20
* Use **K-Means (k=2)** to cluster malignant vs benign samples
* Train **KNN (k=5)** classifier
* Evaluate KNN using:

  * Accuracy
  * Precision
  * Recall
  * F1-score

---

## 📂 **Dataset Description**

The dataset includes features computed from digitized images of breast tumors, such as:

* Mean radius
* Texture
* Smoothness
* Compactness
* Symmetry
* Fractal dimension

The target label:

* **M (Malignant)** → encoded to **1**
* **B (Benign)** → encoded to **0**

---

## 🛠️ **Preprocessing Steps**

### 1️⃣ Drop Unnecessary Columns

Removed:

* `id`
* `Unnamed: 32`

### 2️⃣ Encode Diagnosis

```python
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
```

### 3️⃣ Min-Max Normalization

All features except the diagnosis column were normalized.

### 4️⃣ Train-Test Split

Dataset split into:

* **80% Training**
* **20% Testing**

```python
train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

---

## 🚀 **K-Means Clustering (k = 2)**

K-Means was used to group samples into two clusters without using the diagnosis labels.

After clustering, we compared clusters with actual labels using:

```python
pd.crosstab(df['diagnosis'], df['cluster'])
```

This provides insight into how well the unsupervised algorithm separated malignant vs benign groups.

---

## 🤖 **KNN Classification (k = 5)**

A **K-Nearest Neighbors classifier** was trained on the normalized data.

```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

Predictions were made on the test set.

---

## 📊 **Model Evaluation**

The following metrics were used:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**

Example evaluation code:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
```

📌 *Your actual results should be inserted here based on your model outputs.*

---

## 📈 **Conclusion**

* K-Means clustering gives insight into the separability of the dataset without labels.
* KNN (k=5) provides reliable classification performance when trained on normalized features.
* This dataset is well-structured and shows clear distinction between malignant and benign cases, leading to high performance using simple models like KNN.

---

## 💻 **Files Included**

```
📁 breast-cancer-knn
 ├── README.md
 ├── breast_cancer_knn.ipynb
 └── Dataset.csv   (optional)
```

---

## 🔗 **Google Colab Notebook**

(Insert your Colab link here after uploading the notebook.)

---


