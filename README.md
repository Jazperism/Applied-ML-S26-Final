# Applied-ML-S26-Final

# Credit Card Fraud & Telco Churn Analysis
### CSCI 164 Final Project | Comparative Classification Analysis
**Author:** Jazper Malone  
**Institution:** California State University, Fresno

## Project Overview
This project evaluates and compares the performance of several supervised learning models across two distinct classification challenges: **Telco Customer Churn** and **Credit Card Fraud Detection**. The goal is to analyze how different mathematical approaches (Discriminative vs. Generative) handle issues like class imbalance and PCA-masked feature spaces.

## Datasets
1.  **Telco Customer Churn:** A mixed-type dataset with 7,043 entries focused on predicting customer retention using demographic and service-related features.
2.  **Credit Card Fraud Detection:** A highly imbalanced dataset (0.17% fraud) consisting of PCA-transformed features ($V1$-$V28$), $Time$, and $Amount$.

## Implemented Models
### 1. Telco Customer Churn (Mixed Data)
For this dataset, I compared two distinct approaches to handle consumer behavior data:
* **Logistic Regression:** Used to establish a linear baseline, identifying global patterns in features like contract type and monthly charges.
* **k-Nearest Neighbors (kNN):** A non-parametric, distance-based algorithm selected to capture local patterns and non-linear relationships by looking at the neighborhood of similar customers.

### 2. Credit Card Fraud Detection (Imbalanced/PCA)
Three algorithms were utilized to navigate the mathematical wall of a 0.17% fraud rate:
* **Logistic Regression (Weighted):** Tuned with `class_weight='balanced'` to act as a high-sensitivity "wide net" for catching minority fraud cases.
* **Linear Discriminant Analysis (LDA):** Selected for its mathematical robustness in PCA-transformed spaces. Since the features are centered and scaled, LDA's Gaussian assumptions make it an ironic specialist for this task.
* **Gaussian Naive Bayes (GNB):** A generative baseline that leverages the underlying distribution of the masked features to provide a stable precision-recall balance.

## Key Findings: Credit Card Fraud
* **The Accuracy Paradox:** While all models achieved near-perfect accuracy, the confusion matrices reveal that Logistic Regression achieved a superior **0.918 recall** at the cost of a precision crash to **0.06**.
* **Surgical Precision:** LDA emerged as the strongest overall performer with a mean F1-score of **0.814**, proving that generative models are a natural fit for the "geometry" of PCA-masked data.
* **Operational Trade-offs:** The project establishes that for fraud detection, the choice between Logistic Regression and LDA depends on whether the priority is catching every possible fraud case (Recall) or minimizing operational friction from false alarms (Precision).

## Technical Stack
* **Language:** Python
* **Environment:** Jupyter Notebook / Google Colab
* **Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Techniques:** Hyperparameter Tuning (`GridSearchCV`), Cross-Validation, Preprocessing (StandardScaler, OneHotEncoding), and Model Evaluation (ROC-AUC, Precision-Recall Curves).

## How to Use
1. Clone the repository.
2. Ensure `scikit-learn`, `pandas`, and `matplotlib` are installed.
3. Open `Applied_ML_S26_Final_Notebook.ipynb` to view the full pipeline, from data preprocessing to the final comparative analysis.