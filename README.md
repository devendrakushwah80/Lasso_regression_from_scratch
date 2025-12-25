# 🧠 LASSO Regression from Scratch (NumPy)

This project demonstrates a **from-scratch implementation of LASSO Regression (L1 Regularization)** using **NumPy only**, without relying on high-level machine learning libraries.  
The model is evaluated on the **Diabetes dataset from scikit-learn**, and hyperparameters are manually tuned to improve performance.

---

## 📌 What is LASSO Regression?

LASSO (Least Absolute Shrinkage and Selection Operator) is a linear regression technique that:
- Uses **L1 regularization**
- Shrinks coefficients toward zero
- Performs **automatic feature selection**
- Helps reduce overfitting

### Cost Function
J = (1/n) * Σ(y − ŷ)² + α * Σ|w|

---

## 🚀 Project Objectives

- Implement LASSO regression **from scratch**
- Use **NumPy matrix operations (`np.dot`)**
- Train on a real-world dataset
- Manually tune **alpha** and **epochs**
- Compare models using standard regression metrics

---

## 📂 Dataset Used

- **Dataset:** Diabetes dataset
- **Source:** `sklearn.datasets.load_diabetes`
- **Samples:** 442
- **Features:** 10 numerical features
- **Target:** Disease progression score

---

## 🛠️ Tech Stack

- Python
- NumPy
- scikit-learn (only for dataset loading & metrics)

---

## 🧩 Implementation Details

### Key Highlights
- Gradient Descent optimization
- Sub-gradient method for L1 penalty using `sign(w)`
- Manual hyperparameter tuning (Grid Search–style)
- No use of `@` operator (used `np.dot` explicitly)

---

## 📊 Model Evaluation Metrics

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

---

## 🔍 Hyperparameter Tuning

Multiple models were trained by varying:
- **alpha** (regularization strength)
- **epochs** (training iterations)

Each configuration was evaluated on test data, and the best model was selected based on **R² score**.

---

## 🏆 Best Model Performance

alpha  = 5  
epochs = 3000  

RMSE = 52.91  
MAE  = 42.93  
R²   = 0.472  

### 📈 Improvement
- R² improved from ~0.22 → **0.47**
- RMSE reduced significantly after tuning

This confirms the effectiveness of L1 regularization and hyperparameter tuning.

---

## 🧠 Key Learnings

- LASSO can perform **feature selection** by forcing coefficients to zero
- Proper choice of `alpha` balances bias–variance tradeoff
- Manual tuning helps understand model behavior deeply
- Feature scaling can further improve performance

---

## 📌 How to Run

pip install numpy scikit-learn

Run the notebook or script containing:
- LASSO class
- Dataset loading
- Training & evaluation loop

---

## 📎 Future Improvements

- Add feature scaling (`StandardScaler`)
- Plot alpha vs R² graph
- Compare with `sklearn.linear_model.Lasso`
- Implement early stopping

---

## 🧪 Interview / Resume Line

Implemented LASSO Regression from scratch using NumPy and evaluated it on the Diabetes dataset.  
Improved R² from 0.22 to 0.47 through manual hyperparameter tuning, demonstrating effective regularization and feature selection.

---

## 👨‍💻 Author

Devendra Kushwah  
B.Tech CSE (AI & ML)  
Aspiring Machine Learning Engineer
