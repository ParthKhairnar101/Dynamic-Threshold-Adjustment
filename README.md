# Credit Scoring with LightGBM, MCMC & Bayesian Optimization

This project focuses on building a robust **credit scoring system** using the **Light Gradient Boosting Machine (LightGBM)** model. The system is enhanced with **hyperparameter tuning using a hybrid approach** that combines **Markov Chain Monte Carlo (MCMC)** via the Metropolis-Hastings algorithm and **Bayesian Optimization** using `hyperopt`.

---

## 🚀 Project Features

- **Baseline Model**: LightGBM classifier trained with K-Fold cross-validation.
- **Bayesian Optimization**: Performed using `hyperopt` to tune hyperparameters like `learning_rate`, `reg_alpha`, and `reg_lambda`.
- **MCMC Sampling**: Utilizes Metropolis-Hastings to generate posterior distributions of hyperparameters based on priors.
- **Polynomial Features**: Created using selected continuous variables (`EXT_SOURCE_x`, `DAYS_BIRTH`) to enrich the dataset.
- **Evaluation Metric**: ROC-AUC Score.
- **Explainability**: Feature importances extracted and visualized post training.

---

## 📁 Dataset

Example data format:

**Training (`app_train`) and Test (`app_test`)**

| SK_ID_CURR | EXT_SOURCE_1 | EXT_SOURCE_2 | EXT_SOURCE_3 | DAYS_BIRTH | AMT_CREDIT | AMT_INCOME_TOTAL | AMT_ANNUITY | DAYS_EMPLOYED | TARGET |
|------------|---------------|---------------|---------------|--------------|--------------|--------------------|---------------|------------------|--------|
| 100001     | 0.1           | 0.3           | 0.5           | 1000         | 100000       | 300000             | 5000          | 365243           | 0      |
| ...        | ...           | ...           | ...           | ...          | ...           | ...                 | ...            | ...                | ...    |

---

## 🔧 Installation

```bash
pip install numpy pandas lightgbm hyperopt scikit-learn matplotlib seaborn
```
---

## 📌 Usage

### Train model
submission, feature_importances, metrics = model(app_train, app_test)

- Print ROC AUC Scores
print(metrics)

- Run MCMC to explore posterior distribution
samples = mcmc_sampler(data=app_train, n_iterations=1000, ...)

- Use samples to estimate optimal hyperparameters

- Then apply Bayesian Optimization using hyperopt
best_params = fmin(objective_function, space=search_space, algo=tpe.suggest, max_evals=100)

## 🧪 How It Works
LightGBM is trained using K-Fold cross-validation.

MCMC samples from posterior distribution of key hyperparameters like learning_rate, reg_alpha, reg_lambda.

Bayesian Optimization (via hyperopt) refines hyperparameter selection.

The best parameters are used to retrain the model.

Evaluation metrics (AUC) are printed and saved.

## 🔧 Parameters Considered
- learning_rate
- reg_alpha
- reg_lambda

(Optionally, other LightGBM parameters)

## 📊 Output
submission.csv: Contains predictions for the test set.  
metrics: Training and validation ROC-AUC scores.  
feature_importances.csv: Ranked importance of features.  
MCMC plot and sampled trace (optional).  

## 🧠 Algorithms Used
LightGBM: Fast, distributed, high-performance gradient boosting.  
MCMC (Metropolis-Hastings): Samples from posterior distribution using log-posterior.  
Bayesian Optimization (hyperopt): Efficient global optimization for finding best hyperparams.  

## ✅ Future Improvements
Incorporate more features (e.g., domain-specific credit indicators).  
Add visualization for hyperparameter convergence (via MCMC trace).  
Use real-world datasets with imbalanced classes and apply SMOTE.  

## 🧑‍💻 Author
Made with ❤️ by Parth Khairnar  
Feel free to fork and contribute!
