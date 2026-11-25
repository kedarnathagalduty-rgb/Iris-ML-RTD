# Iris Flower Classification Using Machine Learning

RTD mini-project comparing Logistic Regression and Decision Tree on the built-in scikit-learn Iris dataset (150 samples, four measurements, three species).

## Run on Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe main.py --predict 6.0 2.9 4.5 1.5
```

The four values are sepal length, sepal width, petal length and petal width in cm. No downloaded dataset or API key is required. The program retrains both models, evaluates them and predicts a species using Logistic Regression. Outputs are written to `results/metrics.json` and `results/confusion_matrices.png`.

## Method
- Stratified 80/20 split, random seed 42 (120 train, 30 test).
- Logistic Regression: StandardScaler fitted inside a pipeline, max_iter=1000.
- Decision Tree: maximum depth 3, default Gini criterion.
- Five-fold stratified cross-validation on training data only; held-out test evaluation using macro precision, recall and F1.

## Recorded results (scikit-learn 1.8.0)

| Model | Test accuracy | Macro F1 | Training CV mean accuracy |
|---|---:|---:|---:|
| Logistic Regression | 93.33% | 93.33% | 95.83% |
| Decision Tree | 96.67% | 96.66% | 95.83% |

Both models have 2.64 percentage points CV standard deviation. The one-sample difference on the held-out set does not establish general superiority. This is a learning benchmark, not a novel classifier or field-validated application.

## References
- Fisher, R. (1936). Iris [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://scikit-learn.org/stable/modules/tree.html

Review the report and paper, reproduce the experiment, and check your institution's rules for acknowledging AI-assisted work before submission.
