"""Reproducible comparison of two Iris classifiers; optional interactive prediction."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, ConfusionMatrixDisplay, confusion_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predict', nargs=4, type=float, metavar='CM', help='Sepal length, sepal width, petal length, petal width (cm)')
    args = parser.parse_args()
    if args.predict is not None and (not np.isfinite(args.predict).all() or min(args.predict) <= 0):
        parser.error('Measurements must be positive finite numbers.')
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
    }
    output = Path(__file__).resolve().parent / 'results'
    output.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {'sklearn_version': sklearn.__version__, 'samples': len(y), 'training_samples': len(y_train), 'test_samples': len(y_test), 'class_names': list(data.target_names), 'models': {}}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    print('Model comparison (held-out test set: 30 samples)')
    for (name, model), ax in zip(models.items(), axes):
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        p, r, f, _ = precision_recall_fscore_support(y_test, pred, average='macro', zero_division=0)
        m = {'accuracy': float(accuracy_score(y_test, pred)), 'precision_macro': float(p), 'recall_macro': float(r), 'f1_macro': float(f), 'cv_accuracy_mean': float(scores.mean()), 'cv_accuracy_std': float(scores.std()), 'cv_fold_scores': scores.tolist(), 'confusion_matrix': confusion_matrix(y_test, pred).tolist()}
        results['models'][name] = m
        print(f'{name}: accuracy={m["accuracy"]:.2%}, macro F1={f:.4f}, training CV accuracy={scores.mean():.2%}')
        ConfusionMatrixDisplay(np.array(m['confusion_matrix']), display_labels=data.target_names).plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(name)
    fig.savefig(output / 'confusion_matrices.png', dpi=160)
    plt.close(fig)
    sample = args.predict if args.predict is not None else [5.1, 3.5, 1.4, 0.2]
    # Select the demonstration model in advance, not by looking at the test scores.
    predicted = data.target_names[models['Logistic Regression'].predict([sample])[0]]
    print(f'Prediction for {sample}: {predicted}')
    results['demo_prediction'] = {'measurements_cm': sample, 'model': 'Logistic Regression', 'species': str(predicted)}
    (output / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'Results saved in {output}')

if __name__ == '__main__':
    main()
