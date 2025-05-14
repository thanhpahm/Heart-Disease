import src.common.tools as tools
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, precision_score, recall_score, f1_score
import numpy as np

MODELS = [
    "DecisionTree",
    "RandomForestClassifier",
    "LogisticRegression",
    "NeuralNetwork"
]

RESULTS_PATH = "results/results_evaluated_{}.p"
FIGURE_PATH = "reports/figures/model_comparison.png"

def get_accuracy(result_obj):
    if hasattr(result_obj, 'metrics') and 'accuracy' in result_obj.metrics:
        return result_obj.metrics['accuracy']
    if hasattr(result_obj, 'get_metrics'):
        result_obj.get_metrics()
        return result_obj.metrics.get('accuracy', None)
    return None

def get_metric(result_obj, metric):
    y_true = result_obj.y_true
    y_pred = result_obj.y_pred
    average = 'binary' if len(set(y_true)) == 2 else 'weighted'
    if metric == 'accuracy':
        return get_accuracy(result_obj)
    elif metric == 'recall':
        return recall_score(y_true, y_pred, average=average, zero_division=0)
    elif metric == 'precision':
        return precision_score(y_true, y_pred, average=average, zero_division=0)
    elif metric == 'f1':
        return f1_score(y_true, y_pred, average=average, zero_division=0)
    return None

def plot_roc_curve(result_obj, model_name):
    y_true = result_obj.y_true
    y_pred = result_obj.y_pred
    # Try to get probability predictions if available
    if hasattr(result_obj, 'y_score'):
        y_score = result_obj.y_score
    elif hasattr(result_obj, 'proba'):
        y_score = result_obj.proba
    else:
        # fallback: use y_pred as score (not ideal)
        y_score = y_pred
    # Binary or multiclass
    n_classes = len(np.unique(y_true))
    fig, ax = plt.subplots()
    if n_classes == 2:
        if y_score.ndim > 1:
            y_score = y_score[:,1]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model_name).plot(ax=ax)
        ax.set_title(f"ROC Curve - {model_name} (AUC={roc_auc:.2f})")
    else:
        # One-vs-rest for multiclass
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        if y_score.ndim == 1 or y_score.shape[1] == 1:
            y_score = label_binarize(y_score, classes=np.unique(y_true))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:,i], y_score[:,i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")
        ax.plot([0,1],[0,1],'k--')
        ax.set_title(f"ROC Curve - {model_name}")
        ax.legend()
    plt.tight_layout()
    fig.savefig(f"reports/figures/roc_{model_name}.png", dpi=300)
    plt.close(fig)

def plot_metrics_comparison():
    metrics = ['accuracy', 'recall', 'precision', 'f1']
    metric_names = ['Độ chính xác', 'Recall', 'Precision', 'F1-score']
    values = {m: [] for m in metrics}
    for model in MODELS:
        path = RESULTS_PATH.format(model)
        if not os.path.exists(path):
            for m in metrics:
                values[m].append(0)
            continue
        result = tools.pickle_load(path)
        for m in metrics:
            try:
                v = get_metric(result, m)
            except Exception:
                v = 0
            values[m].append(v if v is not None else 0)
    x = np.arange(len(MODELS))
    width = 0.2
    plt.figure(figsize=(10,6))
    for i, m in enumerate(metrics):
        plt.bar(x + i*width - 1.5*width, values[m], width, label=metric_names[i])
    plt.xticks(x, MODELS)
    plt.ylim(0, 1)
    plt.ylabel('Giá trị')
    plt.title('So sánh các chỉ số đánh giá mô hình')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/figures/model_metrics_comparison.png', dpi=300)
    plt.close()

def main():
    accuracies = []
    for model in MODELS:
        path = RESULTS_PATH.format(model)
        if not os.path.exists(path):
            accuracies.append(0)
            continue
        result = tools.pickle_load(path)
        acc = get_accuracy(result)
        accuracies.append(acc if acc is not None else 0)
        # ROC curve
        try:
            plot_roc_curve(result, model)
        except Exception as e:
            print(f"[WARN] Could not plot ROC for {model}: {e}")
    plt.figure(figsize=(8,5))
    plt.bar(MODELS, accuracies, color=['#4e79a7','#f28e2b','#e15759','#76b7b2'])
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title('So sánh độ chính xác các mô hình')
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=300)
    print(f"Saved to {FIGURE_PATH}")
    # Plot metrics comparison
    plot_metrics_comparison()
    print("Saved to reports/figures/model_metrics_comparison.png")

if __name__ == "__main__":
    main() 