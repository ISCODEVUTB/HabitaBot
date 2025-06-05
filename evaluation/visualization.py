"""
visualization.py

Purpose:
    Generates performance plots: confusion matrix, precision-recall curve, ROC, and per-year accuracy histogram for HabitaBot.

Dependencies:
    - numpy
    - matplotlib
    - scikit-learn
    - metrics.py (for barrios_criticos_gt and consultar_barrios_criticos)

Notes:
    - Assumes you have loaded the trained pipe, the DataFrame, and imported required functions.
    - Run from the evaluation/ directory or set the correct path to metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc
import os
import sys

# Allow import from the same directory
sys.path.append(os.path.dirname(__file__))
from metrics import barrios_criticos_gt, consultar_barrios_criticos

def to_multihot(barrios, all_barrios):
    return [1 if b in barrios else 0 for b in all_barrios]

def plot_all_metrics(df, pipe):
    all_barrios = sorted(df["barrio"].unique())
    gt = []
    pred = []
    anios = sorted(df["año"].unique())
    for año in anios:
        gt_barrios = [b.lower() for b in barrios_criticos_gt(df, año, top_n=10)]
        pred_barrios = [b.lower() for b in consultar_barrios_criticos(pipe, df, año, top_n=10, verbose=False)]
        gt.append(to_multihot(gt_barrios, [b.lower() for b in all_barrios]))
        pred.append(to_multihot(pred_barrios, [b.lower() for b in all_barrios]))

    gt = np.array(gt)
    pred = np.array(pred)

    # Global confusion matrix
    mcm = multilabel_confusion_matrix(gt, pred)
    global_cm = np.sum(mcm, axis=0)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(global_cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No crítico', 'Crítico'])
    ax.set_yticklabels(['No crítico', 'Crítico'])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión Global (multi-label)\nBarrios Críticos vs No Críticos")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(global_cm[i, j]), ha="center", va="center", color="black", fontsize=14)
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(gt.ravel(), pred.ravel())
    ap = average_precision_score(gt, pred, average="micro")
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, marker='.', label=f'AP={ap:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall (micro average)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(gt.ravel(), pred.ravel())
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC (micro average)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Histograma de aciertos por año
    aciertos_por_ano = []
    for año in anios:
        gt_barrios = [b.lower() for b in barrios_criticos_gt(df, año, top_n=10)]
        pred_barrios = [b.lower() for b in consultar_barrios_criticos(pipe, df, año, top_n=10, verbose=False)]
        aciertos = len(set(gt_barrios) & set(pred_barrios))
        aciertos_por_ano.append(aciertos)

    plt.figure(figsize=(8, 4))
    plt.bar(anios, aciertos_por_ano, color="skyblue")
    plt.xlabel("Año")
    plt.ylabel("Aciertos sobre 10")
    plt.title("Cantidad de barrios críticos predichos correctamente por año")
    plt.tight_layout()
    plt.show()