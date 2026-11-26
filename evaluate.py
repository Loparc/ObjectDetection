# evaluate.py
import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from CNN import vgg16
from data_extraction.proposal_dataset import ProposalDataset, PROPOSALS_JSON, fixed_val_subset


def confusion_counts(output, target):
    pred = output.argmax(dim=1)
    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()
    return tp, fp, fn, tn


def compute_metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def evaluate_metrics(model, dataloader, device, loss_fn=None):
    model.eval()
    total_loss, n = 0.0, 0
    tp = fp = fn = tn = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if loss_fn is not None:
                total_loss += loss_fn(output, target).item() * len(target)
            n += len(target)
            b_tp, b_fp, b_fn, b_tn = confusion_counts(output, target)
            tp += b_tp
            fp += b_fp
            fn += b_fn
            tn += b_tn

    results = compute_metrics(tp, fp, fn, tn)
    if loss_fn is not None:
        results['loss'] = total_loss / n
    return results


def load_model_weights(model, filepath, device='cuda'):
    model.load_state_dict(torch.load(filepath, map_location=device, weights_only=True))
    model.eval()
    return model


def print_full_report(model, dataloaders, device, title, output_dir='.'):
    # dataloaders: dict like {'Val': dataloader_val} or {'Val': ..., 'Test': ...}
    print(f"\n================ FINAL REPORT: {title} ================")
    print(f"{'Split':<6} | {'Acc (%)':<8} | {'Prec (%)':<8} | {'Rec (%)':<8} | {'F1 (%)':<8} | "
          f"{'TP':>6} {'FP':>6} {'FN':>6} {'TN':>7}")
    print("-" * 80)

    report_results = {}
    for split_name, loader in dataloaders.items():
        m = evaluate_metrics(model, loader, device)
        report_results[split_name] = m
        print(
            f"{split_name:<6} | {100*m['acc']:<8.2f} | {100*m['precision']:<8.2f} | "
            f"{100*m['recall']:<8.2f} | {100*m['f1']:<8.2f} | "
            f"{m['tp']:>6} {m['fp']:>6} {m['fn']:>6} {m['tn']:>7}"
        )
        plot_confusion_matrix(m, f"{title}_{split_name}", output_dir=output_dir)
    print("=" * 80 + "\n")

    plot_final_metrics_bar(report_results, title, output_dir=output_dir)
    return report_results


def plot_learning_curves(history, title, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    epochs = list(range(1, len(history['train']['loss']) + 1))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    # Loss
    axes[0].plot(epochs, history['train']['loss'], label='Train Loss', marker='o', markersize=3, color='black')
    axes[0].plot(epochs, history['val']['loss'], label='Val Loss', marker='s', markersize=3, color='red', linestyle='--')
    axes[0].set_title(f'{title} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle=':')

    # Metrics
    metric_display = [
        ('acc',       'Accuracy (%)'),
        ('precision', 'Precision (%)'),
        ('recall',    'Recall (%)'),
        ('f1',        'F1 (%)')
    ]

    for idx, (m_key, m_label) in enumerate(metric_display, start=1):
        tr_vals = [x * 100 for x in history['train'][m_key]]
        va_vals = [x * 100 for x in history['val'][m_key]]

        axes[idx].plot(epochs, tr_vals, label=f'Train {m_key}', marker='o', markersize=3, color='tab:blue')
        axes[idx].plot(epochs, va_vals, label=f'Val {m_key}', marker='s', markersize=3, color='tab:orange', linestyle='--')
        axes[idx].set_title(f'{title} - {m_label}')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('%')
        axes[idx].set_ylim(0, 105)
        axes[idx].legend()
        axes[idx].grid(True, linestyle=':')

    axes[5].axis('off')

    filename = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}_learning_curves.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Learning curves saved to {filename}")


def plot_final_metrics_bar(report_results, title, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    metrics = ['acc', 'precision', 'recall', 'f1']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1']

    x = np.arange(len(metrics))
    width = 0.35
    colors = {'Val': '#d95f02', 'Test': '#2ca02c'}

    fig, ax = plt.subplots(figsize=(8, 5))
    splits = list(report_results.keys())
    for i, split in enumerate(splits):
        values = [report_results[split][m] * 100 for m in metrics]
        offset = (i - (len(splits) - 1) / 2) * width
        rects = ax.bar(x + offset, values, width, label=split, color=colors.get(split, '#2b5c8f'))
        ax.bar_label(rects, fmt='%.1f', padding=3, fontsize=8)

    ax.set_ylabel('Score (%)')
    ax.set_title(f'{title} - Final Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 112)
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.7)

    filename = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}_metrics_bar.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics bar chart saved to {filename}")


def plot_confusion_matrix(m, title, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    cm = np.array([[m['tn'], m['fp']],
                   [m['fn'], m['tp']]])

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, f"{cm[i, j]}", ha='center', va='center', color=color, fontsize=12)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Background', 'Pothole'])
    ax.set_yticklabels(['Background', 'Pothole'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{title} - Confusion Matrix')

    filename = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {filename}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

    p = argparse.ArgumentParser()
    p.add_argument('--exp', default='baseline')
    p.add_argument('--test', action='store_true', help='also evaluate on the test set (only at the end!)')
    args = p.parse_args()

    VAL_SAMPLES = 20000   # must be the same as in train.py
    BATCH_SIZE  = 64

    with open(PROPOSALS_JSON) as f:
        proposals = json.load(f)

    dataset_val = ProposalDataset('val', proposals)
    val_idx = fixed_val_subset(dataset_val.labels, VAL_SAMPLES)
    dataloaders = {'Val': DataLoader(Subset(dataset_val, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)}

    if args.test:
        dataset_test = ProposalDataset('test', proposals)
        dataloaders['Test'] = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    del proposals

    model = vgg16(pretrained=False).to(device)
    best_path = os.path.join(output_dir, f"{args.exp}_best.pt")
    model = load_model_weights(model, best_path, device=device)

    print_full_report(model, dataloaders, device, title=args.exp, output_dir=output_dir)