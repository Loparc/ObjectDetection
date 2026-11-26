import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, Subset

from CNN import vgg16
from data_extraction.proposal_dataset import ProposalDataset, PROPOSALS_JSON, fixed_val_subset
from evaluate import evaluate_metrics, compute_metrics, confusion_counts, plot_learning_curves

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def train_model_and_validate(
    model, optimizer, scheduler, loss_fn, dataloader_train, dataloader_val,
    num_epochs=10, device='cuda', best_path='./best_model.pt'
):
    metric_keys = ['loss', 'acc', 'precision', 'recall', 'f1']
    history = {
        'train': {k: [] for k in metric_keys},
        'val':   {k: [] for k in metric_keys}
    }
    best_val_f1 = -1.0

    for epoch in range(num_epochs):
        model.train()
        total_loss, n = 0.0, 0
        tp = fp = fn = tn = 0
        for data, target in dataloader_train:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            # train metrics on the batches seen in this epoch
            total_loss += loss.item() * len(target)
            n += len(target)
            b_tp, b_fp, b_fn, b_tn = confusion_counts(output.detach(), target)
            tp += b_tp
            fp += b_fp
            fn += b_fn
            tn += b_tn

        train_metrics = compute_metrics(tp, fp, fn, tn)
        train_metrics['loss'] = total_loss / n
        val_metrics = evaluate_metrics(model, dataloader_val, device, loss_fn=loss_fn)

        if scheduler is not None:
            scheduler.step(val_metrics['loss'])

        for k in metric_keys:
            history['train'][k].append(train_metrics[k])
            history['val'][k].append(val_metrics[k])

        print(
            f"Epoch {epoch+1:02d}/{num_epochs:02d} | "
            f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
            f"Acc: {100*train_metrics['acc']:.1f}%/{100*val_metrics['acc']:.1f}% | "
            f"Prec: {100*val_metrics['precision']:.1f}% | Rec: {100*val_metrics['recall']:.1f}% | "
            f"F1: {100*val_metrics['f1']:.1f}% | "
            f"TP {val_metrics['tp']} FP {val_metrics['fp']} FN {val_metrics['fn']} TN {val_metrics['tn']}",
            flush=True
        )

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best model saved ({os.path.basename(best_path)} | Val F1: {100*best_val_f1:.2f}%)")

    return history


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    p = argparse.ArgumentParser()
    p.add_argument('--exp', default='baseline')
    args = p.parse_args()
    EXP_NAME = args.exp

    TRAIN_SAMPLES = 50000   # proposals per epoch
    VAL_SAMPLES   = 20000   # fixed validation subset
    BATCH_SIZE    = 64
    NUM_EPOCHS    = 10
    LR            = 1e-4

    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(PROPOSALS_JSON) as f:
        proposals = json.load(f)
    dataset_train = ProposalDataset('train', proposals)
    dataset_val   = ProposalDataset('val', proposals)
    del proposals

    # new random subset every epoch, natural class proportion (no balancing)
    sampler_train = RandomSampler(dataset_train, replacement=False, num_samples=TRAIN_SAMPLES)
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, sampler=sampler_train, num_workers=4)

    val_idx = fixed_val_subset(dataset_val.labels, VAL_SAMPLES)
    dataloader_val = DataLoader(Subset(dataset_val, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = vgg16(pretrained=True).to(device)

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    best_path = os.path.join(OUTPUT_DIR, f"{EXP_NAME}_best.pt")

    history = train_model_and_validate(
        model, optimizer, scheduler, loss_fn, dataloader_train, dataloader_val,
        num_epochs=NUM_EPOCHS, device=device, best_path=best_path
    )

    with open(os.path.join(OUTPUT_DIR, f"{EXP_NAME}_history.json"), 'w') as f:
        json.dump(history, f)
    plot_learning_curves(history, title=EXP_NAME, output_dir=OUTPUT_DIR)