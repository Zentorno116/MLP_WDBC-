import argparse
import os
import numpy as np
import pandas as pd
from model import MLP

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="val.csv" if os.path.exists("val.csv") else "data_validation.csv")
    parser.add_argument("--model", default="saved_model.npy")
    return parser.parse_args()

def main():
    args = parse_args()
    model = MLP.load(args.model)

    df = pd.read_csv(args.dataset, header=None)
    has_labels = str(df.iloc[0, 1]).strip() in ["B", "M"]

    if has_labels:
        raw_y = df.iloc[:, 1].values
        y_true = np.zeros((len(raw_y), 2))
        for i, val in enumerate(raw_y):
            y_true[i] = [1.0, 0.0] if val == "B" else [0.0, 1.0]
        x = df.iloc[:, 2:].to_numpy(dtype=float)
    else:
        y_true = None
        x = df.iloc[:, 1:].to_numpy(dtype=float)

    x_norm = (x - model.mean) / model.std
    probs = model.forward(x_norm)
    preds = np.argmax(probs, axis=1)

    labels = ["B" if p == 0 else "M" for p in preds]

    if has_labels:
        bce = model.compute_loss(y_true, probs)
        acc = model.compute_acc(y_true, probs)
        t = np.argmax(y_true, axis=1)

        tp = int(np.sum((preds == 1) & (t == 1)))
        fp = int(np.sum((preds == 1) & (t == 0)))
        fn = int(np.sum((preds == 0) & (t == 1)))
        tn = int(np.sum((preds == 0) & (t == 0)))

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        print(f"Evaluated samples: {len(x)}")
        print(f"Binary Cross-Entropy Loss: {bce:.4f}")
        print(f"Accuracy:  {acc * 100:.2f}%")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    else:
        for i, lab in enumerate(labels[:10]):
            print(f"Sample {i}: {lab} (P(B)={probs[i, 0]:.4f}, P(M)={probs[i, 1]:.4f})")
        if len(labels) > 10:
            print(f"... total {len(labels)} predictions")

if __name__ == "__main__":
    main()
