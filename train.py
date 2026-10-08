import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import MLP

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="train.csv" if os.path.exists("train.csv") else "data_training.csv")
    parser.add_argument("--val_dataset", default="val.csv" if os.path.exists("val.csv") else "data_validation.csv")
    parser.add_argument("--layer", nargs="+", type=int, default=[24, 24])
    parser.add_argument("--activation", default="sigmoid")
    parser.add_argument("--epochs", type=int, default=84)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.0314)
    parser.add_argument("--loss", default="binaryCrossentropy")
    parser.add_argument("--model_out", default="saved_model.npy")
    parser.add_argument("--seed", type=int, default=321)
    return parser.parse_args()

def parse_csv(path):
    df = pd.read_csv(path, header=None)
    raw_y = df.iloc[:, 1].values
    y = np.zeros((len(raw_y), 2))
    for i, val in enumerate(raw_y):
        y[i] = [1.0, 0.0] if val == "B" else [0.0, 1.0]
    x = df.iloc[:, 2:].to_numpy(dtype=float)
    return x, y

def main():
    args = parse_args()
    np.random.seed(args.seed)

    x_train, y_train = parse_csv(args.dataset)
    if os.path.exists(args.val_dataset):
        x_val, y_val = parse_csv(args.val_dataset)
    else:
        split_idx = int(len(x_train) * 0.8)
        perm = np.random.permutation(len(x_train))
        x_val, y_val = x_train[perm[split_idx:]], y_train[perm[split_idx:]]
        x_train, y_train = x_train[perm[:split_idx]], y_train[perm[:split_idx]]

    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std[std == 0.0] = 1.0

    x_train_norm = (x_train - mean) / std
    x_val_norm = (x_val - mean) / std

    print(f"x_train shape : {x_train_norm.shape}")
    print(f"x_valid shape : {x_val_norm.shape}")

    dims = [x_train_norm.shape[1]] + args.layer + [2]
    model = MLP(dims, activation=args.activation)
    model.mean = mean
    model.std = std

    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
    m = len(x_train_norm)

    for ep in range(1, args.epochs + 1):
        idx = np.random.permutation(m)
        x_shuf = x_train_norm[idx]
        y_shuf = y_train[idx]

        for b in range(0, m, args.batch_size):
            xb = x_shuf[b:b + args.batch_size]
            yb = y_shuf[b:b + args.batch_size]
            out_b = model.forward(xb)
            model.backward(yb, out_b, args.learning_rate)

        train_pred = model.forward(x_train_norm)
        val_pred = model.forward(x_val_norm)

        loss = model.compute_loss(y_train, train_pred)
        vloss = model.compute_loss(y_val, val_pred)
        acc = model.compute_acc(y_train, train_pred)
        vacc = model.compute_acc(y_val, val_pred)

        history["loss"].append(loss)
        history["val_loss"].append(vloss)
        history["acc"].append(acc)
        history["val_acc"].append(vacc)

        print(f"epoch {ep:02d}/{args.epochs:02d} - loss: {loss:.4f} - val_loss: {vloss:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(range(1, args.epochs + 1), history["loss"], label="training loss", color="#2b5c8f")
    ax1.plot(range(1, args.epochs + 1), history["val_loss"], label="validation loss", color="#d9534f", linestyle="--")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, args.epochs + 1), history["acc"], label="training acc", color="#2b5c8f")
    ax2.plot(range(1, args.epochs + 1), history["val_acc"], label="validation acc", color="#d9534f", linestyle="--")
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150)
    plt.close()

    model.save(args.model_out)
    print(f"> saving model '{args.model_out}' to disk...")

if __name__ == "__main__":
    main()
