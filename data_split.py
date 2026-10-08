import argparse
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="sources/data.csv")
    parser.add_argument("--train_out", default="train.csv")
    parser.add_argument("--val_out", default="val.csv")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=321)
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.data, header=None)

    train_parts = []
    val_parts = []

    for label in ["B", "M"]:
        sub = df[df.iloc[:, 1] == label]
        n_val = int(round(len(sub) * args.ratio))
        shuffled = sub.sample(frac=1.0, random_state=args.seed)
        val_parts.append(shuffled.iloc[:n_val])
        train_parts.append(shuffled.iloc[n_val:])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=args.seed)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=args.seed)

    train_df.to_csv(args.train_out, index=False, header=False)
    val_df.to_csv(args.val_out, index=False, header=False)

    print(f"Total: {len(df)} samples")
    print(f"Train: {len(train_df)} -> {args.train_out}")
    print(f"Val:   {len(val_df)} -> {args.val_out}")

if __name__ == "__main__":
    main()
