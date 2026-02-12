import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--player", type=int, default=1)
    parser.add_argument("--output", default="belief_trend.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["player"] == args.player]

    plt.plot(df["turn"], df["unknown_rate"], label="unknown rate")
    plt.plot(df["turn"], df["known_rate"], label="known rate")
    plt.plot(df["turn"], df["avg_margin"], label="avg margin")
    plt.title(f"Belief confidence (player {args.player})")
    plt.xlabel("turn")
    plt.ylabel("rate / margin")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"saved {args.output}")

if __name__ == "__main__":
    main()
