import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Global Vars
INPUT_FILE = Path("scraped_content.json")
RESULTS_FILE = Path("accounts_hashtags_results.json")
PLOT_DIR = Path("plots_accounts_hashtags")

TOP_N = 10 # Number of Top Users and Hashes to calculate

# load scraped data from storagefile into list
def load_posts(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# counts occurance from all accounts
def count_accounts(posts: list[dict]) -> Counter:

    counter = Counter()
    for post in posts:
        acct = post.get("account", {}).get("acct")
        if acct:
            counter[acct] += 1
    return counter

# looks additionally up if account is taged as bot
def is_bot_lookup(posts: list[dict]) -> dict:

    lookup = {}
    for post in posts:
        account = post.get("account", {})
        acct = account.get("acct")
        if acct is not None:
            lookup[acct] = bool(account.get("bot", False))
    return lookup

# counts occurance from all hastags
def count_hashtags(posts: list[dict]) -> Counter:

    counter = Counter()
    for post in posts:
        for tag in post.get("tags", []):
            name = tag.get("name")
            if name:
                counter[name] += 1
    return counter

# Plots top N accounts and their type
def plot_top_accounts(top_accounts: list[tuple[str, int]], bot_lookup: dict, out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    labels = [acct for acct, _ in top_accounts][::-1]
    values = [count for _, count in top_accounts][::-1]
    colors = ["#d62728" if bot_lookup.get(acct, False) else "#1f77b4" for acct in labels]

    plt.figure(figsize=(8, 5))
    plt.barh(labels, values, color=colors)
    plt.title(f"Top {len(top_accounts)} Accounts nach Post-Anzahl")
    plt.xlabel("Anzahl Posts")

    legend_handles = [
        mpatches.Patch(color="#1f77b4", label="Account"),
        mpatches.Patch(color="#d62728", label="Bot-Account (Mastodon-Flag)"),
    ]
    plt.legend(handles=legend_handles, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "top_accounts.png", dpi=150)
    plt.close()

# Plots top N hastags
def plot_top_hashtags(top_hashtags: list[tuple[str, int]], out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    labels = [tag for tag, _ in top_hashtags][::-1]
    values = [count for _, count in top_hashtags][::-1]

    plt.figure(figsize=(8, 5))
    plt.barh(labels, values, color="#ff7f0e")
    plt.title(f"Top {len(top_hashtags)} Hashtags nach Häufigkeit")
    plt.xlabel("Anzahl Posts")
    plt.tight_layout()
    plt.savefig(out_dir / "top_hashtags.png", dpi=150)
    plt.close()


def main() -> None:
    posts = load_posts(INPUT_FILE)
    print(f"{len(posts)} Posts geladen.")

    account_counts = count_accounts(posts)
    hashtag_counts = count_hashtags(posts)
    bot_lookup = is_bot_lookup(posts)

    top_accounts = account_counts.most_common(TOP_N)
    top_hashtags = hashtag_counts.most_common(TOP_N)

    print(f"\n--- Top {TOP_N} Accounts ---")
    for acct, count in top_accounts:
        bot_marker = " [Bot]" if bot_lookup.get(acct) else ""
        print(f"{count:>4}  {acct}{bot_marker}")

    print(f"\n--- Top {TOP_N} Hashtags ---")
    for tag, count in top_hashtags:
        print(f"{count:>4}  #{tag}")

    plot_top_accounts(top_accounts, bot_lookup, PLOT_DIR)
    plot_top_hashtags(top_hashtags, PLOT_DIR)

    results = {
        "total_posts": len(posts),
        "top_accounts": [
            {"acct": acct, "count": count, "is_bot": bot_lookup.get(acct, False)}
            for acct, count in top_accounts
        ],
        "top_hashtags": [{"hashtag": tag, "count": count} for tag, count in top_hashtags],
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nErgebnisse gespeichert in '{RESULTS_FILE}', Plots in '{PLOT_DIR}/'.")


if __name__ == "__main__":
    main()