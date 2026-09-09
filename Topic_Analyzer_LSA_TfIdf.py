import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Global Vars
INPUT_FILE = Path("preprocessed_posts.json")
RESULTS_FILE = Path("lsa_results.json")
PLOT_DIR = Path("plots_lsa")

N_TOPICS = 5    # Amount of Topics to find
N_TOP_WORDS = 12  # Amount of terms per Topic 
MIN_DF = 3        # minimum term occurance in corpus total
MAX_DF = 0.5       # max. term occurance in corpus percentage

# load preprocessed data from storagefile into tuple of lists, build text out of processed tokens
def load_documents(path: Path) -> tuple[list[str], list[dict]]:

    with open(path, "r", encoding="utf-8") as f:
        posts = json.load(f)

    documents = [" ".join(post["tokens"]) for post in posts]
    return documents, posts

# builds tfidf matrix utilizing vectorizer
def build_tfidf_matrix(documents: list[str]):

    vectorizer = TfidfVectorizer(
        min_df=MIN_DF,
        max_df=MAX_DF,
        token_pattern=r"(?u)\b\w+\b",
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix

# takes the processed tfidf and runs LSA on it, returns topic matrix and SVD after truncating
def run_lsa(tfidf_matrix, n_topics: int):
    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    doc_topic_matrix = svd.fit_transform(tfidf_matrix)
    return svd, doc_topic_matrix

# sorting of term loadings and returning top loadings as topics identified
def get_top_words_per_topic(svd, feature_names, n_top_words: int) -> list[list[tuple[str, float]]]:

    topics = []
    for component in svd.components_:
        # sort LSA loadings including negative room
        top_indices_abs = sorted(range(len(component)), key=lambda i: abs(component[i]), reverse=True)
        top_words = [(feature_names[i], float(component[i])) for i in top_indices_abs[:n_top_words]]
        topics.append(top_words)
    return topics

# attach identified topic per post (document)
def assign_dominant_topic(doc_topic_matrix) -> list[int]:

    return list(abs(doc_topic_matrix).argmax(axis=1))

# plot term distribution per topic (5) into grafic
def plot_topic_words(topics: list[list[tuple[str, float]]], out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    for idx, words in enumerate(topics):
        labels = [w for w, _ in words][::-1]
        values = [v for _, v in words][::-1]
        colors = ["#d62728" if v < 0 else "#1f77b4" for v in values]

        plt.figure(figsize=(7, 5))
        plt.barh(labels, values, color=colors)
        plt.title(f"Thema {idx + 1} – stärkste Begriffe (LSA)")
        plt.xlabel("Ladung (Betrag = Relevanz, Farbe = Vorzeichen)")
        plt.tight_layout()
        plt.savefig(out_dir / f"topic_{idx + 1}.png", dpi=150)
        plt.close()

# plot topic distribution over all posts into grafic
def plot_topic_distribution(topic_counts: dict, out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    labels = [f"Thema {t + 1}" for t in sorted(topic_counts)]
    values = [topic_counts[t] for t in sorted(topic_counts)]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values, color="#2ca02c")
    plt.title("Anzahl Posts pro dominantem Thema (LSA)")
    plt.ylabel("Anzahl Posts")
    plt.tight_layout()
    plt.savefig(out_dir / "topic_distribution.png", dpi=150)
    plt.close()


def main() -> None:
    documents, posts = load_documents(INPUT_FILE)
    print(f"{len(documents)} Dokumente geladen.")

    vectorizer, tfidf_matrix = build_tfidf_matrix(documents)
    feature_names = vectorizer.get_feature_names_out()
    print(f"Vokabulargröße nach min_df/max_df-Filterung: {len(feature_names)}")

    svd, doc_topic_matrix = run_lsa(tfidf_matrix, N_TOPICS)
    explained_var = svd.explained_variance_ratio_
    print(f"Erklärte Varianz je Thema: {[round(v, 3) for v in explained_var]}")
    print(f"Summe erklärte Varianz: {round(sum(explained_var), 3)}")

    topics = get_top_words_per_topic(svd, feature_names, N_TOP_WORDS)
    dominant_topics = assign_dominant_topic(doc_topic_matrix)

    # Count and print out 5 most identified topics 
    topic_counts: dict = {}
    for t in dominant_topics:
        topic_counts[t] = topic_counts.get(t, 0) + 1

    print("\n--- Themen (sortiert nach Häufigkeit) ---")
    for topic_idx, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        top_words_str = ", ".join(w for w, _ in topics[topic_idx][:8])
        print(f"Thema {topic_idx + 1} ({count} Posts): {top_words_str}")

    # save results in structured way to compare with other methods later
    results = {
        "method": "TF-IDF + LSA (TruncatedSVD)",
        "n_topics": N_TOPICS,
        "explained_variance_ratio": [float(v) for v in explained_var],
        "topics": [
            {"topic_id": i, "top_words": [{"word": w, "loading": v} for w, v in topics[i]]}
            for i in range(N_TOPICS)
        ],
        "topic_counts": {int(k): v for k, v in topic_counts.items()},
        "document_assignments": [
            {"id": posts[i]["id"], "dominant_topic": int(dominant_topics[i])}
            for i in range(len(posts))
        ],
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # plot grafic output files
    plot_topic_words(topics, PLOT_DIR)
    plot_topic_distribution(topic_counts, PLOT_DIR)

    print(f"\nErgebnisse gespeichert in '{RESULTS_FILE}', Plots in '{PLOT_DIR}/'.")


if __name__ == "__main__":
    main()