import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Global Vars
INPUT_FILE = Path("preprocessed_posts.json")
RESULTS_FILE = Path("lda_results.json")
PLOT_DIR = Path("plots_lda")

N_TOPICS = 5    # Amount of Topics to find
N_TOP_WORDS = 12    # Amount of terms per Topic
MIN_DF = 3  # minimum term occurance in corpus total
MAX_DF = 0.5    # max. term occurance in corpus percentage

# sums all posts per day into pseudo document if true to address potential "short text" issues
POOL_BY_DAY = False

# load preprocessed data from storagefile into list
def load_posts(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# builds documents out of processed tokens (depending on daily summtion or not)
def build_training_documents(posts: list[dict]) -> list[str]:

    if not POOL_BY_DAY:
        return [" ".join(post["tokens"]) for post in posts]

    pooled = defaultdict(list)
    for post in posts:
        day = post["created_at"][:10]  # YYYY-MM-DD
        pooled[day].extend(post["tokens"])

    print(f"Pooling aktiv: {len(posts)} Posts -> {len(pooled)} Tages-Pseudo-Dokumente.")
    return [" ".join(tokens) for tokens in pooled.values()]

# builds BoW matrix
def build_bow_matrix(documents: list[str], vectorizer: CountVectorizer | None = None):
    if vectorizer is None:
        vectorizer = CountVectorizer(
            min_df=MIN_DF,
            max_df=MAX_DF,
            token_pattern=r"(?u)\b\w+\b",
        )
        matrix = vectorizer.fit_transform(documents)
    else:
        matrix = vectorizer.transform(documents)
    return vectorizer, matrix

# runs LDA algorithm itself
def run_lda(bow_matrix, n_topics: int):
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch",
        max_iter=50,
    )
    lda.fit(bow_matrix)
    return lda

# sorting of term loadings and returning top loadings as topics identified
def get_top_words_per_topic(lda, feature_names, n_top_words: int) -> list[list[tuple[str, float]]]:

    topics = []
    for component in lda.components_:  
        probabilities = component / component.sum() # added for normalizing x-axies in plot
        top_indices = probabilities.argsort()[::-1][:n_top_words]
        top_words = [(feature_names[i], float(probabilities[i])) for i in top_indices]
        topics.append(top_words)
    return topics


# attach identified topic per post (document)
def assign_dominant_topic(doc_topic_matrix) -> list[int]:

    return list(doc_topic_matrix.argmax(axis=1))

# plot term distribution per topic (5) into grafic
def plot_topic_words(topics: list[list[tuple[str, float]]], out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    for idx, words in enumerate(topics):
        labels = [w for w, _ in words][::-1]
        values = [v for _, v in words][::-1]
 
        plt.figure(figsize=(7, 5))
        plt.barh(labels, values, color="#9467bd")
        plt.title(f"Thema {idx + 1} – stärkste Begriffe (LDA)")
        plt.xlabel("P(Wort | Thema)") # normalized likeliehood for term in  topic
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
    plt.title("Anzahl Posts pro dominantem Thema (LDA)")
    plt.ylabel("Anzahl Posts")
    plt.tight_layout()
    plt.savefig(out_dir / "topic_distribution.png", dpi=150)
    plt.close()


def main() -> None:
    posts = load_posts(INPUT_FILE)
    print(f"{len(posts)} Posts geladen.")

    training_documents = build_training_documents(posts)
    vectorizer, train_bow_matrix = build_bow_matrix(training_documents)
    feature_names = vectorizer.get_feature_names_out()
    print(f"Vokabulargröße nach min_df/max_df-Filterung: {len(feature_names)}")

    lda = run_lda(train_bow_matrix, N_TOPICS)
    print(f"Log-Likelihood (Training): {round(lda.score(train_bow_matrix), 1)}")
    print(f"Perplexity (Training): {round(lda.perplexity(train_bow_matrix), 1)}")

    # attach topic per post
    single_post_documents = [" ".join(post["tokens"]) for post in posts]
    _, single_post_bow_matrix = build_bow_matrix(single_post_documents, vectorizer=vectorizer)
    doc_topic_matrix = lda.transform(single_post_bow_matrix)

    topics = get_top_words_per_topic(lda, feature_names, N_TOP_WORDS)
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
        "method": "LDA (Bag-of-Words)",
        "pooled_by_day_for_training": POOL_BY_DAY,
        "n_topics": N_TOPICS,
        "perplexity": float(lda.perplexity(train_bow_matrix)),
        "topics": [
            {"topic_id": i, "top_words": [{"word": w, "probability": v} for w, v in topics[i]]}
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