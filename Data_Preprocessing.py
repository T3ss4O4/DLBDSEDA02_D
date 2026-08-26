import json
import re
import string
from html.parser import HTMLParser
from pathlib import Path
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

# Global Vars
INPUT_FILE = Path("scraped_content.json")
OUTPUT_FILE = Path("preprocessed_posts.json")
STATS_FILE = Path("word_frequencies.json")

MIN_TOKEN_LENGTH = 2 # Post contents quality
ONLY_GERMAN = True  # optional filter for only german posts (problems with stopword list)

# fixes issue with NLTK not properly loading needed resource-sets
def ensure_nltk_data() -> None:
    required = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, package in required:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)

# extracts only text from HTML-"content", ignores HTML-formating and special parts
class HTMLStripper(HTMLParser):

    def __init__(self):
        super().__init__()
        self.text_parts: list[str] = []
        self._link_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "a":
            self._link_depth += 1
        elif tag in ("br", "p") and self._link_depth == 0:
            self.text_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._link_depth > 0:
            self._link_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._link_depth == 0:
            self.text_parts.append(data)

    def get_text(self) -> str:
        return " ".join(self.text_parts)

# function to call the html-sanitizer class in procedure
def strip_html(html_content: str) -> str:
    stripper = HTMLStripper()
    stripper.feed(html_content)
    return stripper.get_text()

# typical social media text elements not feasible for topic analysis
URL_PATTERN = re.compile(r"https?://\S+")
MENTION_PATTERN = re.compile(r"@[\w.-]+(@[\w.-]+)?")
HASHTAG_PATTERN = re.compile(r"#\w+")

# runs all clean operations (HTML and social media elements)
def clean_text(raw_html: str) -> str:
    text = strip_html(raw_html)
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(" ", text)
    return text

# actual preprocessing after sanitizing (tokenize, stop words removal, stemming etc.)
def tokenize_and_filter(text: str, stopword_set: set, stemmer: SnowballStemmer) -> list[str]:
    text = text.lower()
    tokens = word_tokenize(text, language="german")

    cleaned = []
    for token in tokens:
        if token in string.punctuation:
            continue
        if not token.isalpha():  # removes posts which only contain numbers
            continue
        if len(token) < MIN_TOKEN_LENGTH:
            continue
        if token in stopword_set:
            continue
        cleaned.append(stemmer.stem(token))
    return cleaned

# load posts from scraped storage file
def load_posts(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# saves results from preprocessing to new file
def save_json(data, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# definition of main procedure
def main() -> None:
    ensure_nltk_data()
    stopword_set = set(stopwords.words("german"))
    stemmer = SnowballStemmer("german")

    posts = load_posts(INPUT_FILE)
    print(f"{len(posts)} Posts geladen.")

    processed = []
    overall_counter: Counter = Counter()
    skipped_lang = 0
    skipped_empty = 0

    for post in posts:
        if ONLY_GERMAN and post.get("language") != "de":
            skipped_lang += 1
            continue

        raw_content = post.get("content", "") or ""
        cleaned_text = clean_text(raw_content)
        tokens = tokenize_and_filter(cleaned_text, stopword_set, stemmer)

        if not tokens:
            skipped_empty += 1
            continue

        overall_counter.update(tokens)

        processed.append({
            "id": post.get("id"),
            "created_at": post.get("created_at"),
            "hashtag": post.get("_fetched_via_hashtag"),
            "language": post.get("language"),
            "cleaned_text": cleaned_text.strip(),
            "tokens": tokens,
        })

    print(
        f"{len(processed)} Posts nach Preprocessing übrig "
        f"({skipped_lang} wegen Sprache, {skipped_empty} wegen leerem Ergebnis übersprungen)."
    )

    save_json(processed, OUTPUT_FILE)

    top_terms = overall_counter.most_common(30)
    save_json(top_terms, STATS_FILE)

    print("Top 15 Begriffe im Korpus (nach Stemming):")
    for term, count in top_terms[:15]:
        print(f"  {term}: {count}")


if __name__ == "__main__":
    main()