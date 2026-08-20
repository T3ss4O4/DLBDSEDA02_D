import time
import json
import logging
import requests
from pathlib import Path


# Global Vars
INSTANCE = "https://mastodon.social"
HASHTAGS = ["koeln", "köln"]
INTERVAL = 1800  # 30 minutes in seconds
FILE_LOCATION = Path("scraped_content.json")
REQUEST_TIMEOUT = 15 # connection errorhandling
LIMIT_PER_CALL = 40 # max. possible entries per request

# debugging console logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# load already existing data from storagefile into list
def load_existing_data() -> list:

    if not FILE_LOCATION.exists():
        return []
    try:
        with open(FILE_LOCATION, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Konnte bestehende Datei nicht lesen (%s) – starte mit leerer Liste.", e)
        return []

# identify id of latest post already in file
def get_last_seen_ids(existing_data: list) -> dict:
    last_ids = {}
    for entry in existing_data:
        tag = entry.get("_fetched_via_hashtag")
        post_id = entry.get("id")
        if tag and post_id:
            if tag not in last_ids or int(post_id) > int(last_ids[tag]):
                last_ids[tag] = post_id
    return last_ids

# actual fetch activity for one hashtag
def fetch_hashtag_posts(hashtag: str, since_id: str | None) -> list:
    params = {"limit": LIMIT_PER_CALL}
    if since_id:
        params["since_id"] = since_id

    url = f"{INSTANCE}/api/v1/timelines/tag/{hashtag}"
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error("Fehler beim Abrufen von #%s: %s", hashtag, e)
        return []

    posts = resp.json()
    
    # add value to trace the hash fetched 
    for post in posts:
        post["_fetched_via_hashtag"] = hashtag
    return posts


# fetchloop for all hash values
def get_hashtag_data(last_ids: dict) -> list:
    all_posts = []
    for hashtag in HASHTAGS:
        since_id = last_ids.get(hashtag)
        posts = fetch_hashtag_posts(hashtag, since_id)
        log.info("‣ #%s: %d neue Posts", hashtag, len(posts))
        all_posts.extend(posts)
    return all_posts

# dedup posts e.g. with multiple hashes
def deduplicate(existing_ids: set, new_posts: list) -> list:
    unique = []
    seen_in_this_batch = set()
    for post in new_posts:
        pid = post.get("id")
        if pid in existing_ids or pid in seen_in_this_batch:
            continue
        seen_in_this_batch.add(pid)
        unique.append(post)
    return unique

# save data to storagefile
def save_data(data_set: list) -> None:
    with open(FILE_LOCATION, "w", encoding="utf-8") as f:
        json.dump(data_set, f, ensure_ascii=False, indent=2)

# definition of main procedure
def run_once() -> None:
    existing_data = load_existing_data()
    existing_ids = {entry.get("id") for entry in existing_data}
    last_ids = get_last_seen_ids(existing_data)

    new_posts = get_hashtag_data(last_ids)
    new_posts = deduplicate(existing_ids, new_posts)

    if new_posts:
        existing_data.extend(new_posts)
        save_data(existing_data)
        log.info("%d neue Posts gespeichert (insgesamt %d).", len(new_posts), len(existing_data))
    else:
        log.info("Keine neuen Posts gefunden.")


def main() -> None:
    log.info("Starte Mastodon-Scraper (Intervall: %ds)", INTERVAL)
    while True:
        try:
            run_once()
        except Exception as e:  # Exeptionhandling for loop continuity 
            log.exception("Unerwarteter Fehler im Durchlauf: %s", e)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()