import os
import time
import json
import requests

INSTANCE = "https://mastodon.social"



url = f"{INSTANCE}/api/v1/timelines/tag/koeln"
response = requests.get(url, timeout=15) 

print(response.text)