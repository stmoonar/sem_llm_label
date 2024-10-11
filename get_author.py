import requests
import json
from time import sleep
import random

def get_author(authorId):
    url = f"https://api.semanticscholar.org/graph/v1/author/{authorId}"
    query_params = {
            # 'fields': 'title,year,abstract,fieldsOfStudy,authors',
            'fields': 'papers.title,papers.fieldsOfStudy,papers.abstract,papers.authors,papers.publicationTypes',
    }
    headers = {
        'x-api-key': "n9k3NKbRXh9Nkl95rkMtW1ZFNakKUa211QPe0E5Z"
    }
    
    retries = 10
    delay = random.randint(1, 3)
    for i in range(retries):
        try:
            response = requests.get(url, params=query_params, headers=headers)
            if response.status_code == 200:
                break
            else:
                print(f"Failed to get {authorId}, retrying...")
                sleep(delay)
        except:
            print(f"Failed to get {authorId}, retrying...")
            sleep(delay)

    return response.json()

# data = get_author("145325584")

# with open("author.json", "w") as f:
#     json.dump(data, f, indent=4)