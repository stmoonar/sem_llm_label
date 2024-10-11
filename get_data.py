import json
from get_author import get_author
from tqdm import tqdm
from time import sleep
import random
import os

root = "145325584"

# 清洗数据
def clean_data(authorId, data):
    cleaned_data = {}
    if 'papers' not in data:
        return clean_data
    papers = data['papers']
    cleaned_data['authorId'] = authorId
    cleaned_data['papers'] = []
    field = "Computer Science"
    for paper in papers:
        if (paper['abstract'] is None) or (paper['fieldsOfStudy'] is None or field not in paper['fieldsOfStudy']):
            continue
        authorIds = [author['authorId'] for author in paper['authors']]
        if authorId in authorIds:
            cleaned_data['papers'].append(paper)

    cleaned_data['papers'] = cleaned_data['papers'][:10]

    return cleaned_data

def run(root, depth):
    if root is None:
        return
    if depth == 3:
        return
    if f"{root}.json" in os.listdir("./authors"):
        data = json.load(open(f"./authors/{root}.json"))
    else:
        sleep(random.randint(1, 3))
        data = get_author(root)
    if 'papers' not in data:
        print(f"[Error] Author {root}, Data: {data}")
        return
    cleaned_data = clean_data(root, data)
    with open(f"./authors/{root}.json", "w") as f:
        json.dump(cleaned_data, f, indent=4)
    for paper in tqdm(cleaned_data['papers'], desc=f"Depth {depth}, Author {root}"):
        for author in paper['authors']:
            authorId = author['authorId']
            run(authorId, depth + 1)

run(root, 0)