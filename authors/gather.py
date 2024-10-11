import json
import os
from tqdm import tqdm

# only get json files, and ignore 'papers.json'
files = os.listdir()
files = [file for file in files if file.endswith('.json') and file != 'papers.json']

paperIds = set()
papers = []

for file in tqdm(files):
    with open(file, 'r') as f:
        data = json.load(f)
        for paper in data['papers']:
            if paper['paperId'] not in paperIds:
                papers.append(paper)
                paperIds.add(paper['paperId'])

print(f"Total papers: {len(papers)}")

# # save to papers.json
# with open('papers.json', 'w') as f:
#     json.dump(papers, f)