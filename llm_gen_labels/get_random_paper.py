import json
import random

# # read '../authors/papers.json'
# with open('../authors/papers.json', 'r') as f:
#     papers = json.load(f)

# # randomly select 10 papers
# random_papers = random.sample(papers, 10)

# # print out the titles and abstracts in JSON format
# random_papers = [{'title': paper['title'], 'abstract': paper['abstract']} for paper in random_papers]
# print(json.dumps(random_papers, indent=4))

def get_random_paper():
    with open('../authors/papers.json', 'r') as f:
        papers = json.load(f)

    random_papers = random.sample(papers, 3)

    random_papers = [{'title': paper['title'], 'abstract': paper['abstract']} for paper in random_papers]
    return json.dumps(random_papers)

def read_prompt():
    with open('prompt.json', 'r') as f:
        return f.read()
