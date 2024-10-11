import requests
from get_random_paper import get_random_paper, read_prompt


API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/829d82ec33fa53aa709ccdde59329bdf/ai/run/"
headers = {"Authorization": "Bearer sohWm6dfOSZ-zP5Z8Eng7bq--djUEenjPzF-WAtx"}


def run(model, inputs):
    input = { "messages": inputs }
    response = requests.post(f"{API_BASE_URL}{model}", headers=headers, json=input)
    return response.json()

system_prompt = 'You are a paper classification assistant. Given the title and abstract of a paper (in JSON format), determine which of the following fields it belongs to. If there are multiple fields, choose the most relevant one: [{"0": "Artificial Intelligence and Machine Learning"}, {"1": "Data Science and Big Data"}, {"2": "Computer Networks and Distributed Systems"}, {"3": "Computer Systems and Architecture"}, {"4": "Software Engineering and Programming Languages"}, {"5": "Human-Computer Interaction and Virtual Reality"}, {"6": "Theoretical Computer Science"}, {"7": "Computer Graphics and Image Processing"}], You only need to respond with a single number, without any additional information. For example, if the paper belongs to the "Artificial Intelligence and Machine Learning" field, you should respond with "0". If you are given a list of titles and abstracts, respond with a corresponding list of numbers, e.g., [0, 1].'

# system_prompt = 'You are a paper classification assistant. Given the title and abstract of a paper (in JSON format), determine which of the following fields it belongs to. Consider the definitions provided for each field when making your decision. If there are multiple fields, choose the most relevant one: [{"0": "Artificial Intelligence and Machine Learning: Includes research on AI models, neural networks, machine learning algorithms, and their applications in tasks like classification, prediction, and decision-making."}, {"1": "Data Science and Big Data: Focuses on data analysis techniques, large-scale data processing, data mining, and statistical modeling, as well as handling massive datasets."}, {"2": "Computer Networks and Distributed Systems: Involves topics like network protocols, communication systems, distributed algorithms, cloud computing, and Internet architecture."}, {"3": "Computer Systems and Architecture: Covers hardware design, low-level systems engineering, parallel computing, microprocessor and microcontroller development, and performance optimization."}, {"4": "Software Engineering and Programming Languages: Encompasses software development methodologies, software testing, programming paradigms, compiler design, and formal methods for software verification."}, {"5": "Human-Computer Interaction and Virtual Reality: Includes user interface design, usability studies, virtual environments, augmented reality, and human-centered computing."}, {"6": "Theoretical Computer Science: Focuses on computational theory, algorithms, formal logic, computational complexity, cryptography, and mathematical aspects of computing."}, {"7": "Computer Graphics and Image Processing: Deals with image analysis, 3D modeling, rendering, visualization, and the development of algorithms for manipulating graphical data."}] You only need to respond with a single number, without any additional information. If you are given a list of titles and abstracts, respond with a corresponding list of numbers, e.g., [0, 1].'

# # Prepare your prompts
# prompt = '''[
# {"title": "A Surprisingly Simple yet Effective Multi-Query Rewriting Method for Conversational Passage Retrieval",
#             "abstract": "Conversational passage retrieval is challenging as it often requires the resolution of references to previous utterances and needs to deal with the complexities of natural language, such as coreference and ellipsis. To address these challenges, pre-trained sequence-to-sequence neural query rewriters are commonly used to generate a single de-contextualized query based on conversation history. Previous research shows that combining multiple query rewrites for the same user utterance has a positive effect on retrieval performance. We propose the use of a neural query rewriter to generate multiple queries and show how to integrate those queries in the passage retrieval pipeline efficiently. The main strength of our approach lies in its simplicity: it leverages how the beam search algorithm works and can produce multiple query rewrites at no additional cost. Our contributions further include devising ways to utilize multi-query rewrites in both sparse and dense first-pass retrieval. We demonstrate that applying our approach on top of a standard passage retrieval pipeline delivers state-of-the-art performance without sacrificing efficiency."},
# ]'''

# prompt = get_random_paper()
# with open('prompt.json', 'w') as f:
#     f.write(prompt)

prompt = read_prompt()

inputs = [
    { "role": "system", "content": system_prompt },
    { "role": "user", "content": prompt }
];
output = run("@cf/meta/llama-3.1-8b-instruct", inputs)
print(output)