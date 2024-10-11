from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# model_path = "/home/xxy/data/Qwen2.5-3B-Instruct"
# model_path = "/data/models/Llama-3.2-3B-Instruct"
model_path = "/data/models/Qwen2.5-7B-Instruct"

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model=model_path, gpu_memory_utilization=0.2, tensor_parallel_size=4)
# llm = LLM(model=model_path, gpu_memory_utilization=0.2)

system_prompt = '''你是一个论文分类助手，通过给定论文的标题和摘要(json格式)，判断其属于以下哪个领域，如果有多个领域，请选择最相关的一个:
[
    {
        "0": "Artificial Intelligence and Machine Learning"
    },
    {
        "1": "Data Science and Big Data"
    },
    {
        "2": "Computer Networks and Distributed Systems"
    },
    {
        "3": "Computer Systems and Architecture"
    },
    {
        "4": "Software Engineering and Programming Languages"
    },
    {
        "5": "Human-Computer Interaction and Virtual Reality"
    },
    {
        "6": "Theoretical Computer Science"
    },
    {
        "7": "Computer Graphics and Image Processing"
    }
]
你只需回答一个数字即可，不要回答任何多余的东西。例如论文属于Artificial Intelligence and Machine Learning领域，你只需回答"0"。如果同时给你多个论文的标题和摘要列表，你也回答一个对应长度的列表，格式[0, 1]。'''

# Prepare your prompts
prompt = '''[
{"title": "Tutorials at The Web Conference 2023",
            "abstract": "This paper summarizes the content of the 28 tutorials that have been given at The Web Conference 2023."},
{"title": "Finding Densest Subgraphs with Edge-Color Constraints",
            "abstract": "We consider a variant of the densest subgraph problem in networks with single or multiple edge attributes. For example, in a social network, the edge attributes may describe the type of relationship between users, such as friends, family, or acquaintances, or different types of communication. For conceptual simplicity, we view the attributes as edge colors. The new problem we address is to find a diverse densest subgraph that fulfills given requirements on the numbers of edges of specific colors. When searching for a dense social network community, our problem will enforce the requirement that the community is diverse according to criteria specified by the edge attributes. We show that the decision versions for finding exactly, at most, and at least h colored edges densest subgraph, where h is a vector of color requirements, are NP-complete, for already two colors. For the problem of finding a densest subgraph with at least h colored edges, we provide a linear-time constant-factor approximation algorithm when the input graph is sparse. On the way, we introduce the related at least h (non-colored) edges densest subgraph problem, show its hardness, and also provide a linear-time constant-factor approximation. In our experiments, we demonstrate the efficacy and efficiency of our new algorithms."},
{"title": "Toward Tweet Entity Linking With Heterogeneous Information Networks",
            "abstract": "Twitter, a microblogging platform, has developed into an increasingly invaluable information source, where millions of users post a great quantity of tweets with various topics per day. Heterogeneous information networks consisting of multi-type objects and relations are becoming more and more prevalent as an organization form of knowledge and information. The task of linking an entity mention in a tweet with its corresponding entity in a heterogeneous information network is of great importance, for the purpose of enriching heterogeneous information networks with the abundant and fresh knowledge embedded in tweets. However, the entity mention is ambiguous. Additionally, tweets are short and informal, making it difficult to mine enough information from a single tweet for entity linking. In this paper, we propose an unsupervised iterative clustering framework TELHIN to link multiple similar tweets with a heterogeneous information network jointly. Our framework takes three dimensions of tweet similarity into consideration: (1) content similarity, (2) temporal similarity, and (3) user similarity. The appropriate weights of different similarity dimensions for each entity mention are learned iteratively based on the metric learning algorithm by leveraging the pairwise constraints generated automatically. Experiments on real data demonstrate the effectiveness of our framework in comparison with the baselines."},
]'''
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# generate outputs
outputs = llm.generate([text], sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print(generated_text)
