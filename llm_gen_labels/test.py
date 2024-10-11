from transformers import AutoModelForCausalLM, AutoTokenizer

# model_path = "/home/xxy/data/Qwen2.5-3B-Instruct"
model_path = "/data/models/Llama-3.2-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

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
你只需回答一个数字即可，不要回答任何多余的东西。例如论文属于"Artificial Intelligence and Machine Learning"领域，你只需回答"0"。如果同时给你多个论文的标题和摘要列表，你也回答一个对应长度的列表，格式[0, 1]。'''

# Prepare your prompts
prompt = '''[
{"title": "A Surprisingly Simple yet Effective Multi-Query Rewriting Method for Conversational Passage Retrieval",
            "abstract": "Conversational passage retrieval is challenging as it often requires the resolution of references to previous utterances and needs to deal with the complexities of natural language, such as coreference and ellipsis. To address these challenges, pre-trained sequence-to-sequence neural query rewriters are commonly used to generate a single de-contextualized query based on conversation history. Previous research shows that combining multiple query rewrites for the same user utterance has a positive effect on retrieval performance. We propose the use of a neural query rewriter to generate multiple queries and show how to integrate those queries in the passage retrieval pipeline efficiently. The main strength of our approach lies in its simplicity: it leverages how the beam search algorithm works and can produce multiple query rewrites at no additional cost. Our contributions further include devising ways to utilize multi-query rewrites in both sparse and dense first-pass retrieval. We demonstrate that applying our approach on top of a standard passage retrieval pipeline delivers state-of-the-art performance without sacrificing efficiency."}
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
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)