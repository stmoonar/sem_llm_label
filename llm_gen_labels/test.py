from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/home/xxy/data/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

system_prompt = '''你是一个论文分类助手，通过给定论文的标题和摘要，判断其属于以下哪个领域，如果有多个领域，请选择最相关的一个:
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
你只需回答一个数字即可，不要回答任何多余的东西。例如论文属于"Artificial Intelligence and Machine Learning"领域，你只需回答"0".如果同时给你多个论文的标题和摘要列表，你要回答一个列表。如[0, 1]。'''

# Prepare your prompts
prompt = '''{"title": "Tutorials at The Web Conference 2023",
            "abstract": "This paper summarizes the content of the 28 tutorials that have been given at The Web Conference 2023."},'''
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