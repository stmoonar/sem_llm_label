from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_path = "/home/xxy/data/Qwen2.5-3B-Instruct"

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model=model_path, dtype="float16")

system_prompt = '''你是一个论文分类助手，通过给定论文的标题和摘要将其归为以下几个领域中的一个。:
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
你只需回答一个数字即可，不要回答任何多余的东西。例如论文属于Artificial Intelligence and Machine Learning领域，你只需回答"0"。如果同时给你一个列表，你也回答一个列表。如[0, 1]。'''

# Prepare your prompts
prompt = '''[

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
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
