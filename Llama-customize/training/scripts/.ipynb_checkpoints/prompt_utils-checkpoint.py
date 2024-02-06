"""
A dedicated helper script to manage templates and prompt building.
"""

system_prompt_tutor = "You are a helpful, respectful and honest MBA Graduate Teaching Assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.If you don't know the answer to a question, please don't share false information."

system_prompt_tutor_2 = "You are a helpful, respectful and honest MBA Graduate Teaching Assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.If you don't know the answer to a question, please don't share false information. If the communication is outside the accounting domain then answer as helpful as possible."

system_prompt_descriptor = "Given a set of metadata about a product, craft a vibrant and engaging description. The description should transform the provided features into an alluring narrative, emphasizing the product's unique qualities. Focus on highlighting the practicality and quality of the product, along with any unique selling points. Your goal is to make the description irresistible to potential buyers. Remember to maintain a positive and persuasive tone throughout."


def format_prompt_instruct(user_instruction="", answer="", system_prompt=""):
    assert len(system_prompt), "No system prompt provided."
    assert len(user_instruction), "No user instruction provided."
    
    prompt = f"""<s>[INST] <<SYS>>
    {system_prompt}
    <</SYS>>

    {user_instruction} [/INST] {answer}"""
    return prompt

def format_prompt(sample_instruction = "", sample_input ="", sample_output=""):
    assert len(sample_instruction), "No instruction provided."
    
    instruction = f"### Instruction\n{sample_instruction}"
    context = f"### Input\n{sample_input}" if len(sample_input) > 0 else None
    response = f"### Answer\n{sample_output}"
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
    return prompt