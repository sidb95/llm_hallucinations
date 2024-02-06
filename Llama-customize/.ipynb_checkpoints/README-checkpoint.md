# Llama-2 fine-tining and deployment

This repository has been created based on a [notebook](https://github.com/philschmid/huggingface-llama-2-samples/blob/master/training/sagemaker-notebook.ipynb) by Phil Schmid. 

The repository contains two main notebooks, which enable customise and interact with a Llama-2 model:
- /training/finateune_llama.ipynb, 
- /training/deploy_llama.ipynb. 

## Setup
1. Ensure that you have access to Llama-2: [link](https://huggingface.co/meta-llama/Llama-2-70b-hf)

## Evaluate
The fine-tuned Llama model was evaluted using [Eleuther AI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). We run the 7 benchmarks, "arc", "hellaswag", "truthfulqa", "mmlu", "winogrande", "gsm8k", "drop", report on huggingface open llm leaderboard [link](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). For comparison, each benchmark was run for the three models: 
- llama_cust_1010: llama model finetuned on 10/10. The model currently deployed to teh students. 
    - huggingface-qlora-2023-10-10-12-58-13-2023-10-10-12-58-17-351/output/model.tar.gz
- llama_cust2: llama model finetened on 10/27. This time instead of setting tokenizer.pad_token = tokenizer.eos_token, we defined tokenizer.pad_token = '[PAD]' 
    - huggingface-qlora-2023-10-27-20-47-45-2023-10-27-20-47-46-997/output/model.tar.gz
- llama-hf: the orginal model hosted by meta, which was the base for the finitunning 
    - meta-llama/Llama-2-13b-hf 

