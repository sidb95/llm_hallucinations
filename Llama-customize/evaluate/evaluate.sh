declare -A tasks
tasks["arc"]="arc_challenge"
tasks["hellaswag"]="hellaswag"
tasks["truthfulqa"]="truthfulqa_mc"
tasks["mmlu"]="hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions"
tasks["winogrande"]="xwinograd_en"
tasks["gsm8k"]="gsm8k"
tasks["drop"]="drop"
tasks_selected=("arc")

declare -A tasks shots
shots["arc"]=25
shots["hellaswag"]=10
shots["truthfulqa"]=0
shots["mmlu"]=5
shots["winogrande"]=5
shots["gsm8k"]=5
shots["drop"]=3

# Before this works you need to run the following command in cmd
# export HUGGINGFACE_TOKEN=your_token_here
# huggingface-cli login --token $HUGGINGFACE_TOKEN

for group_name in "${tasks_selected[@]}"; do
    echo $group_name
    nohup python3 /home/ubuntu/lm-evaluation-harness/main.py \
        --model hf-causal-experimental \
        --model_args pretrained=/home/ubuntu/Llama-customize/model,tokenizer=/home/ubuntu/Llama-customize/model,use_accelerate=True,device_map_option=auto \
        --tasks "${tasks[${group_name}]}" \
        --output_path "/home/ubuntu/Llama-customize/evaluate/results/llama_cust_1010-${group_name}_results.json" \
        --batch_size auto \
        --num_fewshot "${shots[$group_name]}"
done


