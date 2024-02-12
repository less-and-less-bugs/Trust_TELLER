# For LIAR Dataset
## No evidence
#python General_Questions.py --model_name="flan-t5-large"  --openbook="False" --device="cuda"
#python General_Questions.py --model_name="flan-t5-xxl"  --openbook="False" --device="cuda"
#python General_Questions.py --model_name="flan-t5-xl"  --openbook="False" --device="cuda"
#python General_Questions.py --model_name="Llama-2-7b-chat-hf"  --openbook="False" --device="cuda"
#python General_Questions.py --model_name="Llama-2-13b-chat-hf"  --openbook="False" --device="cuda"


## evidence
#python General_Questions.py --model_name="flan-t5-xxl"  --openbook="True" --device="cpu"
#python General_Questions.py --model_name="flan-t5-xl"  --openbook="True" --device="cpu" &
#python General_Questions.py --model_name="flan-t5-large"  --openbook="True" --device="cuda"
#python General_Questions.py --model_name="Llama-2-7b-chat-hf"  --openbook="True" --device="cuda"
#python General_Questions.py --model_name="Llama-2-13b-chat-hf"  --openbook="True" --device="cuda"

#python General_Questions.py --model_name="gpt-3.5-turbo"  --openbook="True" --device="cuda" --mode="sampling"
#python General_Questions.py --model_name="gpt-3.5-turbo"  --openbook="False" --device="cuda" --mode="sampling"



## Evo
#python General_Questions.py --model_name="flan-t5-large"  --openbook="True" --device="cuda" --evo_flag
#python General_Questions.py --model_name="flan-t5-large"  --openbook="False" --device="cuda" --evo_flag
#python General_Questions.py --model_name="flan-t5-xl"  --openbook="True" --device="cuda" --evo_flag
#python General_Questions.py --model_name="flan-t5-xl"  --openbook="False" --device="cuda" --evo_flag
#python General_Questions.py --model_name="flan-t5-xxl"  --openbook="True" --device="cuda" --evo_flag
#python General_Questions.py --model_name="flan-t5-xxl"  --openbook="False" --device="cuda" --evo_flag
#python General_Questions.py --model_name="Llama-2-7b-chat-hf"  --openbook="True" --device="cuda" --evo_flag
#python General_Questions.py --model_name="Llama-2-7b-chat-hf"  --openbook="False" --device="cuda" --evo_flag
#python General_Questions.py --model_name="Llama-2-13b-chat-hf"  --openbook="True" --device="cuda" --evo_flag
#python General_Questions.py --model_name="Llama-2-13b-chat-hf"  --openbook="False" --device="cuda" --evo_flag

# Constraint
## No evidence
#python General_Questions.py --model_name="flan-t5-xxl"  --openbook="False"  --device="cuda" --dataset_name "Constraint"
#python General_Questions.py --model_name="flan-t5-xl"   --openbook="False"  --device="cuda" --dataset_name "Constraint"
#python General_Questions.py --model_name="flan-t5-large" --openbook="False" --device="cuda" --dataset_name "Constraint"
#python General_Questions.py --model_name="Llama-2-13b-chat-hf"  --openbook="False"  --device="cuda" --dataset_name "Constraint"
#python General_Questions.py --model_name="Llama-2-7b-chat-hf"  --openbook="False" --device="cuda" --dataset_name "Constraint"
## evidence
#python General_Questions.py --model_name="flan-t5-xxl"  --openbook="True"  --device="cuda" --dataset_name "Constraint"
#python General_Questions.py --model_name="flan-t5-xl"   --openbook="True"   --device="cuda" --dataset_name "Constraint"
#python General_Questions.py --model_name="flan-t5-large" --openbook="True"  --device="cuda" --dataset_name "Constraint"
#python General_Questions.py --model_name="Llama-2-13b-chat-hf"  --openbook="True"   --device="cuda" --dataset_name "Constraint"
#python General_Questions.py --model_name="Llama-2-7b-chat-hf"  --openbook="True"  --device="cuda" --dataset_name "Constraint"

## Evo
#python General_Questions.py --model_name="flan-t5-large" --openbook="False" --device="cuda" --dataset_name "Constraint" --evo_flag
#python General_Questions.py --model_name="flan-t5-xl"   --openbook="False"  --device="cuda" --dataset_name "Constraint" --evo_flag
#python General_Questions.py --model_name="flan-t5-xxl"  --openbook="False"  --device="cuda" --dataset_name "Constraint" --evo_flag
#python General_Questions.py --model_name="Llama-2-13b-chat-hf"  --openbook="False"  --device="cuda" --dataset_name "Constraint" --evo_flag
#python General_Questions.py --model_name="Llama-2-7b-chat-hf"  --openbook="False" --device="cuda" --dataset_name "Constraint" --evo_flag


# POLITIFACT
## No evidence xxl ok

#python General_Questions.py --model_name="flan-t5-large" --openbook="False" --device="cuda" --dataset_name "POLITIFACT"
#python General_Questions.py --model_name="flan-t5-xxl"  --openbook="False"  --device="cuda" --dataset_name "POLITIFACT"
#python General_Questions.py --model_name="flan-t5-xl"   --openbook="False"  --device="cuda" --dataset_name "POLITIFACT"
#python General_Questions.py --model_name="Llama-2-7b-chat-hf"  --openbook="False" --device="cuda" --dataset_name "POLITIFACT"
python General_Questions.py --model_name="Llama-2-13b-chat-hf"  --openbook="False"  --device="cuda" --dataset_name "POLITIFACT"

## evidence
#python General_Questions.py --model_name="flan-t5-xxl"  --openbook="True"  --device="cuda" --dataset_name "POLITIFACT"
#python General_Questions.py --model_name="flan-t5-xl"   --openbook="True"  --device="cuda" --dataset_name "POLITIFACT"
#python General_Questions.py --model_name="flan-t5-large" --openbook="True" --device="cuda" --dataset_name "POLITIFACT"
#python General_Questions.py --model_name="Llama-2-7b-chat-hf"  --openbook="True" --device="cuda" --dataset_name "POLITIFACT"
#python General_Questions.py --model_name="Llama-2-13b-chat-hf"  --openbook="True"  --device="cuda" --dataset_name "POLITIFACT"

## Evo
#python General_Questions.py --model_name="flan-t5-large" --openbook="False" --device="cuda" --dataset_name "POLITIFACT" --evo_flag
#python General_Questions.py --model_name="flan-t5-xxl"  --openbook="False"  --device="cuda" --dataset_name "POLITIFACT" --evo_flag
#python General_Questions.py --model_name="flan-t5-xl"   --openbook="False"  --device="cuda" --dataset_name "POLITIFACT" --evo_flag
#python General_Questions.py --model_name="Llama-2-7b-chat-hf"  --openbook="False" --device="cuda" --dataset_name "POLITIFACT" --evo_flag
#python General_Questions.py --model_name="Llama-2-13b-chat-hf"  --openbook="False"  --device="cuda" --dataset_name "POLITIFACT" --evo_flag


# GOSSIPCOP
## No evidence
#python General_Questions.py --model_name="flan-t5-xxl"  --openbook="False"  --device="cuda" --dataset_name "GOSSIPCOP"
#python General_Questions.py --model_name="flan-t5-xl"   --openbook="False"  --device="cuda" --dataset_name "GOSSIPCOP"
#python General_Questions.py --model_name="flan-t5-large" --openbook="False" --device="cuda" --dataset_name "GOSSIPCOP"
#python General_Questions.py --model_name="Llama-2-7b-chat-hf"  --openbook="False"  --device="cuda" --dataset_name "GOSSIPCOP"
#python General_Questions.py --model_name="Llama-2-13b-chat-hf"  --openbook="False" --device="cuda" --dataset_name "GOSSIPCOP"

## evidence
#python General_Questions.py --model_name="flan-t5-xxl"  --openbook="True"  --device="cuda" --dataset_name "GOSSIPCOP"
#python General_Questions.py --model_name="flan-t5-xl"   --openbook="True"  --device="cuda" --dataset_name "GOSSIPCOP"
#python General_Questions.py --model_name="flan-t5-large" --openbook="True" --device="cuda" --dataset_name "GOSSIPCOP"
#python General_Questions.py --model_name="Llama-2-7b-chat-hf"  --openbook="True"  --device="cuda" --dataset_name "GOSSIPCOP"
#python General_Questions.py --model_name="Llama-2-13b-chat-hf"  --openbook="True" --device="cuda" --dataset_name "GOSSIPCOP"

## Evo
#
#python General_Questions.py --model_name="flan-t5-large" --openbook="False" --device="cuda" --dataset_name "GOSSIPCOP" --evo_flag
#python General_Questions.py --model_name="flan-t5-xl"   --openbook="False"  --device="cuda" --dataset_name "GOSSIPCOP" --evo_flag
#python General_Questions.py --model_name="flan-t5-xxl"  --openbook="False"  --device="cuda" --dataset_name "GOSSIPCOP" --evo_flag
#python General_Questions.py --model_name="Llama-2-13b-chat-hf"  --openbook="False"  --device="cuda" --dataset_name "GOSSIPCOP" --evo_flag
#python General_Questions.py --model_name="Llama-2-7b-chat-hf"  --openbook="False" --device="cuda" --dataset_name "GOSSIPCOP" --evo_flag