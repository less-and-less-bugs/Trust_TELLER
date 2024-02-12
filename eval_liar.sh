# for baselines flan-t5 series

# use knowledge
#python drive_liar.py --model_name "flan-t5-xl" --evi_flag --eval_mode "logics" --device "cpu" --dataset_name "LIAR-PLUS"
#python drive_liar.py --model_name "flan-t5-xxl" --evi_flag --eval_mode "logics" --device "cpu" --dataset_name "LIAR-PLUS"
#python drive_liar.py --model_name "flan-t5-large" --evi_flag --eval_mode "logics" --dataset_name "LIAR-PLUS"
#python drive_liar.py --model_name "flan-t5-base" --evi_flag --eval_mode "logics" --dataset_name "LIAR-PLUS"
#python drive_liar.py --model_name "flan-t5-small" --evi_flag --eval_mode "logics" --dataset_name "LIAR-PLUS"

# not use knowledge
#python drive_liar.py --model_name "flan-t5-xl"  --eval_mode "logics" --dataset_name "LIAR-PLUS"
#python drive_liar.py --model_name "flan-t5-xxl"  --eval_mode "logics" --device "cpu" --dataset_name "LIAR-PLUS"
#python drive_liar.py --model_name "flan-t5-large" --eval_mode "logics" --dataset_name "LIAR-PLUS"
#python drive_liar.py --model_name "flan-t5-base" --eval_mode "logics" --dataset_name "LIAR-PLUS"
#python drive_liar.py --model_name "flan-t5-small" --eval_mode "logics" --dataset_name "LIAR-PLUS"
#python drive_liar.py --model_name "flan-t5-xl"  --eval_mode "sampling" --dataset_name "LIAR-PLUS"
#python drive_liar.py --model_name "flan-t5-xxl"  --eval_mode "sampling" --dataset_name "LIAR-PLUS"
#python drive_liar.py --model_name "flan-t5-large"  --eval_mode "sampling" --dataset_name "LIAR-PLUS"
#python drive_liar.py --model_name "flan-t5-base"  --eval_mode "sampling" --dataset_name "LIAR-PLUS"
#python drive_liar.py --model_name "flan-t5-small"  --eval_mode "sampling" --dataset_name "LIAR-PLUS"
#
# for baselines llama2 series binary classification
# not use knowledge
#python drive_liar.py --model_name "Llama-2-7b-chat-hf"  --eval_mode "logics" --device "cuda" --mode "binary" --dataset_name "LIAR-PLUS"
#python drive_liar.py --model_name "Llama-2-13b-chat-hf"  --eval_mode "logics" --device "cpu" --dataset_name "LIAR-PLUS"
# use knowledge
#python drive_liar.py --model_name "Llama-2-7b-chat-hf"  --evi_flag  --eval_mode "logics" --device "cuda" --dataset_name "LIAR-PLUS"
#python drive_liar.py --model_name "Llama-2-13b-chat-hf"  --evi_flag  --eval_mode "logics" --device "cpu" --dataset_name "LIAR-PLUS"
# for baselines llama2 series multi-classification classification
## not use knowledge
#python drive_liar.py --model_name "Llama-2-13b-chat-hf"  --eval_mode "logics" --device "cpu" --mode "multiple" --dataset_name "LIAR-PLUS"
## use knowledge
#python drive_liar.py --model_name "Llama-2-7b-chat-hf"  --eval_mode "logics" --device "cuda" --mode "multiple" --evi_flag --dataset_name "LIAR-PLUS"
#python drive_liar.py --model_name "Llama-2-13b-chat-hf"  --eval_mode "logics" --device "cpu" --mode "multiple" --evi_flag --dataset_name "LIAR-PLUS"
#

# for baselines turbo series
# not use knowledge
#python drive_liar.py --model_name "Llama-2-7b-chat-hf"  --eval_mode "logics" --device "cuda" --mode "binary "--dataset_name "LIAR-PLUS"
#python drive_liar.py --model_name "Llama-2-7b-chat-hf"  --eval_mode "logics" --device "cuda" --mode "multiple" --evi_flag --dataset_name "LIAR-PLUS"
#python drive_liar.py --model_name "Llama-2-7b-chat-hf"  --evi_flag  --eval_mode "logics" --device "cuda" --dataset_name "LIAR-PLUS"
# use knowledge
#python drive_liar.py --model_name "Llama-2-7b-chat-hf"  --eval_mode "logics" --device "cuda" --mode "binary" --evi_flag --dataset_name "LIAR-PLUS"
#python drive_liar.py --model_name "Llama-2-7b-chat-hf"  --eval_mode "logics" --device "cuda" --mode "multiple" --evi_flag --dataset_name "LIAR-PLUS"

# for our logic model
# first generate special question
# then generate general question

##### for logic model
# for LIAR-PLUS dataset
#python drive_liar.py --n_out 2 --eval_mode "logics" --device "cuda" --save_path "2false_logic_model.json" --dataset_name "LIAR-PLUS"
#python drive_liar.py --n_out 6 --eval_mode "logics" --device "cuda" --save_path "2multi_false_logic_model.json" --dataset_name "LIAR-PLUS"
#python drive_liar.py --n_out 2 --eval_mode "logics" --device "cuda" --save_path "2true_logic_model.json" --evi_flag --dataset_name "LIAR-PLUS"
#python drive_liar.py --n_out 6 --eval_mode "logics" --device "cuda" --save_path "2multi_true_logic_model.json" --evi_flag --dataset_name "LIAR-PLUS"

## Evo
#python drive_liar.py --n_out 2  --eval_mode "logics" --device="cuda" --evo_flag --dataset_name "LIAR-PLUS" --save_path "1evo_true_logic_model.json" --evi_flag
#python drive_liar.py --n_out 2   --eval_mode "logics" --device="cuda" --evo_flag --dataset_name "LIAR-PLUS" --save_path "1evo_false_logic_model.json"
#python drive_liar.py --n_out 6 --eval_mode "logics" --device="cuda" --evo_flag --dataset_name "LIAR-PLUS" --save_path "1evo_multi_false_logic_model.json"
#python drive_liar.py --n_out 6 --eval_mode "logics" --device="cuda" --evo_flag --dataset_name "LIAR-PLUS" --save_path "1evo_multi_true_logic_model.json" --evi_flag



# for constraint dataset
#python drive_liar.py --n_out 2 --eval_mode "logics" --device "cuda" --save_path "4false_logic_model.json" --dataset_name "Constraint"
#python drive_liar.py --n_out 2 --eval_mode "logics" --device "cuda" --save_path "4evo_false_logic_model.json" --dataset_name "Constraint" --evo_flag


python drive_liar.py --n_out 2 --eval_mode "logics" --device "cuda" --save_path "true_logic_model.json" --evi_flag --dataset_name "Constraint"

# for POLITIFACT dataset
#python drive_liar.py --n_out 2 --eval_mode "logics" --device "cuda" --save_path "5false_logic_model.json" --dataset_name "POLITIFACT"
#python drive_liar.py --n_out 2 --eval_mode "logics" --device "cuda" --save_path "4evo_false_logic_model.json" --dataset_name "POLITIFACT" --evo_flag
#python drive_liar.py --n_out 2 --eval_mode "logics" --device "cuda" --save_path "true_logic_model.json" --evi_flag --dataset_name "POLITIFACT"

# for GOSSIPCOP dataset
#python drive_liar.py --n_out 2 --eval_mode "logics" --device "cuda" --save_path "4false_logic_model.json" --dataset_name "GOSSIPCOP"
#python drive_liar.py --n_out 2 --eval_mode "logics" --device "cuda" --save_path "4evo_false_logic_model.json" --dataset_name "GOSSIPCOP"  --evo_flag

#python drive_liar.py --n_out 2 --eval_mode "logics" --device "cuda" --save_path "true_logic_model.json" --evi_flag --dataset_name "GOSSIPCOP"







##### for baseline

# for constraint dataset
## close-world setting
#python drive_liar.py --dataset_name "Constraint"  --model_name "flan-t5-xl"  --eval_mode "logics" --device "cuda"
#python drive_liar.py --dataset_name "Constraint"  --model_name "flan-t5-xxl"  --eval_mode "logics" --device "cuda"
#python drive_liar.py --dataset_name "Constraint"  --model_name "flan-t5-large" --eval_mode "logics" --device "cuda"
#python drive_liar.py --dataset_name "Constraint"  --model_name "Llama-2-7b-chat-hf"  --eval_mode "logics" --device "cuda"
#python drive_liar.py --dataset_name "Constraint"  --model_name "Llama-2-13b-chat-hf"  --eval_mode "logics" --device "cuda"

## open-world setting
#python drive_liar.py --dataset_name "Constraint"  --model_name "flan-t5-xl"  --eval_mode "logics" --device "cuda" --evi_flag
#python drive_liar.py --dataset_name "Constraint"  --model_name "flan-t5-xxl"  --eval_mode "logics" --device "cuda" --evi_flag
#python drive_liar.py --dataset_name "Constraint"  --model_name "flan-t5-large" --eval_mode "logics" --device "cuda" --evi_flag
#python drive_liar.py --dataset_name "Constraint"  --model_name "Llama-2-7b-chat-hf"  --eval_mode "logics" --device "cuda" --evi_flag
#python drive_liar.py --dataset_name "Constraint"  --model_name "Llama-2-13b-chat-hf"  --eval_mode "logics" --device "cuda" --evi_flag

# for POLITIFACT dataset
## close-world setting
#python drive_liar.py --dataset_name "POLITIFACT"  --model_name "flan-t5-xl"  --eval_mode "logics" --device "cuda"
#python drive_liar.py --dataset_name "POLITIFACT"  --model_name "flan-t5-xxl"  --eval_mode "logics" --device "cuda"
#python drive_liar.py --dataset_name "POLITIFACT"  --model_name "flan-t5-large" --eval_mode "logics" --device "cuda"
#python drive_liar.py --dataset_name "POLITIFACT" --model_name "Llama-2-7b-chat-hf"  --eval_mode "logics" --device "cuda"
#python drive_liar.py --dataset_name "POLITIFACT"  --model_name "Llama-2-13b-chat-hf"  --eval_mode "logics" --device "cuda"

## open-world setting
#python drive_liar.py --dataset_name "POLITIFACT"  --model_name "flan-t5-xl"  --eval_mode "logics" --device "cuda" --evi_flag
#python drive_liar.py --dataset_name "POLITIFACT"  --model_name "flan-t5-xxl"  --eval_mode "logics" --device "cuda" --evi_flag
#python drive_liar.py --dataset_name "POLITIFACT"  --model_name "flan-t5-large" --eval_mode "logics" --device "cuda" --evi_flag
#python drive_liar.py --dataset_name "POLITIFACT" --model_name "Llama-2-7b-chat-hf"  --eval_mode "logics" --device "cuda" --evi_flag
#python drive_liar.py --dataset_name "POLITIFACT"  --model_name "Llama-2-13b-chat-hf"  --eval_mode "logics" --device "cuda" --evi_flag


# for GOSSIPCOP dataset
## close-world setting
#python drive_liar.py --dataset_name "GOSSIPCOP"  --model_name "flan-t5-large" --eval_mode "logics" --device "cuda"
#python drive_liar.py --dataset_name "GOSSIPCOP"  --model_name "flan-t5-xl"  --eval_mode "logics" --device "cuda"
#python drive_liar.py --dataset_name "GOSSIPCOP"  --model_name "flan-t5-xxl"  --eval_mode "logics" --device "cuda"
#python drive_liar.py --dataset_name "GOSSIPCOP" --model_name "Llama-2-7b-chat-hf"  --eval_mode "logics" --device "cuda"
#python drive_liar.py --dataset_name "GOSSIPCOP"  --model_name "Llama-2-13b-chat-hf"  --eval_mode "logics" --device "cuda"

## open-world setting
#python drive_liar.py --dataset_name "GOSSIPCOP"  --model_name "flan-t5-large" --eval_mode "logics" --device "cuda" --evi_flag
#python drive_liar.py --dataset_name "GOSSIPCOP"  --model_name "flan-t5-xl"  --eval_mode "logics" --device "cuda" --evi_flag
#python drive_liar.py --dataset_name "GOSSIPCOP"  --model_name "flan-t5-xxl"  --eval_mode "logics" --device "cuda" --evi_flag
#python drive_liar.py --dataset_name "GOSSIPCOP" --model_name "Llama-2-7b-chat-hf"  --eval_mode "logics" --device "cuda" --evi_flag
#python drive_liar.py --dataset_name "GOSSIPCOP"  --model_name "Llama-2-13b-chat-hf"  --eval_mode "logics" --device "cuda" --evi_flag