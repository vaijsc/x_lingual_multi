BASE_PATH=${1}

# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/reward_eval/reward_eval.py \
#     --prompt_path ${BASE_PATH}/data/self-inst/valid.jsonl \
#     --answer1_path ${BASE_PATH}/results/tinyllama2/eval_main/minillm/self_inst/self_inst-512/stu/10/answers.jsonl \
#     --answer2_path ${BASE_PATH}/results/tinyllama2/eval_main/sft/self_inst/self_inst-512/stu/10/answers.jsonl \
#     --load_data_answer_1 ${BASE_PATH}/eval_with_reward/self_inst/minillm.csv \
#     --model_1_type_id minillm \
#     --model_2_type_id sft \
#     --data_name self_inst \
#     --output_path ${BASE_PATH}/eval_with_reward/self_inst

# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/reward_eval/reward_eval.py \
#     --prompt_path ${BASE_PATH}/data/sinst/11_/valid.jsonl \
#     --answer1_path ${BASE_PATH}/results/tinyllama2/eval_main/minillm/sinst_11_/sinst_11_-512/stu/10/answers.jsonl \
#     --answer2_path ${BASE_PATH}/results/tinyllama2/eval_main/sft/sinst_11_/sinst_11_-512/stu/10/answers.jsonl \
#     --model_1_type_id minillm \
#     --model_2_type_id sft \
#     --data_name sinst \
#     --output_path ${BASE_PATH}/eval_with_reward/sinst

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/reward_eval/reward_eval.py \
    --prompt_path ${BASE_PATH}/data/uinst/11_/valid.jsonl \
    --answer1_path ${BASE_PATH}/results/tinyllama2/eval_main/minillm/uinst_11_/uinst_11_-512/stu/10/answers.jsonl \
    --answer2_path ${BASE_PATH}/results/tinyllama2/eval_main/sft/uinst_11_/uinst_11_-512/stu/10/answers.jsonl \
    --model_1_type_id minillm \
    --model_2_type_id sft \
    --data_name uinst \
    --output_path ${BASE_PATH}/eval_with_reward/uinst

# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/reward_eval/reward_eval.py \
#     --prompt_path ${BASE_PATH}/data/vicuna/valid.jsonl \
#     --answer1_path ${BASE_PATH}/results/tinyllama2/eval_main/minillm/vicuna/vicuna-512/stu/10/answers.jsonl \
#     --answer2_path ${BASE_PATH}/results/tinyllama2/eval_main/sft/vicuna/vicuna-512/stu/10/answers.jsonl \
#     --load_data_answer_1 ${BASE_PATH}/eval_with_reward/vicuna/minillm.csv \
#     --model_1_type_id minillm \
#     --model_2_type_id sft \
#     --data_name vicuna \
#     --output_path ${BASE_PATH}/eval_with_reward/vicuna
    