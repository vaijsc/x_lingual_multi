# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/alpaca/minillm/tinyllama2-7B/-llama2-7B-llama2-7B/bs2-lr5e-06-G4-N2-NN1-lm1-len512/pe4_rs0.5_nr256_ln_sr_tm0.2/5000 minillm
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/alpaca/sft/tinyllama2/e10-bs8-lr5e-05-G1-N2-NN1/31870 sft
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/llama2/train/alpaca/gkd/tinyllama2-7B/31870 gkd
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/alpaca/seqkd/tinyllama2-7B/31870 seqkd
# bash scripts/phi/eval/run_eval.sh . 0 results/phi/train/alpaca/minillm/phi/stu-phi-1.5/bs1-lr5e-06-G8-N2-NN1-lm1-len1024-lora-16-64-0.1/pe4_rs0.5_nr256_ln_sr_tm0.2/5000 minillm
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T non_distill

##### CNN ########
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 meta-llama/Llama-2-7b-hf cnn_teacher
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/cnn-random/teacher-sft/e5-bs1-lr0.0005-G8-N2-NN1-lora-16-64-0.1/15310 cnn_random_teacher_sft
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/cnn-random/sft/e13-bs2-lr0.0005-G4-N2-NN1-lora-16-64-0.1/39806 cnn_random_sft
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/cnn-random/gkd/9186 cnn_random_gkd
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/cnn-cluster/sft/e13-bs2-lr0.0005-G4-N2-NN1-lora-16-64-0.1/39806 cnn_cluster_sft
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/cnn-cluster/minillm/stu-tiny_cluster/bs1-lr5e-06-G8-N2-NN1-lm1-len1024-lora-16-64-0.1/pe4_rs0.5_nr256_ln_sr_tm0.2/5000 cnn_cluster_minillm
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/cnn-random/minillm/stu-tiny_random/bs1-lr5e-06-G8-N2-NN1-lm1-len1024-lora-16-64-0.1/pe4_rs0.5_nr256_ln_sr_tm0.2/5000 cnn_random_minillm

##### Feedback ######
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/cnn-feedback/sft/e13-bs4-lr0.0005-G2-N1-NN1-lora-16-64-0.1/50960 cnn_feedback_sft
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/cnn-feedback/minillm/stu-tiny_random/bs1-lr5e-06-G8-N1-NN1-lm1-len1024-lora-16-64-0.1/pe4_rs0.5_nr256_ln_sr_tm0.2/5000 cnn_feedback_minillm
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/revise-cnn/sft/e10-bs4-lr0.0005-G2-N1-NN1-lora-16-64-0.1/58370 cnn_revise
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/revise-cnn/sft_cont/stu-tiny_random/e10-bs4-lr0.0005-G2-N1-NN1-lora-16-64-0.1/58370 cnn_revise_cont
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/cnn-fb-v123/sft_cont/stu-tiny_random/e5-bs4-lr0.0005-G2-N2-NN1-lora-16-64-0.1/22660 cnn_fb_v123_cont
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/cnn-strategy-v1/sft_cont/stu-tiny_random/e5-bs4-lr0.0005-G2-N2-NN1-lora-16-64-0.1/17760 cnn_strategy_v1_cont
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/cnn-fb-v23/sft_cont/stu-tiny_random/e5-bs4-lr0.0005-G2-N2-NN1-lora-16-64-0.1/20210 cnn_fb_v23_cont
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/cnn-strategy-v1/sft_cont/stu-tiny_random/e10-bs4-lr0.0005-G2-N2-NN1-lora-16-64-0.1/35520 cnn_strategy_v1_e10
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/cnn-fb-v23/sft_cont/stu-tiny_random/e10-bs4-lr0.0005-G2-N2-NN1-lora-16-64-0.1/40420 cnn_fb_v23_e10

######## SciQ #########
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/sciq/teacher-sft/e5-bs4-lr0.0005-G2-N2-NN1-lora-16-64-0.1/3645 sciq_teacher_sft
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/sciq/sft/e13-bs8-lr0.0005-G1-N2-NN1-lora-16-64-0.1/9477 sciq_sft
# bash scripts/tiny_llama2/eval/run_eval.sh . 0 results/tinyllama2/train/sciq/minillm/stu-tiny_random/bs4-lr5e-06-G2-N2-NN1-lm1-len300-lora-16-64-0.1/pe4_rs0.5_nr256_ln_sr_tm0.2/5000 sciq_minillm

####### Text2SQL ###########
# bash scripts/codes/eval/run_eval.sh . 0 seeklhy/codes-1b-bird sql_sft
# bash scripts/codes/eval/run_eval.sh . 0 seeklhy/codes-15b-spider sql_teacher_sft
# bash scripts/codes/eval/run_eval.sh . 0 results/codes/train/gkd/v0/2344 sql_gkd
# bash scripts/codes/eval/run_eval.sh . 0 results/codes/train/gkd/v1/2340 sql_gkd_v1
bash scripts/codes/eval/run_eval.sh . 0 results/codes/train/gkd/v5/2340 sql_gkd_v5