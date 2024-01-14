nohup python train.py --epochs 150 --model VQAGAP_bert_Model --dataset LR --gpu 3 > ../data/logs/LR/vqa_gap_bert_model_log_epoch_150.txt &
nohup python train.py --epochs 150 --model VQAGAP_bert_Model --dataset HR --gpu 5 > ../data/logs/HR/vqa_gap_bert_model_log_epoch_150.txt &
nohup python train.py --epochs 150 --model VQAGAP_qbert_Model --dataset LR --gpu 3 > ../data/logs/LR/vqa_gap_qbert_model_log_epoch_150.txt &
nohup python train.py --epochs 150 --model VQAGAP_qbert_Model --dataset HR --gpu 7 > ../data/logs/HR/vqa_gap_qbert_model_log_epoch_150.txt &

nohup python train.py --epochs 150 --model VQAGAP_qbert_max_avg_pooled --dataset LR --gpu 5 > ../data/logs/LR/vqa_gap_qbert_max_avg_pooled_log_epoch_150.txt & 
nohup python train.py --epochs 150 --model VQAGAP_bert_ca_max_avg_pooled --dataset LR --gpu 5 > ../data/logs/LR/vqa_gap_bert_ca_max_avg_pooled_log_epoch_150.txt & 
