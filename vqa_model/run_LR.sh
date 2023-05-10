nohup python train.py --epochs 150 --model VQAModel --dataset LR --gpu 6 > ../data/logs/lr_vqa_model_log_epoch_150.txt &
nohup python train.py --epochs 150 --model VQAGAP_bert_Model --dataset LR --gpu 6 > ../data/logs/LR/vqa_gap_bert_model_log_epoch_150.txt &
nohup python train.py --epochs 150 --model VQAGAP_bert_Model_finetune --dataset LR --gpu 6 > ../data/logs/LR/vqa_gap_bert_model_finetune_log_epoch_150.txt &
nohup python train.py --epochs 150 --model VQAGAP_qbert_Model --dataset LR --gpu 6 > ../data/logs/LR/vqa_gap_qbert_model_log_epoch_150.txt &
nohup python train.py --epochs 150 --model VQAGAP_qbert_Model_finetune --dataset LR --gpu 6 > ../data/logs/LR/vqa_gap_qbert_model_finetune_log_epoch_150.txt &
nohup python train.py --epochs 150 --model VQAGAPModel --dataset LR --gpu 6 > ../data/logs/LR/vqa_gap_model_log_epoch_150.txt &
nohup python train.py --epochs 150 --model VQAGAPModel_finetune --dataset LR --gpu 6 > ../data/logs/LR/vqa_gap_model_finetune_log_epoch_150.txt &
