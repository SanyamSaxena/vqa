nohup python train.py --epochs 150 --model VQAModel --dataset LR --gpu 3 > ../data/logs/LR/vqa_model_log_epoch_150.txt &
nohup python train.py --epochs 150 --model VQAModel --dataset HR --gpu 5 > ../data/logs/HR/vqa_model_log_epoch_150.txt &
nohup python train.py --epochs 150 --model VQAGAPModel --dataset LR --gpu 5 > ../data/logs/LR/vqa_gap_model_log_epoch_150.txt &
nohup python train.py --epochs 150 --model VQAGAPModel --dataset HR --gpu 7 > ../data/logs/HR/vqa_gap_model_log_epoch_150.txt &
