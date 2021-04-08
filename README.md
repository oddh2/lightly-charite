# Self-supervised learning

Using lightly

`pip install -r requirements.txt`

# Models

SimCLR

`python src/train_self_supervised.py --model_name simclr --model_size 50 --data_path /projects/self_supervised/data/bitewings_caries/bw_U --max_epochs 200 --saving_dir /projects/self_supervised/results --batch_size 16 --input_size 128 --num_ftrs 512 --hf_prob 0.5 --vf_prob 0.5 --cj_prob 0.0  --rr_prob 0.0 --experiment_name bitewings_caries_simclr_50_input_size_128`

MoCo

`python src/train_self_supervised.py --model_name moco --model_size 50 --data_path /projects/self_supervised/data/bitewings_caries/bw_U --max_epochs 200 --saving_dir /projects/self_supervised/results --batch_size 32 --input_size 128 --num_ftrs 512 --hf_prob 0.5 --vf_prob 0.5 --cj_prob 0.0  --rr_prob 0.0 --experiment_name bitewings_caries_moco_50_input_size_128`

# Fine-tuning

`python src/run_finetuning.py --saving_dir /projects/self_supervised/results/distillation_experiment_weighted_loss --L_dir /projects/self_supervised/data/bitewings_caries/bw_L --U_dir /projects/self_supervised/data/bitewings_caries/bw_U --pretrained_dir /projects/self_supervised/results/bitewings_caries_moco_50 --max_epochs 50 --train_size 0.5 --n_splits 10`

# Image augmentation
`--input_size` : Size for reshaping the image
`--vf_prob` : Vertical flipping probability
`--hf_prob` : Horizontal flipping probability
`--rr_prob` : Random rotation probability
`--cj_prob` : Color jitter probability







