# Self-supervised learning

Using lightly

`pip install -r requirements.txt`


# Image augmentation
`--input_size` : Size for reshaping the image
`--vf_prob` : Vertical flipping probability
`--hf_prob` : Horizontal flipping probability
`--rr_prob` : Random rotation probability

# Datasets


## jaw_side: 

images: `/projects/self_supervised/data/jaw_side`

Temptative configuration:
`--input_size 64`
`--vf_prob 0.0`
`--hf_prob 0.0`
`--rr_prob 0.0`

`python src/train.py --model_name simclr --data_path /projects/self_supervised/data/jaw_side --max_epochs 300 --saving_dir /projects/self_supervised/results --batch_size 128 --input_size 64 --num_ftrs 512 --hf_prob 0.0 --vf_prob 0.0 --rr_prob 0.0 --experiment_name jaw_side_simclr`

Notebook at `notebooks/inference_jaw_side.ipynb`


## bitewings_caries

images: `/projects/self_supervised/data/bitewings_caries/images`

masks: `/projects/self_supervised/data/bitewings_caries/masks`

Temptative configuration:
`--input_size 64`
`--vf_prob 0.5`
`--hf_prob 0.5`
`--rr_prob 0.5`

SimCLR

`python src/train.py --model_name simclr --data_path /projects/self_supervised/data/bitewings_caries/images --max_epochs 300 --saving_dir /projects/self_supervised/results --batch_size 128 --input_size 64 --num_ftrs 512 --hf_prob 0.5 --vf_prob 0.5 --rr_prob 0.5 --experiment_name bitewings_caries_simclr`

Given the large input size, the batch size should be small because of GPU memory constraints. The recommendation is to use MoCo

`python src/train.py --model_name moco --model_size 50 --data_path /projects/self_supervised/data/bitewings_caries/images --max_epochs 400 --saving_dir /projects/self_supervised/results --batch_size 32 --input_size 64 --num_ftrs 512 --hf_prob 0.5 --cj_prob 0.5 --vf_prob 0.5 --rr_prob 0.5 --experiment_name bitewings_caries_moco_50_cj`


Notebook at `notebooks/inference_bw_caries.ipynb`






