# Self-supervised learning

`pip install -r requirements.txt`


# References

* [Lightly Github](https://github.com/lightly-ai/lightly)
* [Lightly Tutorials](https://docs.lightly.ai/tutorials/package.html)
* [A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)](https://arxiv.org/pdf/2002.05709.pdf)
* [Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)](https://arxiv.org/pdf/1911.05722.pdf)
* [Big Self-Supervised Models are Strong Semi-Supervised Learners](https://arxiv.org/pdf/2006.10029.pdf)



## Train self supervised model with benchmarking

Self-supervised learning while testing the features with k-nearest neighbours. [Lightly docs for benchmarking](https://docs.lightly.ai/lightly.utils.html#module-lightly.utils.benchmarking)

`nohup python src/self_supervised_benchmarking.py --data_path /projects/sortifier/interim/bw_pan_per_cbct_dataset --saving_dir /projects/self_supervised/results/sortifier_byol --train_knn_data_path /projects/sortifier/interim/bw_pan_per_cbct_train_dataset --test_knn_data_path /projects/sortifier/interim/bw_pan_per_cbct_test_dataset --knn_k 500 --ss_model byol --batch_size 128 --input_size 224 --num_ftrs 512 --sample 1.0 --max_epochs 100 --model_size 18 --vf_prob 0.0 --hf_prob 0.5 --cj_prob 0.0 --gb_prob 0.0 &> sortifier.out &`

Arguments:
* `data_path`: data for training the self-supervised model
* `train_knn_data_path`: data for training k-nn
* `test_knn_data_path`: data for evaluating k-nn
* `knn_k`: number of neigbours to consider



## Linear probing 

Training a linear classifier on top of the frozen feature extractor

`nohup python src/linear_probing.py --saving_dir /projects/self_supervised/results/sortifier_linearprobing --L_dir /projects/sortifier/interim/bw_pan_per_cbct_train_dataset --pretrained_dir /projects/self_supervised/results/sortifier_byol --input_size 224 --learning_rate 1e-4 --n_splits 10 --batch_size 32 --max_epochs 100 --crossvalidation False --baselines True --vf_prob 0.0 --hf_prob 0.5 --cj_prob 0.0 --gb_prob 0.0 &> sortifier_linear_probing.out &`

Arguments:
* `pretrained_dir`: directory containing the self supervised checkpoint and configuration files
* `L_dir`: labeled data
* `input_size`: dimensions of the input tensor
* `max_epochs`: maximum number of epochs for training
* `patience`: patience parameter for early stopping
* `learning_rate`:
* `batch_size`:
* `baselines`: Boolean
* `n_splits`: number of splits for crossvalidation. Default 10
* `crossvalidation`: Boolean. If False it trains for a single split



## Finetuning classification

Once we have the backbone pretrained we can use it for classification, by finetuning it with labeled data

`nohup python src/finetuning_classification.py --saving_dir /projects/self_supervised/results/sortifier_finetuning --L_dir /projects/sortifier/interim/bw_pan_per_cbct_train_dataset --pretrained_dir /projects/self_supervised/results/sortifier_byol --input_size 224 --learning_rate 1e-4 --n_splits 10 --batch_size 64 --max_epochs 100 --crossvalidation False --baselines True --sample 1.0 &> sortifier_finetuning.out &`

Arguments:
* `pretrained_dir`: directory containing the self supervised checkpoint and configuration files
* `L_dir`: labeled data
* `input_size`: dimensions of the input tensor
* `max_epochs`: maximum number of epochs for training
* `patience`: patience parameter for early stopping
* `learning_rate`:
* `batch_size`:
* `baselines`:
* `n_splits`: number of splits for crossvalidation. Default 10
* `crossvalidation`: Boolean. If False it trains for a single split
