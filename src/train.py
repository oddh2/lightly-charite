import os
import glob
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from PIL import Image
import numpy as np
import fire
import lightly.utils.io as io
import json

def train(**kwargs):

    data_path = kwargs.get('data_path',None)
    saving_dir = kwargs.get('saving_dir',None)
    max_epochs = kwargs.get('max_epochs',200)
    experiment_name = kwargs.get('experiment_name','experiment')
    input_size = kwargs.get('input_size',64)
    num_ftrs = kwargs.get('num_ftrs',512)
    batch_size = kwargs.get('batch_size',64)
    hf_prob = kwargs.get('hf_prob',0.5) # horizontal flip
    vf_prob = kwargs.get('vf_prob',0.5) # vertical flip
    rr_prob = kwargs.get('rr_prob',0.5) # random rotation
    cj_prob = kwargs.get('cj_prob',0.0) # random rotation
    memory_bank_size = kwargs.get('memory_bank_size',4096)
    model_name = kwargs.get('model_name','simclr') # simclr or moco
    model_size = kwargs.get('model_size',18) # 18, 34, 50, 101, 152

    if model_name == 'moco':
        print('memory_bank_size:',memory_bank_size)

    saving_dir = os.path.join(saving_dir,experiment_name)
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)

    model_filename = 'checkpoint.pth'
    embeddings_filename = 'embeddings.csv'

    conf = {
        'num_ftrs':num_ftrs,
        'model_name':model_name,
        'max_epochs':max_epochs,
        'input_size':input_size,
        'batch_size':batch_size,
        'hf_prob':hf_prob,
        'vf_prob':vf_prob,
        'rr_prob':rr_prob,
        'cj_prob':cj_prob,
        'model_size':model_size
        }

    print('Configuration:')
    for k,v in conf.items():
        print(f'{k}: {v}')


    with open(os.path.join(saving_dir,'conf.json'),'w') as f:
        json.dump(conf,f)


    num_workers = 8
    #seed = 1
    #pl.seed_everything(seed)

    collate_fn = lightly.data.SimCLRCollateFunction(
        input_size=input_size,
        vf_prob=vf_prob,
        rr_prob=rr_prob,
        hf_prob=hf_prob,
        cj_prob=cj_prob,
        random_gray_scale=0.0
    )


    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=lightly.data.collate.imagenet_normalize['mean'],
            std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])

    dataset_train_simclr = lightly.data.LightlyDataset(
        input_dir=data_path
    )

    dataset_test = lightly.data.LightlyDataset(
        input_dir=data_path,
        transform=test_transforms
    )


    dataloader_train_simclr = torch.utils.data.DataLoader(
        dataset_train_simclr,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    print(model_size)
    resnet = lightly.models.ResNetGenerator('resnet-'+str(model_size))
    last_conv_channels = list(resnet.children())[-1].in_features
    backbone = nn.Sequential(
        *list(resnet.children())[:-1],
        nn.Conv2d(last_conv_channels, num_ftrs, 1),
        nn.AdaptiveAvgPool2d(1)
    )
    
    if model_name == 'simclr':
        criterion = lightly.loss.NTXentLoss()
        model = lightly.models.SimCLR(backbone, num_ftrs=num_ftrs)
    elif model_name == 'moco':
        criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)
        model = lightly.models.MoCo(backbone, num_ftrs=num_ftrs, m=0.99, batch_shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    encoder = lightly.embedding.SelfSupervisedEmbedding(
        model,
        criterion,
        optimizer,
        dataloader_train_simclr
    )


    gpus = 1 if torch.cuda.is_available() else 0
    encoder.train_embedding(gpus=gpus, 
                            progress_bar_refresh_rate=100,
                            max_epochs=max_epochs)


    device = 'cuda' if gpus==1 else 'cpu'
    encoder = encoder.to(device)

    embeddings, labels, fnames = encoder.embed(dataloader_test, device=device)
    #embeddings = normalize(embeddings)

    torch.save(encoder.model.state_dict(),os.path.join(saving_dir,model_filename))
    io.save_embeddings(os.path.join(saving_dir,embeddings_filename), embeddings, labels, fnames)

def main(**kwargs):

    data_path = kwargs.get('data_path',None)
    saving_dir = kwargs.get('saving_dir',None)
    max_epochs = kwargs.get('max_epochs',200)
    input_size = kwargs.get('input_size',64)
    num_ftrs = kwargs.get('num_ftrs',512)
    batch_size = kwargs.get('batch_size',64)
    hf_prob = kwargs.get('hf_prob',0.5) # horizontal flip
    vf_prob = kwargs.get('vf_prob',0.5) # vertical flip
    rr_prob = kwargs.get('rr_prob',0.5) # random rotation
    cj_prob = kwargs.get('cj_prob',0.0) # random color jitter
    memory_bank_size = kwargs.get('memory_bank_size',4096)
    model_name = kwargs.get('model_name','simclr') # simclr or moco
    model_size = kwargs.get('model_size',18)
    experiment_name = kwargs.get('experiment_name','experiment')

    if not data_path:
        raise Exception('data_path not provided')

    if not saving_dir:
        raise Exception('saving_dir not provided')

    train(
        data_path = data_path,
        saving_dir = saving_dir,
        max_epochs = max_epochs,
        input_size = input_size,
        num_ftrs = num_ftrs,
        batch_size = batch_size,
        hf_prob = hf_prob,
        vf_prob = vf_prob,
        rr_prob = rr_prob,
        cj_prob = cj_prob,
        model_name = model_name,
        model_size = model_size,
        memory_bank_size = memory_bank_size,
        experiment_name = experiment_name
        )

if __name__=="__main__":

    fire.Fire(main)

