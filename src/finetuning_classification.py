import fire
from pathlib import Path
import json

import torch
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import lightly

from sklearn.model_selection import StratifiedKFold

from ss_losses import *
from utilities import *
from ss_models import *

# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def main(**kwargs):

    L_dir = kwargs.get('L_dir')
    saving_dir = kwargs.get('saving_dir')
    pretrained_dir = kwargs.get('pretrained_dir')

    input_size = kwargs.get('input_size',224)
    batch_size = kwargs.get('batch_size',32)
    num_workers = kwargs.get('num_workers',8)
    max_epochs = kwargs.get('max_epochs',50)
    n_splits = kwargs.get('n_splits',5)
    learning_rate = kwargs.get('learning_rate',1e-4)
    crossvalidation = kwargs.get('crossvalidation',False)
    baselines = kwargs.get('baselines',False)
    sample = kwargs.get('sample',1.0)
    hf_prob = kwargs.get('hf_prob',0.0)
    vf_prob = kwargs.get('vf_prob',0.0)
    cj_prob = kwargs.get('cj_prob',0.0)
    gb_prob = kwargs.get('gb_prob',0.0)

    L_dir = Path(L_dir)
    pretrained_dir = Path(pretrained_dir)
    saving_dir = Path(saving_dir)
    saving_dir.mkdir(parents=True, exist_ok=True)

    color_jitter = torchvision.transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.2)

    kernel_size = 3

    gaussian_blur = torchvision.transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.RandomHorizontalFlip(p=hf_prob),
        torchvision.transforms.RandomVerticalFlip(p=vf_prob),
        torchvision.transforms.RandomApply([color_jitter], p=cj_prob),
        torchvision.transforms.RandomApply([gaussian_blur], p=gb_prob),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=lightly.data.collate.imagenet_normalize['mean'],
            std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=lightly.data.collate.imagenet_normalize['mean'],
            std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])

    resnet_dict = {
        18: torchvision.models.resnet18(pretrained=False),
        34: torchvision.models.resnet34(pretrained=False),
        50: torchvision.models.resnet50(pretrained=False),
        101: torchvision.models.resnet101(pretrained=False),
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #get L images and categories 
    imgs,labels = parse_dataset(L_dir)
    labels, class_index_dict = label_encoding(labels)
    weights = get_class_weights(labels,class_index_dict)

    n_categories = len(weights)

    crossent_loss = nn.CrossEntropyLoss(reduction ='sum',weight=torch.FloatTensor(weights).to(device)) 
    
    kf = StratifiedKFold(n_splits=n_splits)

    for i,(train_val_index,test_index) in enumerate(kf.split(imgs,labels)):
        
        imgs_test = [imgs[idx] for idx in test_index]
        labels_test = [labels[idx] for idx in test_index]

        imgs_train_val = [imgs[idx] for idx in train_val_index]
        labels_train_val = [labels[idx] for idx in train_val_index]

        train_index, val_index = next(kf.split(imgs_train_val,labels_train_val))

        imgs_train = [imgs_train_val[idx] for idx in train_index]
        labels_train = [labels_train_val[idx] for idx in train_index]

        imgs_val = [imgs_train_val[idx] for idx in val_index]
        labels_val = [labels_train_val[idx] for idx in val_index]

        if crossvalidation == False and i>0:
            break

        print(f'split_{i}')
        split_dir = saving_dir.joinpath(f'split_{i}')
        split_dir.mkdir(parents=True, exist_ok=True)

        save_json(class_index_dict, split_dir, 'class_index_dict.json')

        save_json({
            'train':[str(path) for path in imgs_train],
            'val':[str(path) for path in imgs_val],
            'test':[str(path) for path in imgs_test]}, split_dir, 'splits.json')

        trainset = BWDataset(imgs_train,labels_train,train_transform)
        idx = np.random.randint(len(trainset),size = int(sample*len(trainset)))
        trainset = torch.utils.data.Subset(trainset,idx)
        trainloader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn
        )

        valset = BWDataset(imgs_val,labels_val,test_transform)
        valloader = torch.utils.data.DataLoader(
            valset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn
        )

        testset = BWDataset(imgs_test,labels_test,test_transform)
        testloader = torch.utils.data.DataLoader(
            testset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn
        )

        print('training:',len(trainset))
        print('validation:',len(valset))
        print('testing:',len(testset))


        print('finetuning self-supervised model')

        conf = {'model_size':18}

        # load model

        model,conf = load_self_supervised(pretrained_dir,device)
  
        ss_clf = SelfSupervisedClassifier(model.backbone,conf['num_ftrs'],n_categories).to(device)
        optimizer = optim.Adam(ss_clf.parameters(), lr=learning_rate)
        ss_clf, _ = fine_tune(
            model = ss_clf,
            max_epochs = max_epochs,
            optimizer = optimizer,
            loss_function = crossent_loss,
            trainloader = trainloader,
            valloader = valloader,
            checkpoint_name = 'self_supervised',
            device = device,
            saving_dir = split_dir
        )

        #eval on test data
        test_metrics = evaluate(ss_clf,testloader,crossent_loss,device)
        print('\nTest metrics:')
        print_metrics(test_metrics)
        save_json(test_metrics, split_dir, 'self_supervised_metrics.json')
        print(50*'*')

        if baselines:

            model_size = conf['model_size']

            print(f'finetuning resnet {model_size} Imagenet init')
            backbone = torchvision.models.resnet18(pretrained=True)
            student = Classifier(backbone,n_categories).to(device)
            optimizer = optim.Adam(student.parameters(), lr=learning_rate)
            student, _ = fine_tune(
                model = student,
                max_epochs = max_epochs,
                optimizer = optimizer,
                loss_function = crossent_loss,
                trainloader = trainloader,
                valloader = valloader,
                checkpoint_name = 'resnet18_imagenet',
                device = device,
                saving_dir = split_dir
            )

            #eval on test data
            test_metrics = evaluate(student,testloader,crossent_loss,device)
            print('\nTest metrics:')
            print_metrics(test_metrics)
            save_json(test_metrics, split_dir, 'resnet18_imagenet_metrics.json')
            print(50*'*')

            
            backbone = resnet_dict[model_size]

            print(f'finetuning resnet {model_size} random init')
            
            student = Classifier(backbone,n_categories).to(device)
            optimizer = optim.Adam(student.parameters(), lr=learning_rate)
            student, _ = fine_tune(
                model = student,
                max_epochs = max_epochs,
                optimizer = optimizer,
                loss_function = crossent_loss,
                trainloader = trainloader,
                valloader = valloader,
                checkpoint_name = 'resnet18_random',
                device = device,
                saving_dir = split_dir
            )

            #eval on test data
            test_metrics = evaluate(student,testloader,crossent_loss,device)
            print('\nTest metrics:')
            print_metrics(test_metrics)
            save_json(test_metrics, split_dir, 'resnet18_random_metrics.json')
            print(50*'*')


if __name__=="__main__":
    fire.Fire(main)