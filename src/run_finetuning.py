import fire
from pathlib import Path
import torchvision.transforms as transforms
import torchvision
import lightly
import torch
import torchvision.datasets as datasets
import torch.optim as optim
import json

from ss_losses import *
from utilities import *
from ss_models import *

def load_data(**kwargs):
    input_size = kwargs.get('input_size')
    U_dir = kwargs.get('U_dir')
    imgs = kwargs.get('imgs')
    labels = kwargs.get('labels')
    train_size = kwargs.get('train_size')
    batch_size = kwargs.get('batch_size')
    num_workers = kwargs.get('num_workers')

    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])

    # train test validation splits
    train_imgs, train_labels, eval_imgs,eval_labels = make_random_split(imgs,labels,train_size = train_size)

    val_imgs, val_labels, test_imgs,test_labels = make_random_split(eval_imgs,eval_labels,train_size = 0.5)

    trainset = BWDataset(train_imgs,train_labels,train_transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True
    )

    valset = BWDataset(val_imgs,val_labels,test_transform)
    valloader = torch.utils.data.DataLoader(
        valset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True
    )

    testset = BWDataset(test_imgs,test_labels,test_transform)
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True
    )

    unlabeledset = datasets.ImageFolder(root=U_dir, transform=train_transform)
    unlabeledloader = torch.utils.data.DataLoader(
        unlabeledset,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers
    )

    print('training:',len(trainset))
    print('validation:',len(valset))
    print('testing:',len(testset))
    print('unlabeled:',len(unlabeledset))

    return unlabeledloader, trainloader, valloader, testloader

def save_metrics(metrics_dict, saving_dir, fname):
    with open(saving_dir.joinpath(fname),'w') as f:
        json.dump(metrics_dict,f)

def get_imgs_per_cat(y_encoded):
    #count the images in each category
    data_dict = {}
    for el in y_encoded:
        if el not in data_dict.keys():
            data_dict.update({el:1})
        else:
            data_dict[el] += 1
    return data_dict


def get_class_weights(y_encoded,encoding_dict):
    """Calculates the weights for the Cross Entropy loss """
    data_dict = get_imgs_per_cat(y_encoded)       
    N = sum(data_dict.values())
    print('Percentage of images in each category:\n')              
    #calculate weights as the inverse of the frequency of each class
    weights = []
    for k in range(len(data_dict)):
        v = data_dict[k]
        weights.append(N/v)
        print('{}: {:.6g} %'.format(encoding_dict[k],100.0*v/N))    
    print('Weights: {}\n'.format(weights))
    print('\n')      
    return weights


def main(**kwargs):

    U_dir = kwargs.get('U_dir')
    L_dir = kwargs.get('L_dir')
    saving_dir = kwargs.get('saving_dir')
    pretrained_dir = kwargs.get('pretrained_dir')
    train_size = kwargs.get('train_size',0.5)
    input_size = kwargs.get('input_size',128)
    batch_size = kwargs.get('batch_size',16)
    num_workers = kwargs.get('num_workers',2)
    max_epochs = kwargs.get('max_epochs',50)
    n_splits = kwargs.get('n_splits',5)

    L_dir = Path(L_dir)
    pretrained_dir = Path(pretrained_dir)
    saving_dir = Path(saving_dir)
    saving_dir.mkdir(parents=True, exist_ok=True)

    L_dir = Path(L_dir)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    #get L images and categories 
    imgs,labels = parse_dataset(L_dir)
    labels, class_index_dict = label_encoding(labels)

    weights = get_class_weights(labels,class_index_dict)

    crossent_loss = nn.CrossEntropyLoss(reduction ='sum',weight=torch.FloatTensor(weights).to(device)) 
    
    for split in range(n_splits):

        split_dir = saving_dir.joinpath(f'split_{split}')
        split_dir.mkdir(parents=True, exist_ok=True)

        save_metrics(class_index_dict, split_dir, 'class_index_dict.json')

        unlabeledloader, trainloader, valloader, testloader = load_data(
            input_size = input_size,
            U_dir = U_dir,
            imgs = imgs,
            labels = labels,
            train_size = train_size,
            batch_size = batch_size,
            num_workers = num_workers
        )

        
        n_categories = len([cat for cat in L_dir.iterdir()])
        #crossent_loss = CrossEntropyLoss(n_categories)

        print('training student 50 on L')
        student = Student(50,n_categories).to(device)
        optimizer = optim.Adam(student.parameters(), lr=0.0001)
        student, _ = fine_tune(
            model = student,
            max_epochs = max_epochs,
            optimizer = optimizer,
            loss_function = crossent_loss,
            trainloader = trainloader,
            valloader = valloader,
            checkpoint_name = 'student50',
            device = device,
            saving_dir = split_dir
        )

        #eval on test data
        test_metrics = evaluate(student,testloader,crossent_loss,device)
        print('\nTest metrics:')
        print_metrics(test_metrics)
        save_metrics(test_metrics, split_dir, 'metrics_student_50.json')
        print(50*'*')

        

        #train student on L as baseline 
        print('training student 18 on L')
        student = Student(18,n_categories).to(device)
        optimizer = optim.Adam(student.parameters(), lr=0.0001)
        student, _ = fine_tune(
            model = student,
            max_epochs = max_epochs,
            optimizer = optimizer,
            loss_function = crossent_loss,
            trainloader = trainloader,
            valloader = valloader,
            checkpoint_name = 'student18',
            device = device,
            saving_dir = split_dir
        )

        #eval on test data
        test_metrics = evaluate(student,testloader,crossent_loss,device)
        print('\nTest metrics:')
        print_metrics(test_metrics)
        save_metrics(test_metrics, split_dir, 'metrics_student_18.json')
        print(50*'*')

        #train teacher on L
        print('fine tuning self-supervised teacher on L')
        teacher = load_simcrl(pretrained_dir,n_categories,device,model_size = 50)
        optimizer = optim.Adam(teacher.parameters(), lr=0.00001)
        teacher, _ = fine_tune(
            model = teacher,
            max_epochs = max_epochs,
            optimizer = optimizer,
            loss_function = crossent_loss,
            trainloader = trainloader,
            valloader = valloader,
            checkpoint_name = 'teacher',
            device = device,
            saving_dir = split_dir
        )

        #eval on test data
        test_metrics = evaluate(teacher,testloader,crossent_loss,device)
        print('\nTest metrics:')
        print_metrics(test_metrics)
        save_metrics(test_metrics, split_dir, 'metrics_teacher.json')
        print(50*'*')

        # #distillation
        # print('distillation of student model on U')
        # #init student
        # student = Student(18,n_categories).to(device)
        # #load teacher checkpoint
        # teacher = load_simcrl(pretrained_dir,n_categories,device,model_size = 50)
        # teacher.load_state_dict(torch.load(split_dir.joinpath('teacher.pth')))
        # teacher = teacher.freeze_weights()
        # distill_loss = DistilLoss()
        # optimizer = optim.Adam(student.parameters(), lr=0.0001)
        # distilled, _ = distillation(
        #     teacher = teacher,
        #     student = student,
        #     max_epochs = max_epochs,
        #     optimizer = optimizer,
        #     distill_loss = distill_loss,
        #     classification_loss = crossent_loss,
        #     U_loader = unlabeledloader,
        #     valloader = valloader,
        #     checkpoint_name = 'distilled',
        #     device = device,
        #     saving_dir = split_dir
        # )

        # #eval on test data
        # test_metrics = evaluate(distilled,testloader,crossent_loss,device)
        # print('\nTest metrics:')
        # print_metrics(test_metrics)
        # save_metrics(test_metrics, split_dir, 'metrics_distilled.json')


if __name__=="__main__":

    fire.Fire(main)