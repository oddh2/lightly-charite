import numpy as np
import torch
from PIL import Image
import json
import lightly
import torch.nn as nn
import matplotlib.pyplot as plt

import torchvision

from ss_models import *

import imblearn
import sklearn
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


def save_json(metrics_dict, saving_dir, fname):
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

def label_encoding(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    encoding_dict = {}
    for cat in le.classes_:
        label = le.transform(np.array([cat]))[0]
        encoding_dict.update({int(label):cat}) 

    return y_encoded, encoding_dict

def parse_dataset(data_path):

    img_paths = []
    labels = []
    for label in data_path.iterdir():
        for img in data_path.joinpath(label).iterdir():
            img_paths.append(img)
            labels.append(label.name)

    return img_paths, labels

def make_random_split(imgs,labels,train_size = 0.5):

    sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - train_size)
    idx_train, idx_test = next(sss.split(np.array(imgs),np.array(labels)))

    train_imgs = [imgs[i] for i in idx_train]
    train_labels = [labels[i] for i in idx_train]

    test_imgs = [imgs[i] for i in idx_test]
    test_labels = [labels[i] for i in idx_test]
    
    return train_imgs, train_labels, test_imgs, test_labels

def print_metrics(metrics):
    loss = metrics['val_loss']
    acc = metrics['acc']
    f1 = metrics['f1']
    recall = metrics['recall']
    precision = metrics['precision']
    print(f'loss: {loss:.3f} acc:{acc:.3f} f1:{f1:.3f} precision:{precision:.3f} recall:{recall:.3f}')


def load_self_supervised(pretrained_dir,device):

    model_path = pretrained_dir.joinpath('checkpoint.pth')
    conf_path = pretrained_dir.joinpath('conf.json')
    with open(conf_path,'r') as f:
        conf = json.load(f)

    resnet_dict = {
        18: torchvision.models.resnet18(pretrained=False),
        34: torchvision.models.resnet34(pretrained=False),
        50: torchvision.models.resnet50(pretrained=False),
        101: torchvision.models.resnet101(pretrained=False),
    }

    backbone = resnet_dict[conf['model_size']]

    if conf['benchmarking']:
        model = MoCoModel_benchmarking(backbone,None,1,num_ftrs=conf['num_ftrs'],).to(device)
    else:
        model = MoCoModel(backbone,num_ftrs=conf['num_ftrs']).to(device)

    model.backbone.load_state_dict(torch.load(model_path))

    return model, conf


# to do: change name
class BWDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_path,labels, transform):
   
        self.imgs_path = np.array(imgs_path)
        self.labels = np.array(labels)                
        self.transform = transform
        
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        img_loc = self.imgs_path[idx]
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        label = torch.tensor(self.labels[idx])
        return tensor_image,label


# to do: add classification to name
def fine_tune(**kwargs):
    model = kwargs.get('model')
    trainloader = kwargs.get('trainloader')
    valloader = kwargs.get('valloader')
    device = kwargs.get('device')
    optimizer = kwargs.get('optimizer')
    loss_function = kwargs.get('loss_function')
    saving_dir = kwargs.get('saving_dir')
    max_epochs = kwargs.get('max_epochs',50)
    patience = kwargs.get('patience',10)
    checkpoint_name = kwargs.get('checkpoint_name','fine_tuned')

    count = 0
    best_loss = 1e9
    for epoch in range(max_epochs):
        train_loss = 0.0
        for image,label in trainloader:
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(image), label)/image.shape[0]
            loss.backward()
            optimizer.step()
                
        metrics_dict = evaluate(model,valloader,loss_function,device)
        val_loss = metrics_dict['val_loss']
        if val_loss < best_loss:
                if saving_dir:
                    torch.save(model.state_dict(),saving_dir.joinpath(f'{checkpoint_name}.pth'))
                best_loss = val_loss
                best_metrics = metrics_dict
                
                count = 0
        else:
            count += 1
        if count > patience:
            break

    #load best model
    if saving_dir:
        model.load_state_dict(torch.load(saving_dir.joinpath(f'{checkpoint_name}.pth')))

    return model, best_metrics



def evaluate(model,testloader,loss_function,device):
  val_loss = 0
  ground_truth_list = []
  predictions_list =  []
  model.eval()
  for image,label in testloader:
      image, label = image.to(device), label.to(device)
      outputs = model(image)
      probabilities, predicted = torch.max(outputs.data, 1)
      val_loss += loss_function(outputs, label.long()).item()
      ground_truth_list += list(label.cpu())
      predictions_list += list(predicted.cpu())

  val_loss /= len(testloader.dataset)

  acc = sklearn.metrics.accuracy_score(ground_truth_list,predictions_list)
  f1 = sklearn.metrics.f1_score(ground_truth_list,predictions_list,average = 'macro')
  precision = sklearn.metrics.precision_score(ground_truth_list,predictions_list,average = 'macro')
  recall = sklearn.metrics.recall_score(ground_truth_list,predictions_list,average = 'macro')
  cm = sklearn.metrics.confusion_matrix(ground_truth_list,predictions_list)
  sensitivity = imblearn.metrics.sensitivity_score(ground_truth_list,predictions_list,average = 'macro')
  specificity = imblearn.metrics.specificity_score(ground_truth_list,predictions_list,average = 'macro')

  if cm[1,1]+cm[0,1] == 0:
      ppv = 0.0
  else:
      ppv = cm[1,1]/(cm[1,1]+cm[0,1])

  if cm[0,0]+cm[1,0] == 0:
      npv = 0.0
  else:
      npv = cm[0,0]/(cm[0,0]+cm[1,0])

  #_,auc = compute_ROC(ground_truth_list, predictions_list,{0:'a',1:'b'},title = '')
  auc = 0.0

  ground_truth_list = [int(item) for item in ground_truth_list]
  predictions_list = [int(item) for item in predictions_list]

  print(f'loss: {val_loss:.3f} acc:{acc:.3f} f1:{f1:.3f} precision:{precision:.3f} recall:{recall:.3f} auc:{auc:.3f}')

  metrics_dict = {
      'val_loss':val_loss,
      'acc':acc,
      'f1':f1,
      'precision':precision,
      'recall':recall,
      'ppv':ppv,
      'npv':npv,
      'auc':auc,
      'sensitivity':sensitivity,
      'specificity':specificity,
      'TP':int(cm[1,1]),
      'TN':int(cm[0,0]),
      'FP':int(cm[0,1]),
      'FN':int(cm[1,0]),
      'ground_truth_list':ground_truth_list,
      'predictions_list':predictions_list
      }

  return metrics_dict

def distillation(**kwargs):
    #teacher = kwargs.get('teacher')
    model = kwargs.get('model')
    trainloader = kwargs.get('trainloader')
    valloader = kwargs.get('valloader')
    device = kwargs.get('device')
    optimizer = kwargs.get('optimizer')
    distill_loss = kwargs.get('distill_loss')
    classification_loss = kwargs.get('classification_loss')
    saving_dir = kwargs.get('saving_dir')
    max_epochs = kwargs.get('max_epochs',50)
    patience = kwargs.get('patience',3)
    checkpoint_name = kwargs.get('checkpoint_name','distilled')

    #teacher.eval()
    model.train()

    count = 0
    best_loss = 1e9
    for epoch in range(max_epochs):
        for image,label in trainloader:
            image = image.to(device)
            label = label.to(device)
            #label = teacher(image).to(device)
            #print(teacher(image).shape)
            loss = distill_loss(model(image),label)
            loss.backward()
            optimizer.step()
                
        metrics_dict = evaluate(model,valloader,classification_loss,device)
        val_loss = metrics_dict['val_loss']
        if val_loss < best_loss:
                if saving_dir:
                    torch.save(model.state_dict(),saving_dir.joinpath(f'{checkpoint_name}.pth'))
                best_loss = val_loss
                best_metrics = metrics_dict
                
                count = 0
        else:
            count += 1
        if count > patience:
            break

    #load best model
    if saving_dir:
        model.load_state_dict(torch.load(saving_dir.joinpath(f'{checkpoint_name}.pth')))

    return model, best_metrics


def compute_ROC(ground_truth_list, predictions_list,encoding_dict,title = ''):
    # code adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    n_labels = len(encoding_dict)
    #print(n_labels)

    ground_truth_list_binarized = sklearn.preprocessing.label_binarize(ground_truth_list, classes=range(n_labels))
    predictions_list_binarized = sklearn.preprocessing.label_binarize(predictions_list, classes=range(n_labels))

    #print(predictions_list_binarized.shape)

    if n_labels == 2:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr[0], tpr[0], _ = sklearn.metrics.roc_curve(ground_truth_list_binarized, predictions_list_binarized)
        roc_auc[0] = auc(fpr[0], tpr[0])

        auc_value = roc_auc[0]

        fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(ground_truth_list_binarized.ravel(), predictions_list_binarized.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        fig,ax = plt.subplots(figsize = (7,7))
        fontsize = 20

        lw = 2
        ax.plot(fpr[0], tpr[0], color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
        ax.plot([0, 1], [0, 1], 'k--', lw=lw)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate',fontsize = fontsize)
        ax.set_ylabel('True Positive Rate',fontsize = fontsize)
        ax.set_title('Receiver Operating Characteristic',fontsize = fontsize)
        ax.legend(loc="lower right",fontsize = fontsize)
        ax.set_xticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=fontsize)
        ax.set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=fontsize)
        #plt.show()
        return fig,auc_value

    else:

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_labels):
            fpr[i], tpr[i], _ = roc_curve(ground_truth_list_binarized[:,i], predictions_list_binarized[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(ground_truth_list_binarized.ravel(), predictions_list_binarized.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        auc_value = roc_auc["micro"]
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_labels)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_labels):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_labels

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        lw = 4
        fontsize = 30
        #plt.figure()
        
        fig,ax = plt.subplots(figsize=(15,15))

        #c
        # ax.plot(fpr["micro"], tpr["micro"],
        #         label='micro-average ROC curve (area = {0:0.2f})'
        #             ''.format(roc_auc["micro"]),
        #         color='deeppink', linestyle=':', linewidth=4)

        ax.plot(fpr["macro"], tpr["macro"],
                label='macro-average (area = {0:0.2f})'
                    ''.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue','green'])
        for i, color in zip(range(n_labels), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                    label=' {0} (area = {1:0.2f})'
                    ''.format(encoding_dict[i], roc_auc[i]))

        ax.plot([0, 1], [0, 1], 'k--', lw=lw)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate',fontsize = fontsize)
        ax.set_ylabel('True Positive Rate',fontsize = fontsize)
        ax.set_title(title,fontsize = fontsize)
        ax.legend(loc="lower right",fontsize = fontsize)
        ax.set_xticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=fontsize)
        ax.set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=fontsize)
        #plt.show()
        return fig,auc_value
    

