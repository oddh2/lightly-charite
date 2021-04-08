import torch
import torch.nn as nn
import torchvision.models as models

class Student(nn.Module):
    def __init__(self,model_size,output_size):
        super().__init__()
        if model_size == 18:
            self.net = models.resnet18(pretrained=True)
        elif model_size == 50:
            self.net = models.resnet50(pretrained=True)

        
        self.net.fc = nn.Linear(self.net.fc.in_features, output_size)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.net(x)
        out = self.sm(out)
        return out


class Teacher(nn.Module):
    def __init__(self, model,num_ftrs,output_dim):
        super().__init__()
        
        self.freeze = False
        self.net = model
        self.fc1 = nn.Linear(num_ftrs, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)
        self.sm = nn.Softmax(dim=1)
        
    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                y_hat = self.net.backbone(x).squeeze()
                y_hat = self.fc1(y_hat)
                y_hat = self.relu(y_hat)
                y_hat = self.fc2(y_hat)
                out = self.sm(out)
                return y_hat
        else:
            y_hat = self.net.backbone(x).squeeze()
            y_hat = self.fc1(y_hat)
            y_hat = self.relu(y_hat)
            y_hat = self.fc2(y_hat)
            out = self.sm(out)
            return y_hat
            
    def freeze_weights(self):
      for p in self.net.parameters():
          p.requires_grad = False
      self.freeze = True
      return self

    def unfreeze_weights(self):
      for p in self.net.parameters():
          p.requires_grad = True
      self.freeze = False
      return self