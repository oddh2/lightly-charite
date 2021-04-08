import torch

def P(x,tau = 1.0):
  return torch.exp(x/tau)/(torch.exp(x/tau).sum())

class CrossEntropyLoss(torch.nn.Module):

    def __init__(self,num_classes):
        super(CrossEntropyLoss,self).__init__()
        self.num_classes = num_classes

    def forward(self, prediction, label):
      label = torch.nn.functional.one_hot(label,num_classes=self.num_classes)
      loss = -label*torch.log(P(prediction))
      return loss.sum().sum()

class DistilLoss(torch.nn.Module):

    def __init__(self):
        super(DistilLoss,self).__init__()

    def forward(self, t_output, s_output):
      loss = -P(t_output)*torch.log(P(s_output))
      return loss.sum().sum()