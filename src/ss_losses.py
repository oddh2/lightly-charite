import torch

def P(x,tau = 1.0):
  return torch.exp(x/tau)/(torch.exp(x/tau).sum())

class CrossEntropyLoss(torch.nn.Module):

    def __init__(self,num_classes,weights=None):
        super(CrossEntropyLoss,self).__init__()
        self.num_classes = num_classes
        self.weights = weights

    def forward(self, prediction, label):
      label = torch.nn.functional.one_hot(label,num_classes=self.num_classes)
      loss = -label*torch.log(P(prediction))
      if self.weights is not None:
        loss = self.weights*loss
      return loss.sum().sum()

class DistilLoss(torch.nn.Module):

    def __init__(self,weights=None):
        super(DistilLoss,self).__init__()
        self.weights = weights

    def forward(self,s_output,t_output):
      #loss = -P(t_output)*torch.log(P(s_output))
      loss = -t_output*torch.log(s_output)
      if self.weights is not None:
        loss = self.weights*loss
      return loss.sum().sum()