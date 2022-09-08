import torchvision.models as models
import torch.nn as nn
import torch
class ArtDLClassifier(nn.Module):
  def __init__(self, num_classes):
    super(ArtDLClassifier, self).__init__()
    # Loading the pretrained model
    self.resnet = models.resnet50(pretrained=True)

    # Taking all the layers except the linear layer
    self.net = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

    # Replacing linear layer with 1x1 conv channels
    self.fc_conv = nn.Conv2d(in_channels = 2048, out_channels = num_classes, kernel_size=1)
    self.activations = {}

    # Setting the trainable params for the optimizer
    self.tr_params = nn.ModuleList([self.net[4][2:], self.net[5], self.net[6], self.net[7], self.fc_conv])
  def forward(self, image):
    # Forward prop
    out = self.net(image)
    out = self.fc_conv(out)
    return out
  def trainable_params(self):
    return (list(self.tr_params.parameters()))
  
  def get_activations(self,key):
        '''
            Description: Method to get activations for a particular layer
            Args: 
                key -> nn.Module(the module to calculate activations for)
        '''
        def hook(module, input, out):
            self.activations[key] = out.detach()
        return hook