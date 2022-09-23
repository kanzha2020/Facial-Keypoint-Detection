## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        ####################################################################################
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        # 224->220->136
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F+2*P)/S +1 = (224-3+2*1)/1 +1 = 224
        # L1 ImgIn shape=(?, 224, 224, 1)
        # Conv -> (?, 224, 224, 32)
        # Pool -> (?, 112, 112, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        ## output size = (W-F+2P)/S +1 = (112-2+0)/1 +1 = 110
        # L2 ImgIn shape=(?, 110, 110, 32)
        # Conv -> (?, 110, 110, 64)
        # Pool -> (?, 55, 55, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        ## output size = (W-F)/S +1 = (55-2+2)/1 +1 = 56
        # L3 ImgIn shape=(?, 55, 55, 64)
        # Conv ->(?, 56, 56, 128)
        # Pool ->(?, 28, 28, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        ## output size = (W-F)/S +1 = (28-2)/1 +1 = 27
        # L3 ImgIn shape=(?, 28, 28, 128)
        # Conv ->(?, 28, 28, 256)
        # Pool ->(?, 13, 13, 256)
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        ## output size = (W-F)/S +1 = (13-2+0)/1 +1 = 12
        # L3 ImgIn shape=(?, 13, 13, 256)
        # Conv ->(?, 13, 13, 512)
        # Pool ->(?, 6, 6, 512)
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        ## output size = (W-F)/S +1 = (6-1+0)/1 +1 = 6
        # L3 ImgIn shape=(?, 6, 6, 512)
        # Conv ->(?, 6, 6, 512)
        # Pool ->(?, 3, 3, 1024)
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Dropout
        self.dropout = torch.nn.Dropout(p=0.25)
        # Fully-connected (linear) layers
        self.fc1 = torch.nn.Linear(1024*3*3, 2048)
        self.fc2 = torch.nn.Linear(2048, 1024)
        self.fc3 = torch.nn.Linear(1024, 512)
        self.fc4 = torch.nn.Linear(512, 68*2)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:        
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        
        # Prep for linear layer / Flatten
        out = out.view(out.size(0), -1)
        
        # linear layers with dropout in between
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = F.relu(self.fc3(out))
        out = self.dropout(out)
        out = self.fc4(out)
        return out
