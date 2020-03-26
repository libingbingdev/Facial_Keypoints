## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool = nn.MaxPool2d(2, 2)
        ## output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # the output tensor will have dimensions: (64, 106, 106)
        # after another pool layer this becomes (64, 53, 53); 
        self.conv2 = nn.Conv2d(32, 64, 5)
        ## output size = (W-F)/S +1 = (53-5)/1 +1 = 49
        # the output tensor will have dimensions: (128, 49, 49)
        # after another pool layer this becomes (128, 24, 24); 
        self.conv3 = nn.Conv2d(64, 128, 5)
        ## output size = (W-F)/S +1 = (24-5)/1 +1 = 20
        # the output tensor will have dimensions: (256, 20, 20)
        # after another pool layer this becomes (256, 10, 10); 
        self.conv4 = nn.Conv2d(128, 256, 5)
        ## output size = (W-F)/S +1 = (10-5)/1 +1 = 6
        # the output tensor will have dimensions: (512, 6, 6)
        # after another pool layer this becomes (512, 3, 3); 
        self.conv5 = nn.Conv2d(256, 512, 5)
        self.conv1_drop = nn.Dropout(p = 0.1)
        self.conv2_drop = nn.Dropout(p = 0.2)
        self.conv3_drop = nn.Dropout(p = 0.25)
        self.conv4_drop = nn.Dropout(p = 0.3)
        self.conv5_drop = nn.Dropout(p = 0.35)
        self.fc1 = nn.Linear(512*3*3, 1024)
        self.fc1_drop = nn.Dropout(p = 0.4)
        self.fc2 = nn.Linear(1024, 136)
        
        #卷积层各层初始化权重和偏差
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                I.xavier_normal_(m.weight.data)
                I.constant_(m.bias.data,0.)


        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1_drop(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv2_drop(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.conv3_drop(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.conv4_drop(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = self.conv5_drop(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
