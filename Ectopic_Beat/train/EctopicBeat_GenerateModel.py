import torchvision
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import copy

import warnings
warnings.filterwarnings(action='ignore')

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def GenerateModel(model_name, num_classes, feature_extract, use_pretrained = True):   # 모델을 다 만들 필요는 없어서, model_type을 받아서 그것만 만드는게 나을 것 같은데, if 문으로 하면 너무 조건이 많은 것 같은데
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    #print(str(model_name))

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        model_ft.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    if model_name == "resnet50":
        """ Resnet50
        """
        # print("True")
        model_ft = models.resnet50(pretrained=use_pretrained)
        model_ft.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    if model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        model_ft.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    elif model_name == "CDD1d":
        model_ft = CNN1d(num_classes)
        input_size = None

    elif model_name == "CNN_LSTM":
        model_ft = Conv1d_LSTM()
        input_size = None

    # else:
    #     print("Invalid model name, exiting...")
    #     exit()

    return model_ft, input_size

class CNN1d(nn.Module):
    """1D convolutional neural network. Classifier of the gravitational waves.
    Architecture from there https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.120.141103
    """

    def __init__(self, num_classes, debug=False):
        super().__init__()
        self.num_classes = num_classes
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5),
            nn.BatchNorm1d(64),
            nn.ELU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.AvgPool1d(kernel_size=8),
            nn.BatchNorm1d(64),
            nn.ELU(),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ELU(),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3),
            nn.AvgPool1d(kernel_size=6),
            nn.BatchNorm1d(128),
            nn.ELU(),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ELU(),
        )
        self.cnn6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3),
            nn.MaxPool1d(kernel_size=4),
            nn.BatchNorm1d(256),
            nn.ELU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.ELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.ELU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, self.num_classes),
        )
        self.debug = debug

    def forward(self, x, pos=None):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class Conv1d_LSTM(nn.Module):
    def __init__(self, in_channel=1, out_channel=2):
        super(Conv1d_LSTM, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=in_channel,
                                out_channels=16,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=16,
                                out_channels=32,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        
        self.lstm = nn.LSTM(input_size=32,
                            hidden_size=50,
                            num_layers=1,
                            bias=True,
                            bidirectional=False,
                            batch_first=True)
        
        self.dropout = nn.Dropout(0.5)

        self.dense1 = nn.Linear(50, 32)
        self.dense2 = nn.Linear(32, out_channel)

    def forward(self, x):
        # Raw x shape : (B, S, F) => (B, 10, 3)
        
        # Shape : (B, F, S) => (B, 3, 10)
#         x = x.transpose(1, 2)
        # Shape : (B, F, S) == (B, C, S) // C = channel => (B, 16, 10)
        x = self.conv1d_1(x)
        # Shape : (B, C, S) => (B, 32, 10)
        x = self.conv1d_2(x)
        # Shape : (B, S, C) == (B, S, F) => (B, 10, 32)
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        # Shape : (B, S, H) // H = hidden_size => (B, 10, 50)
        _, (hidden, _) = self.lstm(x)
        # Shape : (B, H) // -1 means the last sequence => (B, 50)
        x = hidden[-1]
        
        # Shape : (B, H) => (B, 50)
        x = self.dropout(x)
        
        # Shape : (B, 32)
        x = self.dense1(x)
        # Shape : (B, O) // O = output => (B, 1)
        x = self.dense2(x)

        return x