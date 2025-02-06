
import torch
import torch.nn as nn

# CNN 모델 정의

class DynamicCNN(nn.Module):   # layer 8
    def __init__(self, num_layers=8, input_channels=1, num_classes = 8):
        super(DynamicCNN, self).__init__()
        self.num_layers = num_layers
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                # self.layers.append(nn.Conv1d(self.input_channels, 16, kernel_size=3, stride=1, padding=1))
                self.layers.append(nn.Conv1d(self.input_channels, 64, kernel_size=2, stride=1, padding=1))
            else:
                # self.layers.append(nn.Conv1d(16 * (2 ** (i-1)), 16 * (2 ** i), kernel_size=3, stride=1, padding=1))
                self.layers.append(nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1))
                # self.layers.append(nn.Conv1d(64, 64, kernel_size=2, stride=1, padding=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool1d(kernel_size=2, stride=2))   # 이 부분이 있으면 여러개 layer가 64, 64 로 고정된 크기로 쌓이는데 자꾸 작아지게 되어서 결국 ouptput size 가 0 이 되는 오류가 나는 것 같음
        
        # Output layer
        # self.output_conv = nn.Conv1d(16 * (2 ** (self.num_layers - 1)), self.num_classes, kernel_size=3, stride=1, padding=1)
        self.output_conv = nn.Conv1d(64, self.num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        output = self.output_conv(x)
        # print(output.shape)
        # Flatten the output tensor
        # output = output.view(output.size(0), -1)
        output = torch.squeeze(output)
        # print(output.shape)

        return output

class CNN18(nn.Module):
    def __init__(self, input_channels=1, num_classes=8):
        super(CNN18, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        
        # 첫 번째 CNN 레이어
        self.layers.append(nn.Conv1d(input_channels, 64, kernel_size=3, stride=3, padding=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        
        # 17개의 중간 CNN 레이어
        for _ in range(17):
            self.layers.append(nn.Conv1d(64, 64, kernel_size=2, stride=1, padding=1))
            self.layers.append(nn.ReLU())
            #self.layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            if _ % 4 == 0:  # 매 2번째 레이어에서만 MaxPool1d 추가
                self.layers.append(nn.MaxPool1d(kernel_size=3, stride=3))
        # 출력 레이어
        self.layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        self.output_conv = nn.Conv1d(64, num_classes, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        output = self.output_conv(x)
        return output

class CNNModel10(nn.Module):   # layer 10
    def __init__(self, input_channels=1, num_classes=8):
        super(CNNModel10, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.layers = nn.ModuleList()

        # CNN layers
        for i in range(10):
            if i == 0:
                self.layers.append(nn.Conv1d(self.input_channels, 64, kernel_size=3, stride=1, padding=1))
            else:
                self.layers.append(nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.ReLU())
            if i % 2 == 0:  # 매 2번째 레이어에서만 MaxPool1d 추가   4번째에서 추가하면 성능 하락
                self.layers.append(nn.MaxPool1d(kernel_size=3, stride=3))        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # CNN layers
        for layer in self.layers:
            x = layer(x)
            # print(x.shape)
        
        # Flatten the output tensor
        x = x.view(x.size(0), -1)
        print(x.shape)
        # Fully connected layers
        x = self.fc1(x); # print(x.shape)
        x = torch.relu(x); # print(x.shape)
        x = self.fc2(x); # print(x.shape)

        return x

class CNNModel12(nn.Module):   # layer 12
    def __init__(self, input_channels=1, num_classes=8):
        super(CNNModel12, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.layers = nn.ModuleList()

        # CNN layers
        for i in range(12):
            if i == 0:
                self.layers.append(nn.Conv1d(self.input_channels, 64, kernel_size=3, stride=1, padding=1))
            else:
                self.layers.append(nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.ReLU())
            if i % 4 == 0:  # 매 2번째 레이어에서만 MaxPool1d 추가   4번째에서 추가하면 성능 하락
                self.layers.append(nn.MaxPool1d(kernel_size=3, stride=3))        
        # Fully connected layers
        self.fc1 = nn.Linear(64*13, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # CNN layers
        for layer in self.layers:
            x = layer(x)
            print(x.shape)
        
        # Flatten the output tensor
        x = x.view(x.size(0), -1)
        print(x.shape)
        # Fully connected layers
        x = self.fc1(x); print(x.shape)
        x = torch.relu(x); print(x.shape)
        x = self.fc2(x); print(x.shape)

        return x

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):   # layer 4
    def __init__(self, input_channels=1, num_classes=8):
        super(SimpleCNN, self).__init__()
        self.input_channels = 1
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.fc = nn.Linear(512 * 22, 3584)
        self.fc2 = nn.Linear(3584, self.num_classes)  # 수정
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # print(x.shape)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # print(x.shape)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        # print(x.shape)
        
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        # print(x.shape)
        
        # Flatten
        x = x.view(x.size(0), -1)
        # print(x.shape)
        
        # Fully connected layer
        x = self.fc(x); # print(x.shape)
        x = self.fc2(x); # print(x.shape)
        
        return x

import torch
import torch.nn as nn

class SimpleCNN2(nn.Module):   # layer 2
    def __init__(self, input_channels=1, num_classes=8):
        super(SimpleCNN2, self).__init__()
        self.input_channels = 1
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.fc = nn.Linear(128 * 90, 3584)
        self.fc2 = nn.Linear(3584, self.num_classes)  # 수정
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # print(x.shape)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # print(x.shape)
        
       
        # Flatten
        x = x.view(x.size(0), -1)
        # print(x.shape)
        
        # Fully connected layer
        x = self.fc(x); # print(x.shape)
        x = self.fc2(x); # print(x.shape)
        
        return x

import torch
import torch.nn as nn

class SimpleCNN3(nn.Module):   # layer 6
    def __init__(self, input_channels=1, num_classes=8):
        super(SimpleCNN3, self).__init__()
        self.input_channels = 1
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv6 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()
        self.maxpool6 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.fc = nn.Linear(512 * 5, 3584)
        self.fc2 = nn.Linear(3584, self.num_classes)  # 수정
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # print(x.shape)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # print(x.shape)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        # print(x.shape)
        
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        # print(x.shape)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)
        # print(x.shape)
        
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.maxpool6(x)
        # print(x.shape)
        
        # Flatten
        x = x.view(x.size(0), -1)
        # print(x.shape)
        
        # Fully connected layer
        x = self.fc(x); # print(x.shape)
        x = self.fc2(x); # print(x.shape)
        
        return x

class CNNModel10(nn.Module):   # layer 10
    def __init__(self, input_channels=1, num_classes=8):
        super(CNNModel10, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.layers = nn.ModuleList()

        # CNN layers
        for i in range(10):
            if i == 0:
                self.layers.append(nn.Conv1d(self.input_channels, 64, kernel_size=3, stride=1, padding=1))
            else:
                self.layers.append(nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.ReLU())
            if i % 2 == 0:  # 매 2번째 레이어에서만 MaxPool1d 추가   4번째에서 추가하면 성능 하락
                self.layers.append(nn.MaxPool1d(kernel_size=3, stride=3))        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # CNN layers
        for layer in self.layers:
            x = layer(x)
            # print(x.shape)
        
        # Flatten the output tensor
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # Fully connected layers
        x = self.fc1(x); # print(x.shape)
        x = torch.relu(x); # print(x.shape)
        x = self.fc2(x); # print(x.shape)

        return x

class CNNModel12(nn.Module):   # layer 12
    def __init__(self, input_channels=1, num_classes=8):
        super(CNNModel12, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.layers = nn.ModuleList()

        # CNN layers
        for i in range(12):
            if i == 0:
                self.layers.append(nn.Conv1d(self.input_channels, 64, kernel_size=3, stride=1, padding=1))
            else:
                self.layers.append(nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.ReLU())
            if i % 4 == 0:  # 매 2번째 레이어에서만 MaxPool1d 추가   4번째에서 추가하면 성능 하락
                self.layers.append(nn.MaxPool1d(kernel_size=3, stride=3))        
        # Fully connected layers
        self.fc1 = nn.Linear(64*13, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # CNN layers
        for layer in self.layers:
            x = layer(x)
            # print(x.shape)
        
        # Flatten the output tensor
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # Fully connected layers
        x = self.fc1(x); # print(x.shape)
        x = torch.relu(x); # print(x.shape)
        x = self.fc2(x); # print(x.shape)

        return x