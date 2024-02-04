from torch.utils.data import Dataset
import cv2
import os
from typing import Callable, Tuple, Dict
import torch
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, signal_data, signal_list, signal_type, label_, transform):#: Callable = None):
        self.signals = signal_data
        self.list_ = signal_list
        self.signal_type = signal_type
        self.transform = transform
        self.label_ = label_
        
    def __len__(self):
        return len(self.list_)
    
    def __getitem__(self, index):
#         print(self.transform)
        signal = self.signals[self.list_[index]]['signal'][self.signal_type]
        #print(signal)
        
        if self.transform is not None:
            signal = self.transform(signal)

        else:
            signal =  torch.from_numpy(np.array(signal))
#             signal = signal.unsqueeze(1)
            
        #label = self.signals[self.list_[index]]['label'] <- dict 에서 label을 바로 뽑아오는 경우
        label = self.label_[index]   # <- encoding 해서 label을 붙이는 경우
        sub_id = self.list_[index]
        
        return signal, label, sub_id

def generate_dataset(data, signal_list, signal_type, label_, transform):#: Callable = None):
    
    return CustomDataset(data, signal_list, signal_type, label_, transform)#=None)