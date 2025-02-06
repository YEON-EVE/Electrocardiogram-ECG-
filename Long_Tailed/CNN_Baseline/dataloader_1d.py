from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
# import cv2
import os
import random
from typing import Callable, Tuple, Dict
import torch
import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.neighbors import NearestNeighbors

data_transforms = {}
# data_transforms['CWT'] = transforms.Compose([   # resize 필요할 것 같은데
#    return_CWT(),
#    transforms.Resize((input_size,input_size)),
#    transforms.ToTensor()
#])
#data_transforms['Raw'] = None

class CustomDataset(Dataset):
    def __init__(self, signal_data, signal_list, signal_type, label_, transform):#: Callable = None):
        self.signals = signal_data
        self.list_ = signal_list
        self.signal_type = signal_type
        self.transform = transform
        self.labels = label_
        
    def __len__(self):
        return len(self.list_)
    
    def __getitem__(self, index):
#         print(self.transform)
        if isinstance(self.signals, pd.DataFrame):
            signal = self.signals.iloc[index, :360]
        else:
            signal = self.signals[self.list_[index]]['signal'][self.signal_type]
        #print(signal)

        if self.transform is not None:
            signal = self.transform(signal)

        else:
            signal =  torch.from_numpy(np.array(list(signal)))
#             signal = signal.unsqueeze(1)
        # signal = signal.unsqueeze(1)   # 여기서 바꾸면 안됨.이러면 신호 하나만 형태가 바껴서 다 합치면 원하는 shape이 안 나옴
        #label = self.signals[self.list_[index]]['label'] <- dict 에서 label을 바로 뽑아오는 경우
        label = self.labels[index]   # <- encoding 해서 label을 붙이는 경우
        sub_id = self.list_[index]
        
        return signal, label, sub_id

def load_data1d(data, dataset, phase, batch_size, sampler_dic=None, num_workers=4, test_open=False, shuffle=True):   # 이 부분을 다시 만들어야 하는데,,, dataset 을 그냥 csv 경로로 받으면 될듯??

    # else까지는 smote 할 때만 주석처리
    #if os.path.isfile(dataset):   # 여기는 dataset이 경로인 경우 읽어오고, dataframe인 경우 바로 진행
    if type(dataset) == str:
        csv = pd.read_excel(dataset)
    else:
        csv = dataset.copy().reset_index(drop = True)

    # sampling 이 training set에만 적용되어야 함
    # over sampling
    if phase == 'train' and sampler_dic == 'over':
        y = csv['taskB']; x = csv
        ros = RandomOverSampler(random_state=0)
        csv, y_resampled = ros.fit_resample(x, y)
        # print(phase, csv.shape)

    elif phase == 'train' and sampler_dic == 'under':
        y = csv['taskB']; x = csv
        ros = RandomUnderSampler(random_state=0)
        csv, y_resampled = ros.fit_resample(x, y)
        # print(phase, csv.shape)
        
    elif phase == 'train' and sampler_dic == 'smote':
        ecg_few4 = csv[csv['taskB'].isin([4])].reset_index(drop = True)
        ecg_few4 = pd.get_dummies(ecg_few4, columns = ['taskB'])
        # print(ecg_few4.head())
        y4_sub = ecg_few4[['taskB_4']]
        X4_sub = ecg_few4.iloc[:, :360]
        # print(X4_sub.head())
        X4_res, y4_res = MLSMOTE(X4_sub, y4_sub, 1000, 5)  # Applying MLSMOTE to augment the dataframe

        ecg_few6 = csv[csv['taskB'].isin([6])].reset_index(drop = True)
        ecg_few6 = pd.get_dummies(ecg_few6, columns = ['taskB'])
        y6_sub = ecg_few6[['taskB_6']]
        X6_sub = ecg_few6.iloc[:, :360]
        X6_res, y6_res = MLSMOTE(X6_sub, y6_sub, 1000, 5)  # Applying MLSMOTE to augment the dataframe

        X4_res['taskB'] = 4; X6_res['taskB'] = 6
        X4_res['subject_num'] = [i+'_'+j for i, j in zip(list(map(str, range(1000))), ['SM4']*1000)]
        X6_res['subject_num'] = [i+'_'+j for i, j in zip(list(map(str, range(1000))), ['SM6']*1000)]
        csv = pd.concat([csv, X4_res, X6_res])

    else:
        pass



    file_list = list(csv['subject_num'])   # column 이름 확인해서 수정. 1d 는 pickle 파일에 subject_num으로 특정되어 있음
    label_list = list(csv['taskB'])

    #print('Loading data from %s' % (dataset))

    # set_ = CustomDataset(data, file_list, 'MLII', label_list, transform = None)   # 원래 CNN baseline
    set_ = CustomDataset(csv, file_list, 'MLII', label_list, transform = None)   # smote 할 때


 
    return DataLoader(dataset=set_, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)

 
def get_tail_label(df: pd.DataFrame, ql=[0.05, 1.]) -> list:
    """
    Find the underrepresented targets.
    Underrepresented targets are those which are observed less than the median occurance.
    Targets beyond a quantile limit are filtered.
    """
    irlbl = df.sum(axis=0)
    irlbl = irlbl[(irlbl > irlbl.quantile(ql[0])) & ((irlbl < irlbl.quantile(ql[1])))]  # Filtering
    irlbl = irlbl.max() / irlbl
    threshold_irlbl = irlbl.median()
    tail_label = irlbl[irlbl > threshold_irlbl].index.tolist()
    return tail_label

def get_minority_samples(X: pd.DataFrame, y: pd.DataFrame, ql=[0.05, 1.]):
    """
    return
    X_sub: pandas.DataFrame, the feature vector minority dataframe
    y_sub: pandas.DataFrame, the target vector minority dataframe
    """
    tail_labels = get_tail_label(y, ql=ql)
    index = y[y[tail_labels].apply(lambda x: (x == 1).any(), axis=1)].index.tolist()
    
    X_sub = X[X.index.isin(index)].reset_index(drop = True)
    y_sub = y[y.index.isin(index)].reset_index(drop = True)
    return X_sub, y_sub

def nearest_neighbour(X: pd.DataFrame, neigh) -> list:
    """
    Give index of 10 nearest neighbor of all the instance
    
    args
    X: np.array, array whose nearest neighbor has to find
    
    return
    indices: list of list, index of 5 NN of each element in X
    """
    nbs = NearestNeighbors(n_neighbors=neigh, metric='euclidean', algorithm='kd_tree').fit(X)
    euclidean, indices = nbs.kneighbors(X)
    return indices

def MLSMOTE(X, y, n_sample, neigh=5):
    """
    Give the augmented data using MLSMOTE algorithm
    
    args
    X: pandas.DataFrame, input vector DataFrame
    y: pandas.DataFrame, feature vector dataframe
    n_sample: int, number of newly generated sample
    
    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
    """
    indices2 = nearest_neighbour(X, neigh=5)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0, n-1)
        neighbor = random.choice(indices2[reference, 1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis = 0, skipna = True)
        target[i] = np.array([1 if val > 0 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference,:] - X.loc[neighbor,:]
        new_X[i] = np.array(X.loc[reference,:] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    return new_X, target