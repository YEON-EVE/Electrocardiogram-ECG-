import random
import numpy as np
from torch.utils.data.sampler import Sampler

##################################
## Class-aware sampling, partly implemented by frombeijingwithlove
##################################


class RandomCycleIter:   # 이거는 data 반환하는 class 같은데, phase에 따라서 random하게 반환할지를 정하는 것 같음
    
    def __init__(self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode
        
    def __iter__(self):
        return self
    
    def __next__(self):
        self.i += 1
        
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)
            
        return self.data_list[self.i]


def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):   # num_samples_cls 는 class 당 sample 개수

    i = 0
    j = 0
    while i < n:
        if j >= num_samples_cls:
            j = 0
    
        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]]*num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]
        
        i += 1
        j += 1


class ClassAwareSampler(Sampler):

    def __init__(self, data_source, num_samples_cls=1,):   # data_source 는 dataset
        num_classes = len(np.unique(data_source.labels))
        self.class_iter = RandomCycleIter(range(num_classes))
        cls_data_list = [list() for _ in range(num_classes)]   # class 수만큼 빈 리스트 생성
        for i, label in enumerate(data_source.labels):
            cls_data_list[label].append(i)   # [0, 1, 2, 4, 5, 6] class 만 있었던 경우 cls_data_list[6] 이면 list index out of range 오류 발생 -> group k fold 때문에 발생하는 문제(fold별로 class가 골고루 있을 수 없어서)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)
        self.num_samples_cls = num_samples_cls

    def __iter__(self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)
    
    def __len__(self):
        return self.num_samples


def get_sampler():
    return ClassAwareSampler

##################################