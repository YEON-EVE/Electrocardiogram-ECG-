import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pandas as pd
import numpy as np

from data.ClassAwareSampler import get_sampler

import models as Sig1D
import copy
import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

seed = 2021
deterministic = True

# random.seed(seed)
# np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

class EarlyStopping(object):
    def __init__(self, patience=2, save_path="model.pth"):
        self._min_loss = np.inf
        self._patience = patience
        self._path = save_path
        self.__counter = 0
 
    def should_stop(self, model, loss):
        if loss < self._min_loss:
            self._min_loss = loss
            self.__counter = 0
            self.__counter = 0
            # torch.save(model.state_dict(), self._path)
        elif loss > self._min_loss:
            self.__counter += 1
            if self.__counter >= self._patience:
                return True
        return False
   
    def load(self, model):
        model.load_state_dict(torch.load(self._path))
        return model
    
    @property
    def counter(self):
        return self.__counter

def pretrain_model(data_type, save_root, fold_num, dataloaders, layers = 8, num_cls = 8, n_epoch = 15, device = 'cuda'):

    print('Start pretraining fold {} model {}'.format(fold_num, layers))

    sampler_dic = {'type': 'ClassAwareSampler', 'def_file': './data/ClassAwareSampler.py', 'num_samples_cls': 4, 'sampler':get_sampler()}

    early_stopper = EarlyStopping(patience = 10)

    # model = Sig1D.DynamicCNN(num_layers=layers)
    # model = Sig1D.CNN18()
    model = Sig1D.CNNModel12()

    #if torch.cuda.device_count() > 1:   # GPU 병렬 사용을 위한 설정
    #    model = nn.DataParallel(model)
    #else:
    #    pass

    num_layers = sum(1 for _ in model.children())
    # print(f"The model has {num_layers} layers.")
    # print(model)

    model.to(device)


    # weights_ = [0.273, 0.948, 0.909, 0.970, 1.0, 0.931, 0.995, 0.974]
    # weights_ = torch.FloatTensor().to(device)
    loss_lambda = 0.5
    ce_loss = nn.CrossEntropyLoss()

    #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    train_running_corrects = 0
    val_running_corrects = 0
    running_corrects = 0

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1.0   # 평균적으로 1보다 작은 loss가 나오기 때문에 초기값을 1로 설정
    best_epoch = 0
    best_acc = 0.0

    for epoch in range(n_epoch):
        for phase in ['train', 'val']:   # epoch 중 최적의 모델을 찾기 위해 validation 과정 추가
            #print(phase)
            running_corrects = 0
            if phase == 'train':
                model.train()
            else:
                model.eval()   # validation phase에서는 가중치 update X
 
            loss_all = 0.0

            for i, (images, labels, paths) in enumerate(dataloaders['{}_loader'.format(phase)], 0):
                # print(i)
                # 1d signal인 경우 shape을 맞춰주는 작업 필요
                if (data_type == '1D') | (data_type == 'csv'): images = images.unsqueeze(1)
                else: pass

                images, labels = images.to(device, dtype = torch.float), labels.to(device)   # random sampling 에서 구한 loss와

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(images)   # output shape: (batch_size, 512, 1)?
                    output_all = output   # dataloader에 따라 아래에서 output_c 와 output_r 로 나눠지는데, epoch acc를 구할 때 전체가 필요하므로 하나 남겨둠
                    # print(output.shape)

                    loss = ce_loss(output, labels)
                    # loss = ce_loss(output.squeeze(2), labels)

                    if phase == 'train':   # phase가 train인 경우 update
                        loss.backward()
                        optimizer.step()

                # epoch loss
                loss_all += loss.item()

                # epoch accuracy   **
                probs, preds = F.softmax(output_all.detach(), dim=1).max(dim=1)   # output의 shape때문에 softmax를 거쳐야 class를 예측할 수 있음  
                # print(len(preds), len(labels_all))
                # running_corrects += torch.sum(preds == labels.data)
                # running_corrects += torch.sum(preds == labels.data[0])   # CNN18 돌릴 때
                running_corrects += torch.sum(preds == labels.data)
                #print(running_corrects)
                #print(len(dataloaders['{}_plain_loader'.format(phase)].dataset))

            scheduler.step(epoch)
            # epoch accuracy   **
            epoch_acc = running_corrects.double() / (len(dataloaders['{}_loader'.format(phase)].dataset))   # acc 를 구할 때 쓴 output_all 은 batchsize가 동일한 dataloader를 2개 쓴거나 마찬가지이므로, 2배 해줘야 함

            # loss 계산해서 epoch loss 확인
            epoch_loss = loss_all / len(dataloaders['{}_loader'.format(phase)].dataset)   # loss 는 위에서 보면 M개씩 나눠서 더하기 때문에 결국 개수는 batch size와 동일 -> 전체 데이터셋 길이로 나눠도 됨

            if phase == 'val' and epoch_loss < best_loss:   # validation phase에서 성능이 더 높은 모델 가중치 저장
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                val_loss = epoch_loss

            else:
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)

        if early_stopper.should_stop(model, val_loss):
            print(f"EarlyStopping: [Epoch: {epoch - early_stopper.counter}]")
            break
    print('Epoch[{}, {}] {} best loss: {:.4f} and best acc: {:.4f}'.format(best_epoch, n_epoch, fold_num, best_loss, best_acc))

    # dataframe save
    epoch_df = pd.DataFrame(None)
    epoch_df['epoch_loss'] = [train_loss_history[i] for i in range(len(train_loss_history))]
    epoch_df['epoch_acc'] = [train_acc_history[i].item() for i in range(len(train_acc_history))]
    epoch_df['epoch_loss'] = [val_loss_history[i] for i in range(len(val_loss_history))];
    epoch_df['epoch_acc'] = [val_acc_history[i].item() for i in range(len(val_acc_history))]
    epoch_df['fold'] = fold_num

    # model save
    model.load_state_dict(best_model_wts)
    torch.save(model, os.path.join(save_root, 'BaseLine_{}_fold{}_{}.pth'.format(data_type, fold_num, layers)))
    #loaded_model = torch.load('model.pth')
    return model, epoch_df