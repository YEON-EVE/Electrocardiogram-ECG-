import numpy as np
import pandas as pd
import time
import random
import os
from sklearn.metrics import roc_auc_score

from PIL import Image
import copy
import cv2

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

from EctopicBeat_GenerateDataset import generate_dataset
from EctopicBeat_GenerateModel import GenerateModel
from EctopicBeat_foldsplit import return_fold

import pywt
import matplotlib.pyplot as plt

import skimage.io

from ssqueezepy import cwt
from ssqueezepy.visuals import plot, imshow

import warnings
warnings.filterwarnings(action='ignore')


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

from datetime import datetime
import pickle

# class config:
#    seed = 11
        
#    #lr = 1e-3
#    #epochs = 25
#    #batch_size = 32
#    #num_workers = 4
#    #train_5_folds = True

# def seed_everything(seed: int = 42):
#    random.seed(seed)
#    np.random.seed(seed)
#    os.environ["PYTHONHASHSEED"] = str(seed)
#    torch.manual_seed(seed)
#    torch.cuda.manual_seed(seed)  # type: ignore
#    torch.backends.cudnn.deterministic = True  # type: ignore
#    torch.backends.cudnn.benchmark = True  # type: ignore


def fold_train(model_name,
               device,
               num_classes,
               num_epochs,
               feature_extract,
               train_shuffle,
               batch_size, 
               num_workers,
               csv_path,
               pickle_path,
               target,
               data_transform_):

    print("Start")
    #seed_everything(config.seed)
    #output path 지정
    save_root = "./output_{}".format(datetime.today().strftime("%Y%m%d"))
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    csv_save = os.path.join(save_root, "CSV")
    model_save = os.path.join(save_root, "Model")

    for dir_ in [csv_save, model_save]:
        if not os.path.exists(dir_):
            os.mkdir(dir_)

    #fold 구분
    test_csv, train_valid, fold2 = return_fold(csv_path, nsplit_1 = 5, nsplit_2 = 5, target = target)
    if num_classes == 2:
        class_names = ['Normal', 'Abnormal']
    else:
        class_names = ['Normal', 'A', 'V']
    columns_list = ['ACC', 'ACC2', 'PPV', 'SEN', 'NPV', 'SPC', 'F1', 'Pred_time', 'AUC'] #'TP', 'TN', 'FP', 'FN', 
    score_df = pd.DataFrame(None, columns = columns_list)

    #df for prob_list; test dataset
    df_prob = pd.DataFrame(None)
    df_prob['subject_num'] = test_csv['subject_num']
    df_prob['true_label'] = test_csv[target]

    # load
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    for i in range(len(fold2)):
        model_ft, input_size = GenerateModel(model_name, num_classes, feature_extract, use_pretrained = True)

        data_transform = {}
        data_transform['CWT'] = transforms.Compose([   # resize 필요할 것 같은데
            return_CWT(),
            transforms.Resize((input_size,input_size)),
            transforms.ToTensor()
        ])
        data_transform['Raw'] = None
        
        train_idx = fold2[i][0]
        valid_idx = fold2[i][1]

        train_dataset = generate_dataset(data, 
                                         signal_list = list(train_valid.loc[train_idx]['subject_num']), 
                                         signal_type = 'MLII', 
                                         label_ = list(train_valid.loc[train_idx][target]),
                                         transform = data_transform[data_transform_])
        valid_dataset = generate_dataset(data, 
                                         signal_list = list(train_valid.loc[valid_idx]['subject_num']), 
                                         signal_type = 'MLII', 
                                         label_ = list(train_valid.loc[valid_idx][target]),
                                         transform = data_transform[data_transform_])
        test_dataset = generate_dataset(data, 
                                         signal_list = list(test_csv['subject_num']), 
                                         signal_type = 'MLII', 
                                         label_ = list(test_csv[target]),
                                         transform = data_transform[data_transform_])

        image_datasets = {'train' : train_dataset,
                          'val' : valid_dataset}
                          #'test' : test_dataset}
        dataloaders = {x : DataLoader(image_datasets[x], batch_size = batch_size,
                                           shuffle = train_shuffle, num_workers = num_workers, drop_last = True) for x in ['train', 'val']}
        dataloaders_test = DataLoader(test_dataset, batch_size = batch_size,
                                              shuffle = False, num_workers = num_workers, collate_fn = collate_fn)
        # ddd = iter(dataloaders['test'])
        # ddd.next()
        if target == 'Score4':
            weights = [0.5, 1.0]
            class_weights = torch.FloatTensor(weights).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            # print(target)
        else:
            criterion = nn.CrossEntropyLoss()
            # print('No Weight')
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.005, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        #model training
        model_ft = model_ft.to(device)
        model, val_acc_history = train(model_ft, device, dataloaders, criterion, optimizer_ft, num_epochs)

        #test prediction
        confusion_matrix, pred_time, label_list, prob_list, sub_list = pred_test(model, device, dataloaders_test, num_classes)

        #print(confusion_matrix)
        con_df = pd.DataFrame(confusion_matrix, index = class_names, columns = class_names).astype(int)
        #print(con_df)
        score_dic = cal_score(con_df, num_classes, pred_time)

        if num_classes == 2:
            roc_score = roc_auc_score(label_list, prob_list)
            score_dic['AUC'] = roc_score
        else:
            score_dic['AUC'] = np.nan
        print(score_dic)
        score_df = score_df.append(score_dic, ignore_index = True)

        #df_prob update
        df_prob_temp = pd.DataFrame({'subject_num' : sub_list,
                                     'prob_{0}'.format(i) : prob_list})
        df_prob = pd.merge(df_prob, df_prob_temp, on = 'subject_num', how = 'outer')
        #model save
        torch.save(model, os.path.join(model_save, "{}_{}_{}.pt".format(target, model_name, i+1)))
        torch.save(model.state_dict(), os.path.join(model_save, "{}_{}_{}_state_dict.pt".format(target, model_name, i+1)))
        # print(score_df)
    #score dataframe save
    score_df.to_excel(os.path.join(csv_save, "{}_{}_result.xlsx".format(target, model_name)))
    df_prob.to_excel(os.path.join(csv_save, "{}_{}_prob_result.xlsx".format(target, model_name)))
    print("{} {} Success".format(target, model_name))
    return score_df

def train(model, device, dataloaders, criterion, optimizer, num_epochs):#, is_inception=False):
    # training code
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    print('Train start')

    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('train Start')
        # print('-'*10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels, sub_id in dataloaders[phase]:
                if len(inputs.shape) == 2:
                    inputs = inputs.unsqueeze(1)
                inputs = inputs.to(device, dtype = torch.float)
                labels = labels.to(device, dtype = torch.int64)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        # print()
    time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def pred_test(model, device, dataloaders, num_classes):
    label_list = []
    sub_list = []
    prob_list = []
    confusion_matrix = np.zeros((num_classes, num_classes))
    start = time.time()
    with torch.no_grad():
        for inputs, labels, sub_id in dataloaders:
            if len(inputs.shape) == 2:
                inputs = inputs.unsqueeze(1)
            inputs = inputs.to(device, dtype = torch.float)
            labels = labels.to(device)

            label_list.extend(labels.to("cpu").detach().numpy())
            sub_list.extend(sub_id)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        
            sig_outputs = torch.sigmoid(outputs)
            sig_prob = sig_outputs[:, 1]
            prob_list.extend(sig_prob.squeeze().to("cpu").detach().numpy()
                            ) 
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    end = time.time() - start
    return confusion_matrix, end, label_list, prob_list, sub_list

def cal_score(df, num_classes, pred_time):
    if num_classes ==2:
        TP = df.iloc[1, 1]; TN = df.iloc[0, 0]; FP = df.iloc[0, 1]; FN = df.iloc[1, 0]
        ACC = (TP + TN)/(TP + TN + FP + FN)
        ACC2 = np.nan
        PPV = TP/(TP + FP)
        SEN = TP/(TP + FN)
        NPV = TN/(TN + FN)
        SPC = TN/(TN + FP)
        F1 = 2*PPV*SEN/(PPV + SEN)

    else:
        FP = df.sum(axis = 0) - np.diag(df); FN = df.sum(axis = 1) - np.diag(df)
        TP = np.diag(df); TN = np.array(df).sum() - (FP + FN + TP)
        ACC = np.nanmean(np.array((TP + TN)/(TP + TN + FP + FN)))
        ACC2 = TP.sum()/(np.array(df).sum())
        PPV = np.nanmean(np.array(TP/(TP+FP)))
        SEN = np.nanmean(np.array(TP/(TP+FN)))
        NPV = np.nanmean(np.array(TN/(TN+FN)))
        SPC = np.nanmean(np.array(TN/(TN+FP)))
        F1 = np.nanmean(np.array(2*PPV*SEN/(PPV + SEN)))

    return {#'TP' : TP, 'TN' : TN, 'FP' : FP, 'FN' : FN,
            'ACC' : ACC, 'ACC2' : ACC2, 'PPV' : PPV, 'SEN' : SEN, 'NPV' : NPV,
            'SPC' : SPC, 'F1' : F1, 'Pred_time' : pred_time}

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def return_CWTimg(signal):
    data1 = np.array(signal)
    Wx, scales = cwt(data1, 'morlet')
    Wx = abs(Wx)
    
    img = scale_minmax(Wx, 0, 255).astype(np.uint8)
    img = np.flip(img, axis = 0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy
    img = Image.fromarray(img)
    
    return img

class return_CWT(object):
    def __call__(self, signal):
        img = return_CWTimg(signal)
        return img

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)