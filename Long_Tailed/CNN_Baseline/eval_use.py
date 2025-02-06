
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np

from utils import mic_acc_cal, shot_acc
from utils2 import pred_test, cal_score

import time

def pred_test(data_type, model, device, dataloaders, num_classes):
    label_list = []
    file_list = []
    prob_list = []
    pred_list = []
    confusion_matrix = np.zeros((num_classes, num_classes))
    start = time.time()

    model.eval()
    # model = model.float()   # input(torch.FloatTensor), model type(torch.cuda.FloatTensor)이 안 맞아서 맞춰주는 코드
    with torch.no_grad():
        # for inputs, labels, file_id in dataloaders:
        for i, (images, labels, paths) in enumerate(dataloaders, 0):
            
            if data_type == '1D': images = images.unsqueeze(1)
            else: pass

            inputs, labels = images.to(device, dtype = torch.float), labels.to(device)

            label_list.extend(labels.to("cpu").detach().numpy())
            file_list.extend(paths)
            outputs = model(inputs)
            # logits, _ = model(inputs)
            # _, preds = torch.max(outputs, 1)

#             total_logits = torch.cat((total_logits, logits))
#             total_labels = torch.cat((total_labels, labels))
            
            # probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1)
            probs, preds = F.softmax(outputs.detach(), dim=1).max(dim=1)
            prob_list.extend(probs)
            pred_list.extend(preds.to("cpu").detach().numpy())
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
                # print(t, p)
    end = time.time() - start
    return confusion_matrix, file_list, pred_list, label_list

def cal_score(df):
    FP = df.sum(axis = 0) - np.diag(df); FN = df.sum(axis = 1) - np.diag(df)
    TP = np.diag(df); TN = np.array(df).sum() - (FP + FN + TP)
    ACC = np.array((TP + TN)/(TP + TN + FP + FN))
    ACC2 = TP.sum()/(np.array(df).sum())
    PPV = np.array(TP/(TP+FP))
    SEN = np.array(TP/(TP+FN))
    NPV = np.array(TN/(TN+FN))
    SPC = np.array(TN/(TN+FP))
    F1 = np.array(2*PPV*SEN/(PPV + SEN))

    return round(pd.DataFrame([ACC, PPV, SEN, NPV, SPC, F1]).T, 4)

def cal_score2(df, num_classes, pred_time):
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