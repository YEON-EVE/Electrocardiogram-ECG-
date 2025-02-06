
import torch
import torch.nn as nn
import torch.optim as optim
import os

from data.dataloader_1d import load_data1d   # signal 형태의 데이터로더
from data.ClassAwareSampler import get_sampler
# from pretrain import pretrain_model
from pretrain_validation import pretrain_model

from foldsplit import return_fold
from eval_use import *
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import warnings
warnings.filterwarnings('ignore')


def fold_train(data_type, data_root, save_root, csv_path, device, batch_size, layers, sampler_dic, n_epoch, num_workers):

    # 저장위치

    # fold split

    # data_root = "E:\\ECG\\data\\Long_Tailed\\signal_image"
    # save_root = "E:\\ECG\\LongTailed\\GIST"
    # sampler_dic = {'type': 'ClassAwareSampler', 'def_file': '.\\data\\ClassAwareSampler.py', 'num_samples_cls': int(70), 'sampler':get_sampler()}   # 이건 GISTNet 일 때 설정. CNN에서는 필요 없음
    # device = 'cuda:0'

    # test_csv, train_valid, fold2 = return_fold(csv_path, nsplit_1 = 5, nsplit_2 = 5) # <- test dataset을 따로 떼어둘 때
    # 아래는 test dataset을 따로 떼어두지 않을 때
    train_valid = pd.read_excel(data_root)
    train_valid = train_valid.dropna(subset = ['taskB']).reset_index(drop = True)
    train_valid['taskB'] = train_valid['taskB'].astype(int)
    fold2 = []
    cvS = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
    y = train_valid['taskB']

    for train_idxs, test_idxs in cvS.split(train_valid, y):
        fold2.append([train_idxs, test_idxs])

    val_col = ['file_list', 'pred_list', 'label_list', 'fold']
    val_df = pd.DataFrame(None, columns = val_col)
    epoch_df_pre = pd.DataFrame(None)
    epoch_df = pd.DataFrame(None)
    #class_names = ['1', '2', '3', '4', '5', '6', '7', '8']
    #class_names = ['N', 'L', 'R', 'A', 'E', 'V', '!', '/']
    score_col = ['ACC', 'PPV', 'SEN', 'NPV', 'SPC', 'F1', 'fold']
    test_score = pd.DataFrame(None, columns = score_col)
    # evaluation code 추가해야하고
    # 각 fold 에서 test set의 정답과 pred 값을 저장해야함. 왜냐하면 다른 비교 논문에서 그런식으로 진행한 것으로 보임
    # fold마다 csv를 빼면 너무 복잡해지니까 concat 해서 하나만 마지막에 저장하는걸로

    if data_type == '1D':
        with open(data_root, 'rb') as f:
            data = pickle.load(f)
    # elif data_type == 'csv':
    #     data = pd.read_excel(data_root)
    else:
        data = 'pass'
        # pass

    for i in range(len(fold2)):
        print("fold {} start".format(i))
        train_idx = fold2[i][0]
        valid_idx = fold2[i][1]

        # print(train_valid.head())

        dataloaders = {}
        if (data_type == '1D') | (data_type == 'csv'):
            dataloaders['train_loader'] = load_data1d(data, dataset=train_valid.iloc[train_idx], phase='train', batch_size=batch_size, sampler_dic=sampler_dic, num_workers=num_workers, test_open=False, shuffle=True)
            dataloaders['val_loader'] = load_data1d(data, dataset=train_valid.iloc[valid_idx], phase='test', batch_size=batch_size, sampler_dic=sampler_dic, num_workers=num_workers, test_open=False, shuffle=True)
        else:
            dataloaders['train_loader'] = load_data(data_root, dataset=train_valid.iloc[train_idx], phase='train', batch_size=batch_size, sampler_dic=sampler_dic, num_workers=num_workers, test_open=False, shuffle=True)
            dataloaders['val_loader'] = load_data(data_root, dataset=train_valid.iloc[valid_idx], phase='test', batch_size=batch_size, sampler_dic=sampler_dic, num_workers=num_workers, test_open=False, shuffle=True)

        #train_loader = load_data(data_root, dataset=train_valid.iloc[train_idx], phase='train', batch_size=batch_size, sampler_dic=sampler_dic, num_workers=0, test_open=False, shuffle=True)
        #train_plain_loader = load_data(data_root, dataset=train_valid.iloc[train_idx], phase='train', batch_size=batch_size, sampler_dic=None, num_workers=0, test_open=False, shuffle=True)

        ## model 에서 저장한 모델을 model_f 학습할 때 넘겨줘야함. save root를 model_root 로 넘겨주면 될 것 같음
        model, epoch_df_tmp = pretrain_model(data_type, save_root, i, dataloaders, layers = layers, num_cls = 8, n_epoch = n_epoch, device = device)
        
        #model_root = os.path.join(save_root, 'BaseLine_{}_fold{}_{}.pth'.format(data_type, fold_num, sum(layers)))
        #model_f = torch.load(model_root)

        epoch_df = pd.concat([epoch_df, epoch_df_tmp], axis = 0)
        ## evaluation
        
        # validation
        conf_matrix, file_list, pred_list, label_list = pred_test('1D', model, device, dataloaders['val_loader'], 8)
        temp = pd.DataFrame({"file_list":file_list, "pred_list":pred_list, "label_list":label_list})
        temp['fold'] = i

        val_df = pd.concat([val_df, temp])

        # test- test dataset을 따로 떼어두지 않을 때는 아래 코드를 통으로 실행 X
        #test_loader = load_data1d(data, dataset=test_csv, phase='test', batch_size=batch_size, sampler_dic=none, num_workers=0, test_open=false, shuffle=true)
        #conf_matrix, file_list, pred_list, label_list = pred_test('1d', model_f, device, test_loader, 8)
        #temp_test = pd.dataframe(conf_matrix).astype(int)
        #score_df = cal_score(temp_test)
        #score_df['fold'] = i
        #score_df.columns = score_col

        #test_score = pd.concat([test_score, score_df])

    #test_score.to_excel(os.path.join(save_root, 'test_score_{}_{}.xlsx'.format(data_type, sum(layers))), index = True)
    val_df.to_excel(os.path.join(save_root, 'BaseLine_val_pred_{}_{}.xlsx'.format(data_type, layers)), index = False)
    epoch_df.to_excel(os.path.join(save_root, 'BaseLine_{}_{}_epochINFO.xlsx'.format(data_type, layers)), index = True)