from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import pandas as pd


def return_fold(csv_path, nsplit_1 = 10, nsplit_2 = 10):
    all_csv = pd.read_excel(csv_path)
    all_csv = all_csv.dropna(subset = ['taskB']).reset_index(drop = True)
    all_csv['taskB'] = all_csv['taskB'].astype(int)

    y = all_csv['taskB']
    x = all_csv

    fold = []; fold2 = []
    cvS = StratifiedKFold(n_splits=nsplit_1, shuffle=True, random_state=11)
    y = all_csv['taskB']

    for train_idxs, test_idxs in cvS.split(all_csv, y):
        fold.append([train_idxs, test_idxs])

    test_csv = all_csv.loc[fold[3][1]].reset_index(drop = True)

    train_valid = all_csv.loc[fold[3][0]].reset_index(drop = True)
    
    cvS2 = StratifiedKFold(n_splits=nsplit_2, shuffle=True, random_state=11)#, random_state=11)
    y2 = train_valid['taskB']

    for train_idxs, test_idxs in cvS2.split(train_valid, y2):
        fold2.append([train_idxs, test_idxs])

    return test_csv, train_valid, fold2


def return_groupfold(csv_path, nsplit_1 = 10, nsplit_2 = 10):
    all_csv = pd.read_excel(csv_path)
    all_csv = all_csv.dropna(subset = ['taskB']).reset_index(drop = True)
    all_csv['taskB'] = all_csv['taskB'].astype(int)
    fold = []; fold2 = []
    group_cv = StratifiedGroupKFold(n_splits=nsplit_1, shuffle=True, random_state=11)
    groups = all_csv['subject']
    y = all_csv['taskB']

    for train_idxs, test_idxs in group_cv.split(all_csv, y, groups):
        fold.append([train_idxs, test_idxs])

    test_csv = all_csv.loc[fold[3][1]].reset_index(drop = True)

    train_valid = all_csv.loc[fold[3][0]].reset_index(drop = True)
    
    group_cv2 = StratifiedGroupKFold(n_splits=nsplit_2, shuffle=True, random_state=11)#, random_state=11)
    groups2 = train_valid['subject']
    y2 = train_valid['taskB']
    
    for train_idxs, test_idxs in group_cv2.split(train_valid, y2, groups2):
        fold2.append([train_idxs, test_idxs])

    return test_csv, train_valid, fold2