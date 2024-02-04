from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd

def return_fold(csv_path, target, nsplit_1 = 8, nsplit_2 = 8):

    df_final = pd.read_excel(csv_path)
    # test set 분리
    fold = []; fold2 = []
    group_cv = StratifiedGroupKFold(n_splits=nsplit_1, shuffle=True, random_state=11)
    groups = df_final['subject_id']
    y = df_final['label_bin']
    for train_idxs, test_idxs in group_cv.split(df_final, y, groups):
        fold.append([train_idxs, test_idxs])
    
    test_csv = df_final.loc[fold[4][1]].reset_index(drop = True)
    test_csv = test_csv.dropna(subset=[target]).reset_index(drop = True)
    
    # validation 을 위한 k fold
    train_valid = df_final.loc[fold[4][0]].reset_index(drop = True)
    train_valid = train_valid.dropna(subset = [target]).reset_index(drop = True)

    group_cv2 = StratifiedGroupKFold(n_splits=nsplit_2, shuffle=True, random_state=11)#, random_state=11)
    groups2 = train_valid['subject_id']
    y2 = train_valid['label_bin']
    for train_idxs, test_idxs in group_cv2.split(train_valid, y2, groups2):
        fold2.append([train_idxs, test_idxs])
            
    print(test_csv.shape, train_valid.shape)
    return test_csv, train_valid, fold2