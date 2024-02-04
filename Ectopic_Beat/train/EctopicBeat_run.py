import os
import argparse
import torch

from EctopicBeat_train import fold_train

#image_dir = "E:\\image_classfication\\original"
#csv_path = "C:\\Users\\Yeon\\Desktop\\Yeon\\학부연구원\\DDH\\all_csv.csv"
#target = 'Score1'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required = True)
    parser.add_argument('--num_classes', required = True)
    parser.add_argument('--num_epochs', default = '10', required = False)
    parser.add_argument('--feature_extract', default = 'True', required = False)
    parser.add_argument('--train_shuffle', default = True, required = False)
    parser.add_argument('--batch_size', default = 128, required = False)
    parser.add_argument('--num_workers', default = 0, required = False)
    parser.add_argument('--csv_path', required = True)
    parser.add_argument('--pickle_path', required = True)
    parser.add_argument('--target', default = 'label', required = False)
    parser.add_argument('--data_transform_', required = True)
    args = parser.parse_args()
    #print(type(args.model_name))
    # if args.model_name == "resnet18":
    #     print(args.model_name)
    # else:
    #     print("wrong")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)
    #print(type(int(args.num_classes)))
    fold_train(model_name = str(args.model_name),
               device = device,
               num_classes = int(args.num_classes),
               num_epochs = int(args.num_epochs),
               feature_extract = args.feature_extract,
               train_shuffle = args.train_shuffle,
               batch_size = int(args.batch_size),
               num_workers = int(args.num_workers),
               csv_path = args.csv_path,
               pickle_path = args.pickle_path,
               target = args.target,
               data_transform_ = args.data_transform_)