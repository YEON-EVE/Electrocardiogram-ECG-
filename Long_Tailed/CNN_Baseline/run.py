import os
import argparse
import torch

from fold_train import fold_train

#image_dir = "E:\\image_classfication\\original"
#csv_path = "C:\\Users\\Yeon\\Desktop\\Yeon\\학부연구원\\DDH\\all_csv.csv"
#target = 'Score1'

print("run")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', required = True)
    parser.add_argument('--data_root', required = True)
    parser.add_argument('--save_root', required = True)
    parser.add_argument('--csv_path', required = False)
    parser.add_argument('--layers', required = True)
    parser.add_argument('--sampler_dic', default = None, required = False)
    # parser.add_argument('--n_struct', default = 0, required = False)
    parser.add_argument('--num_epochs', default = 15, required = False)
    parser.add_argument('--batch_size', default = 128, required = False)
    parser.add_argument('--num_workers', default = 4, required = False)

    args = parser.parse_args()

    # layers = list(map(int, list((str(args.layers)[0], str(args.layers)[1], str(args.layers)[2], str(args.layers)[3]))))
    # layers = list(map(int, str(args.layers).split("_")))
    # print(layers)
    #print(type(args.model_name))
    # if args.model_name == "resnet18":
    #     print(args.model_name)
    # else:
    #     print("wrong")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    #print(type(int(args.num_classes)))
    fold_train(data_type = str(args.data_type),
               data_root = str(args.data_root), 
               save_root = str(args.save_root), 
               csv_path = str(args.csv_path), 
               device = device, 
               # n_struct = int(args.n_struct), 
               batch_size = int(args.batch_size), 
               layers = int(args.layers),
               sampler_dic = str(args.sampler_dic),
               n_epoch = int(args.num_epochs), 
               num_workers = int(args.num_workers))
    