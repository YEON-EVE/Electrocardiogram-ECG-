import os

#model 종류 : resnet18, resnet50, vgg, squeezenet, densenet - resnet101, alexnet 제외

# Model 1d CNN
# os.system("python EctopicBeat_run.py --model_name CDD1d --num_classes 3 --csv_path ***.xlsx --pickle_path *** --target label_tri --num_workers 0 --num_epochs 10 --batch_size 16 --data_transform_ Raw")

# Model resnet18
# os.system("python EctopicBeat_run.py --model_name resnet50 --num_classes 3 --csv_path *** --pickle_path *** --target label_tri --num_workers 0 --num_epochs 1 --batch_size 16 --data_transform_ CWT")

# squeezenet
os.system("python EctopicBeat_run.py --model_name squeezenet --num_classes 3 --csv_path *** --pickle_path *** --target label_tri --num_workers 0 --num_epochs 1 --batch_size 16 --data_transform_ CWT")
os.system("python EctopicBeat_run.py --model_name densenet --num_classes 3 --csv_path *** --pickle_path *** --target label_tri --num_workers 0 --num_epochs 1 --batch_size 16 --data_transform_ CWT")
