
import os
import time


print("demo")

print('Baseline', time.strftime('%Y.%m.%d - %H:%M:%S'))
# n_struct = 0

# print('layer 8')
# os.system("python run.py --data_type 1D --data_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_preprocessed.pickle --save_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\LongTailed\\CNN_BaseLine --csv_path C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_task_f.xlsx --batch_size 512 --layers 8 --num_epochs 100 --num_workers 0")
# print('layer 18')
# os.system("python run.py --data_type 1D --data_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_preprocessed.pickle --save_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\LongTailed\\CNN_BaseLine --csv_path C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_task_f.xlsx --batch_size 512 --layers 18 --num_epochs 100 --num_workers 0")
# print('layer 32')
# os.system("python run.py --data_type 1D --data_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_preprocessed.pickle --save_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\LongTailed\\CNN_BaseLine --csv_path C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_task_f.xlsx --batch_size 512 --layers 32 --num_epochs 100 --num_workers 0")
# print('layer 50')
# os.system("python run.py --data_type 1D --data_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_preprocessed.pickle --save_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\LongTailed\\CNN_BaseLine --csv_path C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_task_f.xlsx --batch_size 512 --layers 50 --num_epochs 100 --num_workers 0")
# print('layer 101')
# os.system("python run.py --data_type 1D --data_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_preprocessed.pickle --save_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\LongTailed\\CNN_BaseLine --csv_path C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_task_f.xlsx --batch_size 512 --layers 101 --num_epochs 100 --num_workers 0")

# print('layer 2')
# os.system("python run.py --data_type 1D --data_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_preprocessed.pickle --save_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\LongTailed\\CNN_BaseLine --csv_path C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_task_f.xlsx --batch_size 512 --layers 2 --num_epochs 100 --num_workers 0")

# print('layer 4')
# os.system("python run.py --data_type 1D --data_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_preprocessed.pickle --save_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\LongTailed\\CNN_BaseLine --csv_path C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_task_f.xlsx --batch_size 512 --layers 4 --num_epochs 100 --num_workers 0")

# print('layer 6')
# os.system("python run.py --data_type 1D --data_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_preprocessed.pickle --save_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\LongTailed\\CNN_BaseLine --csv_path C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_task_f.xlsx --batch_size 512 --layers 6 --num_epochs 100 --num_workers 0")

# print('layer 6 under sampling')
# os.system("python run.py --data_type 1D --data_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_preprocessed.pickle --save_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\LongTailed\\CNN_sampling_under --csv_path C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_task_f.xlsx --batch_size 512 --layers 6 --sampler_dic under --num_epochs 100 --num_workers 0")

# print('layer 6 over sampling')
# os.system("python run.py --data_type 1D --data_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_preprocessed.pickle --save_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\LongTailed\\CNN_sampling_over --csv_path C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_task_f.xlsx --batch_size 512 --layers 6 --sampler_dic over --num_epochs 100 --num_workers 0")

print('layer 6 smote')
os.system("python run.py --data_type csv --data_root C:\\Users\\Yeon\\code\\SNUH\\ECG\\Long_Tailed\\smote_dataset.xlsx --save_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\LongTailed\\CNN_SMOTE --csv_path C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_task_f.xlsx --batch_size 512 --layers 6 --sampler_dic smote --num_epochs 100 --num_workers 0")

# print('layer 10')
# os.system("python run.py --data_type 1D --data_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_preprocessed.pickle --save_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\LongTailed\\CNN_BaseLine --csv_path C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_task_f.xlsx --batch_size 512 --layers 10 --num_epochs 100 --num_workers 0")

# print('layer 12')
# os.system("python run.py --data_type 1D --data_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_preprocessed.pickle --save_root C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\LongTailed\\CNN_BaseLine --csv_path C:\\Users\\Yeon\\Desktop\\From_Daegu\\research\\FromE\\ECG\\data\\Long_Tailed\\labelindex_1s_task_f.xlsx --batch_size 512 --layers 12 --num_epochs 100 --num_workers 0")


