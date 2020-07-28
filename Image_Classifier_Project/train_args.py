#                                                                             
# PROGRAMMER: Juhyeon Kim
# DATE CREATED: 21.07.2020                                  
# REVISED DATE: 
# PURPOSE: Create a function that retrieves the following command line inputs 
#          from the user using the Argparse Python module. Arguments:
#     1. Image Folder as data_dir with default value 'flowers'
#     2. Checkpoint Directory as --save_dir with default value 'model.pth'
#     3. Pretrained Model as --arch with default value 'vgg16' (supporting only 'vgg16' or 'densenet121')
#     4. Learning Rate as --learning_rate with default value 0.003
#     5. HIdden Units as --hidden_units with default value 4096
#     6. Epochs as --epochs with default value 20
#     7. GPU Programming as --gpu with default value False
#     8. Load checkpoint and keep on train as --load with defaul value False
#

import argparse

def get_parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type = str, default = './flowers')
    parser.add_argument('--save_dir', type = str, default = 'model.pth')
    parser.add_argument('--arch', type = str, default = 'vgg16')
    parser.add_argument('--learning_rate', type = float, default = 0.003)
    parser.add_argument('--hidden_units', type = int, default = 4096)
    parser.add_argument('--epochs', type = int, default = 20)                    
    parser.add_argument('--gpu', action = 'store_true', default = False)
    parser.add_argument('--load', action = 'store_true', default = False)

    return parser.parse_args()