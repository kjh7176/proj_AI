#                                                                             
# PROGRAMMER: Juhyeon Kim
# DATE CREATED: 27.07.2020                                  
# REVISED DATE: 
# PURPOSE: Create a function that retrieves the following command line inputs 
#          from the user using the Argparse Python module. Arguments:
#     1. Image Folder as image_path with default value 'flowers/test/11/image_03098.jpg'
#     2. Checkpoint Directory as checkpoint with default value 'model.pth'
#     3. Top K most likely classes as --top_k with default value 1
#     4. Mapping categories to real names as --category_names with default value 'cat_to_names.json'
#     5. GPU Programming as --gpu with default value False
#

import argparse

def get_parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_path', type = str, default = './flowers/test/11/image_03098.jpg')
    parser.add_argument('checkpoint', type = str, default = 'model.pth')
    parser.add_argument('--top_k', type = int, default = 5)
    parser.add_argument('--category_names', type = str, default = './cat_to_name.json')
    parser.add_argument('--gpu', action = 'store_true', default = False)
    
    return parser.parse_args()