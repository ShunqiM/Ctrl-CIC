import json
import os
from imageio import imread, imsave
import pickle
import shutil

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def read_image(path):
    return imread(path)

def save_image(path, img):
    imsave(path, img)

def save_pkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_path_size(path='.'):
    total = 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    else:
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += get_path_size(entry.path)
        return total
    
def get_shard_index(index, n_per_shard):
    return index//n_per_shard

def create_dir_if_not_exists(dir):
    # Create the directory if it does not exist
    if not os.path.exists(dir):
        os.makedirs(dir)

def create_dir_if_not_exists_given_file_path(file_path):
    file_directory = os.path.dirname(file_path)
    if file_directory == '': return # case when file_path=tmp.txt
    create_dir_if_not_exists(file_directory)

def remove_dir_or_file_if_exists(dir_or_file):
    if os.path.exists(dir_or_file):
        if os.path.isdir(dir_or_file):
            shutil.rmtree(dir_or_file) # note: this might fail for read-only files
            #os.rmdir(dir_or_file) # this only works for empty directories
        else: os.remove(dir_or_file)
        
def get_n_innermost_paths(file_path, n):
    parth_parts = file_path.split(os.sep)
    innermost_paths = parth_parts[-n:]
    return os.sep.join(innermost_paths)

def get_n_outtermost_paths(file_path, n):
    parth_parts = file_path.split(os.sep)
    outtermost_paths = parth_parts[:n]
    return os.sep.join(outtermost_paths)