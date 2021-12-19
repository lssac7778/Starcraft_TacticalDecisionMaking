import random

import numpy as np
import torch

from dataconvertor import decompress_pickle
from dataprocessor import print_matrix
import os
import multiprocessing as mp
import time
from torch.utils.data import Dataset, DataLoader
from Const import Labels

def load_data(filenames, result_queue):
    for filename in filenames:
        game = decompress_pickle(filename)
        single_input = type(game['input']) == np.ndarray
        if single_input:
            game['input'] = [game['input']]

        game['label'] = [[Labels.old_to_new[int(l)]] for l in game['label']]
        game['label_weight'] = np.array([[Labels.weight[l[0]]] for l in game['label']])
        game['label'] = np.array(game['label'])
        result_queue.put((filename, game))

def multi_load_data(load_dir, num_process=24):
    # get all filenames
    all_filenames = []
    for filename in os.listdir(load_dir):
    #for filename in os.listdir(load_dir)[:100]:

        all_filenames.append(os.path.join(load_dir, filename))
    file_num = len(all_filenames)

    # divide filenames
    interval = file_num // num_process
    proc_filenames = []
    for i in range(0, file_num, interval):
        proc_filenames.append(all_filenames[i:i+interval])
    proc_filenames[-1] += all_filenames[i+interval:]

    # run process
    processes = []
    result_queue = mp.Queue()
    for filenames in proc_filenames:
        p = mp.Process(target=load_data, args=(filenames, result_queue))
        processes.append(p)
    for process in processes:
        process.start()

    data_len = 1_000_000
    data = {
        'inputs': None,
        'label': None,
        'label_weight': None
    }
    mappings = []

    # gather data
    i = 0
    map_idx = 0
    while i < file_num:
        filename, game = result_queue.get()
        game_len = game['label'].shape[0]

        if i == 0:
            label_shape = list(game['label'].shape)
            input_shapes = [list(k.shape) for k in game['input']]

            data['label'] = np.zeros([data_len] + label_shape[1:], dtype=np.float32)
            data['label_weight'] = np.zeros([data_len] + label_shape[1:], dtype=np.float32)
            data['inputs'] = [np.zeros([data_len] + shape[1:], dtype=np.float32) for shape in input_shapes]

        data['label'][map_idx:map_idx+game_len] = game['label']
        data['label_weight'][map_idx:map_idx+game_len] = game['label_weight']
        for j in range(len(data['inputs'])):
            data['inputs'][j][map_idx:map_idx+game_len] = game['input'][j]

        mappings.append((map_idx, map_idx+game_len, i))
        print(f"{i + 1:>5}/{file_num:>5} | {map_idx:>7} | loaded {filename}")
        map_idx += game_len
        i += 1

    data['label'] = data['label'][:mappings[-1][1]]
    data['label_weight'] = data['label_weight'][:mappings[-1][1]]
    data['inputs'] = [input[:mappings[-1][1]] for input in data['inputs']]

    # join process
    for process in processes:
        process.join()

    return data, mappings

class Stardata(Dataset):
    def __init__(self, load_dir, window_size=1, preprocess_func='simple_preprocess', augment=True):
        self.data, self.mappings = multi_load_data(load_dir)
        self.window_size = window_size
        self.input_len = len(self.data['inputs'])
        self.preprocess_func = preprocess_func
        self.augment = augment

    def find_game(self, i):
        for start, end, idx in self.mappings:
            if i >= start and i < end:
                return start, end, idx
        raise IndexError

    def process_label(self, label):
        return torch.LongTensor(label)[0]

    def preprocess(self, inputs):
        if self.window_size == 1:
            for i in range(len(inputs)):
                inputs[i] = np.expand_dims(inputs[i], axis=0)

        if self.augment:
            for i in range(len(inputs)):
                if len(inputs[i].shape) > 3:
                    inputs[i] = self.augment_matrix(inputs[i])

        result = [torch.FloatTensor(array) for array in inputs]

        if self.window_size == 1:
            for i in range(len(result)):
                result[i] = result[i][0]
        return tuple(result)

    def augment_matrix(self, matrix):
        if random.randint(0, 3) != 0:
            matrix = np.rot90(matrix, random.randint(1, 3), (-2, -1))
        if random.randint(0, 1):
            matrix = np.flip(matrix, axis=-1)
        return matrix.copy()

    def __len__(self):
        return self.mappings[-1][1]

    def __getitem__(self, idx):
        if not self.window_size == 1:
            start, end, game_idx = self.find_game(idx)
            window_start = idx - self.window_size + 1
            window_end = idx + 1
            inputs = []
            for input_idx in range(self.input_len):
                if start > window_start:
                    original_shape = list(self.data['inputs'][input_idx][idx].shape)
                    zero_inputs = np.zeros([start - window_start] + original_shape)
                    inputt = np.concatenate((zero_inputs, self.data['inputs'][input_idx][start:window_end]))
                else:
                    inputt = self.data['inputs'][input_idx][window_start:window_end]
                inputs.append(inputt)

        else:
            inputs = [self.data['inputs'][input_idx][idx] for input_idx in range(self.input_len)]

        label = self.data['label'][idx]
        result = list(self.preprocess(inputs)) + [self.process_label(label)]
        return result

if __name__ == "__main__":
    path = "./data/simple_preprocess/TT"
    dataset = Stardata(path, window_size=3, preprocess_func='simple_preprocess')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    for x, y, z in dataloader:
        print(x.shape, y.shape, z.shape)


