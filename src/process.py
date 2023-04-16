import numpy as np
import random


def random_split_list(a, random_seed, split=0.5):
    random.seed(random_seed)
    random.shuffle(a)
    split_idx = int(len(a) * split)
    return a[:split_idx], a[split_idx:]


def get_windowed_data(array, label, win_len=100, slide_frac=None, slide_step=1):
    win_arr = []
    win_label = []
    init_idx = 0
    end_idx = win_len
    if slide_frac is not None:
        slide = (int)(win_len * slide_frac)
    else:
        slide = slide_step
    while end_idx <= len(array):
        win_arr.append(array[init_idx:end_idx])
        win_label.append(label[init_idx:end_idx])
        init_idx += slide
        end_idx += slide
    win_arr = np.array(win_arr)
    label_median = np.median(np.array(win_label), axis=1)
    win_label = label_median

    return win_arr, win_label
