import numpy as np


def get_windowed_data(array, label, win_len=100, slide_frac=0.5):
    win_arr = []
    win_label = []
    init_idx = 0
    end_idx = win_len
    slide = (int)(win_len * slide_frac)
    while end_idx <= len(array):
        win_arr.append(array[init_idx:end_idx])
        win_label.append(label[init_idx:end_idx])
        init_idx += slide
        end_idx += slide
    win_arr = np.array(win_arr)
    win_label = np.median(np.array(win_label), axis=1)

    return win_arr, win_label
