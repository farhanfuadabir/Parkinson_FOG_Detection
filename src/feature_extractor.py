import numpy as np
import pandas as pd
import scipy as sp
import librosa as lb


def correlation(x):
    cor = []
    for n in range(x.shape[0]):
        cor.append(np.correlate(x[n, :], x[n, :])[0])
    return np.array(cor)


def mean_crossing_rate(x):
    mcr = []
    for n in range(x.shape[0]):
        mcr.append(lb.feature.zero_crossing_rate(
            x[n, :] - np.mean(x[n, :]))[0, 0])
    return np.array(mcr)


def get_entropy(x, axis=1):
    x_sum = np.sum(x, axis=axis, keepdims=True)
    x_sum[x_sum == 0] = 0.000001
    x = x / x_sum
    entropy = np.sum(sp.special.entr(x), axis=axis)
    return entropy


def number_of_peaks(x):
    npk = []
    for n in range(x.shape[0]):
        thres = (np.max(x[n, :]) / 3)
        peaks, _ = sp.signal.find_peaks(x[n, :], thres)
        npk.append(len(peaks))
    return np.array(npk, dtype=float)


def get_stat_features(x, axis=1, prefix=''):

    # print('Calculating Features...', end = " ")
    min = np.min(x, axis=axis)
    max = np.max(x, axis=axis)
    std = np.std(x, axis=axis)
    avg = np.mean(x, axis=axis)
    var = np.var(x, axis=axis)
    ptp = np.ptp(x, axis=axis)
    mrc = np.max(np.diff(x, axis=axis), axis=axis)
    arc = np.mean(np.diff(x, axis=axis), axis=axis)
    src = np.std(np.diff(x, axis=axis), axis=axis)
    mad = sp.stats.median_abs_deviation(x, axis=axis)
    iqr = sp.stats.iqr(x, axis=axis)
    cor = correlation(x)
    mcr = mean_crossing_rate(x)
    rms = np.sum(np.square(x), axis=axis)
    skw = sp.stats.skew(x, axis=axis)
    kut = sp.stats.kurtosis(x, axis=axis)
    # print('Done!')

    feature_names = ['min', 'max', 'std', 'avg', 'var',
                     'ptp', 'mrc', 'arc', 'src', 'mad',
                     'iqr', 'cor', 'mcr', 'rms', 'skw', 'kut']
    columnName = [prefix + '_' + sub for sub in feature_names]

    stat_features = pd.DataFrame(np.stack((min, max, std, avg,
                                           var, ptp, mrc, arc,
                                           src, mad, iqr, cor,
                                           mcr, rms, skw, kut), axis=1), columns=columnName)

    # feature_names = ['std']
    # columnName = [prefix + '_' + sub for sub in feature_names]

    # stat_features = pd.DataFrame(std, columns=columnName)

    # if (stat_features.isna().sum().sum()) > 0:
    #     NaN_columnName = stat_features.columns[stat_features.isna(
    #     ).any()].tolist()
    #     raise ValueError(
    #         f'NaN detected while calculating {prefix} stat features - {NaN_columnName}')

    return stat_features


def get_freq_features(x, axis=1, fs=100, nperseg=20, prefix=''):

    # print('Calculating Features...', end = " ")
    # nperseg = x.shape[1]

    freq, psd = sp.signal.welch(x, fs, nperseg=nperseg, axis=axis)
    mpw = np.max(psd, axis=axis)
    ent = get_entropy(psd, axis=axis)
    psd_sum = np.sum(psd, axis=axis)
    psd_sum[psd_sum == 0] = 0.000001
    ctf = np.divide(np.sum((freq * psd), axis=axis), psd_sum)
    mxf = np.argmax(psd, axis=axis)
    enr = np.sum(np.square(psd), axis=axis) / nperseg
    npk = number_of_peaks(psd)
    # print('Done!')

    feature_names = ['mpw', 'ent', 'ctf', 'mxf', 'enr', 'npk']
    columnName = [prefix + '_' + sub for sub in feature_names]

    freq_features = pd.DataFrame(
        np.stack((mpw, ent, ctf, mxf, enr, npk), axis=1), columns=columnName)

    # if (freq_features.isna().sum().sum()) > 0:
    #     NaN_columnName = freq_features.columns[freq_features.isna(
    #     ).any()].tolist()
    #     raise ValueError(
    #         f'NaN detected while calculating {prefix} freq features - {NaN_columnName}')

    return freq_features


def get_mutual_features(x, y, z, axis=1, prefix=''):
    cxy = []
    cxz = []
    cyz = []
    vxy = []
    vxz = []
    vyz = []
    # print('Calculating Features...', end = " ")
    nperseg = x.shape[1]

    for n in range(x.shape[0]):
        cxy.append(np.corrcoef(x[n, :].ravel(), y[n, :].ravel())[0, 1])
        cxz.append(np.corrcoef(x[n, :].ravel(), z[n, :].ravel())[0, 1])
        cyz.append(np.corrcoef(y[n, :].ravel(), z[n, :].ravel())[0, 1])
        vxy.append(np.cov(x[n, :].ravel(), y[n, :].ravel())[0, 1])
        vxz.append(np.cov(x[n, :].ravel(), z[n, :].ravel())[0, 1])
        vyz.append(np.cov(y[n, :].ravel(), z[n, :].ravel())[0, 1])
    cxy = np.array(cxy)
    cxz = np.array(cxz)
    cyz = np.array(cyz)
    vxy = np.array(vxy)
    vxz = np.array(vxz)
    vyz = np.array(vyz)
    sma = (np.trapz(x, axis=axis) + np.trapz(x, axis=axis) +
           np.trapz(x, axis=axis)) / nperseg
    # print('Done!')

    feature_names = ['cxy', 'cxz', 'cyz', 'vxy', 'vxz', 'vyz', 'sma']
    columnName = [prefix + '_' + sub for sub in feature_names]

    mutual_features = pd.DataFrame(np.stack((cxy, cxz, cyz, vxy, vxz, vyz, sma),
                                            axis=1), columns=columnName)

    if (mutual_features.isna().sum().sum()) > 0:
        NaN_columnName = mutual_features.columns[mutual_features.isna(
        ).any()].tolist()
        raise ValueError(
            f'NaN detected while calculating {prefix} mutual features - {NaN_columnName}')

    return mutual_features


def get_velocity(x, axis=1):
    nperseg = x.shape[1]
    return np.trapz(x, axis=axis) / nperseg


def get_fft_coefs(x, fs=100, n=20, axis=1):
    # print('Calculating Features...', end = " ")
    _, psd = sp.signal.welch(x, fs, nperseg=n, axis=axis)
    # print('Done!')
    return psd
