"""
Helping functions.
"""

import matplotlib.pyplot as plt
import matplotlib

import os
import pickle
import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
import sklearn.metrics as metrics

def rcparams(r=0.5):
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams['font.size'] = 25 * r
    matplotlib.rcParams['xtick.labelsize'] = 20 * r
    matplotlib.rcParams['ytick.labelsize'] = 20 * r
    matplotlib.rcParams['axes.labelsize'] = 25 * r
    matplotlib.rcParams['legend.title_fontsize'] = 17 * r
    matplotlib.rcParams['legend.fontsize'] = 17 * r
    matplotlib.rcParams['axes.axisbelow'] = True
    matplotlib.rcParams['figure.figsize'] = [6 * r, 6 * r]
    matplotlib.rcParams['lines.linewidth'] = 2
    matplotlib.rcParams['axes.prop_cycle'] = plt.cycler(color=['#515151', '#df5048', '#3370d8', '#5baa71',
                                                    '#a87bd8', '#c49b33', '#5bc8ca', '#76504f',
                                                    '#8e8c2b', '#ea6f2d', '#7099c8', '#80b537']) 


def format_axis(ax):
    from matplotlib.ticker import (AutoMinorLocator)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3)
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)

def format_axis_im(ax):
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=6)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
        
def plot_loss(hist, titles=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    t = np.arange(len(hist['loss']))
    ax.plot(t, hist['loss'], label='Train loss')
    ax.plot(t, hist['val_loss'], label='Val loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss (MSE)')
    ax.set_xlim([t.min(), t.max()])
    ax.legend()
    if titles != None:
        ax.set_title(titles)
    format_axis(ax)
    plt.tight_layout()
    
def plot_cumerr(err, n_bins, xlim, xlabel='DA Error ($)', labels=['Std', 'MinMax']):
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, err_ in enumerate(err):
        ax.hist(err_, n_bins, density=True, histtype='step', cumulative=True, label=labels[i], linewidth=2)
    ax.set_xlim([0, xlim])
    ax.set_ylim([0, 1.1])
    ax.legend(loc=4)
    ax.grid()

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative Frequency')
    format_axis(ax)
    plt.tight_layout()
        
def remove_outlier(x):
    out = []
    for i in range(x.shape[-1]):
        x_ = x[..., i]
        x_[x_ <= -999] = np.nan
        mask = np.isnan(x_)
        x_[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), x_[~mask])
        out.append(x_)
    return np.array(out).T

def read_csv(site='LZ_HOUSTON', 
             scalar='std',
             xvar=['Demand', 'Outage', 'Wind', 'Temp', 'DA_Price', 'RT_Price'],
             yvar=['RT_Price']):
    
    df = pd.read_csv('./data/Data_Extract_NDE_Raw.csv', delimiter='|')
    df['TimeStamp'] = pd.DatetimeIndex(pd.to_datetime(df.DateTime)).astype(np.int64) // 10 ** 9

    df_ = df[df.Site == site]
    df_ = df_.sort_values(by=['TimeStamp'])

    # training
    t0 = pd.to_datetime('2019-08-19 00:00:00').value // 10 ** 9
    t1 = pd.to_datetime('2021-08-19 00:00:00').value // 10 ** 9
    dtrain = df_.loc[(df_.TimeStamp >= t0) & (df_.TimeStamp < t1)]

    # testing
    t2 = pd.to_datetime('2021-08-19 00:00:00').value // 10 ** 9
    t3 = pd.to_datetime('2022-05-30 00:00:00').value // 10 ** 9
    dtest = df_.loc[(df_.TimeStamp >= t2) & (df_.TimeStamp < t3)]

    xtrain = remove_outlier(dtrain[xvar].values)
    ytrain = remove_outlier(dtrain[yvar].values)
    xtest = remove_outlier(dtest[xvar].values)
    ytest = remove_outlier(dtest[yvar].values)
    
    
    ymin = np.min([ytrain.min(), ytest.min()])
    
    ytrain = np.log10(ytrain - ymin + 1)
    ytest = np.log10(ytest - ymin + 1)

    if scalar == 'std':
        scalar1 = pp.StandardScaler()
        scalar2 = pp.StandardScaler()
    elif scalar == 'minmax':
        scalar1 = pp.MinMaxScaler()
        scalar2 = pp.MinMaxScaler()
    elif scalar == 'robust':
        scalar1 = pp.RobustScaler()
        scalar2 = pp.RobustScaler()
        
    xtrain = scalar1.fit_transform(xtrain)
    xtest = scalar1.transform(xtest)
    ytrain = scalar2.fit_transform(ytrain)
    ytest = scalar2.transform(ytest)

    xvalid = xtrain[8784:]
    xtrain = xtrain[:8784]
    yvalid = ytrain[8784:]
    ytrain = ytrain[:8784]
    
    return xtrain, xvalid, xtest, ytrain, yvalid, ytest, scalar2


def create_dataset(x, y, time_steps=1, shuffle=False):
    xs, ys = [], []
    for i in range(0, len(x) - time_steps, 24):
        v = x[i:(i + time_steps)]
        xs.append(v)
        ys.append(y[i:(i + 24)].squeeze())
    xs = np.array(xs)
    ys = np.array(ys)
    
    if shuffle:
        xs = np.random.RandomState(0).permutation(xs)
        ys = np.random.RandomState(0).permutation(ys)
    return xs, ys
