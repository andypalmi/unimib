import os
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns

sns.set_theme()




def get_src(df, src_prefix):
    ''' gets hyperspectral signals using cols starting with src_prefix '''
    # get x points
    src_cols = [col for col in df.columns if col.startswith(src_prefix)]
    x = np.array([float(col[len(src_prefix):]) for col in src_cols])
    # sort x points in increasing order
    pos = np.argsort(x)
    src_cols = [src_cols[cur_pos] for cur_pos in pos]
    # extract x and y values
    src_x = x[pos]
    src_y = df[src_cols].to_numpy()
    # convert to tensor
    src_x = torch.from_numpy(src_x).float()
    src_y = torch.from_numpy(src_y).float()
    # return
    return src_cols, src_x, src_y


def get_tgt(df, tgt_vars):
    ''' gets target variables using specified columns '''
    # extract variables
    tgt_vars = df[tgt_vars].to_numpy()
    # convert to torch
    tgt_vars = torch.from_numpy(tgt_vars).float()
    # return them
    return tgt_vars


def load_data(fn):
    # read csv
    df = pd.read_csv(fn)
    # get vars
    src_cols, src_x, src_y = get_src(df, src_prefix)
    # return
    return df, src_cols, src_x, src_y



if __name__ == '__main__':
    # define input file
    data_fn = './data/lucas_dataset_val.csv'
    src_prefix = 'spc.'
    n_components = 0.9
    # read file
    df, src_cols, src_x, src_y = load_data(data_fn)
    # apply pca
    pca = PCA(n_components=n_components)
    # fit and transform
    components = pca.fit_transform(src_y)
    # print shape
    print('Original signal had {} components, PCA reduced it to {} components.'.format(src_y.shape[1], components.shape[1]))
    # invert components
    inverted = pca.inverse_transform(components)
    # plot according to values
    plt.plot(src_x, src_y[0], 'b', label='original')
    plt.plot(src_x, inverted[0], 'r', label='PCA')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pca.png', dpi=300)
    # plt.show()
        

        
    
    
    
    
    
    
    
    
    
    
    
    