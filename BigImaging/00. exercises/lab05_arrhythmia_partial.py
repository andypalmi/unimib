import os,sys,math,time,io,argparse,json,traceback,collections
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms, utils, models, ops
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import cpu_count, Pool
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import seaborn as sns
from datetime import datetime

sns.set()


class Dataset(torch.utils.data.Dataset):

    def __init__(self, csv):
        # read the csv file
        self.df = pd.read_csv(csv, sep=',')
        # save cols
        self.output_cols = ['class']
        # get columns of dataframe
        self.input_cols = list(set(self.df.columns) - set(self.output_cols))


    def __len__(self):
        # here i will return the number of samples in the dataset
        return len(self.df)


    def __getitem__(self, idx):
        # here i will load the file in position idx
        cur_sample = self.df.iloc[idx]
        # split in input / ground-truth
        cur_sample_x = cur_sample[self.input_cols]
        cur_sample_y = cur_sample[self.output_cols]
        # convert to torch format
        cur_sample_x = torch.tensor(cur_sample_x.tolist()).unsqueeze(0)
        cur_sample_y = torch.tensor(cur_sample_y.tolist()).squeeze()
        # return values
        return cur_sample_x, cur_sample_y
	