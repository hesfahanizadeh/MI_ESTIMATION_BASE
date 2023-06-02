# Title: Base framework for estimation of Mutual Information using Neural Networks
# Last Update: 06/02/2023
# Developers: William (Bill) Wu and Homa Esfahanizadeh
# Based on and Inspired by the public repositories of the following papers:
#  I. Belghazi, S. Rajeswar, A.  Baratin, R. D. Hjelm, and A. C. Courville, “MINE: mutual information neural estimation,” PMLR, 2018.
#  K. Choi and S. Lee, “Regularized mutual information neural estimation,” Openreview, 2021.

# %% Import Necessary Libraries
import numpy as np
import json
import torch
import matplotlib.pyplot as plt

import torchvision
import pytorch_lightning as pl
from tqdm import tqdm
from pytorch_lightning import Trainer, seed_everything

import torch.nn as nn
from torch.nn import functional as F

import math
import logging
logging.getLogger().setLevel(logging.ERROR)

from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# %% Set Cpmputing Device and Set Random Seed
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'   #cuda:0 or cuda:1 for our 2-GPU testbed
num_gpus = 1 if device=='cuda:0' else 0
print(device)

seed = 1
seed_everything(seed, workers=True)
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Numpy RNG
np.random.seed(seed)

# %% Parameters and Constants
DATA_SAVE_PATH = "data/"  # Location for saved classification datapoints
FIG_SAVE_PATH = "figures/" # Location for saving graphical results

EPS = 1e-6 # Used in computing the gradient of loss when numerically estimating MI
MINE_EPOCHS = 2000 # Number of ietartions for numerically estimating MI
MINE_BATCH_SIZE = 5000  # batch size for loading dataset into MINE
lr = 1e-4 # learning rate

# %% Mutual Information Estimation Setup: Inspired by MINE and ReMINE papers

class T(nn.Module): # This is the function that its parameters are optimized to be used in estimation of MI
    def __init__(self, size1, size2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(size1+size2, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x1, x2):
        x1, x2 = x1.float().to(device), x2.float().to(device)
        x1 = x1.view(x1.size(0), -1).to(device)
        x2 = x2.view(x2.size(0), -1).to(device)
        cat = torch.cat((x1, x2), 1).to(device)
        return self.layers(cat).to(device)    

class EMALoss(torch.autograd.Function): # exponential moving average.
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()
        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output): # Gradient
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None

def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach() # The second term is for going from sum to average
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = alpha * t_exp + (1.0 - alpha) * running_mean.item()
    t_log = EMALoss.apply(x, running_mean) # Forward

    return t_log, running_mean

class Mine(nn.Module):
    def __init__(self, stats_network, loss='mine', alpha=0.01, lam=0.1, C=0):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha  # Used for ema during MINE iterations
        # Both lambda and C are a part of the regularization in ReMINE's objective
        self.lam = lam # Lambda
        self.C = C
        self.stats_network = stats_network # Function stat_net

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])] # Permutation of z for marginal distribution

        stats_network_score = self.stats_network(x, z).mean() # The first terms in Remine Estimation
        t_marg = self.stats_network(x, z_marg)
        

        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(
                t_marg, 0) - math.log(t_marg.shape[0])

        # Introducing ReMINE regularization here
        return -stats_network_score + second_term + self.lam * (second_term - self.C) ** 2 # Minus sign is because of the minimization
    
class MutualInformationEstimator(pl.LightningModule):
    def __init__(self, loss='mine', **kwargs):
        super().__init__()
        self.energy_loss = kwargs.get('mine')
        self.file_name = kwargs.get('file_name') + ".txt"
        self.kwargs = kwargs
        self.gradient_batch_size = kwargs.get('gradient_batch_size', 1)
        self.train_loader = kwargs.get('train_loader')
        assert self.energy_loss is not None
        assert self.train_loader is not None
        print("energy loss: ", self.energy_loss)
        with open(DATA_SAVE_PATH + self.file_name, 'w') as f:
            pass # clear the file

    def forward(self, x, z):
        if self.on_gpu:
            x = x.to(device)
            z = z.to(device)

        return self.energy_loss(x, z)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.kwargs['lr'])

    def training_step(self, batch, batch_idx):
        x, z = batch
        if self.on_gpu:
            x = x.to(device)
            z = z.to(device)

        loss = self.energy_loss(x, z).to(device)
        mi = -loss
        tensorboard_logs = {'loss': loss, 'mi': mi}
        tqdm_dict = {'loss_tqdm': loss, 'mi': mi}
        self.last_mi = mi
        self.logger.experiment.add_scalar(
            f"MI Train",
            self.current_epoch,
            mi
        )
        self.logger.log_metrics(tensorboard_logs, self.current_epoch)
        
        if batch_idx % self.gradient_batch_size == 0:
            with open(DATA_SAVE_PATH + self.file_name, 'a') as f:
                f.write(str(self.current_epoch)+'\t'+str(mi.tolist())+'\n')
        return {
            **tensorboard_logs, 'log': tensorboard_logs, 'progress_bar': tqdm_dict
        }
        
    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_idx: int = 0, optimizer_closure = None, on_tpu: bool = False, using_native_amp: bool = False, using_lbfgs: bool = False):
        if batch_idx % self.gradient_batch_size == 0:
            optimizer.step(closure=optimizer_closure)
        else:
            # REFACTOR: Aassumes optimizer closure always non-null
            optimizer_closure()

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer, optimizer_idx: int):
        if batch_idx % self.gradient_batch_size == 0:
            optimizer.zero_grad()

    def train_dataloader(self):
        assert self.train_loader is not None
        return self.train_loader

# %% Constructing/Loading Dataset 1: MNIST Dataset
# The goal here is comparing I(X,Y)
# X is a handwritten images showing a digit in {0,...,9}
# Y = L(X) represents if the the digit is even or odd

class SyntheticDataset(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        
    def __getitem__(self, index):
        x = self.data1[index]
        z = self.data2[index]
        return x, z
    
    def __len__(self):
        return len(self.data1)

# Digit MNIST Dataset
train = torchvision.datasets.MNIST('data', train=True, download=True)

N_FEATURES = train.data.shape[1]*train.data.shape[2] # Number of features per sample 28x28
GRADIENT_BATCH_SIZE =  train.data.shape[0]/MINE_BATCH_SIZE  # How many gradients get accumulated before update (zero_grad)

X_train = train.data.reshape(-1, 784).float() / 255.0
y_train = train.targets

# label shows if the digit is even or odd
y_train = np.where(y_train % 2 == 0, np.ones(y_train.shape), np.zeros_like(y_train))

og_train_dataset = SyntheticDataset(X_train, y_train)

# %% Calculate H(L(X))=I(X,L(X)

EXPERIMENT = f"Dataset1 MI Estimations"
train_loader_HLx = DataLoader(og_train_dataset, batch_size=MINE_BATCH_SIZE, shuffle=True)

t = T(N_FEATURES,1).to(device)
mi_estimator = Mine(t, loss='mine').to(device)
func_str = f"I(X;L(X))"

kwargs = {
    'mine': mi_estimator,
    'lr': lr,
    'batch_size': MINE_BATCH_SIZE,
    'alpha': 0.1,
    'func': func_str,
    'train_loader': train_loader_HLx,
    # Determines how many minibatches (MINE iters) of gradients get accumulated before optimizer step gets applied
    # Meant to stabilize the MINE curve for better encoder training performance
    'gradient_batch_size': GRADIENT_BATCH_SIZE,
    'file_name': 'I(X,L(X))'
}

logger = TensorBoardLogger(
    "lightning_logs",
    name=f"{EXPERIMENT} utility BS={MINE_BATCH_SIZE}",
    version=f"{func_str}, BS: {MINE_BATCH_SIZE}"
)

model = MutualInformationEstimator(loss='mine', **kwargs).to(device)

trainer = Trainer(max_epochs=MINE_EPOCHS, logger=logger, gpus=1)
trainer.fit(model)

# # %% Printing MI Estimations

itr_vec = range(MINE_EPOCHS)

MI_epoch_est = []

for line in open(DATA_SAVE_PATH+'I(X,L(X)).txt', "r"):
    itr, est = line.split()
    MI_epoch_est.append(float(est))

plt.figure()
plt.plot(itr_vec, MI_epoch_est)
plt.ylabel('MI Estimation')
plt.xlabel('Iteration' )
plt.savefig(FIG_SAVE_PATH+'MI_Dataset1.png')
