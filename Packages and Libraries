# Install Packages
"""

pip install torch

pip install torch_geometric

pip install sentence_transformers

!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
!pip install -q git+https://github.com/pyg-team/pytorch_geometric.git

!pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html



"""# Import Libraries"""

import torch

import torch_geometric

import torch_geometric.transforms as T
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.datasets import MovieLens
from torch_geometric.nn import SAGEConv, to_hetero, GATConv
