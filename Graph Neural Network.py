"GNN Code"


"""# Set Device"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device

"""# Load Data"""

from google.colab import files
  
  
uploaded = files.upload()

#from google.colab import drive
#drive.mount('/content/gdrive')

filepath ="/content/gdrive/My Drive/ciao_rating.xlsx"
# df = pd.read_excel(filepath)

main_df = pd.read_excel('ciao_rating_3.xlsx',names=['user_id','movie_id','category','rating','helpfulness','timestamp','session'])

df = main_df.drop(columns='helpfulness')

"""# Data Preprocessing"""

df = main_df.drop(columns='helpfulness')

df




"""# Train Data"""

df

train_df = df.copy()

train_df['weight'] = train_df['timestamp']/(60*60*24)
min_weight = train_df.groupby('session').min()['weight']
min_weight

new_weight = train_df.apply(lambda x : x.weight -min_weight[x.session]+1, axis = 1)

train_df['weight'] = new_weight

train_df = train_df.sort_values('timestamp',ascending= True).reset_index(drop = True)
train_df.reset_index(inplace = True)

filter = train_df.groupby('session').max()['index'].reset_index()['index']
filter = list(filter)
test = []
train_df['test'] = 0
for index,i in enumerate (train_df['test']):
  if index in (filter):
    test.append(index)
    train_df.loc[index,"test"]=1
  else:
    train_df.loc[index,"test"] = 0

total_test_df = train_df.loc[train_df['test'] == 1]

train_df = train_df.loc[train_df['test'] == 0].drop(columns = 'test')

train_df

#assign weight

# train_df['weight'] = train_df['timestamp']/(60*60*24)
# min_weight = train_df.groupby('session').min()['weight']
# min_weight

# new_weight = train_df.apply(lambda x : x.weight -min_weight[x.session]+1, axis = 1)

# train_df['weight'] = new_weight

train_df_copy = train_df.copy()

# train_df = train_df_copy.copy()

train_df

#set my practice train data

train_df = train_df.sort_values('user_id')
train_df.reset_index(drop = True,inplace = True)



train_df = train_df.iloc[0:2013]
train_df

filter = list(train_df['session'].unique())
user_test_df = total_test_df[~total_test_df.session.isin(filter)]
filter = list(user_test_df.groupby('user_id').max()['session'])
user_test_df = user_test_df[user_test_df.session.isin(filter)]
user_test_df = user_test_df[user_test_df.user_id.isin(train_df['user_id'])]



"""# Define HeteroData

"""

train_data = HeteroData().to(device)
train_data['user'].num_nodes = len(train_user_mapping)  # Users do not have any features.
train_data['session'].num_nodes = len(train_session_mapping)
train_data['movie'].num_nodes = len(train_movie_x)


train_data['user', 'rates', 'movie'].edge_index = t_ui_edge_index
train_data['user', 'rates', 'movie'].edge_attr = t_ui_edge_label
train_data['user', 'rates', 'movie'].edge_label= t_rui_edge_label

train_data['movie', 'rated in', 'session'].edge_index = t_is_edge_index
train_data['movie', 'rated in', 'session'].edge_attr = t_is_edge_label
train_data['movie', 'rated in', 'session'].edge_label = train_edge_label

train_data['movie', 'links', 'movie'].edge_index = t_ii_edge_index
train_data['session', 'links', 'session'].edge_index = t_ss_edge_index


print(train_data)


test_data = HeteroData().to(device)
test_data['user'].num_nodes = len(train_user_mapping)  # Users do not have any features.
test_data['session'].num_nodes = len(train_session_mapping)
test_data['movie'].num_nodes = len(train_movie_x)


test_data['user', 'rates', 'movie'].edge_index = t_ui_edge_index
test_data['user', 'rates', 'movie'].edge_attr = t_ui_edge_label
test_data['user', 'rates', 'movie'].edge_label= t_rui_edge_label

test_data['movie', 'rated in', 'session'].edge_index = target_attr_index
test_data['movie', 'rated in', 'session'].edge_attr = target_edges_attr
test_data['movie', 'rated in', 'session'].edge_label = target_edge_label

test_data['movie', 'links', 'movie'].edge_index = t_ii_edge_index
test_data['session', 'links', 'session'].edge_index = t_ss_edge_index

print(test_data)

t_rui_edge_label

train_data['user'].x = (torch.eye(train_data['user'].num_nodes ))
del train_data['user'].num_nodes


train_data['session'].x = torch.eye(train_data['session'].num_nodes)
del train_data['session'].num_nodes


test_movie_x = train_movie_x.float()

train_data['movie'].x = train_movie_x.float()
# train_movie_x = train_movie_x.resize(16861)
train_movie_x = train_movie_x.resize(len(train_movie_x))
train_data['movie'].y= train_movie_x.long()
del train_data['movie'].num_nodes



 


test_data['user'].x = (torch.eye(test_data['user'].num_nodes ))
del test_data['user'].num_nodes

test_data['session'].x = torch.eye(test_data['session'].num_nodes)
del test_data['session'].num_nodes

test_data['movie'].x = test_movie_x.float()
# train_movie_x = train_movie_x.resize(16861)
train_movie_x = test_movie_x.resize(len(test_movie_x))
test_data['movie'].y= test_movie_x.long()
del test_data['movie'].num_nodes

train_movie_x

train_data = T.ToUndirected()(train_data)

test_data = T.ToUndirected()(test_data)

train_data
del train_data['session', 'rev_rated in', 'movie'].edge_label
del train_data['movie', 'rev_rates', 'user'].edge_label # Remove "reverse" label.


test_data
del test_data['session', 'rev_rated in', 'movie'].edge_label
del test_data['movie', 'rev_rates', 'user'].edge_label # Remove "reverse" label.

train_data

test_data



"""# target edges

"""

target_edge_label_index, target_edge_label = load_edge_csv(
    target_edges_df,
    src_index_col='movie_id',
    src_mapping=train_movie_mapping,
    dst_index_col='session',
    dst_mapping=train_session_mapping,
    encoders={'rating': IdentityEncoder(dtype=torch.long)},
)

(target_edge_label_index)

# target_edge_label

test_data



"""# ciao label prediction

"""

import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.datasets import MovieLens
from torch_geometric.nn import SAGEConv, to_hetero, GATConv

parser = argparse.ArgumentParser()
parser.add_argument('-f')
parser.add_argument('--use_weighted_loss', action='store_true',
                    help='Whether to use weighted MSE loss.')
args = parser.parse_args()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# We have an unbalanced dataset with many labels for rating 3 and 4, and very
# few for 0 and 1. Therefore we use a weighted MSE loss.
if args.use_weighted_loss:
    weight = torch.bincount(train_data['movie', 'session'].edge_label.resize(len(train_data['movie', 'session'].edge_label)))
    weight = weight.max() / weight
else:
    weight = None


def weighted_mae_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * abs((pred - target.to(pred.dtype)))).mean()

weight = torch.bincount(train_data['movie', 'session'].edge_label.resize(len(train_data['movie', 'session'].edge_label)))
# weight = weight.max() / weight
weight

from sklearn.metrics import mean_absolute_error

train_data['movie', 'rated in', 'session'].edge_attr = train_data['movie', 'rated in', 'session'].edge_attr.float()
train_data['movie', 'rev_rates', 'user'].edge_attr = train_data['movie', 'rev_rates', 'user'].edge_attr.float()
train_data['session', 'rev_rated in', 'movie'].edge_attr = train_data['session', 'rev_rated in', 'movie'].edge_attr.float()
train_data['user','rates','movie'].edge_attr = train_data['user','rates','movie'].edge_attr.float()

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels,add_self_loops = False, edge_dim = -1)
       %self.conv2 = GATConv((-1, -1), out_channels,add_self_loops = False,edge_dim = -1)

    def forward(self, x, edge_index,edge_attr):
        x = self.conv1(x, edge_index,edge_attr).relu()
        # x = self.conv2(x, edge_index,edge_attr)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['movie'][row], z_dict['session'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        z_dict['movie'] = train_movie_x
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, train_data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict,edge_attr_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict,edge_attr_dict)
        pred = self.decoder(z_dict, edge_label_index)

        return pred


# class GNNEncoder(torch.nn.Module):
#     def __init__(self, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GATConv((-1, -1), hidden_channels)
#         self.conv2 = GATConv((-1, -1), out_channels)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index).relu()
#         x = self.conv2(x, edge_index)
#         return x


# class EdgeDecoder(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super().__init__()
#         self.lin1 = Linear(2 * hidden_channels, hidden_channels)
#         self.lin2 = Linear(hidden_channels, 1)

#     def forward(self, z_dict, edge_label_index):
#         row, col = edge_label_index
#         z = torch.cat([z_dict['movie'][row], z_dict['session'][col]], dim=-1)

#         z = self.lin1(z).relu()
#         z = self.lin2(z)
#         return z.view(-1)


# class Model(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super().__init__()
#         self.encoder = GNNEncoder(hidden_channels, hidden_channels)
#         self.encoder = to_hetero(self.encoder, train_data.metadata(), aggr='sum')
#         self.decoder = EdgeDecoder(hidden_channels)

#     def forward(self, x_dict, edge_index_dict, edge_label_index):
#         z_dict = self.encoder(x_dict, edge_index_dict)
#         pred = self.decoder(z_dict, edge_label_index)

#         return pred

train_data

test_data['movie', 'session'].edge_index = target_edge_label_index
test_data['movie', 'session'].edge_label = target_edge_label
test_data['movie', 'session'].edge_attr = target_edges_attr

test_data['movie', 'rated in', 'session'].edge_attr = test_data['movie', 'rated in', 'session'].edge_attr.float()
test_data['movie', 'rev_rates', 'user'].edge_attr = test_data['movie', 'rev_rates', 'user'].edge_attr.float()
test_data['session', 'rev_rated in', 'movie'].edge_attr = test_data['session', 'rev_rated in', 'movie'].edge_attr.float()
test_data['user','rates','movie'].edge_attr = test_data['user','rates','movie'].edge_attr.float()

train_data

# Due to lazy initialization, we need to run one model step so the number
# of parameters can be inferred:
# with torch.no_grad():
#     model.encoder(train_data.x_dict, train_data.edge_index_dict,train_data.edge_attr_dict)


model = Model(hidden_channels=64).to(device)
with torch.no_grad():
    model.encoder(train_data.x_dict, train_data.edge_index_dict, train_data.edge_attr_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.05,weight_decay=0.01)

def check_pos_list(my_list):
  for i in my_list:
    if  i>0 :
      ans = 'positive'
    if i<0  :
      ans = 'neither'
      break

def scale(my_tensor,max_range):
   
  max_number = max(my_tensor.tolist())
  ratio = max_range / max_number

  for i in range(len(my_tensor)):
    my_tensor[i] = torch.round(my_tensor[i] * ratio)
  # max_number = max(my_tensor.tolist())
  # for i in range(len(my_tensor)):
  #   ratio = maximum / my_tensor[i]
  #   my_tensor[i] = torch.round(5 * ratio)


  return my_tensor

pred

def train():

    model.train()
    optimizer.zero_grad()
    # pred = model(train_data.x_dict, train_data.edge_index_dict,train_data.edge_attr_dict,
    #              train_data['movie', 'session'].edge_index)

    pred = model(train_data.x_dict, train_data.edge_index_dict,train_data.edge_attr_dict,
                 train_data['movie', 'session'].edge_index)
    target = train_data['movie', 'session'].edge_label

    # pred = scale_zero
    pred = scale_rate(pred)
    # pred = pred.clamp(min=1, max=5)


    # loss = mean_absolute_error(target.tolist(), pred.tolist())
    loss = weighted_mae_loss(pred, target, weight)
    loss.backward()
    optimizer.step()
    return float(loss)

pred = model(train_data.x_dict, train_data.edge_index_dict,train_data.edge_attr_dict,
                 train_data['movie', 'session'].edge_index)
pred

def scale_zero(my_tensor):
   
  min_number = abs(0 - min(my_tensor.tolist()))

  for i in range(len(my_tensor)):
    my_tensor[i] = my_tensor[i] + min_number
  # max_number = max(my_tensor.tolist())
  # for i in range(len(my_tensor)):
  #   ratio = maximum / my_tensor[i]
  #   my_tensor[i] = torch.round(5 * ratio)


  return my_tensor

scale_zero(pred)

abs(0-min(pred.tolist()))

pred

from sklearn.preprocessing import MinMaxScaler

def scale_rate(my_tensor):
  dataf = my_tensor.tolist()
  scaler = MinMaxScaler(feature_range=(1, 5))
  scaled = scaler.fit_transform([[x] for x in dataf])
  scaled = [round(j) for sub in scaled for j in sub]
  # scaled = [[x] for x in scaled]

  for i in range(len(my_tensor)):
    my_tensor[i] = torch.round(my_tensor[i] * 0 + scaled[i])

  # scaled = torch.FloatTensor(scaled)
  return my_tensor

a = scale_rate(pred)
print(a)

test_list = pred
new_list = []
max_num = max(test_list)
for i in test_list:
  new = 5*(1/(i/max_num))
  new_list.append(new)

pred.dtype

max_number = max(pred.tolist())
max_number

max_number = max(my_tensor.tolist())
  for i in range(len(my_tensor)):
    ratio = maximum / my_tensor[i]
    my_tensor[i] = torch.round(5 * ratio)

scale(pred,5)

min(pred)

# maximum =  max(pred)
maximum = -2
x = -6
ratio = maximum/ x
new_x = 5 * ratio

scale(pred,-5)

scale(pred,5)

scale(pred,5)

@torch.no_grad()
def test(data):
    model.eval()
    # pred = model(data.x_dict, data.edge_index_dict,
    #              data.edge_attr_dict,
    #              data['movie', 'session'].edge_index)
    
    pred = model(data.x_dict, data.edge_index_dict,data.edge_attr_dict,
              data['movie', 'session'].edge_index)
    # pred = scale_zero(pred)    
    pred = scale_rate(pred)
   

    target = data['movie', 'session'].edge_label
    # rmse = F.mse_loss(pred, target).sqrt()
    # rmse = mean_absolute_error(target.tolist(), pred.tolist())
    rmse = weighted_mae_loss(pred, target, weight)
    return pred,float(rmse)

train_data['movie', 'session'].edge_label

pred

test_data

for epoch in range(1, 101):
    loss = train()
    pred1,mae_train = test(train_data)
    pred2,mae_test = test(test_data)
    print(mae_train,mae_test)

# @torch.no_grad()
# def test():
#     model.eval()
#     pred = model(train_data.x_dict, train_data.edge_index_dict,
#                  train_data.edge_attr_dict,
#                  train_data['movie', 'session'].edge_index)
#     pred = pred.clamp(min=0, max=5)
#     target = train_data['movie', 'session'].edge_label.float()
#     # rmse = F.mse_loss(pred, target).sqrt()
#     rmse = mean_absolute_error(target.tolist(), pred.tolist())
#     return float(rmse)

train_data['movie', 'rated in', 'session'].edge_attr = train_data['movie', 'rated in', 'session'].edge_attr.float()
train_data['movie', 'rev_rates', 'user'].edge_attr = train_data['movie', 'rev_rates', 'user'].edge_attr.float()
train_data['session', 'rev_rated in', 'movie'].edge_attr = train_data['session', 'rev_rated in', 'movie'].edge_attr.float()
train_data['user','rates','movie'].edge_attr = train_data['user','rates','movie'].edge_attr.float()



"""# rating prediction


"""

import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.datasets import MovieLens
from torch_geometric.nn import SAGEConv, to_hetero, GATConv

parser = argparse.ArgumentParser()
parser.add_argument('-f')
parser.add_argument('--use_weighted_loss', action='store_true',
                    help='Whether to use weighted MSE loss.')
args = parser.parse_args()

# We have an unbalanced dataset with many labels for rating 3 and 4, and very
# few for 0 and 1. Therefore we use a weighted MSE loss.
if args.use_weighted_loss:
    weight = torch.bincount(train_data['movie', 'session'].edge_label.resize(len(train_data['movie', 'session'].edge_label)))
    weight = weight.max() / weight
else:
    weight = None


def weighted_mae_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * abs((pred - target.to(pred.dtype)))).mean()

from sklearn.metrics import mean_absolute_error

train_data['movie', 'rated in', 'session'].edge_attr = train_data['movie', 'rated in', 'session'].edge_attr.float()
train_data['movie', 'rev_rates', 'user'].edge_attr = train_data['movie', 'rev_rates', 'user'].edge_attr.float()
train_data['session', 'rev_rated in', 'movie'].edge_attr = train_data['session', 'rev_rated in', 'movie'].edge_attr.float()
train_data['user','rates','movie'].edge_attr = train_data['user','rates','movie'].edge_attr.float()

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels,add_self_loops = False, edge_dim = -1,dropout = 0.7)
       #self.conv2 = GATConv((-1, -1), out_channels,add_self_loops = False,edge_dim = -1,dropout = 0.7)

    def forward(self, x, edge_index,edge_attr):
        x = self.conv1(x, edge_index,edge_attr).relu()
        x = self.conv2(x, edge_index,edge_attr)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['movie'][row], z_dict['session'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, train_data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict,edge_attr_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict,edge_attr_dict)
        pred = self.decoder(z_dict, edge_label_index)

        return pred

train_data['movie', 'rated in', 'session'].edge_attr = train_data['movie', 'rated in', 'session'].edge_attr.float()
train_data['movie', 'rev_rates', 'user'].edge_attr = train_data['movie', 'rev_rates', 'user'].edge_attr.float()
train_data['session', 'rev_rated in', 'movie'].edge_attr = train_data['session', 'rev_rated in', 'movie'].edge_attr.float()
train_data['user','rates','movie'].edge_attr = train_data['user','rates','movie'].edge_attr.float()

test_data['movie', 'session'].edge_index = target_edge_label_index
test_data['movie', 'session'].edge_label = target_edge_label
test_data['movie', 'session'].edge_attr = target_edges_attr

test_data['movie', 'rated in', 'session'].edge_attr = test_data['movie', 'rated in', 'session'].edge_attr.float()
test_data['movie', 'rev_rates', 'user'].edge_attr = test_data['movie', 'rev_rates', 'user'].edge_attr.float()
test_data['session', 'rev_rated in', 'movie'].edge_attr = test_data['session', 'rev_rated in', 'movie'].edge_attr.float()
test_data['user','rates','movie'].edge_attr = test_data['user','rates','movie'].edge_attr.float()

def scale_rate(my_tensor):
  dataf = my_tensor.tolist()
  scaler = MinMaxScaler(feature_range=(1, 5))
  scaled = scaler.fit_transform([[x] for x in dataf])
  scaled = [round(j) for sub in scaled for j in sub]
  # scaled = [[x] for x in scaled]

  for i in range(len(my_tensor)):
    my_tensor[i] = torch.round(my_tensor[i] * 0 + scaled[i])

  # scaled = torch.FloatTensor(scaled)
  return my_tensor

from sklearn.preprocessing import MinMaxScaler

model = Model(hidden_channels=128).to(device)

with torch.no_grad():
    model.encoder(train_data.x_dict, train_data.edge_index_dict,train_data.edge_attr_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    # pred = model(train_data.x_dict, train_data.edge_index_dict,train_data.edge_attr_dict,
    #              train_data['movie', 'session'].edge_index)

    pred = model(train_data.x_dict, train_data.edge_index_dict,train_data.edge_attr_dict,
                 train_data['movie', 'session'].edge_index)
    target = train_data['movie', 'session'].edge_label

    pred = scale_rate(pred)


    # loss = mean_absolute_error(target.tolist(), pred.tolist())
    loss = weighted_mae_loss(pred, target, weight)
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(data):
    model.eval()
    # pred = model(data.x_dict, data.edge_index_dict,
    #              data.edge_attr_dict,
    #              data['movie', 'session'].edge_index)
    
    pred = model(data.x_dict, data.edge_index_dict,data.edge_attr_dict,
              data['movie', 'session'].edge_index)
        
    pred = scale_rate(pred)
   

    target = data['movie', 'session'].edge_label.float()
    # rmse = F.mse_loss(pred, target).sqrt()
    # rmse = mean_absolute_error(target.tolist(), pred.tolist())
    rmse = weighted_mae_loss(pred, target, weight)
    return pred,float(rmse)

def scale(my_tensor,max_range):
   
  max_number = max(my_tensor.tolist())
  ratio = max_range / max_number

  for i in range(len(my_tensor)):
    my_tensor[i] = torch.round(my_tensor[i] * ratio)
  # max_number = max(my_tensor.tolist())
  # for i in range(len(my_tensor)):
  #   ratio = maximum / my_tensor[i]
  #   my_tensor[i] = torch.round(5 * ratio)


  return my_tensor

for epoch in range(1, 21):
    loss = train()
    pred1,mae_train = test(train_data)
    pred2,mae_test = test(test_data)
    print(mae_train,mae_test)
    if (epoch == 10):
      print(pred2)

pred_s = pred2

"""# New Section"""

if args.use_weighted_loss:
    weight = torch.bincount(train_data['user', 'movie'].edge_label.resize(len(train_data['user', 'movie'].edge_label)))
    weight = weight.max() / weight
else:
    weight = None


def weighted_mae_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * abs((pred - target.to(pred.dtype)))).mean()

train_data

test_data

test_data['user','rates','movie'].edge_attr = ui_target_edges_attr
test_data['user','rates','movie'].edge_label = ui_target_edge_label
test_data['user','rates','movie'].edge_index = ui_target_edge_label_index

test_data

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels,add_self_loops = False, edge_dim = -1)
        self.conv2 = GATConv((-1, -1), out_channels,add_self_loops = False,edge_dim = -1)

    def forward(self, x, edge_index,edge_attr):
        x = self.conv1(x, edge_index,edge_attr).relu()
        x = self.conv2(x, edge_index,edge_attr)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, train_data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict,edge_attr_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict,edge_attr_dict)
        pred = self.decoder(z_dict, edge_label_index)

        return pred

model = Model(hidden_channels=32).to(device)

with torch.no_grad():
    model.encoder(train_data.x_dict, train_data.edge_index_dict,train_data.edge_attr_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    # pred = model(train_data.x_dict, train_data.edge_index_dict,train_data.edge_attr_dict,
    #              train_data['movie', 'session'].edge_index)

    pred = model(train_data.x_dict, train_data.edge_index_dict,train_data.edge_attr_dict,
                 train_data['user', 'movie'].edge_index)
    target = train_data['user', 'movie'].edge_label

    pred = scale_rate(pred)


    # loss = mean_absolute_error(target.tolist(), pred.tolist())
    loss = weighted_mae_loss(pred, target, weight)
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(data):
    model.eval()
    # pred = model(data.x_dict, data.edge_index_dict,
    #              data.edge_attr_dict,
    #              data['movie', 'session'].edge_index)
    
    pred = model(data.x_dict, data.edge_index_dict,data.edge_attr_dict,
              data['user', 'movie'].edge_index)
        
    pred = scale_rate(pred)
   

    target = data['user', 'movie'].edge_label.float()
    # rmse = F.mse_loss(pred, target).sqrt()
    # rmse = mean_absolute_error(target.tolist(), pred.tolist())
    rmse = weighted_mae_loss(pred, target, weight)
    return pred,float(rmse)

# filter = list(train_df['session'].unique())
# user_test_df = total_test_df[~total_test_df.session.isin(filter)]
# filter = list(user_test_df.groupby('user_id').max()['session'])
# user_test_df = user_test_df[user_test_df.session.isin(filter)]
# user_test_df = user_test_df[user_test_df.user_id.isin(train_df['user_id'])]

test_data['movie', 'rated in', 'session'].edge_attr = test_data['movie', 'rated in', 'session'].edge_attr.float()
test_data['movie', 'rev_rates', 'user'].edge_attr = test_data['movie', 'rev_rates', 'user'].edge_attr.float()
test_data['session', 'rev_rated in', 'movie'].edge_attr = test_data['session', 'rev_rated in', 'movie'].edge_attr.float()
test_data['user','rates','movie'].edge_attr = test_data['user','rates','movie'].edge_attr.float()

for epoch in range(1, 16):
    loss = train()
    pred1,mae_train = test(train_data)
    pred2r,mae_test = test(test_data)
    print(mae_train,mae_test)
    if (epoch == 15):
      print(pred2r)

pred2r

test_data
