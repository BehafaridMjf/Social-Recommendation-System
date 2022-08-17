def load_node_csv(df, index_col, encoders=None, **kwargs):
    df_node = df.set_index(index_col,drop = True)
    mapping = {index: i for i, index in enumerate(df_node.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df_node[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return df_node,x, mapping

class IdentityEncoder(object):
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)

def load_edge_csv(df, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None,label = None, **kwargs):
    df_edge = df

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)


    return edge_index, edge_attr
    
    
"""# Load Nodes

"""

user_test_df

filter = list(train_df['session'].unique())
target_edges_df = total_test_df.loc[total_test_df['session'].isin(filter)]
movie_df = pd.concat([train_df[['movie_id','category']],target_edges_df[['movie_id','category']]])
# movie_df = pd.DataFrame(movie_df)
movie_df = pd.concat([movie_df[['movie_id','category']],user_test_df[['movie_id','category']]])
movie_df = pd.DataFrame(movie_df)

movie_df.drop_duplicates(inplace = True)
movie_df.reset_index(drop = True, inplace = True)
movie_df

train_df_node1 ,train_user_x, train_user_mapping = load_node_csv(train_df, index_col='user_id')
# test_df_node1 ,test_user_x, test_user_mapping = load_node_csv(test_df, index_col='user_id')

train_movie_df = pd.DataFrame(train_df[['movie_id','category']])
train_movie_df = train_movie_df.drop_duplicates()

# test_movie_df = pd.DataFrame(test_df[['movie_id','category']])
# test_movie_df = test_movie_df.drop_duplicates()

train_df_node2 ,train_movie_x, train_movie_mapping = load_node_csv(movie_df, index_col='movie_id',encoders={'category':IdentityEncoder(dtype=torch.long)})

# test_df_node2 ,test_movie_x, test_movie_mapping = load_node_csv(test_movie_df, index_col='movie_id',encoders={'category':IdentityEncoder(dtype=torch.long)})

session_df = train_df['session']
session_column = session_df.drop_duplicates()
session_df = pd.DataFrame(session_column)

# test_session_df = test_df['session']
# test_session_column = test_session_df.drop_duplicates()
# test_session_df = pd.DataFrame(test_session_column)

train_df_node3 ,train_session_x, train_session_mapping = load_node_csv(session_df, index_col='session')
# test_df_node3 ,test_session_x, test_session_mapping = load_node_csv(test_session_df, index_col='session')





"""# Load Edges"""

t_ui_edge_index, t_ui_edge_label = load_edge_csv(
    train_df,
    src_index_col='user_id',
    src_mapping=train_user_mapping,
    dst_index_col='movie_id',
    dst_mapping=train_movie_mapping,
    encoders={'weight':IdentityEncoder(dtype=torch.long)}
)


# test_ui_edge_index, test_ui_edge_label = load_edge_csv(
#     test_df,
#     src_index_col='user_id',
#     src_mapping=test_user_mapping,
#     dst_index_col='movie_id',
#     dst_mapping=test_movie_mapping,
#     encoders={'rating': IdentityEncoder(dtype=torch.long),'weight':IdentityEncoder(dtype=torch.long)}
# )

t_is_edge_index, t_is_edge_label = load_edge_csv(
    train_df,
    src_index_col='movie_id',
    src_mapping= train_movie_mapping,
    dst_index_col='session',
    dst_mapping= train_session_mapping,
    encoders={'weight':IdentityEncoder(dtype=torch.long)},
)


# test_is_edge_index, test_is_edge_label = load_edge_csv(
#     test_df,
#     src_index_col='movie_id',
#     src_mapping= test_movie_mapping,
#     dst_index_col='session',
#     dst_mapping= test_session_mapping,
#     encoders={'weight':IdentityEncoder(dtype=torch.long)},
# )

train_df

t_edge_label_index, t_edge_label = load_edge_csv(
    train_df,
    src_index_col='movie_id',
    src_mapping=train_movie_mapping,
    dst_index_col='session',
    dst_mapping=train_session_mapping,
    encoders={'rating': IdentityEncoder(dtype=torch.long)},
)

t_rui_edge_label_index, t_rui_edge_label = load_edge_csv(
    train_df,
    src_index_col='user_id',
    src_mapping=train_user_mapping,
    dst_index_col='movie_id',
    dst_mapping=train_movie_mapping,
    encoders={'rating': IdentityEncoder(dtype=torch.long)},
)

train_edge_label_index, train_edge_label = load_edge_csv(
    train_df,
    src_index_col='movie_id',
    src_mapping=train_movie_mapping,
    dst_index_col='session',
    dst_mapping=train_session_mapping,
    encoders={'rating': IdentityEncoder(dtype=torch.long)}
)

target_edge_label_index, target_edge_label = load_edge_csv(
    target_edges_df,
    src_index_col='movie_id',
    src_mapping=train_movie_mapping,
    dst_index_col='session',
    dst_mapping=train_session_mapping,
    encoders={'rating': IdentityEncoder(dtype=torch.long)},
)

target_attr_index, target_edges_attr = load_edge_csv(
    target_edges_df,
    src_index_col='movie_id',
    src_mapping=train_movie_mapping,
    dst_index_col='session',
    dst_mapping=train_session_mapping,
    encoders={'weight':IdentityEncoder(dtype=torch.long)})

ui_target_edge_label_index, ui_target_edge_label = load_edge_csv(
    user_test_df,
    src_index_col='user_id',
    src_mapping=train_user_mapping,
    dst_index_col='movie_id',
    dst_mapping=train_movie_mapping,
    encoders={'rating': IdentityEncoder(dtype=torch.long)}
)

ui_target_attr_index, ui_target_edges_attr = load_edge_csv(
    user_test_df,
    src_index_col='user_id',
    src_mapping=train_user_mapping,
    dst_index_col='movie_id',
    dst_mapping=train_movie_mapping,
    encoders={'weight':IdentityEncoder(dtype=torch.long)})

session_link_df = session_df.copy()
session_link_df.insert(0,'src',session_df['session'])
session_link_df = session_link_df.rename(columns = {'session':'dst'})

t_ss_edge_index, t_ss_edge_label = load_edge_csv(
    session_link_df,
    src_index_col='src',
    src_mapping= train_session_mapping,
    dst_index_col='dst',
    dst_mapping= train_session_mapping,
)





"""# Add Movie Movie Links"""

def get_category(movie_df):
  mov_cat_1 = movie_df.loc[movie_df['category'] == 1]['movie_id'].tolist()
  mov_cat_2 = movie_df.loc[movie_df['category'] == 2]['movie_id'].tolist()
  mov_cat_3 = movie_df.loc[movie_df['category'] == 3]['movie_id'].tolist()
  mov_cat_4 = movie_df.loc[movie_df['category'] == 4]['movie_id'].tolist()
  mov_cat_5 = movie_df.loc[movie_df['category'] == 5]['movie_id'].tolist()
  mov_cat_6 = movie_df.loc[movie_df['category'] == 6]['movie_id'].tolist()

  return mov_cat_1,mov_cat_2,mov_cat_3,mov_cat_4,mov_cat_5,mov_cat_6

filter = train_df_node2.reset_index()
mov_cat_1,mov_cat_2,mov_cat_3,mov_cat_4,mov_cat_5,mov_cat_6 = get_category(filter)

# test_filter = test_df_node2.reset_index()
# test_mov_cat_1,test_mov_cat_2,test_mov_cat_3,test_mov_cat_4,test_mov_cat_5,test_mov_cat_6 = get_category(test_filter)

from itertools import combinations
cat1_links = list(combinations(mov_cat_1, 2))
cat2_links = list(combinations(mov_cat_2, 2))
cat3_links = list(combinations(mov_cat_3, 2))
cat4_links = list(combinations(mov_cat_4, 2))
cat5_links = list(combinations(mov_cat_5, 2))
cat6_links = list(combinations(mov_cat_6, 2))


# test_cat1_links = list(combinations(test_mov_cat_1, 2))
# test_cat2_links = list(combinations(test_mov_cat_2, 2))
# test_cat3_links = list(combinations(test_mov_cat_3, 2))
# test_cat4_links = list(combinations(test_mov_cat_4, 2))
# test_cat5_links = list(combinations(test_mov_cat_5, 2))
# test_cat6_links = list(combinations(test_mov_cat_6, 2))

mov_cat1_df_src = list(zip(*cat1_links))[0]
mov_cat1_df_dst = list(zip(*cat1_links))[1]

mov_cat1_df = pd.DataFrame(columns = ['src','dst'])

mov_cat1_df['src'] = mov_cat1_df_src
mov_cat1_df['dst'] = mov_cat1_df_dst




# test_mov_cat1_df_src = list(zip(*test_cat1_links))[0]
# test_mov_cat1_df_dst = list(zip(*test_cat1_links))[1]

# test_mov_cat1_df = pd.DataFrame(columns = ['src','dst'])

# test_mov_cat1_df['src'] = test_mov_cat1_df_src
# test_mov_cat1_df['dst'] = test_mov_cat1_df_dst

mov_cat2_df_src = list(zip(*cat2_links))[0]
mov_cat2_df_dst = list(zip(*cat2_links))[1]

mov_cat2_df = pd.DataFrame(columns = ['src','dst'])

mov_cat2_df['src'] = mov_cat2_df_src
mov_cat2_df['dst'] = mov_cat2_df_dst



# test_mov_cat2_df_src = list(zip(*test_cat2_links))[0]
# test_mov_cat2_df_dst = list(zip(*test_cat2_links))[1]

# test_mov_cat2_df = pd.DataFrame(columns = ['src','dst'])

# test_mov_cat2_df['src'] = test_mov_cat2_df_src
# test_mov_cat2_df['dst'] = test_mov_cat2_df_dst

mov_cat3_df_src = list(zip(*cat3_links))[0]
mov_cat3_df_dst = list(zip(*cat3_links))[1]

mov_cat3_df = pd.DataFrame(columns = ['src','dst'])

mov_cat3_df['src'] = mov_cat3_df_src
mov_cat3_df['dst'] = mov_cat3_df_dst


# test_mov_cat3_df_src = list(zip(*test_cat3_links))[0]
# test_mov_cat3_df_dst = list(zip(*test_cat3_links))[1]

# test_mov_cat3_df = pd.DataFrame(columns = ['src','dst'])

# test_mov_cat3_df['src'] = test_mov_cat3_df_src
# test_mov_cat3_df['dst'] = test_mov_cat3_df_dst

mov_cat4_df_src = list(zip(*cat4_links))[0]
mov_cat4_df_dst = list(zip(*cat4_links))[1]

mov_cat4_df = pd.DataFrame(columns = ['src','dst'])

mov_cat4_df['src'] = mov_cat4_df_src
mov_cat4_df['dst'] = mov_cat4_df_dst



# test_mov_cat4_df_src = list(zip(*test_cat4_links))[0]
# test_mov_cat4_df_dst = list(zip(*test_cat4_links))[1]

# test_mov_cat4_df = pd.DataFrame(columns = ['src','dst'])

# test_mov_cat4_df['src'] = test_mov_cat4_df_src
# test_mov_cat4_df['dst'] = test_mov_cat4_df_dst

mov_cat5_df_src = list(zip(*cat5_links))[0]
mov_cat5_df_dst = list(zip(*cat5_links))[1]

mov_cat5_df = pd.DataFrame(columns = ['src','dst'])

mov_cat5_df['src'] = mov_cat5_df_src
mov_cat5_df['dst'] = mov_cat5_df_dst


# test_mov_cat5_df_src = list(zip(*test_cat5_links))[0]
# test_mov_cat5_df_dst = list(zip(*test_cat5_links))[1]

# test_mov_cat5_df = pd.DataFrame(columns = ['src','dst'])

# test_mov_cat5_df['src'] = test_mov_cat5_df_src
# test_mov_cat5_df['dst'] = test_mov_cat5_df_dst

mov_cat6_df_src = list(zip(*cat6_links))[0]
mov_cat6_df_dst = list(zip(*cat6_links))[1]

mov_cat6_df = pd.DataFrame(columns = ['src','dst'])

mov_cat6_df['src'] = mov_cat6_df_src
mov_cat6_df['dst'] = mov_cat6_df_dst


# test_mov_cat6_df_src = list(zip(*test_cat6_links))[0]
# test_mov_cat6_df_dst = list(zip(*test_cat6_links))[1]

# test_mov_cat6_df = pd.DataFrame(columns = ['src','dst'])

# test_mov_cat6_df['src'] = test_mov_cat6_df_src
# test_mov_cat6_df['dst'] = test_mov_cat6_df_dst

frames = [mov_cat1_df,mov_cat2_df,mov_cat3_df,mov_cat4_df,mov_cat5_df,mov_cat6_df]
ii_df = pd.concat(frames)


# test_frames = [test_mov_cat1_df,test_mov_cat2_df,test_mov_cat3_df,test_mov_cat4_df,test_mov_cat5_df,test_mov_cat6_df]
# test_ii_df = pd.concat(test_frames)

t_ii_edge_index, t_ii_edge_label = load_edge_csv(
    ii_df,
    src_index_col='src',
    src_mapping=train_movie_mapping,
    dst_index_col='dst',
    dst_mapping= train_movie_mapping)


# test_ii_edge_index, test_ii_edge_label = load_edge_csv(
#     test_ii_df,
#     src_index_col='src',
#     src_mapping=test_movie_mapping,
#     dst_index_col='dst',
#     dst_mapping= test_movie_mapping)
