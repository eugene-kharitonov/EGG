# not necessary, but could be useful
def dict2string(d):
    s = []

    for k, v in d.items():
        if type(v) in (int, float):
            s.append(f"--{k}={v}")
        elif type(v) is bool and v:
            s.append(f"--{k}")
        elif type(v) is str:
            assert '"' not in v, f"Key {k} has string value {v} which contains forbidden quotes."
            s.append(f'--{k}={v}')
        else:
            raise Exception(
                f"Key {k} has value {v} of unsupported type {type(v)}.")
    return s

# only grid() is called from nest


grid_json = \
"""[{
    "n_attributes": [4],
    "n_values": [10],
    "vocab_size" : [5,10,50,100],
    "max_len" : [6,8],
    "batch_size" : [5120],
    "data_scaler": [60],
    "n_epochs": [3000],
    "random_seed": [0,1,2,3,4,5,6,7,8,9],
    "sender_hidden": [500],
    "receiver_hidden": [500],
    "sender_entropy_coeff": [0.5],
    "sender_cell": ["gru"],
    "receiver_cell": ["gru"],
    "lr":[0.001],
    "receiver_emb": [30],
    "sender_emb": [5]
},
{
    "n_attributes": [4],
    "n_values": [10],
    "vocab_size" : [50],
    "max_len" : [3,4],
    "batch_size" : [5120],
    "data_scaler": [60],
    "n_epochs": [3000],
    "random_seed": [0,1,2,3,4,5,6,7,8,9],
    "sender_hidden": [500],
    "receiver_hidden": [500],
    "sender_entropy_coeff": [0.5],
    "sender_cell": ["gru"],
    "receiver_cell": ["gru"],
    "lr":[0.001],
    "receiver_emb": [30],
    "sender_emb": [5]
},
{
    "n_attributes": [4],
    "n_values": [10],
    "vocab_size" : [100],
    "max_len" : [2,3,4],
    "batch_size" : [5120],
    "data_scaler": [60],
    "n_epochs": [3000],
    "random_seed": [0,1,2,3,4,5,6,7,8,9],
    "sender_hidden": [500],
    "receiver_hidden": [500],
    "sender_entropy_coeff": [0.5],
    "sender_cell": ["gru"],
    "receiver_cell": ["gru"],
    "lr":[0.001],
    "receiver_emb": [30],
    "sender_emb": [5]
},
{
    "n_attributes": [3],
    "n_values": [10],
    "vocab_size" : [50],
    "max_len" : [2,3,4,6,8],
    "batch_size" : [5120],
    "data_scaler": [60],
    "n_epochs": [3000],
    "random_seed": [0,1,2,3,4,5,6,7,8,9],
    "sender_hidden": [500],
    "receiver_hidden": [500],
    "sender_entropy_coeff": [0.5],
    "sender_cell": ["gru"],
    "receiver_cell": ["gru"],
    "lr":[0.001],
    "receiver_emb": [30],
    "sender_emb": [5]
},
{
    "n_attributes": [3],
    "n_values": [10],
    "vocab_size" : [100],
    "max_len" : [2],
    "batch_size" : [5120],
    "data_scaler": [60],
    "n_epochs": [3000],
    "random_seed": [0,1,2,3,4,5,6,7,8,9],
    "sender_hidden": [500],
    "receiver_hidden": [500],
    "sender_entropy_coeff": [0.5],
    "sender_cell": ["gru"],
    "receiver_cell": ["gru"],
    "lr":[0.001],
    "receiver_emb": [30],
    "sender_emb": [5]
},
{
    "n_attributes": [3],
    "n_values": [10],
    "vocab_size" : [10],
    "max_len" : [4,6,8],
    "batch_size" : [5120],
    "data_scaler": [60],
    "n_epochs": [3000],
    "random_seed": [0,1,2,3,4,5,6,7,8,9],
    "sender_hidden": [500],
    "receiver_hidden": [500],
    "sender_entropy_coeff": [0.5],
    "sender_cell": ["gru"],
    "receiver_cell": ["gru"],
    "lr":[0.001],
    "receiver_emb": [30],
    "sender_emb": [5]
},
{
    "n_attributes": [3],
    "n_values": [10],
    "vocab_size" : [5],
    "max_len" : [6,8],
    "batch_size" : [5120],
    "data_scaler": [60],
    "n_epochs": [3000],
    "random_seed": [0,1,2,3,4,5,6,7,8,9],
    "sender_hidden": [500],
    "receiver_hidden": [500],
    "sender_entropy_coeff": [0.5],
    "sender_cell": ["gru"],
    "receiver_cell": ["gru"],
    "lr":[0.001],
    "receiver_emb": [30],
    "sender_emb": [5]
},
{
    "n_attributes": [2],
    "n_values": [10],
    "vocab_size" : [50],
    "max_len" : [2,3],
    "batch_size" : [5120],
    "data_scaler": [60],
    "n_epochs": [3000],
    "random_seed": [0,1,2,3,4,5,6,7,8,9],
    "sender_hidden": [500],
    "receiver_hidden": [500],
    "sender_entropy_coeff": [0.5],
    "sender_cell": ["gru"],
    "receiver_cell": ["gru"],
    "lr":[0.001],
    "receiver_emb": [30],
    "sender_emb": [5]
}
{
    "n_attributes": [2],
    "n_values": [10],
    "vocab_size" : [100],
    "max_len" : [2],
    "batch_size" : [5120],
    "data_scaler": [60],
    "n_epochs": [3000],
    "random_seed": [0,1,2,3,4,5,6,7,8,9],
    "sender_hidden": [500],
    "receiver_hidden": [500],
    "sender_entropy_coeff": [0.5],
    "sender_cell": ["gru"],
    "receiver_cell": ["gru"],
    "lr":[0.001],
    "receiver_emb": [30],
    "sender_emb": [5]
},
{
    "n_attributes": [2],
    "n_values": [10],
    "vocab_size" : [10],
    "max_len" : [2,3,4,6,8],
    "batch_size" : [5120],
    "data_scaler": [60],
    "n_epochs": [3000],
    "random_seed": [0,1,2,3,4,5,6,7,8,9],
    "sender_hidden": [500],
    "receiver_hidden": [500],
    "sender_entropy_coeff": [0.5],
    "sender_cell": ["gru"],
    "receiver_cell": ["gru"],
    "lr":[0.001],
    "receiver_emb": [30],
    "sender_emb": [5]
},
{
    "n_attributes": [2],
    "n_values": [10],
    "vocab_size" : [5],
    "max_len" : [3,4,6,8],
    "batch_size" : [5120],
    "data_scaler": [60],
    "n_epochs": [3000],
    "random_seed": [0,1,2,3,4,5,6,7,8,9],
    "sender_hidden": [500],
    "receiver_hidden": [500],
    "sender_entropy_coeff": [0.5],
    "sender_cell": ["gru"],
    "receiver_cell": ["gru"],
    "lr":[0.001],
    "receiver_emb": [30],
    "sender_emb": [5]
}]"""

    