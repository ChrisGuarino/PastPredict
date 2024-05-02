import random
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import yfinance as yf


def make_seqs(data, tau):
    x = torch.tensor(data, dtype=torch.float32)
    x = x.flatten()
    T = len(x)
    
    # generate sequences and labels (yrev = x0 yfwd = xt, [xt-tau...xt-1])
    features = torch.zeros((T - tau, tau))
    for i in range(tau):
        features[:, i] = x[i: T - tau + i]

    labels_fwd = x[tau:].reshape((-1, 1))
    labels_rev = x[0: T - (tau+1)].reshape((-1, 1))

    features = features.reshape(-1, tau, 1)

    labels_fwd = labels_fwd[1:]
    features = features[1:]

    labels = torch.cat((labels_rev, labels_fwd), dim=1).reshape(-1, 2, 1)

    return features, labels

def get_stock_data(tau=100):
    ticker = '^GSPC'
    start = '1974-01-01'
    end = '2024-01-01'

    df = yf.download(ticker, start=start, end=end)
    df['Percent Change'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    df.drop(['Adj Close', 'Open', 'High', 'Low', 'Volume', 'Close'], axis=1, inplace=True)
    train_percent = 0.6  
    val_percent = 0.2    
    test_percent = 0.2   

    total_entries = len(df)

    train_cutoff = int(train_percent * total_entries)
    val_cutoff = int((train_percent + val_percent) * total_entries)

    train_df = df.iloc[:train_cutoff]
    val_df = df.iloc[train_cutoff:val_cutoff]
    test_df = df.iloc[val_cutoff:]


    scaler = MinMaxScaler(feature_range=(-1, 1))

    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    X_train, y_train = make_seqs(train_scaled, tau)
    X_val, y_val = make_seqs(val_scaled, tau)
    X_test, y_test = make_seqs(test_scaled, tau)

    batch_size = 32
    # datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    # loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_loader= DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class Exp2Dataset(Dataset):
    def __init__(self, X_fwd, y_fwd, X_rev, y_rev):
        self.X_fwd = X_fwd
        self.y_fwd = y_fwd
        self.X_rev = X_rev
        self.y_rev = y_rev

        assert len(X_fwd) == len(y_fwd)
        assert len(X_rev) == len(y_rev)
        self.length = len(X_fwd)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.X_fwd[idx], self.y_fwd[idx], self.X_rev[idx], self.y_rev[idx]

def get_stock_data_e2a(tau):
    ticker = '^GSPC'
    start = '1974-01-01'
    end = '2024-01-01'

    df = yf.download(ticker, start=start, end=end)
    df['Percent Change'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    df.drop(['Adj Close', 'Open', 'High', 'Low', 'Volume', 'Close'], axis=1, inplace=True)
    train_percent = 0.6  
    val_percent = 0.2    
    test_percent = 0.2   

    total_entries = len(df)

    train_cutoff = int(train_percent * total_entries)
    val_cutoff = int((train_percent + val_percent) * total_entries)

    train_df = df.iloc[:train_cutoff]
    val_df = df.iloc[train_cutoff:val_cutoff]
    test_df = df.iloc[val_cutoff:]

    scaler = MinMaxScaler(feature_range=(-1, 1))

    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    X_train_fwd, y_train_fwd = make_seq_e2a(train_scaled, tau)
    X_val_fwd, y_val_fwd = make_seq_e2a(val_scaled, tau)
    X_test_fwd, y_test_fwd = make_seq_e2a(test_scaled, tau)

    X_train_rev, y_train_rev = make_seq_e2a(train_scaled, tau, rev=True)
    X_val_rev, y_val_rev = make_seq_e2a(val_scaled, tau, rev=True)
    X_test_rev, y_test_rev = make_seq_e2a(test_scaled, tau, rev=True)

    batch_size = 32
    # datasets
    train_dataset = Exp2Dataset(X_train_fwd, y_train_fwd, X_train_rev, y_train_rev)
    val_dataset = Exp2Dataset(X_val_fwd, y_val_fwd, X_val_rev, y_val_rev)
    test_dataset = Exp2Dataset(X_test_fwd, y_test_fwd, X_test_rev, y_test_rev)
    # loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_loader= DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def make_seq_e2a(data, tau, rev=False):
    x = torch.tensor(data, dtype=torch.float32)
    x = x.flatten()
    T=len(data)

    if rev:
        x = reversed(x)

    # generate sequences and labels (y = xt, xt[xt-tau...xt-1])
    features = torch.zeros((T - tau, tau))
    for i in range(tau):
        features[:, i] = x[i: T - tau + i]
    labels = x[tau:].reshape((-1, 1))
    features = features.reshape(-1, tau, 1)

    return features, labels
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    # for cuda
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
       
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def gen_sin_data(T,tau, graph=True, rev=False):

    # generate data
    time = torch.arange(0, T, dtype=torch.float32)
    x = torch.sin(0.005 * time) + torch.randn(T) * 0.05

    if rev:
        x = reversed(x)

    
    # generate sequences and labels (y = xt, xt[xt-tau...xt-1])
    features = torch.zeros((T - tau, tau))
    for i in range(tau):
        features[:, i] = x[i: T - tau + i]
    labels = x[tau:].reshape((-1, 1))
    features = features.reshape(-1, tau, 1)

    # split size
    train_size = int(0.6 * len(features))
    val_size = int(0.2 * len(features))
    test_size = len(features) - train_size - val_size

    # split
    X_train, y_train = features[:train_size], labels[:train_size]
    X_val, y_val = features[train_size:train_size+val_size], labels[train_size:train_size+val_size]
    X_test, y_test = features[train_size+val_size:], labels[train_size+val_size:]

    if  graph:
        # graph data
        graph_sin(time, x, train_size)


    return X_train, y_train, X_val, y_val, X_test, y_test

def graph_sin(X, y, n_train):
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    ax.plot(X_train,y_train, lw=3, label='train data')
    ax.plot(X_test, y_test,  lw=3, label='test data')
    ax.legend(loc="lower left")
    plt.show();

def get_e2_sin_data(T, n_train, tau):
    # generate data
    time = torch.arange(0, T, dtype=torch.float32)
    x = torch.sin(0.01 * time) # add noise here if wanted to -- + torch.randn(T) * 0.2

    # graph data
    graph_sin(time, x, n_train)
    
    # generate sequences and labels (yrev = x0 yfwd = xt, xt[xt-tau...xt-1])
    features = torch.zeros((T - tau, tau))
    for i in range(tau):
        features[:, i] = x[i: T - tau + i]

    labels_fwd = x[tau:].reshape((-1, 1))
    labels_rev = x[0: T - (tau+1)].reshape((-1, 1))

    features = features.reshape(-1, tau, 1)

    labels_fwd = labels_fwd[1:]
    features = features[1:]

    labels = torch.cat((labels_rev, labels_fwd), dim=1).reshape(-1, 2, 1)

    # split to train and test
    X_train = features[:n_train]
    X_test = features[n_train:]
    y_train = labels[:n_train]
    y_test = labels[n_train:]

    return X_train, y_train, X_test, y_test