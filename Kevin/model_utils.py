
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import random
import copy

######### All

def get_metrics(preds, true):
        mae = mean_absolute_error(true, preds)
        mse = mean_squared_error(true, preds)

        return mae, mse

######### E1

def train_e1_model(model, train_loader, val_loader, loss_fn, opt, n_epochs=100):
    losses_train = []
    losses_val = []
    best_epoch = -1
    best_val_loss = float('inf')
    best_model_state_dict = None

    # forward training loop
    for epoch in range(n_epochs):
        tot_train = 0.0
        tot_val = 0.0
        
        # Training 
        model.train()  
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            tot_train += loss.item()
        
        losses_train.append(tot_train / len(train_loader.sampler))
        
        # Validation 
        model.eval()  
        with torch.no_grad():
            for inputs_val, labels_val in val_loader:
                outputs_val = model(inputs_val)
                val_loss = loss_fn(outputs_val, labels_val)
                tot_val += val_loss.item()
        
        val_loss = tot_val / len(val_loader.sampler)
        losses_val.append(val_loss)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    return best_model_state_dict, best_epoch, losses_train, losses_val

def make_preds(model, test_loader):
        preds = []
        true = []

        # Set the model to evaluation mode
        model.eval()

        # Make predictions
        with torch.no_grad():
            for inputs, labels in test_loader:
                out = model(inputs)

                preds.append(out.cpu().numpy())
                true.append(labels.cpu().numpy())

        preds = np.concatenate(preds).flatten()
        true = np.concatenate(true).flatten()

        return preds, true

############ E2 a
# TODO make for fwd and rev

def train_e2a_model(model_fwd, model_rev, train_loader, val_loader, loss_fn, opt_fwd, opt_rev, n_epochs=100):
    losses_train_fwd = []
    losses_train_rev = []
    losses_train_tot = []

    losses_val_fwd = []
    losses_val_rev = []
    losses_val_tot = []

    best_model_state_dict_fwd = None
    best_model_state_dict_rev = None

    best_epoch = -1
    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        epoch_loss_fwd = 0.
        epoch_loss_rev = 0.
        epoch_loss_tot = 0.

        val_epoch_loss_fwd = 0.
        val_epoch_loss_rev = 0.
        val_epoch_loss_tot = 0.

        model_fwd.train() 
        model_rev.train() 
        
        for inputs_fwd, labels_fwd, inputs_rev, labels_rev in train_loader:

            out_fwd = model_fwd(inputs_fwd)
            out_rev = model_rev(inputs_rev)
 
            loss_fwd = loss_fn(out_fwd, labels_fwd)
            loss_rev = loss_fn(out_rev, labels_rev)
            loss_tot = loss_fwd + loss_rev

            # Backward and optimize
            opt_fwd.zero_grad()
            opt_rev.zero_grad()

            loss_tot.backward()

            opt_fwd.step()
            opt_rev.step()
            
            epoch_loss_fwd += loss_fwd.item()
            epoch_loss_rev += loss_rev.item()
            epoch_loss_tot += loss_tot.item()
        
        losses_train_fwd.append(epoch_loss_fwd / len(train_loader.sampler))
        losses_train_rev.append(epoch_loss_rev / len(train_loader.sampler))
        losses_train_tot.append(epoch_loss_tot / len(train_loader.sampler))

        # Validation 
        model_fwd.eval()  
        model_rev.eval() 

        with torch.no_grad():
            for inputs_fwd_val, labels_fwd_val, inputs_rev_val, labels_rev_val in val_loader:
                out_fwd_val = model_fwd(inputs_fwd_val)
                out_rev_val = model_rev(inputs_rev_val)

                val_loss_fwd = loss_fn(out_fwd_val, labels_fwd_val)
                val_loss_rev = loss_fn(out_rev_val, labels_rev_val)
                val_loss_tot = val_loss_fwd + val_loss_rev

                val_epoch_loss_fwd += val_loss_fwd.item()
                val_epoch_loss_rev += val_loss_rev.item()
                val_epoch_loss_tot += val_loss_tot.item()

        val_loss = val_epoch_loss_tot / len(val_loader.sampler)

        losses_val_fwd.append(val_epoch_loss_fwd / len(val_loader.sampler))
        losses_val_rev.append(val_epoch_loss_rev / len(val_loader.sampler))
        losses_val_tot.append(val_loss)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict_fwd = copy.deepcopy(model_fwd.state_dict())
            best_model_state_dict_rev = copy.deepcopy(model_rev.state_dict())
            best_epoch = epoch

    fwd_result = best_model_state_dict_fwd, best_epoch, losses_train_fwd, losses_val_fwd
    rev_result = best_model_state_dict_rev, best_epoch, losses_train_rev, losses_val_rev

    return fwd_result, rev_result

def make_preds_e2(model, test_loader, rev=False):
    preds = []
    true = []

    # eval mode
    model.eval()
    if rev:
        # make preds
        with torch.no_grad():
            for _, _, inputs, labels in test_loader:
                
                out = model(inputs)

                preds.append(out.cpu().numpy())
                true.append(labels.cpu().numpy())

        preds = np.concatenate(preds).flatten()
        true = np.concatenate(true).flatten()

    else:
         # make preds
        with torch.no_grad():
            for inputs, labels, _, _ in test_loader:
                
                out = model(inputs)

                preds.append(out.cpu().numpy())
                true.append(labels.cpu().numpy())

        preds = np.concatenate(preds).flatten()
        true = np.concatenate(true).flatten()


    return preds, true

    ################# E2B1
def train_model_e2b1(model, train_loader, val_loader, loss_fn, opt, epochs=100):
    losses_fwd_train = []
    losses_rev_train = []
    losses_tot_train = []

    losses_fwd_val = []
    losses_rev_val = []
    losses_tot_val = []

    best_model_state_dict = None

    best_epoch = -1
    best_val_loss = float('inf')

    for epoch in range(epochs):
        epoch_loss_fwd = 0.
        epoch_loss_rev = 0.
        epoch_loss_tot = 0.


        tot_val_fwd = 0.
        tot_val_rev = 0.
        tot_val_tot = 0.

        model.train()
        for inputs, labels in train_loader:

            out = model(inputs)
            out_fwd = out[:, 0:1]
            out_rev = out[:, 1:2]
            
            labels_fwd = labels[:, 1, :]
            labels_rev = labels[:, 0, :]
            
            loss_fwd = loss_fn(out_fwd, labels_fwd)
            loss_rev = loss_fn(out_rev, labels_rev)
            loss_tot = loss_fwd + loss_rev

            opt.zero_grad()
            loss_tot.backward()
            opt.step()
            
            epoch_loss_fwd += loss_fwd.item()
            epoch_loss_rev += loss_rev.item()
            epoch_loss_tot += loss_tot.item()
        
        losses_fwd_train.append(epoch_loss_fwd / (len(train_loader.sampler)))
        losses_rev_train.append(epoch_loss_rev / (len(train_loader.sampler)))
        losses_tot_train.append(epoch_loss_tot / (len(train_loader.sampler)))
        
        # validation 
        model.eval()  
        with torch.no_grad():
            for inputs_val, labels_val in val_loader:
            
                out_val = model(inputs_val)

                out_fwd_val = out_val[:, 0:1]
                out_rev_val = out_val[:, 1:2]
                
                labels_fwd_val = labels_val[:, 1, :]
                labels_rev_val = labels_val[:, 0, :]

                val_loss_fwd = loss_fn(out_fwd_val, labels_fwd_val)
                val_loss_rev = loss_fn(out_rev_val, labels_rev_val)
                val_loss_tot = val_loss_rev + val_loss_fwd

                tot_val_fwd += val_loss_fwd.item()
                tot_val_rev += val_loss_rev.item()
                tot_val_tot += val_loss_tot.item()

        val_loss = tot_val_tot / len(val_loader.sampler)
        
        losses_fwd_val.append(tot_val_fwd / (len(val_loader.sampler)))
        losses_rev_val.append(tot_val_rev / (len(val_loader.sampler)))
        # losses_tot_val.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    result = best_model_state_dict, best_epoch, losses_fwd_train, losses_fwd_val, losses_rev_train, losses_rev_val
   
    return result


def make_preds_e2b1(model, test_loader):
    preds_fwd = []
    true_fwd = []

    preds_rev = []
    true_rev = []

    # eval mode
    model.eval()
    # make preds
    with torch.no_grad():
        for inputs, labels in test_loader:
            
            out = model(inputs)

            out_fwd = out[:, 0:1]
            out_rev = out[:, 1:2]
                
            labels_fwd = labels[:, 1, :]
            labels_rev = labels[:, 0, :]

            preds_fwd.append(out_fwd.cpu().numpy())
            true_fwd.append(labels_fwd.cpu().numpy())

            preds_rev.append(out_rev.cpu().numpy())
            true_rev.append(labels_rev.cpu().numpy())

    preds_fwd = np.concatenate(preds_fwd).flatten()
    true_fwd = np.concatenate(true_fwd).flatten()

    preds_rev = np.concatenate(preds_rev).flatten()
    true_rev = np.concatenate(true_rev).flatten()

    return preds_fwd, true_fwd, preds_rev, true_rev

######### E2b2

def train_model_e2b2(model, train_loader, val_loader, opt, loss_fn, epochs=50):

    losses_fwd_train = []
    losses_rev_train = []
    losses_tot_train = []

    losses_fwd_val = []
    losses_rev_val = []
    losses_tot_val = []

    best_model_state_dict = None

    best_epoch = -1
    best_val_loss = float('inf')

    for epoch in range(epochs):
        epoch_loss_fwd = 0.
        epoch_loss_rev = 0.
        epoch_loss_tot = 0.

        epoch_val_fwd = 0.
        epoch_val_rev = 0.
        epoch_val_tot = 0.
        
        model.train()

        for inputs, labels in train_loader:

            out_fwd, out_rev = model(inputs)
            
            labels_fwd = labels[:, 1, :]
            labels_rev = labels[:, 0, :]
            
            loss_fwd = loss_fn(out_fwd, labels_fwd)
            loss_rev = loss_fn(out_rev, labels_rev)
            loss_tot = loss_fwd + loss_rev

            opt.zero_grad()
            loss_tot.backward()
            opt.step()
            
            epoch_loss_fwd += loss_fwd.item()
            epoch_loss_rev += loss_rev.item()
            epoch_loss_tot += loss_tot.item()
        
        losses_fwd_train.append(epoch_loss_fwd / (len(train_loader.sampler)))
        losses_rev_train.append(epoch_loss_rev / (len(train_loader.sampler)))
        losses_tot_train.append(epoch_loss_tot / (len(train_loader.sampler)))

        # validation 
        model.eval()  
        with torch.no_grad():
            for inputs_val, labels_val in val_loader:
            
                out_val = model(inputs_val)

                out_fwd_val, out_rev_val = model(inputs_val)
            
                labels_fwd_val = labels_val[:, 1, :]
                labels_rev_val = labels_val[:, 0, :]

                val_loss_fwd = loss_fn(out_fwd_val, labels_fwd_val)
                val_loss_rev = loss_fn(out_rev_val, labels_rev_val)
                val_loss_tot = val_loss_rev + val_loss_fwd

                epoch_val_fwd += val_loss_fwd.item()
                epoch_val_rev += val_loss_rev.item()
                epoch_val_tot += val_loss_tot.item()

        val_loss = epoch_val_tot / len(val_loader.sampler)
        
        losses_fwd_val.append(epoch_val_fwd / (len(val_loader.sampler)))
        losses_rev_val.append(epoch_val_rev / (len(val_loader.sampler)))
        losses_tot_val.append(val_loss)
        

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    result = best_model_state_dict, best_epoch, losses_fwd_train, losses_fwd_val, losses_rev_train, losses_rev_val
   
    return result

def make_preds_e2b2(model, test_loader):
    preds_fwd = []
    true_fwd = []

    preds_rev = []
    true_rev = []

    # eval mode
    model.eval()
    # make preds
    with torch.no_grad():
        for inputs, labels in test_loader:

            out_fwd, out_rev = model(inputs)
            
            labels_fwd = labels[:, 1, :]
            labels_rev = labels[:, 0, :]

            preds_fwd.append(out_fwd.cpu().numpy())
            true_fwd.append(labels_fwd.cpu().numpy())

            preds_rev.append(out_rev.cpu().numpy())
            true_rev.append(labels_rev.cpu().numpy())

    preds_fwd = np.concatenate(preds_fwd).flatten()
    true_fwd = np.concatenate(true_fwd).flatten()

    preds_rev = np.concatenate(preds_rev).flatten()
    true_rev = np.concatenate(true_rev).flatten()

    return preds_fwd, true_fwd, preds_rev, true_rev