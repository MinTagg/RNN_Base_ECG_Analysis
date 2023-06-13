import torch
from torch import nn
import numpy as np
from tqdm import tqdm

def train_function(model, train_dataset, val_dataset, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction = 'sum').to(device)
    model.to(device)

    history = [[], []]

    for epoch in range(1, epochs+1):
        model = model.train()

        total_loss = []
        loader = tqdm(train_dataset, total = len(train_dataset))
        for input_data in loader:
            optimizer.zero_grad()

            result = model(input_data)
            loss = criterion(result, input_data)

            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())
        
        total_loss = np.mean(total_loss)
        history[0].append(total_loss)
        
        val_loss = []
        loader = tqdm(val_dataset, total = len(val_dataset))
        model.eval()
        with torch.no_grad():
            for input_data in loader:
                result = model(input_data)
                loss = criterion(result, input_data)
                val_loss.append(loss.item())
        
        val_loss = np.mean(val_loss)
        history[1].append(val_loss)

        print(f'Epoch :: {epoch} Train loss :: {total_loss} Validation loss :: {val_loss}')
        print('')
    
    return model, history