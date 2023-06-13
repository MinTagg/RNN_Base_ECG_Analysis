from tqdm import tqdm
import torch

def get_score(model, test_loader):
    loader = tqdm(test_loader, total = len(test_loader))
    model.eval()
    criterion = torch.nn.L1Loss(reduction = 'sum')

    losses = [] #  Truth, False
    labels = []

    for test_input, label in loader:
        with torch.no_grad():
            prediction = model(test_input)
            for index in range(test_input.shape[0]):
                loss = criterion(prediction[index], test_input[index]).detach().cpu().numpy()
                if label[index] == 0: # Truth input
                    losses.append(loss.item())
                    labels.append(0)
                elif label[index] == 1: # False input
                    losses.append(loss.item())
                    labels.append(1)
    
    return losses, labels