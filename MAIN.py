import time
import torch
from DATASET import *
from LSTM_MODEL import MODEL as LSTM_MODEL
from TRANSFORMER_MODEL import TransformerAutoencoder as TRANSFORMER_MODEL
from TRAIN import train_function
from CONFIG import EPOCH
import os
import matplotlib.pyplot as plt
from PREDICT import predict
from sklearn.metrics import roc_curve, auc
from FOR_ROC import get_score
from SAMPLE import get_sample

def start(model_name, model_parameters, device, train_dataset, val_dataset, test_dataset, abnormal_dataset):

    if model_name == 'lstm':
        model = LSTM_MODEL(model_parameters[0])
    elif model_name == 'transformer':
        model = TRANSFORMER_MODEL(model_parameters[1], model_parameters[2], model_parameters[3], model_parameters[4])

    model, history = train_function(model, train_dataset, val_dataset, EPOCH, device)

    predict_normal_losses = predict(model, train_dataset)
    predict_abnormal_losses = predict(model, abnormal_dataset)

    # get score
    roc_losses, roc_labels = get_score(model, test_dataset)

    fpr, tpr, _ = roc_curve(roc_labels, roc_losses)
    roc_auc = auc(fpr, tpr)

    # get sample
    normal, normal_pred, abnormal, abnormal_pred = get_sample(model, val_dataset, abnormal_dataset)

    # number of models parameter
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # make variable name 'finish' that contain now time in format {year}_{month}_{day}_{hour}_{minute}_{second}
    finish = time.strftime('%Y_%m_%d_%H_%M_%S')

    # make directory in RESULT folder that named with finish time. and make text file that named with model_info.txt. It contain model name and model parameters
    
    os.mkdir(f'RESULT/{finish}')
    with open(f'RESULT/{finish}/model_info.txt', 'w') as f:
        f.write(f'{model_name}\n')
        f.write(f'lstm_embedding_feature, transformer_input_dim, transformer_hidden_dim, transformer_num_heads, transformer_num_layers\n')
        f.write(f'{model_parameters}\n')
        f.write(f'num_parameters : {num_parameters}\n')
    
    # save model
    torch.save(model.state_dict(), f'RESULT/{finish}/model.pt')

    # plot history using plt. history[0] is Train loss, history[1] is Validation loss, x label is epoch, y label is loss
    plt.plot(history[0], label='Train loss')
    plt.plot(history[1], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'RESULT/{finish}/loss.png')
    plt.close()

    # plot two predict_normal_losses and predict_abnormal_losses using plt's histogram, using density. x label is loss, y label is count
    plt.hist(predict_normal_losses, label='Normal', density=True, bins = 100)
    plt.hist(predict_abnormal_losses, label='Abnormal', density=True, bins = 100)
    plt.xlabel('Loss')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(f'RESULT/{finish}/histogram.png')
    plt.close()

    # plot roc curve using plt.
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--') # diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(f'RESULT/{finish}/roc.png')
    plt.close()

    # plot normal, normal_pred, abnormal, abnormal_pred using sub plot. x label is time, y label is value
    plt.subplot(2, 1, 1)
    plt.plot(list(normal[0]), label='Normal')
    plt.plot(list(normal_pred[0]), label='Normal Prediction')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(list(abnormal[0]), label='Abnormal')
    plt.plot(list(abnormal_pred[0]), label='Abnormal Prediction')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()

    plt.savefig(f'RESULT/{finish}/sample.png')
    plt.close()

if __name__ == '__main__':
    

    model_name = [
        'lstm',
        'lstm',
        'lstm',
        'lstm',

        'transformer',
        'transformer',
        'transformer',

        'transformer',
        'transformer',
        'transformer',
    ]

    # lstm_embedding_feature, transformer_input_dim, transformer_hidden_dim, transformer_num_heads, transformer_num_layers
    model_parameters = [
        [32, None, None, None, None],
        [64, None, None, None, None],
        [128, None, None, None, None],
        [256, None, None, None, None],

        [None, 2, 256, 2, 2],
        [None, 4, 256, 2, 2],
        [None, 4, 256, 4, 2],
        
        [None, 2, 512, 2, 2],
        [None, 4, 512, 2, 2],
        [None, 4, 512, 4, 2],
    ]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if len(model_name) != len(model_parameters):
        raise ValueError('Model name and model parameters must have same length')
    
    train, validation, test, abnormal = load_dataloader(device)
    
    for i in range(len(model_name)):
        print(f'Start {model_name[i]}')
        start(model_name[i], model_parameters[i], device, train, validation, test, abnormal)