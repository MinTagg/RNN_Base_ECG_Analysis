# RNN Model Base ECG Data Analyse

- This project analyse ECG data with RNN model.
- The data is from [ECG5000](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000).

## Purpose

![](https://github.com/MinTagg/RNN_Base_ECG_Analysis/assets/98318559/553e72fe-d389-4c68-b888-7f9ddfcc41c4)
- Detecting Normal and others using RNN base model.

## Model

![LSTM](https://github.com/MinTagg/RNN_Base_ECG_Analysis/assets/98318559/b49f3277-bba8-4ef2-88ed-5675a6a89f8e)
- LSTM Model

![Transformer](https://github.com/MinTagg/RNN_Base_ECG_Analysis/assets/98318559/fde46e77-4642-400d-bf22-2dd8636dc6ae)
- Transformer Model

## Requirements

- pandas, numpy, pytorch(1.13.0), matplotlib, scipy, sklearn, tqdm

## How to use

- Download ECG5000 dataset from [here](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000).
- Make 'ECG5000' and save the dataset in 'ECG5000' folder.
- Run 'MAIN.py' file.
    - You can change model parameters at the bottom of the file.

## Results

- Train : Validation = 0.75 : 0.25
- Batch size : 32
- Optimzier : Adam
- Loss function : L1Loss
- Epoch : 150

### LSTM
|     Embedding   Feature Channels    |     Parameters    |     AUC   score    |
|-------------------------------------|-------------------|--------------------|
|     32                              |     63,297        |     0.995          |
|     64                              |     249,473       |     0.996          |
|     128                             |     990,465       |     0.989          |
|     256                             |     3,947,009     |     0.992          |

### Transformer

|     Parameters     [Input   Dim, Hiddden Dim, Head, Layers]     |     Parameters    |     AUC   score    |
|-----------------------------------------------------------------|-------------------|--------------------|
|     [2,   256, 2, 2]                                            |     5,315         |     0.267          |
|     [4,   256, 2, 2]                                            |     9,797         |     0.946          |
|     [4,   256, 4, 2]                                            |     9,797         |     0.950          |
|     [2,   512, 2, 2]                                            |     10,435        |     0.261          |
|     [4, 512, 2, 2]                                              |     19,013        |     0.963          |
|     [4, 512,   4, 2]                                            |     19,013        |     0.591          |

## Analysis

- LSTM model is better than Transformer model.
- This task is based on reconstruction.
- Transfomer model reconstructed too much, so there were no difference between normal and others.

![sample](https://github.com/MinTagg/RNN_Base_ECG_Analysis/assets/98318559/dbdda185-17c3-4386-81a7-f37af81321f8)
- LSTM Model Reconstruction Sample

![sample](https://github.com/MinTagg/RNN_Base_ECG_Analysis/assets/98318559/f5b9fed0-ac10-471a-a483-f4cb1bdfc93e)
- Transformer Model Reconstruction Sample