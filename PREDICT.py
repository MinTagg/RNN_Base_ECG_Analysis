from tqdm import tqdm
import torch

def predict(model, data_loader):
  """
  데이터 로더를 입력하였을 때, 데이터별로 loss값을 측정하여 list 형식으로 loss값들을 출력
  """
  print('Start Prediction')
  loader = tqdm(data_loader, total = len(data_loader))
  losses = []

  criterion = torch.nn.L1Loss(reduction = 'sum')

  for input_data in loader:
    prediction = model(input_data)
    for index in range(input_data.shape[0]):
      losses.append(criterion(prediction[index], input_data[index]).detach().cpu().numpy())
  return losses