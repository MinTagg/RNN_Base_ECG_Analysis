
def get_sample(model, val_loader, abnormal_loader):
    normal_data = []
    normal_prediction = []
    for index, instance in enumerate(val_loader):
        prediction = model(instance)
        normal_data.append(instance[0].detach().cpu().numpy())
        normal_prediction.append(prediction[0].detach().cpu().numpy())
        if index == 1:
            break
    
    abnormal_data = []
    abnormal_prediction = []

    for index, instance in enumerate(abnormal_loader):
        prediction = model(instance)
        abnormal_data.append(instance[0].detach().cpu().numpy())
        abnormal_prediction.append(prediction[0].detach().cpu().numpy())
        if index == 1:
            break

    return normal_data, normal_prediction, abnormal_data, abnormal_prediction