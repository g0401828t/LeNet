import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import optim


import os
from datetime import datetime
from torchsummary import summary

from model import LeNet_5
from plotgraph import plotgraph

# 사용 가능한 gpu 확인 및 설정
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)


# parameters
lr = 0.01
batch_size = 512
epochs = 10
path = "D:/projects"
datapath = path + '/dataset'
resultpath = path + "/lenet/results"
modelpath = path + "/lenet/models/lenet_best_model.h"

# MNIST dataset 불러오기
data_transforms = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])


# traindata, testdata 불러오기
train_data = datasets.MNIST(datapath, train=True, download=True, transform=data_transforms)
train_data, val_data = torch.utils.data.random_split(train_data, [50000, 10000])
test_data = datasets.MNIST(datapath, train=True, download=True, transform=data_transforms)
print("train_data: {}, val_data: {}, test_data: {}".format(
    len(train_data), len(val_data), len(test_data)
))

# data loader 생성
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# model 불러오기
model = LeNet_5()
# model을 cuda로 전달
model.to(device)
print("model set to",next(model.parameters()).device)
# 모델 summary를 확인합니다.
summary(model, input_size=(1, 32, 32))


# loss, optimizer 정의하기
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = lr)


# 학습 함수
def train():
    loss_list, valloss_list, valacc_list = [], [], []
    for epoch in range(epochs):
        model.train()
        avg_loss, val_loss = 0, 0
        best_acc = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            # 순전파
            hypothesis = model(x)
            loss = criterion(hypothesis, y)

            # 역전파
            loss.backward()
            optimizer.step()

            avg_loss += loss
        avg_loss = avg_loss/len(train_loader)

        # validation
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)

                prediction = model(x)
                # for loss
                val_loss += criterion(prediction, y) / len(val_loader)
                # for acc
                correct = prediction.max(1)[1] == y
                val_acc = correct.float().mean()



                # Early Stop
                if val_acc > best_acc:
                    best_acc = val_acc
                    es = 5
                    torch.save(model.state_dict(), modelpath)
                else: 
                    es -= 1
                if es == 0: 
                    model.load_state_dict(torch.load(modelpath))
                    break

            loss_list.append(avg_loss.item())
            valloss_list.append(val_loss.item())
            valacc_list.append(val_acc.item())
        print(datetime.now().time().replace(microsecond=0), "EPOCHS: [{}/{}], avg_loss: [{:.4f}], val_acc: [{:.2f}%]".format(
                epoch+1, epochs, avg_loss.item(), val_acc.item()*100))
        plotgraph(loss_list=loss_list, valloss_list=valloss_list, valacc_list=valacc_list, path = resultpath)



# test 함수
def test():
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    avg_loss = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            # 순전파
            prediction = model(x)
            correct = prediction.max(1)[1] == y
            test_acc = correct.float().mean()
    print("Acc: [{:.2f}%]".format(
        test_acc*100
    ))



# train
train()
test()
