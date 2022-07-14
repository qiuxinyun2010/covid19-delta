import os
from scipy.io import loadmat
import torchio as tio
import torch
import numpy as np
import scipy.ndimage
import scipy.io as sio
from sklearn.model_selection import train_test_split
import torch.utils.data as data
from sklearn import metrics
from resnet import generate_model
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR,LambdaLR
from utils.image_process import clip_and_normalize_mean_std, normalize_min_max_and_clip
import torch.nn as nn
from tqdm import tqdm
from models import resnet,recv
import matplotlib.pylab as plt

join = os.path.join
dir = os.listdir

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True

class MyDataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x
       # self.y = y
        self.y = torch.LongTensor(y)

    def __getitem__(self, index):
        xi = self.x[index]
        xi = clip_and_normalize_mean_std(xi,-1024,1024)[None, ...]
        xi = torch.FloatTensor(xi)
        yi = self.y[index]

        return xi, yi

    def __len__(self):
        return len(self.y)

def train(model,train_loader,val_loader,optimizer,scheduler,criterion,epochs, save_dir = '/media/data/zhiqiang/save_weights/'):
    max_acc=0
    for epoch in range(1, epochs + 1):
        print("epoch:"+str(epoch))
        num_iters = 0
        gts = []
        predicts = []
        model.train()
        epoch_loss=0
        for i, batch in tqdm(enumerate(train_loader)):

            optimizer.zero_grad()
            x = batch[0]
            y = batch[1]

            x = x.type(torch.FloatTensor).to(device)
            y = y.type(torch.LongTensor).to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            outputs_logit = outputs.argmax(dim=1)

            num_iters += 1
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
            predicts.append(outputs_logit.cpu().detach().numpy())
            gts.append(y.cpu().detach().numpy())
            
        predicts = np.concatenate(predicts).flatten().astype(np.int16)
        gts = np.concatenate(gts).flatten().astype(np.int16)  
        print('train_loss:',(epoch_loss)/(i+1))

        acc = metrics.accuracy_score(gts, predicts) 
        f1_score = metrics.f1_score(gts, predicts,average='macro') 
        print('train_acc:',acc,'train_f1:',f1_score)

       # scheduler.step()
        # 验证
        test_acc = test(model,val_loader,criterion)

        # 保存模型
        if test_acc > max_acc:
            torch.save(model.state_dict(), save_dir + 'best_model.pth')
            max_acc = test_acc


def test(model,val_loader,criterion):
    num_iters = 0
    acc = 0
    gts = []
    predicts = []
    model.eval()
    epoch_loss=0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader)):
            x = batch[0]
            y = batch[1]
            #y = torch.zeros(1, 2).scatter_(1,  y[...,None], 1)
            x = x.type(torch.FloatTensor).to(device)
            y = y.type(torch.LongTensor).to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            outputs_logit = outputs.argmax(dim=1)

            num_iters += 1
            epoch_loss+=loss.item()
            predicts.append(outputs_logit.cpu().detach().numpy())
            gts.append(y.cpu().detach().numpy())

        predicts = np.concatenate(predicts).flatten().astype(np.int16)
        gts = np.concatenate(gts).flatten().astype(np.int16)  
        acc = metrics.accuracy_score(gts, predicts)      
        f1_score = metrics.f1_score(gts, predicts,average='macro') 
        print('val_loss:',(epoch_loss)/(i+1),'val_acc:',acc,'val_f1_scores:',f1_score)

    return acc
if __name__ == '__main__':

    # 设置随机数种子
    setup_seed(507)
    model = resnet.generate_model(18,n_input_channels=1,n_classes=3)

    # 加载数据
    train_X = np.load("/media/data/zhiqiang/lung/train_x.npy")
    train_Y = np.load("/media/data/zhiqiang/lung/train_y.npy")

    # 划分训练集、验证集
    train_x, val_x, train_y, val_y = train_test_split(train_X, train_Y, test_size=0.2, random_state=0, stratify=train_Y)

    # 初始化数据集
    train_dataset = MyDataset(train_x,train_y)
    val_dataset = MyDataset(val_x,val_y)
    #val_dataset = MyDataset(test_x,test_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    ## 初始化训练参数
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = CosineAnnealingLR(optimizer, 4, eta_min=0, last_epoch=-1)
    #scheduler = StepLR(optimizer, step_size=12, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    #模型
    train(model,train_dataloader,val_dataloader,optimizer,scheduler,criterion,30) #vgg