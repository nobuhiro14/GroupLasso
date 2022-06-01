from email.headerregistry import Group
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import linalg as LA

import time
import os 
import scipy.io
import numpy as np 
from collections import namedtuple

class GroupLasso():
    def __init__(self,model,lb):
        self.lasso = []
        self.lb = lb 
        for m in model.modules():
            if isinstance(m,nn.Conv2d):
                tmp = torch.zeros((m.weight.shape[0]),device="cuda")
                weight = m.weight
                for j in range(weight.shape[0]):
                    wi = weight[j]
                    tmp[j] = (lb*torch.tensor(weight[j].numel(),dtype=torch.float).sqrt())
                self.lasso.append(tmp)
            
    def get_group_lasso(self,model):
        lasso = 0
        i = 0
        for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nr = torch.linalg.norm(m.weight,dim=(1,2,3))
                
                    lasso += torch.mul(self.lasso[i],nr).sum()
                    i += 1
                elif isinstance(m,nn.Linear):
                    tmp = m.weight.view(-1) 
                    lmb = self.lb * torch.tensor(tmp.numel(),dtype=torch.float).sqrt()
                    lasso += lmb* torch.sqrt(tmp.square().sum())
        
        return lasso
        
    def get_l1_norm(self,model,lb):
        return lb*sum(p.view(-1).abs() for p in model.parameters())



    

def measure_L2_norm(model,save_pth):
    results = {}
    #ls_result = []
    num = 1
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            tmp = LA.norm(m.weight, dim=(1,2,3))
            name = f"conv_No{num:02}"
            results[name] = tmp.cpu().detach().numpy()
            #ls_result.append(tmp)
            num +=1
    fname = f"{save_pth}/L2_norm_result.mat"

    scipy.io.savemat(fname,{f"norm{model_num}":results})
    #scipy.io.savemat(fname,{f"norm{model_num}":ls_result})


def main():
    lr = 0.001
    momentum = 0.9
    num_epoch = 30
    lb = 5*10**-5
    save_pth = "result"
    os.makedirs(save_pth,exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ############# prepare dataset ##############################
    transform = transforms.Compose([
        #transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )
    ])
    train_dataset = datasets.CIFAR10(
    root= './data', train = True,
    download =True, transform = transform)
    train_loader = torch.utils.data.DataLoader(train_dataset
        , batch_size = 64
        , shuffle = True)
    test_dataset = datasets.CIFAR10(
    root= './data', train = False,
    download =True, transform = transform)
    test_loader = torch.utils.data.DataLoader(test_dataset
        , batch_size = 64
        , shuffle = True)

    vgg16 = models.vgg16(pretrained=False)
    vgg16 = vgg16.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=vgg16.parameters(), lr=lr, momentum=momentum)
    lss = GroupLasso(vgg16,lb)

    n_total_step = len(train_loader)
    average_time = 0
    for epoch in range(num_epoch):
        running_loss = 0
        start_time = time.time()
        ls_min = float('inf')
        ls_max = 0
        acc_train  = 0
        for i , (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            labels_hat = vgg16(imgs)
            #labels_hat = model(imgs)
            n_corrects = (labels_hat.argmax(axis=1)==labels).sum().item()
            acc_train += 100*(n_corrects/labels.size(0))
            loss_value = criterion(labels_hat,labels)
            loss_value += lss.get_group_lasso(vgg16)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss_value.item()
            if (i+1) % 250 == 0:
                print(f"epoch {epoch+1}/{num_epoch}, step: {i+1}/{n_total_step}: loss = {loss_value:.5f},acc = {100*(n_corrects/labels.size(0)):.2f}%")
                print()
        end_time = time.time() 
        ver_time = end_time - start_time 
        print(f"times per epoch : {ver_time:.4f} (sec)")
        average_time += ver_time 

        if epoch %2 == 0 :
            with torch.no_grad():
                number_corrects = 0
                number_samples = 0
                for i, (imgs,labels) in enumerate(test_loader):
                    imgs = imgs.to(device)
                    labels = labels.to(device)

                    labels_hat = vgg16(imgs)
                    #labels_hat = model(imgs)
                    acc_train += 100*(n_corrects/labels.size(0))
                    loss_value = criterion(labels_hat,labels)
                    number_corrects += (labels_hat.argmax(axis=1)==labels).sum().item()
                    number_samples += labels.size(0)
                
                print(f"Overall accuracy {(number_corrects / number_samples)*100}%")


    average_time = average_time / (epoch +1)
    print("**************************************")
    print(f"Average training time : {average_time:.4f} (sec)")
    print("**************************************")
    model_save_path = f"{save_pth}/vgg16.model"
    torch.save(vgg16.state_dict(), model_save_path)
    measure_L2_norm(vgg16,save_pth)
        


if __name__ == '__main__':
    main()
