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
import argparse
from collections import namedtuple
import json


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--epoch",type = int, default = 30, help="epoch of supervised learning for base_model")
        parser.add_argument("--bt_size", type=int, default=64, help ="batch size of supervised learning for base_model")
        parser.add_argument("--lr", type=float, default=0.01, help =" learning rate of SGD (supervised)")
        parser.add_argument("--split_ratio", type=float, default=0.8, help =" Split ratio for train valid datasets")
        parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD (supervised)")
        parser.add_argument("--lb_group", type=float, default=9*10**-6, help="parameter of lasso")
        parser.add_argument("--lb_l1", type=float, default=5*10**-5, help="parameter of lasso")
        parser.add_argument("--lasso_flag",type=int, default=1, help="0: no regularization, 1: use group lasso, 2: use L1 norm 3: use L1 norm and Group Lasso 4: linear 5: div")
        parser.add_argument("--save_pth", type =str, default="result", help="directry to save result")
        self.parser = parser 
    
    def get_args(self):
        return self.parser.parse_args()

def split_dataset(data_set, split_ratio, order=None):
    n_examples = len(data_set)
    split_at = int(n_examples * split_ratio)
    if split_at < 0:
        raise ValueError('split_at must be non-negative')
    if split_at > n_examples:
        raise ValueError('split_at exceeds the dataset size')

    if order is not None:
        subset1_indices = order[0:split_at]
        subset2_indices = order[split_at:n_examples]
    else:
        subset1_indices = list(range(0,split_at))
        subset2_indices = list(range(split_at,n_examples))

    subset1 = Subset(data_set, subset1_indices)
    subset2 = Subset(data_set, subset2_indices)

    return subset1, subset2

class GroupLasso():
    def __init__(self,model,lb):
        self.lasso = []
        self.linear = []
        self.devide = 4 
        self.div = []
        self.lb = lb 
        for m in model.modules():
            if isinstance(m,nn.Conv2d):
                tmp = torch.zeros((m.weight.shape[0]),device="cuda")
                linear = torch.zeros((m.weight.shape[0]),device="cuda")
                div = torch.zeros((m.weight.shape[0]),device="cuda")
                weight = m.weight
                quo = weight.size(1)/4 

                
                for j in range(weight.shape[0]):
                   
                    tmp[j] = (lb*torch.tensor(weight[j].numel(),dtype=torch.float).sqrt())
                    linear[j] = lb *(j+1)
                    div[j] = lb*(weight.size(1)/quo)
                self.lasso.append(tmp)
                self.linear.append(linear) 
                self.div.append(div)

            
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
        return lb*sum(p.view(-1).abs().sum() for p in model.parameters())

    def get_filter_channel(self,model):
        lasso = 0
        ch = 0
        
        for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nr = torch.linalg.norm(m.weight,dim=(1,2,3))
                    tmp = torch.linalg.norm(m.weight,dim=(0,2,3))
                    ch += torch.mul(self.lb,nr).sum()
                    lasso += torch.mul(self.lb,nr).sum()
                elif isinstance(m,nn.Linear):
                    tmp = m.weight.view(-1) 
                    lmb = self.lb * torch.tensor(tmp.numel(),dtype=torch.float).sqrt()
                    lasso += lmb* torch.sqrt(tmp.square().sum())
        
        return lasso + ch
    
    def get_linear_lasso(self,model):
        lasso = 0
        i = 0
        for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nr = torch.linalg.norm(m.weight,dim=(1,2,3))
                
                    lasso += torch.mul(self.linear[i],nr).sum()
                    i += 1
                elif isinstance(m,nn.Linear):
                    tmp = m.weight.view(-1) 
                    lmb = self.lb * torch.tensor(tmp.numel(),dtype=torch.float).sqrt()
                    lasso += lmb* torch.sqrt(tmp.square().sum())
        
        return lasso
    
    def get_div_lasso(self,model):
        lasso = 0
        i = 0
        for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nr = torch.linalg.norm(m.weight,dim=(1,2,3))
                
                    lasso += torch.mul(self.div[i],nr).sum()
                    i += 1
                elif isinstance(m,nn.Linear):
                    tmp = m.weight.view(-1) 
                    lmb = self.lb * torch.tensor(tmp.numel(),dtype=torch.float).sqrt()
                    lasso += lmb* torch.sqrt(tmp.square().sum())
        
        return lasso


    

def measure_L2_norm(model,save_pth,epoch):
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
    fname = f"{save_pth}/L2_norm_result_epoch_{epoch}.mat"

    scipy.io.savemat(fname,{f"norm":results})
    #scipy.io.savemat(fname,{f"norm{model_num}":ls_result})


def main():
    opt = Options()
    args = opt.get_args()
    
    os.makedirs(args.save_pth,exist_ok=True)
    with open(f"{args.save_pth}/vgg16parameter.json",mode="w") as f:
        json.dump(args.__dict__, f, indent=4)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    


    ############# prepare dataset ##############################
    transform = transforms.Compose([
        #transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )
    ])
    dataset = datasets.CIFAR10(
    root= './data', train = True,
    download =True, transform = transform)

    train_dataset, val_dataset = split_dataset(dataset,args.split_ratio)

    val_loader = torch.utils.data.DataLoader(val_dataset
        , batch_size = args.bt_size 
        , shuffle = True)


    train_loader = torch.utils.data.DataLoader(train_dataset
        , batch_size = args.bt_size 
        , shuffle = True)
    test_dataset = datasets.CIFAR10(
    root= './data', train = False,
    download =True, transform = transform)
    test_loader = torch.utils.data.DataLoader(test_dataset
        , batch_size = args.bt_size 
        , shuffle = True)

    vgg16 = models.vgg16(pretrained=False)
    vgg16 = vgg16.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=vgg16.parameters(), lr=args.lr, momentum=args.momentum)
    lss = GroupLasso(vgg16,args.lb_group)

    n_total_step = len(train_loader)
    average_time = 0
    acc_tp5 = 0
    train_log = []
    val_log = []
    tp5_log = []
    for epoch in range(args.epoch):
        running_loss = 0
        start_time = time.time()
        acc_tp5 = 0
        
        acc_train  = 0
        acc_val = 0 
        for i , (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            labels_hat = vgg16(imgs)
            #labels_hat = model(imgs)
            n_corrects = (labels_hat.argmax(axis=1)==labels).sum().item()
            acc_train += 100*(n_corrects/labels.size(0))
            loss_value = criterion(labels_hat,labels)
            _,tp_5_pred = labels_hat.topk(5,1,True,True)
            tp_5_pred = tp_5_pred.t()
            tp5_correct = tp_5_pred.eq(labels.view(1,-1).expand_as(tp_5_pred))
            acc_tp5 += (tp5_correct.float().sum().sum())*(100/args.bt_size)

            if args.lasso_flag == 1:
                loss_value += lss.get_group_lasso(vgg16)
            elif args.lasso_flag == 2:
                loss_value += lss.get_l1_norm(vgg16,args.lb_l1)
            elif args.lasso_flag == 3:
                loss_value += lss.get_group_lasso(vgg16) + lss.get_l1_norm(vgg16,args.lb_l1)
            elif args.lasso_flag == 4:
                loss_value += lss.get_filter_channel(vgg16)
            elif args.lasso_flag == 5:
                loss_value += lss.get_linear_lasso(vgg16)
            elif args.lasso_flag == 6:
                loss_value += lss.get_div_lasso(vgg16)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss_value.item()
            if (i+1) % 250 == 0:
                print(f"epoch {epoch+1}/{args.epoch}, step: {i+1}/{n_total_step}: loss = {loss_value:.5f},acc = {100*(n_corrects/labels.size(0)):.2f}%")
                print(f"top 5 acc : {(tp5_correct.float().sum().sum())*(100/args.bt_size):.2f}")
                print()
        
        ######################### end of epoch ################################
        end_time = time.time() 
        ver_time = end_time - start_time 
        print(f"times per epoch : {ver_time:.4f} (sec)")
        print(f"Top 5 accuracy : {acc_tp5/n_total_step:.2f} %")
        average_time += ver_time 

        if epoch %2 == 1 :
            measure_L2_norm(vgg16,args.save_pth,epoch)

        
        with torch.no_grad():
            number_corrects = 0
            number_samples = 0
            for i, (imgs,labels) in enumerate(val_loader):
                imgs = imgs.to(device)
                labels = labels.to(device)

                labels_hat = vgg16(imgs)
                #labels_hat = model(imgs)
                acc_val += 100*(n_corrects/labels.size(0))
                loss_value = criterion(labels_hat,labels)
                number_corrects += (labels_hat.argmax(axis=1)==labels).sum().item()
                number_samples += labels.size(0)
            
            print(f"Val accuracy {(number_corrects / number_samples)*100}%")
        
        train_log.append(acc_train/n_total_step)
        tp5_log.append(acc_tp5/n_total_step)
        val_log.append(acc_val/n_total_step)
    
    loginfo = {}
    loginfo["train"] = train_log 
    loginfo["tp5"] = tp5_log 
    loginfo["val"] = val_log 

    fname = f"{args.save_pth}/training_log.mat"
    scipy.io.savemat(fname,loginfo)
        
        

                


    average_time = average_time / (epoch +1)
    print("**************************************")
    print(f"Average training time : {average_time:.4f} (sec)")
    print("**************************************")
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
        
        print(f"test accuracy {(number_corrects / number_samples)*100}%")
    model_save_path = f"{args.save_pth}/vgg16.model"
    torch.save(vgg16.state_dict(), model_save_path)
    
        


if __name__ == '__main__':
    main()
