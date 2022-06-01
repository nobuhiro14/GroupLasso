import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import time




def main():
    lr = 0.001
    momentum = 0.9
    num_epoch = 40
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
    optimizer = torch.optim.SGD(params=vgg16.parameters, lr=lr, momentum=momentum)

    n_total_step = len(train_loader)

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

            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss_value.item()
            if (i+1) % 250 == 0:
                print(f"epoch {epoch+1}/{num_epoch}, step: {i+1}/{n_total_step}: loss = {loss_value:.5f},acc = {100*(n_corrects/labels.size(0)):.2f}%")
                print()


if __name__ == '__main__':
    main()
