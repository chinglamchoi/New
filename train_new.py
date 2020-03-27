import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms as transforms
import numpy as np
import os
import unet
from sklearn.metrics import adjusted_rand_score

from skimage import io
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import random


class Covid(data.Dataset):
    def __init__(self, imgs, masks, transform=transforms.ToTensor(), img_transform=transforms.Normalize([0.5], [0.5])):
        self.imgs, self.masks = imgs, masks
        self.transform, self.img_transform = transform, img_transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img1, mask1 = self.transform(self.imgs[index]).float(), self.transform(self.masks[index]).float()
        img1 = self.img_transform(img1)
        return (img1, mask1)

def lr_change(alpha):
    global optimizer
    alpha /= 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = alpha
    print("Lr changed to " + str(alpha))
    return alpha

def DiceLoss(a,b):
    smooth = 1.
    a = a.view(-1)
    b = b.view(-1)
    intersection = (a*b).sum()
    return 1 - ((2. * intersection + smooth) / (a.sum() + b.sum() + smooth))

def RandLoss(a,b):
    a = (a>0.5).float()
    a = a.cpu().numpy().flatten()
    b = b.cpu().numpy().flatten()
    c = adjusted_rand_score(a,b)
    c = (c+1)/2
    return 1-c

#torch.multiprocessing.freeze_support()
if __name__ == "__main__":
    mean, std = [0.5], [0.5]
    train_imgs, train_masks, test_imgs, test_masks = np.load("train_imgs.npy"), np.load("train_masks.npy"), np.load("test_imgs.npy"), np.load("test_masks.npy")

    trainset = Covid(imgs=train_imgs, masks=train_masks)
    testset = Covid(imgs=test_imgs, masks=test_masks)
    trainloader = data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=12)
    testloader = data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=12)

    parser = argparse.ArgumentParser()
    parser.add_argument("-pre", metavar="PRE", type=str, default=None, dest="pre")
    parser.add_argument("-lr", metavar="LR", type=float, default=0.001, dest="lr")
    parser.add_argument("-eps", metavar="E", type=int, default=400, dest="eps")
    parser.add_argument("-wd", metavar="WD", type=float, default=1e-8, dest="wd")
    parser.add_argument("-m", metavar="M", type=float, default=0, dest="m")
    parser.add_argument("-opt", metavar="OPT", type=str, default="Adam", dest="opt")
    parser.add_argument("-cuda", metavar="CUDA", type=int, default=0, dest="cuda")
    parser.add_argument("-mul", metavar="MUL", type=bool, default=False, dest="mul") #same thing as GAP
    parser.add_argument("-rec", metavar="REC", type=bool, default=False, dest="rec")
    parser.add_argument("-stp", metavar="STP", type=int, default=100, dest="stp")
    args = parser.parse_args()

    a = "cuda:" + str(args.cuda)
    device = torch.device(a if torch.cuda.is_available() else "cpu")
    criterion1 = nn.BCEWithLogitsLoss().to(device)
    net = unet.run_cnn()
    vall = False
    if args.pre is not None:
        checkpoint = torch.load(args.pre)
        net.load_state_dict(checkpoint["net"])
        vall = True
    net.to(device)

    best_loss = checkpoint["loss"] if vall else 100

    alpha = checkpoint["alpha"] if vall else args.lr
    if args.opt == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=alpha, weight_decay=args.wd)
    elif args.opt == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr=alpha, weight_decay=args.wd)
    elif args.opt == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=alpha, weight_decay=args.wd, momentum=args.m)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=alpha, weight_decay=args.wd, momentum=args.m)
    if vall:
        optimizer.load_state_dict(checkpoint["optimizer"])
    train_loss, val_loss = [], []
    start_ = checkpoint["epoch"] if vall else 1 
    epochs = checkpoint["epoch"]+args.eps if vall else args.eps
    for epoch in range(start_, epochs+1):
        if epoch % args.stp == 0 and epoch != epochs:
            alpha = lr_change(alpha)
        net = net.train()
        epoch_loss = 0.0
        for img, mask in trainloader:
            mask_type = torch.float32 #long if classes > 1
            img, mask = (img.to(device), mask.to(device, dtype=mask_type))
            mask_pred = net(img)
            loss = criterion1(mask_pred, mask)
            t = torch.sigmoid(mask_pred)
            epoch_loss += DiceLoss(t, mask).item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss.append(epoch_loss/640)
        print("Epoch" + str(epoch) + " Train Loss:", epoch_loss/640) 
        
        net = net.eval()
        tot = 0
        tot_val = 0.0
        for img, mask in testloader:
            mask_type = torch.float32
            img, mask = (img.to(device), mask.to(device, dtype=mask_type))
            mask_pred = net(img)
            t = torch.sigmoid(mask_pred)
            tot_val += DiceLoss(t, mask).item()
        loss_ = tot_val/20
        print("Epoch" + str(epoch) + " Val Loss:", loss_)
        if loss_ < best_loss:
            valid = True
            print("New best test loss!")
            best_loss = loss_
        else:
            valid = False
        val_loss.append(loss_)
        print("\n")
        
        if valid:
            for param_group in optimizer.param_groups:
                alpha_ = param_group["lr"]
            state = {
                "net":net.state_dict(),
                "loss": loss_,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "alpha": alpha_
            }
            path_ = "./models/vanilla_" + args.opt + "_lr" + str(args.lr)
            path_ += "/"
            try:
                os.mkdir(path_)
            except:
                pass
            torch.save(state, str(path_ + "best.pt"))
        if epoch == epochs:
            fig = plt.figure()
            plt.plot(train_loss, label="Train")
            plt.plot(val_loss, label="Val")
            plt.xlabel("Epochs")
            plt.ylabel("Dice Loss")
            plt.title("Train-Val Loss")
            fig.savefig(path_+ "train.png")
            print("Saved plots")
