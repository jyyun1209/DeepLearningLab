import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from models.UNet import UNet
from models.UNet_dilat import UNet_dilat
from models.UNet_adv_v3 import UNet_adv_v3
from models.UNet_adv import UNet_adv
from models.UNet_mid import UNet_mid
from models.Simple_seg import Simple_seg
from models.Simple_seg_v2 import Simple_seg_v2
from models.Simple_seg_v2_dilat import Simple_seg_v2_dilat
import pickle

# torch.set_printoptions(profile="full")

## Dataset part
class BratsDataset_seg(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.data_list = list(sorted(os.listdir(os.path.join(root))))

        # print(len(self.data_list))    # 222, 29, 84

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        ###### fill the codes below #####
        # 1. define paths for an image and a label
        # 2. load arrays for the image and label
        # 3. process image array (e.g., normalization, numpy array to tensor)
        # 4. process label array [0,1,2,4] => [0,1,2,3]

        dir_name = self.data_list[idx]
        dir_path = os.path.join(self.root, dir_name)
        npy_files = os.listdir(dir_path)
        for nf in npy_files:
            if nf.startswith('label'):
                label_file = os.path.join(dir_path, nf)
                label = np.load(label_file)
                label = torch.LongTensor(label)
                label = torch.where(label == 3, 0, label)
                label = torch.where(label == 4, 3, label)
                # print(label)
            else:
                img_file = os.path.join(dir_path, nf)
                img = np.load(img_file)
                img = torch.FloatTensor(img)

        #################################
        output = {'img': img, 'label': label}
        return output


def train_one_epoch(model, optimizer, criterion, train_data_loader, valid_data_loader, device, epoch, lr_scheduler, print_freq=10, min_valid_loss=100):
    for train_iter, pack in enumerate(train_data_loader):
        train_loss = 0
        valid_loss = 0
        img = pack['img'].to(device)
        label = pack['label'].to(device)
        optimizer.zero_grad()
        pred = model(img)
        # print(pred.shape) # torch.Size([2, 4, 240, 240])
        # print(label.shape)    # torch.Size([2, 240, 240])
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if (train_iter+1) % print_freq == 0:
            with torch.no_grad():
                model.eval()
                for valid_iter, pack in enumerate(valid_data_loader):
                    img = pack['img'].to(device)
                    label = pack['label'].to(device)
                    pred = model(img)
                    loss = criterion(pred, label)
                    valid_loss += loss.item()

                if min_valid_loss >= valid_loss/len(valid_data_loader):

                    # set file fname for model save 'best_model_{your name}.pth'
                    torch.save(model.state_dict(), 'best_model_junyoung.pth')

                    min_valid_loss = valid_loss/len(valid_data_loader)
                    print('{}th epoch {}/{} iter: train loss={}, valid loss={}, lr={}'
                          .format(epoch+1, train_iter+1, len(train_data_loader), train_loss/print_freq, valid_loss/len(valid_data_loader), lr_scheduler.get_last_lr()),
                          " => model saved")
                else:
                    print('{}th epoch {}/{} iter: train loss={}, valid loss={}, lr={}'
                          .format(epoch+1, train_iter+1, len(train_data_loader), train_loss/print_freq, valid_loss/len(valid_data_loader), lr_scheduler.get_last_lr()))
        lr_scheduler.step()
    return min_valid_loss


def evaluate(model, test_data_loader, device):
    ###### fill the codes below #####
    # for all test images, compute dice similarity score.
    # and obtain class-average DSC
    # DSC = 2*TP / (2*TP + FP + FN)
    #################################
    class_DSC = 0
    dscs = []
    with torch.no_grad():
        for test_iter, pack in enumerate(test_data_loader):
            img = pack['img'].to(device)
            label = pack['label'].to(device)
            pred = model(img)
            pred = torch.squeeze(pred)
            label = torch.squeeze(label)
            _, pred_labels = torch.max(pred, 0)

            Ps = torch.where(pred_labels > 0, 1, 0)
            p_matched = torch.where((pred_labels > 0) & (pred_labels == label), 1, 0)
            TPs = torch.where((Ps == 1) & (Ps == p_matched), 1, 0)
            FNs = torch.where((Ps == 0) & (Ps != label), 1, 0)

            # print(pred_labels)
            P = torch.sum(Ps)
            TP = torch.sum(TPs)
            FP = P - TP
            FN = torch.sum(FNs)
            # print(P, TP, FP, FN)
            # print(2*TP / (2*TP + FP + FN))
            # print(test_iter, "th Img. Dice Similarity Coefficient: ", 2*TP / (2*TP + FP + FN))
            # dscs.append((2*TP / (2*TP + FP + FN)).item())
            class_DSC += 2*TP / (2*TP + FP + FN)

    class_DSC /= len(test_data_loader)
    print("Average Dice Similarity Coefficient: ", class_DSC.item())

    # with open(file='dscs_unet.pickle', mode='wb') as f:
    #     pickle.dump(dscs, f)

    return class_DSC


def visualize(model, test_data_loader, device, skip=0, cmap=None, Max=False, Min=False):
    ###### fill the codes below #####
    # for all test images, compute dice similarity score.
    # and obtain class-average DSC
    # DSC = 2*TP / (2*TP + FP + FN)
    #################################

    class_DSC = 0
    Max_DSC = 0
    Min_DSC = 1
    Max_idx = 0
    Min_idx = 0
    if Max == True and Min == True:
        raise ValueError("Max and Min cannot be both True.")

    if Max == False and Min == False:
        with torch.no_grad():
            for test_iter, pack in enumerate(test_data_loader):
                if test_iter < skip:
                    continue
                img = pack['img'].to(device)
                label = pack['label'].to(device)
                pred = model(img)
                pred = torch.squeeze(pred)
                label = torch.squeeze(label)
                _, pred_labels = torch.max(pred, 0)

                Ps = torch.where(pred_labels > 0, 1, 0)
                p_matched = torch.where((pred_labels > 0) & (pred_labels == label), 1, 0)
                TPs = torch.where((Ps == 1) & (Ps == p_matched), 1, 0)
                FNs = torch.where((Ps == 0) & (Ps != label), 1, 0)

                # print(pred_labels)
                P = torch.sum(Ps)
                TP = torch.sum(TPs)
                FP = P - TP
                FN = torch.sum(FNs)
                # print(P, TP, FP, FN)
                print(test_iter, "th Img. Dice Similarity Coefficient: ", 2*TP / (2*TP + FP + FN))
                class_DSC += 2*TP / (2*TP + FP + FN)
                f, ax = plt.subplots(3, 2)
                img = torch.squeeze(img).cpu()
                ax[0,0].imshow(img[0], cmap=cmap)
                ax[0,1].imshow(img[1], cmap=cmap)
                ax[1,0].imshow(img[2], cmap=cmap)
                ax[1,1].imshow(img[3], cmap=cmap)
                ax[2,0].imshow(label.cpu(), cmap=cmap)
                ax[2,1].imshow(pred_labels.cpu(), cmap=cmap)
                ax[0,0].title.set_text('Image1')
                ax[0,1].title.set_text('Image2')
                ax[1,0].title.set_text('Image3')
                ax[1,1].title.set_text('Image4')
                ax[2,0].title.set_text('Label')
                ax[2,1].title.set_text('Prediction')
                plt.show()
    else:
        with torch.no_grad():
            for test_iter, pack in enumerate(test_data_loader):
                img = pack['img'].to(device)
                label = pack['label'].to(device)
                pred = model(img)
                pred = torch.squeeze(pred)
                label = torch.squeeze(label)
                _, pred_labels = torch.max(pred, 0)

                Ps = torch.where(pred_labels > 0, 1, 0)
                p_matched = torch.where((pred_labels > 0) & (pred_labels == label), 1, 0)
                TPs = torch.where((Ps == 1) & (Ps == p_matched), 1, 0)
                FNs = torch.where((Ps == 0) & (Ps != label), 1, 0)

                # print(pred_labels)
                P = torch.sum(Ps)
                TP = torch.sum(TPs)
                FP = P - TP
                FN = torch.sum(FNs)
                # print(P, TP, FP, FN)
                class_dsc = 2*TP / (2*TP + FP + FN)
                class_DSC += class_dsc
                if class_dsc >= Max_DSC:
                    Max_DSC = class_dsc
                    Max_idx = test_iter
                if class_dsc <= Min_DSC:
                    Min_DSC = class_dsc
                    Min_idx = test_iter
            print("Max_DSC, Max_idx: ", Max_DSC.item(), ",", Max_idx, ", Min_DSC, Min_idx: ", Min_DSC.item(), ",", Min_idx)
            
            if Max == True:
                lim_idx = Max_idx
            if Min == True:
                lim_idx = Min_idx

            for test_iter, pack in enumerate(test_data_loader):
                if test_iter == lim_idx:
                    img = pack['img'].to(device)
                    label = pack['label'].to(device)
                    pred = model(img)
                    pred = torch.squeeze(pred)
                    label = torch.squeeze(label)
                    _, pred_labels = torch.max(pred, 0)

                    f, ax = plt.subplots(3, 2)
                    img = torch.squeeze(img).cpu()
                    
                    ax[0,0].imshow(img[0], cmap=cmap)
                    ax[0,1].imshow(img[1], cmap=cmap)
                    ax[1,0].imshow(img[2], cmap=cmap)
                    ax[1,1].imshow(img[3], cmap=cmap)
                    
                    # ax[0,0].imshow(img[0], cmap=cm.gray)
                    # ax[0,1].imshow(img[1], cmap=cm.gray)
                    # ax[1,0].imshow(img[2], cmap=cm.gray)
                    # ax[1,1].imshow(img[3], cmap=cm.gray)

                    ax[2,0].imshow(label.cpu(), cmap=cmap)
                    ax[2,1].imshow(pred_labels.cpu(), cmap=cmap)

                    ax[0,0].title.set_text('1st channel image')
                    ax[0,1].title.set_text('2nd channel image')
                    ax[1,0].title.set_text('3rd channel image')
                    ax[1,1].title.set_text('4th channel image')
                    ax[2,0].title.set_text('Label')
                    ax[2,1].title.set_text('Prediction')
                    plt.show()
                else:
                    continue

    class_DSC /= len(test_data_loader)
    print("Average Dice Similarity Coefficient: ", class_DSC.item())

    return class_DSC


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # gpu가 사용가능하다면 gpu를 사용하고, 아니라면 cpu를 사용함

    ## Hyper-parameters
    num_epochs = 30
    # num_epochs = 50
    n_channels = 4 # number of modalities
    n_classes = 4

    # model_channel = UNet(n_channels=n_channels, n_classes=n_classes)
    # model_channel = UNet_dilat(n_channels=n_channels, n_classes=n_classes)
    # model_channel = UNet_adv_v3(n_channels=n_channels, n_classes=n_classes)
    # model_channel = UNet_adv(n_channels=n_channels, n_classes=n_classes)
    # model_channel = UNet_mid(n_channels=n_channels, n_classes=n_classes)
    # model_channel = Simple_seg(n_channels=n_channels, n_classes=n_classes)
    # model_channel = Simple_seg_v2(n_channels=n_channels, n_classes=n_classes)
    model_channel = Simple_seg_v2_dilat(n_channels=n_channels, n_classes=n_classes)
    model_channel.to(device)

    optimizer = torch.optim.Adam(model_channel.parameters(), lr=0.0005)#weight_decay=0.0001
    criterion = nn.CrossEntropyLoss()

    # step_size 이후 learning rate에 gamma만큼을 곱해줌 ex) 111번 스텝 뒤에 lr에 gamma를 곱해줌
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=111,
                                                gamma=1) #0.9 ## learning rate decay

    ## data loader
    train_dataset = BratsDataset_seg('./seg_dataset/train')
    valid_dataset = BratsDataset_seg('./seg_dataset/valid')
    test_dataset = BratsDataset_seg('./seg_dataset/test')


    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=4)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, shuffle=True, num_workers=4)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, num_workers=4)

    ########### TRAINING ###########
    min_val_loss = 100
    for epoch in range(num_epochs):
        min_val_loss = train_one_epoch(model_channel, optimizer, criterion, train_data_loader, \
            valid_data_loader, device, epoch, lr_scheduler, print_freq=30, min_valid_loss=min_val_loss)
    ################################

    ########## EVALUATION ##########
    ##### fill the codes below #####
    #1. load the model which has the minimum validation loss
    #2. evaluate the model using DSC or mIOU

    # model_path = 'best_model_UNet_adv_v3.pth'
    # model_channel.load_state_dict(torch.load(model_path))
    # model_channel.eval()
    # model_channel.to(device)

    # evaluate(model_channel, test_data_loader, device=device)
    ################################


    ######### VISUALIZATION #########
    # model_path = 'best_model_simple_seg_v2_dilat.pth'
    # model_channel.load_state_dict(torch.load(model_path))
    # model_channel.eval()
    # model_channel.to(device)

    # # visualize(model_channel, test_data_loader, device=device, cmap=None)
    # visualize(model_channel, test_data_loader, device=device, cmap=None, skip=35)
    # # visualize(model_channel, test_data_loader, device=device, cmap=None, Max=True)
    # # visualize(model_channel, test_data_loader, device=device, cmap=None, Min=True)
    #################################

    
if __name__ == '__main__':
    main()
