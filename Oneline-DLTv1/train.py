# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import cv2
from torch_homography_model import build_model
from datetime import datetime
from dataset import TrainDataset
from utils import display_using_tensorboard

# name of log
train_log_dir = 'train_log_Oneline-FastDLT'

# path of project
exp_name = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
exp_train_log_dir = os.path.join(exp_name, train_log_dir)

LOG_DIR = os.path.join(exp_train_log_dir, 'logs')

# Where to load model
MODEL_LOAD_DIR = os.path.join(exp_name, 'models')
# Where to save new model
MODEL_SAVE_DIR = os.path.join(exp_train_log_dir, 'real_models')

now_time = datetime.now()

writer = SummaryWriter(log_dir=LOG_DIR)

if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


def train(args):

    train_path = os.path.join(exp_name, 'Data/Train_List.txt')
    net = build_model(args.model_name, pretrained=args.pretrained)

    if args.finetune:
        model_path = os.path.join(exp_name, 'models/freeze-mask-first-fintune.pth')
        print(model_path)
        state_dict = torch.load(model_path, map_location='cpu')
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.state_dict().items():
            name_key = k[7:]  # remove `module.`
            new_state_dict[name_key] = v
        # load params
        net = build_model(args.model_name)
        model_dict = net.state_dict()
        new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict.keys()}
        model_dict.update(new_state_dict)
        net.load_state_dict(model_dict)

    net = torch.nn.DataParallel(net)
    if torch.cuda.is_available():
        net = net.cuda()

    train_data = TrainDataset(data_path=train_path, exp_path=exp_name, patch_w=args.patch_size_w, patch_h=args.patch_size_h, rho=16)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=args.cpus, shuffle=True, drop_last=True)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, amsgrad=True, weight_decay=1e-4)  # default as 0.0001
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    print("start training")

    score_print_fre = 200
    model_save_fre = 4000
    glob_iter = 0

    for epoch in range(args.max_epoch):
        net.train()
        loss_sigma = 0.0
        loss_sigma_feature = 0.0

        scheduler.step()  # Note: The initial learning rate should be 1e-4. torch_version==1.0.1 ->init lr == 0.0001; torch_version>=1.2.0 ->init lr == 0.0001*1.25?
        print(epoch, 'lr={:.6f}'.format(scheduler.get_lr()[0]))
        for i, batch_value in enumerate(train_loader):
            # save model
            if (glob_iter % model_save_fre == 0 and glob_iter != 0 ):
                filename = str(args.model_name)+'_iter_' + str(glob_iter) + '.pth'
                model_save_path = os.path.join(MODEL_SAVE_DIR, filename)
                torch.save(net, model_save_path)

                for name, layer in net.named_parameters():
                    if layer.requires_grad == True:

                        writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), glob_iter)
                        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), glob_iter)

            org_imges = batch_value[0].float()
            input_tesnors = batch_value[1].float()
            patch_indices = batch_value[2].float()
            h4p = batch_value[3].float()

            I = org_imges[:, 0, ...]
            I = I[:, np.newaxis, ...]
            I2_ori_img = org_imges[:, 1, ...]
            I2_ori_img = I2_ori_img[:, np.newaxis, ...]
            I1 = input_tesnors[:, 0, ...]
            I1 = I1[:, np.newaxis, ...]
            I2 = input_tesnors[:, 1, ...]
            I2 = I2[:, np.newaxis, ...]

            if torch.cuda.is_available():
                input_tesnors = input_tesnors.cuda()
                patch_indices = patch_indices.cuda()
                h4p = h4p.cuda()
                I = I.cuda()
                I2_ori_img = I2_ori_img.cuda()
                I2 = I2.cuda()

            # forward, backward, update weights
            optimizer.zero_grad()

            batch_out = net(org_imges, input_tesnors, h4p, patch_indices)
            loss_feature = batch_out['feature_loss'].mean()
            pred_I2 = batch_out['pred_I2_d']
            I2_dataMat_CnnFeature = batch_out['patch_2_res_d']
            pred_I2_dataMat_CnnFeature = batch_out['pred_I2_CnnFeature_d']
            triMask = batch_out['mask_ap_d']
            loss_map = batch_out['feature_loss_mat_d']

            total_loss = loss_feature
            total_loss.backward()
            optimizer.step()

            loss_sigma += total_loss.item()
            loss_sigma_feature += loss_feature.item()

            # print loss etc.
            if i % score_print_fre == 0 and i != 0:
                loss_avg_feature = loss_sigma_feature / score_print_fre
                loss_sigma = 0.0
                loss_sigma_feature = 0.0

                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}]/[{:0>3}] Feature Loss: {:.4f} lr={:.8f}".format(epoch + 1,
                                                                                                       args.max_epoch,
                                                                                                       i + 1, len(train_loader), loss_avg_feature,
                                                                                                       scheduler.get_lr()[0]))

            glob_iter += 1

            # using tensorbordX to check the input or output performance during training
            if glob_iter % 200 == 0:
                display_using_tensorboard(I, I2_ori_img, I2, pred_I2, I2_dataMat_CnnFeature, pred_I2_dataMat_CnnFeature,
                                          triMask, loss_map, writer)
            writer.add_scalars('Loss_group', {'feature_loss': loss_feature.item()}, glob_iter)
            writer.add_scalar('learning rate', scheduler.get_lr()[0], glob_iter)

    print('Finished Training')


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=2, help='Number of splits')
    parser.add_argument('--cpus', type=int, default=8, help='Number of cpus')

    parser.add_argument('--img_w', type=int, default=640)
    parser.add_argument('--img_h', type=int, default=360)
    parser.add_argument('--patch_size_h', type=int, default=315)
    parser.add_argument('--patch_size_w', type=int, default=560)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--model_name', type=str, default='resnet34')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained waights?')
    parser.add_argument('--finetune', type=bool, default=False, help='Use pretrained waights?')

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    train(args)


