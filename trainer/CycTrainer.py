#!/usr/bin/python3
import random
import matplotlib.pyplot as plt
import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from .utils import LambdaLR,Logger,ReplayBuffer
from .utils import weights_init_normal,get_config
from .datasets import ImageDataset,ValDataset, EyeDataset, EyeDataset1
from Model.CycleGan import *
from .utils import Resize,ToTensor,smooothing_loss
from .utils import Logger
from .reg import Reg
from torchvision.transforms import RandomAffine,ToPILImage
from .transformer import Transformer_2D
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
import cv2
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy import interp
from itertools import cycle
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
import pandas as pd
import shap
from matplotlib.font_manager import FontProperties
from torchsummary import summary

class Cyc_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        ## def networks
        self.netG_A2B = Generator(config['input_nc'], config['output_nc']).cuda('cuda:0')
        self.netD_B = Discriminator(config['input_nc']).cuda()
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        if config['regist']:
            self.R_A = Reg(config['size'], config['size'],config['input_nc'],config['input_nc']).cuda()
            self.spatial_transform = Transformer_2D().cuda()
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        if config['bidirect']:
            self.netG_B2A = Generator(config['input_nc'], config['output_nc']).cuda()
            self.netD_A = Discriminator(config['input_nc']).cuda()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        else:
            self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))


        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()
        # self.classifier_loss = torch.nn.MSELoss()
        self.classifier_loss = torch.nn.CrossEntropyLoss()
        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.test_input_A = Tensor(1, config['input_nc'], config['size'], config['size'])
        self.test_input_B = Tensor(1, config['output_nc'], config['size'], config['size'])
        self.target_real = Variable(Tensor(1,1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(1,1).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        #Dataset loader
        level = config['noise_level']  # set noise level

        transforms_1 = [ToPILImage(),
                        RandomAffine(degrees=level, translate=[0.02 * level, 0.02 * level],
                                     scale=[1 - 0.02 * level, 1 + 0.02 * level]),
                        ToTensor(),
                        Resize(size_tuple=(config['size'], config['size']))]

        transforms_2 = [ToPILImage(),
                        RandomAffine(degrees=1, translate=[0.02, 0.02], scale=[0.98, 1.02]),
                        ToTensor(),
                        Resize(size_tuple=(config['size'], config['size']))]

        # self.dataloader = DataLoader(EyeDataset(config['dataroot'], level, transforms_1=transforms_1, transforms_2=transforms_2, unaligned=False, type='train'),
        #                         batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'], drop_last=True)
        val_transforms = [ToTensor(),
                          Resize(size_tuple = (config['size'], config['size']))]
        self.transforms = val_transforms

        # self.val_data = DataLoader(EyeDataset(config['val_dataroot'], 0, transforms_1 =val_transforms, transforms_2=None, unaligned=False, type='val'),
        #                         batch_size=1, shuffle=False, num_workers=config['n_cpu'])

        self.test_data = DataLoader(
            EyeDataset(config['val_dataroot'], 0, transforms_1=val_transforms, transforms_2=None, unaligned=False,
                       type='exter'),
            batch_size=1, shuffle=False, num_workers=config['n_cpu'])


       # Loss plot
       #  self.logger = Logger(config['name'],config['port'],config['n_epochs'], len(self.dataloader))

    def train(self):
        ###### Training ######
        best_acc, acc, best_mae = 0, -1, 100
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for i, batch in enumerate(self.dataloader):
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
                class_label = batch['class_label'].cuda()
                if self.config['bidirect']:   # C dir
                    if self.config['regist']:    #C + R
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        # GAN loss
                        fake_B,pre_class = self.netG_A2B(real_A)
                        pred_fake = self.netD_B(fake_B)
                        loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)

                        # classifier_loss = self.classifier_loss(pre_class, class_label)
                        fake_A = self.netG_B2A(real_B)
                        pred_fake = self.netD_A(fake_A)
                        loss_GAN_B2A = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)

                        Trans = self.R_A(fake_B,real_B)
                        SysRegist_A2B = self.spatial_transform(fake_B,Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)

                        # Cycle loss
                        recovered_A = self.netG_B2A(fake_B)
                        loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                        recovered_B = self.netG_A2B(fake_A)
                        loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                        # Total loss
                        loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + SR_loss +SM_loss
                        print("Total loss", loss_Total.item(), 'Gan_A2B', loss_GAN_A2B.item(), 'GAN_B2A', loss_GAN_B2A.item(),
                              'Cycle_ABA', loss_cycle_ABA.item(), 'Cycle_BAB', loss_cycle_BAB.item(), 'SR_loss', SR_loss.item(), 'SM_loss', SM_loss.item())

                        loss_Total.backward()
                        self.optimizer_G.step()
                        self.optimizer_R_A.step()

                        ###### Discriminator A ######
                        self.optimizer_D_A.zero_grad()
                        # Real loss
                        pred_real = self.netD_A(real_A)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                        pred_fake = self.netD_A(fake_A.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_A = (loss_D_real + loss_D_fake)
                        loss_D_A.backward()

                        self.optimizer_D_A.step()
                        ###################################

                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()

                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()

                        self.optimizer_D_B.step()
                        ###################################

                    else: #only  dir:  C
                        self.optimizer_G.zero_grad()
                        # GAN loss
                        fake_B, pre_class = self.netG_A2B(real_A)
                        pred_fake = self.netD_B(fake_B)
                        loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        classifier_loss = self.classifier_loss(pre_class, class_label)

                        fake_A = self.netG_B2A(real_B)
                        pred_fake = self.netD_A(fake_A)
                        loss_GAN_B2A = self.config['Adv_lamda']*self.MSE_loss(pred_fake, self.target_real)

                        # Cycle loss
                        recovered_A = self.netG_B2A(fake_B)
                        loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                        recovered_B = self.netG_A2B(fake_A)
                        loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                        # Total loss
                        loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                        loss_Total.backward()
                        self.optimizer_G.step()

                        ###### Discriminator A ######
                        self.optimizer_D_A.zero_grad()
                        # Real loss
                        pred_real = self.netD_A(real_A)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                        pred_fake = self.netD_A(fake_A.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_A = (loss_D_real + loss_D_fake)
                        loss_D_A.backward()

                        self.optimizer_D_A.step()
                        ###################################

                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()

                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()

                        self.optimizer_D_B.step()
                        ###################################



                else:    # s dir :NC
    
                    if self.config['regist']:    # NC+R
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        #### regist sys loss
                        # pre_class = self.netG_A2B(real_A)
                        fake_B,pre_class = self.netG_A2B(real_A)
                        Trans = self.R_A(fake_B,real_B)
                        SysRegist_A2B = self.spatial_transform(fake_B, Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B, fake_B)###SR
                        classifier_loss = self.classifier_loss(pre_class, class_label)
                        pred_fake0 = self.netD_B(fake_B)
                        adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_real)
                        ####smooth loss
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                        toal_loss = SM_loss+adv_loss+SR_loss+classifier_loss
                        toal_loss.backward()
                        self.optimizer_R_A.step()
                        self.optimizer_G.step()
                        self.optimizer_D_B.zero_grad()
                        with torch.no_grad():
                             fake_B, pre_class = self.netG_A2B(real_A)
                        pred_fake0 = self.netD_B(fake_B)
                        pred_real = self.netD_B(real_B)
                        loss_D_B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_fake)+self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        loss_D_B.backward()
                        self.optimizer_D_B.step()



                    else:        # only NC
                        self.optimizer_G.zero_grad()
                        fake_B = self.netG_A2B(real_A)
                        #### GAN aligin loss
                        pred_fake, _ = self.netD_B(fake_B)
                        adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        adv_loss.backward()
                        self.optimizer_G.step()
                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()
                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)
                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()
                        self.optimizer_D_B.step()
                        ###################################


                self.logger.log({'loss_D_B': loss_D_B,'SR_loss':SR_loss, "class_loss":classifier_loss},
                        images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B})#,'SR':SysRegist_A2B
                # self.logger.log()#,'SR':SysRegist_A2B




            #############val###############
            with (torch.no_grad()):
                MAE = 0
                num = 0
                count = 0
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.test_input_A.copy_(batch['A']))
                    real_B = Variable(self.test_input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                    class_label = batch['class_label']
                    fake_B, pre_label = self.netG_A2B(real_A)
                    fake_B = fake_B.detach().cpu().numpy().squeeze()
                    pre_label = pre_label.detach().cpu().numpy().squeeze()
                    count += (pre_label.argmax() == class_label).item()
                    mae = self.MAE(fake_B,real_B)
                    MAE += mae
                    num += 1
                acc = count/num
                print ('Val MAE:',MAE/num, "acc:", acc, "best_acc", best_acc)


            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
            torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B.pth')
            if acc > best_acc:
                print("best MAE",acc)
                torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'best_netG_A2B.pth')
                best_acc = acc

    def test(self,):
        self.netG_A2B.load_state_dict(torch.load(self.config['checkpoint']))
        with torch.no_grad():
                SSIM = 0
                num = 0
                count = 0
                class_label_ = []
                pre_label_ = []
                pre_class_ = []
                for i, batch in enumerate(self.test_data):
                    real_A = Variable(self.test_input_A.copy_(batch['A']))
                    real_B = Variable(self.test_input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                    class_label = batch['class_label']
                    fake_B,  pre_label = self.netG_A2B(real_A)
                    fake_B = fake_B.detach().cpu().numpy().squeeze()
                    real_A_ = real_A.detach().cpu().numpy().squeeze()
                    norm_image = cv2.normalize(fake_B, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                               dtype=cv2.CV_32F)
                    norm_image1 = cv2.normalize(real_A_, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                               dtype=cv2.CV_32F)
                    norm_image2 = cv2.normalize(real_B, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                               dtype=cv2.CV_32F)
                    cv2.imwrite(os.path.join(self.config['image_save'], batch["name"][0]), norm_image)  # 预测图像
                    cv2.imwrite(os.path.join(self.config['image_save1'], batch["name"][0]), norm_image1) # 干预前图像
                    cv2.imwrite(os.path.join(self.config['image_save2'], batch["name"][0]), norm_image2) # 干预后图像
                    pre_label = pre_label.detach().cpu().numpy().squeeze()
                    count += sum((pre_label.argmax() == class_label)).item()
                    ssim = compare_ssim(fake_B,real_B, data_range=fake_B.max() - fake_B.min())
                    SSIM += ssim
                    num += 1
                    class_label_.append(class_label.item())
                    pre_label_.append(pre_label)
                    pre_class_.append(pre_label.argmax())
                print ('SSIM:',SSIM/num)
                print('acc', count/num)
                self.draw(class_label_, pre_label_)
                self.performance(class_label_, pre_class_)

    def draw(self, real, pred):
        from sklearn.utils import resample
        class_names = ['Stable', 'Ineffective', "Effective"]
        pred = np.array(pred)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        real_ = np.zeros((len(real), 3))
        for i, number in enumerate(real):
            real_[i, real[i]] = 1

        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(real_[:, i], pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(real_.ravel(), pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(3):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= 3
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


        ci = dict()
        n_bootstraps = 1000
        rng_seed = 42  # control reproducibility

        for i in range(3):
            bootstrapped_scores = []
            for j in range(n_bootstraps):
                indices = resample(np.arange(len(real_)), random_state=j * rng_seed)
                if len(np.unique(real_[indices, i])) < 2:
                    continue
                score = auc(*roc_curve(real_[indices, i], pred[indices, i])[:2])
                bootstrapped_scores.append(score)
            sorted_scores = np.array(bootstrapped_scores)
            sorted_scores.sort()
            ci[i] = [sorted_scores[int(0.025 * len(sorted_scores))],
                     sorted_scores[int(0.975 * len(sorted_scores))]]


        micro_bootstrapped_scores = []
        for j in range(n_bootstraps):
            indices = resample(np.arange(len(real_)), random_state=j * rng_seed)
            if len(np.unique(real_[indices])) < 2:
                continue
            score = auc(*roc_curve(real_[indices].ravel(), pred[indices].ravel())[:2])
            micro_bootstrapped_scores.append(score)
        sorted_scores = np.array(micro_bootstrapped_scores)
        sorted_scores.sort()
        ci["micro"] = [sorted_scores[int(0.025 * len(sorted_scores))],
                       sorted_scores[int(0.975 * len(sorted_scores))]]


        macro_bootstrapped_scores = []
        for j in range(n_bootstraps):
            indices = resample(np.arange(len(real_)), random_state=j * rng_seed)
            mean_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
            mean_tpr = np.zeros_like(mean_fpr)
            for i in range(3):
                fpr_i, tpr_i, _ = roc_curve(real_[indices, i], pred[indices, i])
                mean_tpr += interp(mean_fpr, fpr_i, tpr_i)
            mean_tpr /= 3
            score = auc(mean_fpr, mean_tpr)
            macro_bootstrapped_scores.append(score)
        sorted_scores = np.array(macro_bootstrapped_scores)
        sorted_scores.sort()
        ci["macro"] = [sorted_scores[int(0.025 * len(sorted_scores))],
                       sorted_scores[int(0.975 * len(sorted_scores))]]
        lw = 2
        fig, ax = plt.subplots()
        ax.plot(fpr["micro"], tpr["micro"],
                label=f'AUC {format(roc_auc["micro"],".3f")}, 95%CI {ci["micro"][0]:.3f}~{ci["micro"][1]:.3f}',
                color='deeppink', linewidth=1)

        ax.plot(fpr["macro"], tpr["macro"],
                label=f'AUC {format(roc_auc["macro"],".3f")}, 95%CI {ci["macro"][0]:.3f}~{ci["macro"][1]:.3f}',
                color='navy', linewidth=1)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(3), colors):
            ax.plot(fpr[i], tpr[i], color=color, linewidth=1,
                    label=f'AUC {format(roc_auc[i],".3f")}, 95%CI {ci[i][0]:.3f}~{ci[i][1]:.3f}')

        ax.plot([0, 1], [0, 1], 'k--', lw=lw)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        font = FontProperties()
        font.set_size('small')
        plt.legend(loc="lower right", prop=font)

        axin = ax.inset_axes([0.23, 0.58, 0.26, 0.26])
        axin.plot(fpr["micro"], tpr["micro"], color='deeppink', linewidth=1)
        axin.plot(fpr["macro"], tpr["macro"], color='navy', linewidth=1)
        for i, color in zip(range(3), colors):
            axin.plot(fpr[i], tpr[i], color=color, linewidth=1)
        legend = ax.legend(frameon=False,prop={"family":"Arial"})
        axin.set_xlim([0.0, 0.2])
        axin.set_ylim([0.8, 1.0])
        ax.indicate_inset_zoom(axin)
        axin.grid(True)
        plt.savefig(f'{self.config["save_root"]}/NC+R_32', dpi=700)

        # Output the 95% confidence intervals
        for i in range(3):
            print(f'Class {class_names[i]} AUC: {roc_auc[i]:.3f}, 95% CI: [{ci[i][0]:.3f}, {ci[i][1]:.3f}]')
        print(f'Micro-average AUC: {roc_auc["micro"]:.3f}, 95% CI: [{ci["micro"][0]:.3f}, {ci["micro"][1]:.3f}]')
        print(f'Macro-average AUC: {roc_auc["macro"]:.3f}, 95% CI: [{ci["macro"][0]:.3f}, {ci["macro"][1]:.3f}]')

    def performance(self, real, pred):
        class_names = ['Stable', 'Ineffective', "Effective"]
        C = confusion_matrix(real, pred)
        plt.matshow(C, cmap=plt.cm.Blues)
        plt.xticks(np.arange(len(class_names)), class_names)
        plt.yticks(np.arange(len(class_names)), class_names, rotation=90,va='center')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')

        for i in range(len(C)):
            for j in range(len(C)):
                plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
        plt.savefig(f'{self.config["save_root"]}/confusion_matrix.png', dpi=700)
        """sen"""
        sen = []
        for i in range(3):
            tp = C[i][i]
            fn = np.sum(C[i, :]) - tp
            sen1 = tp / (tp + fn)
            sen.append(sen1)

        """spe"""
        spe = []
        for i in range(3):
            number = np.sum(C[:, :])#26
            tp = C[i][i] #13
            fn = np.sum(C[i, :]) - tp #0
            fp = np.sum(C[:, i]) - tp #4
            tn = number - tp - fn - fp #9
            spe1 = tn / (tn + fp)
            spe.append(spe1)
        print(sen)
        print(spe)




class fundus_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        ## def networks
        self.netG_A2B = mm.Generator(config['input_nc'], config['output_nc']).cuda('cuda:0')
        self.netD_B = mm.Discriminator(config['input_nc']).cuda()
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        if config['regist']:
            self.R_A = Reg(config['size'], config['size'],config['input_nc'],config['input_nc']).cuda()
            self.spatial_transform = Transformer_2D().cuda()
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        if config['bidirect']:
            self.netG_B2A = mm.Generator(config['input_nc'], config['output_nc']).cuda()
            self.netD_A = mm.Discriminator(config['input_nc']).cuda()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        else:
            self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))


        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()
        self.classifier_loss = torch.nn.CrossEntropyLoss()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.input_C = Tensor(config['batchSize'], 3, 256, 256)
        self.test_input_A = Tensor(1, config['input_nc'], config['size'], config['size'])
        self.test_input_B = Tensor(1, config['output_nc'], config['size'], config['size'])
        self.test_input_C = Tensor(1, 3, 256, 256)
        self.target_real = Variable(Tensor(1,1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(1,1).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        #Dataset loader
        level = config['noise_level']  # set noise level

        transforms_1 = [ToPILImage(),
                   RandomAffine(degrees=level,translate=[0.02*level, 0.02*level],scale=[1-0.02*level, 1+0.02*level]),
                   ToTensor(),
                   Resize(size_tuple = (config['size'], config['size']))]

        transforms_2 = [ToPILImage(),
                   RandomAffine(degrees=1,translate=[0.02, 0.02],scale=[0.98, 1.02]),
                   ToTensor(),
                   Resize(size_tuple = (config['size'], config['size']))]
        transforms_3 = [ToPILImage(),
                        RandomAffine(degrees=1, translate=[0.02, 0.02], scale=[0.98, 1.02]),
                        ToTensor(),
                        Resize(size_tuple=(config['size'], config['size']))]

        self.dataloader = DataLoader(fundusDataset(config['dataroot'], level, transforms_1=transforms_1, transforms_2=transforms_2, transforms_3=transforms_3,unaligned=False, type='train'),
                                batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'], drop_last=True)
        val_transforms = [ToTensor(),
                          Resize(size_tuple = (config['size'], config['size']))]

        self.val_data = DataLoader(fundusDataset(config['val_dataroot'], 0, transforms_1 =val_transforms, transforms_2=None,transforms_3=None, unaligned=False, type='val'),
                                batch_size=1, shuffle=False, num_workers=config['n_cpu'])


       # Loss plot
        self.logger = Logger(config['name'],config['port'],config['n_epochs'], len(self.dataloader))

    def train(self):
        ###### Training ######
        best_acc, acc = 0, -1
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for i, batch in enumerate(self.dataloader):
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
                fundus = Variable(self.input_C.copy_(batch['fundus']))
                class_label = batch['class_label'].cuda()
                if self.config['bidirect']:   # C dir
                    if self.config['regist']:    #C + R
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        # GAN loss
                        fake_B,pre_class = self.netG_A2B(real_A)
                        pred_fake = self.netD_B(fake_B)
                        loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)

                        classifier_loss = self.classifier_loss(pre_class, class_label)
                        fake_A = self.netG_B2A(real_B)
                        pred_fake = self.netD_A(fake_A)
                        loss_GAN_B2A = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)

                        Trans = self.R_A(fake_B,real_B)
                        SysRegist_A2B = self.spatial_transform(fake_B,Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)

                        # Cycle loss
                        recovered_A = self.netG_B2A(fake_B)
                        loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                        recovered_B = self.netG_A2B(fake_A)
                        loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                        # Total loss
                        loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + SR_loss +SM_loss+classifier_loss
                        print("Total loss", loss_Total.item(), 'Gan_A2B', loss_GAN_A2B.item(), 'GAN_B2A', loss_GAN_B2A.item(),
                              'Cycle_ABA', loss_cycle_ABA.item(), 'Cycle_BAB', loss_cycle_BAB.item(), 'SR_loss', SR_loss.item(), 'SM_loss', SM_loss.item())

                        loss_Total.backward()
                        self.optimizer_G.step()
                        self.optimizer_R_A.step()

                        ###### Discriminator A ######
                        self.optimizer_D_A.zero_grad()
                        # Real loss
                        pred_real = self.netD_A(real_A)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                        pred_fake = self.netD_A(fake_A.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_A = (loss_D_real + loss_D_fake)
                        loss_D_A.backward()

                        self.optimizer_D_A.step()
                        ###################################

                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()

                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()

                        self.optimizer_D_B.step()
                        ###################################

                    else: #only  dir:  C
                        self.optimizer_G.zero_grad()
                        # GAN loss
                        fake_B, pre_class = self.netG_A2B(real_A)
                        pred_fake = self.netD_B(fake_B)
                        loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        classifier_loss = self.classifier_loss(pre_class, class_label)

                        fake_A = self.netG_B2A(real_B)
                        pred_fake = self.netD_A(fake_A)
                        loss_GAN_B2A = self.config['Adv_lamda']*self.MSE_loss(pred_fake, self.target_real)

                        # Cycle loss
                        recovered_A = self.netG_B2A(fake_B)
                        loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                        recovered_B = self.netG_A2B(fake_A)
                        loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                        # Total loss
                        loss_Total = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                        loss_Total.backward()
                        self.optimizer_G.step()

                        ###### Discriminator A ######
                        self.optimizer_D_A.zero_grad()
                        # Real loss
                        pred_real = self.netD_A(real_A)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                        pred_fake = self.netD_A(fake_A.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_A = (loss_D_real + loss_D_fake)
                        loss_D_A.backward()

                        self.optimizer_D_A.step()
                        ###################################

                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()

                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)

                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)
                        loss_D_B.backward()

                        self.optimizer_D_B.step()
                        ###################################



                else:                  # s dir :NC
                    if self.config['regist']:    # NC+R
                        self.optimizer_R_A.zero_grad()
                        self.optimizer_G.zero_grad()
                        #### regist sys loss
                        fake_B, pre_class = self.netG_A2B(real_A, fundus)
                        Trans = self.R_A(fake_B,real_B)
                        SysRegist_A2B = self.spatial_transform(fake_B,Trans)
                        SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B)###SR
                        classifier_loss = self.classifier_loss(pre_class, class_label)
                        pred_fake0 = self.netD_B(fake_B)
                        adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_real)
                        ####smooth loss
                        SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                        toal_loss = SM_loss+adv_loss+classifier_loss+SR_loss
                        toal_loss.backward()
                        self.optimizer_R_A.step()
                        self.optimizer_G.step()
                        self.optimizer_D_B.zero_grad()
                        with torch.no_grad():
                            fake_B, pre_class = self.netG_A2B(real_A, fundus)
                        pred_fake0 = self.netD_B(fake_B)
                        pred_real = self.netD_B(real_B)
                        loss_D_B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, self.target_fake)+self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        loss_D_B.backward()
                        self.optimizer_D_B.step()



                    else:        # only NC
                        self.optimizer_G.zero_grad()
                        fake_B, pre_class = self.netG_A2B(real_A, fundus)
                        #### GAN aligin loss
                        pred_fake = self.netD_B(fake_B)
                        adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                        adv_loss.backward()
                        self.optimizer_G.step()
                        ###### Discriminator B ######
                        self.optimizer_D_B.zero_grad()
                        # Real loss
                        pred_real = self.netD_B(real_B)
                        loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)
                        # Fake loss
                        fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                        pred_fake = self.netD_B(fake_B.clone().detach())
                        loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)
                        classifier_loss = self.classifier_loss(pre_class, class_label)
                        # Total loss
                        loss_D_B = (loss_D_real + loss_D_fake)+classifier_loss
                        loss_D_B.backward()
                        self.optimizer_D_B.step()
                        ###################################


                self.logger.log({'loss_D_B': loss_D_B,'SR_loss':SR_loss, 'classifier_loss':classifier_loss},
                       images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B})#,'SR':SysRegist_A2B
                # self.logger.log({'loss_D_B': loss_D_B, 'classifier_loss': classifier_loss},
                #                 images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B})  # ,'SR':SysRegist_A2B

    #         # Save models checkpoints
            #torch.save(self.R_A.state_dict(), self.config['save_root'] + 'Regist.pth')
            #torch.save(netD_A.state_dict(), 'output/netD_A_3D.pth')
            #torch.save(netD_B.state_dict(), 'output/netD_B_3D.pth')


            #############val###############
            with (torch.no_grad()):
                MAE = 0
                num = 0
                count = 0
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.test_input_A.copy_(batch['A']))
                    real_B = Variable(self.test_input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                    fundus = Variable(self.test_input_C.copy_(batch['fundus']))
                    class_label = batch['class_label']
                    fake_B, pre_label = self.netG_A2B(real_A, fundus)
                    fake_B = fake_B.detach().cpu().numpy().squeeze()
                    pre_label = pre_label.detach().cpu().numpy().squeeze()
                    count += (pre_label.argmax() == class_label).item()
                    mae = self.MAE(fake_B,real_B)
                    MAE += mae
                    num += 1
                acc = count/num
                print ('Val MAE:',MAE/num, "acc:", acc, "best_acc", best_acc)

            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
            torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B.pth')
            if acc > best_acc:
                torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'best_netG_A2B.pth')
                best_acc = acc

    def test(self,):
        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'best_netG_A2B.pth'))

        with torch.no_grad():
                SSIM = 0
                num = 0
                count = 0
                class_label_ = []
                pre_label_ = []
                pre_class_ = []
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.test_input_A.copy_(batch['A']))
                    real_B = Variable(self.test_input_B.copy_(batch['B'])).detach().cpu().numpy().squeeze()
                    fundus = Variable(self.test_input_C.copy_(batch['fundus']))
                    class_label = batch['class_label']
                    fake_B, pre_label = self.netG_A2B(real_A, fundus)
                    fake_B = fake_B.detach().cpu().numpy().squeeze()
                    norm_image = cv2.normalize(fake_B, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                               dtype=cv2.CV_32F)
                    cv2.imwrite(os.path.join(self.config['image_save'],batch["name"][0]), norm_image)
                    pre_label = pre_label.detach().cpu().numpy().squeeze()
                    count += sum((pre_label.argmax() == class_label)).item()
                    ssim = compare_ssim(fake_B,real_B, data_range=fake_B.max() - fake_B.min())
                    SSIM += ssim
                    num += 1
                    class_label_.append(class_label.item())
                    pre_label_.append(pre_label)
                    pre_class_.append(pre_label.argmax())

                print ('SSIM:',SSIM/num)
                print('acc', count/num)
                self.draw(class_label_, pre_label_)
                self.performance(class_label_, pre_class_)

    def draw(self, real, pred):
        pred = np.array(pred)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        real_ = np.zeros((len(real),3))
        for i, number in enumerate(real):
            real_[i, real[i]] = 1

        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(real_[:, i], pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(real_.ravel(), pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(3):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= 3
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        lw = 2
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(3), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('multi-calss ROC')
        plt.legend(loc="lower right")
        plt.savefig(f'{self.config["save_root"]}/NC+R_32')


    def performance(self, real, pred):
        C = confusion_matrix(real, pred)
        plt.matshow(C, cmap=plt.cm.Blues)
        for i in range(len(C)):
            for j in range(len(C)):
                plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(f'{self.config["save_root"]}/confusion_matrix.png')
        """sen"""
        sen = []
        for i in range(3):
            tp = C[i][i]
            fn = np.sum(C[i, :]) - tp
            sen1 = tp / (tp + fn)
            sen.append(sen1)

        """spe"""
        spe = []
        for i in range(3):
            number = np.sum(C[:, :])#26
            tp = C[i][i] #13
            fn = np.sum(C[i, :]) - tp #0
            fp = np.sum(C[:, i]) - tp #4
            tn = number - tp - fn - fp #9
            spe1 = tn / (tn + fp)
            spe.append(spe1)
        print(sen)
        print(spe)
