# refer to https://github.com/zengchang233/xiaoicesing2
# BSD 3-Clause License

import torch
import torch.nn as nn


class GAN_loss_G(nn.Module):
    def __init__(self):
        super(GAN_loss_G, self).__init__()
        self.feat_loss = FeatLoss()
        self.adv_g_loss = LSGANGLoss()
    def forward(self, D_fake):
        feat_loss, _ = self.feat_loss(D_fake)
        adv_g_loss, _ = self.adv_g_loss(D_fake)

        return {'feat_loss': feat_loss, 'adv_g_loss': adv_g_loss}


class GAN_loss_D(nn.Module):
    def __init__(self):
        super(GAN_loss_D, self).__init__()
        self.adv_d_loss = LSGANDLoss()
    def forward(self, D_fake):
        adv_d_loss, _ = self.adv_d_loss(D_fake)
        
        return {'adv_d_loss': adv_d_loss}


class FeatLoss(nn.Module):
    '''
    feature loss (multi-band discriminator) 
    '''
    def __init__(self, feat_loss_weight = (1.0, 1.0, 1.0)):
        super(FeatLoss, self).__init__()
        self.loss_d = nn.MSELoss() #.to(self.device)
        self.feat_loss_weight = feat_loss_weight

    def forward(self, D_fake):
        feat_g_loss = 0.0
        feat_loss = [0.0] * len(D_fake)
        report_keys = {}
        for j in range(len(D_fake)):
            for k in range(len(D_fake[j][0])):
                for n in range(len(D_fake[j][0][k][1])):
                    if len(D_fake[j][0][k][1][n].shape) == 4:
                        t_batch = D_fake[j][0][k][1][n].shape[0]
                        t_length = D_fake[j][0][k][1][n].shape[-1]
                        D_fake[j][0][k][1][n] = D_fake[j][0][k][1][n].view(t_batch, t_length,-1)
                        D_fake[j][1][k][1][n] = D_fake[j][1][k][1][n].view(t_batch, t_length,-1)
                    feat_loss[j] += self.loss_d(D_fake[j][0][k][1][n], D_fake[j][1][k][1][n]) * 2
                feat_loss[j] /= (n + 1)
            feat_loss[j] /= (k + 1)
            feat_loss[j] *= self.feat_loss_weight[j]
            report_keys['feat_loss_' + str(j)] = feat_loss[j]
            feat_g_loss += feat_loss[j]

        return feat_g_loss, report_keys

class LSGANGLoss(nn.Module):
    def __init__(self, adv_loss_weight=(0.1, 0.1, 0.1)):
        super(LSGANGLoss, self).__init__()
        self.loss_d = nn.MSELoss() #.to(self.device)
        self.adv_loss_weight = adv_loss_weight

    def forward(self, D_fake):
        adv_g_loss = 0.0
        adv_loss = [0.0] * len(D_fake)
        report_keys = {}
        for j in range(len(D_fake)):
            for k in range(len(D_fake[j][0])):
                adv_loss[j] += self.loss_d(D_fake[j][0][k][0], D_fake[j][0][k][0].new_ones(D_fake[j][0][k][0].size()))
            adv_loss[j] /= (k + 1)
            adv_loss[j] *= self.adv_loss_weight[j]
            report_keys['adv_g_loss_' + str(j)] = adv_loss[j]
            adv_g_loss += adv_loss[j]
        return adv_g_loss, report_keys

class LSGANDLoss(nn.Module):
    def __init__(self):
        super(LSGANDLoss, self).__init__()
        self.loss_d = nn.MSELoss()

    def forward(self, D_fake):
        adv_d_loss = 0.0
        adv_loss = [0.0] * len(D_fake)
        real_loss = [0.0] * len(D_fake)
        fake_loss = [0.0] * len(D_fake)
        report_keys = {}
        for j in range(len(D_fake)):
            for k in range(len(D_fake[j][0])):
                real_loss[j] += self.loss_d(D_fake[j][1][k][0], D_fake[j][1][k][0].new_ones(D_fake[j][1][k][0].size()))
                fake_loss[j] += self.loss_d(D_fake[j][0][k][0], D_fake[j][0][k][0].new_zeros(D_fake[j][0][k][0].size()))
            real_loss[j] /= (k + 1)
            fake_loss[j] /= (k + 1)
            adv_loss[j] = 0.5 * (real_loss[j] + fake_loss[j])
            report_keys['adv_d_loss_' + str(j)] = adv_loss[j]
            adv_d_loss += adv_loss[j]
        return adv_d_loss, report_keys
