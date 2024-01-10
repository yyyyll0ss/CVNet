import cv2
import torch
import torch.nn.functional as F

from math import log
from torch import nn
from CVNet.backbones import build_backbone
from CVNet.utils.polygon import generate_polygon
from CVNet.utils.polygon import get_pred_junctions
from CVNet.backbones import MultitaskHead
from skimage.measure import label, regionprops


def cross_entropy_loss_for_junction(logits, positive):
    nlogp = -F.log_softmax(logits, dim=1)

    loss = (positive * nlogp[:, None, 1] + (1 - positive) * nlogp[:, None, 0])

    return loss.mean()

def sigmoid_l1_loss(logits, targets, offset = 0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp-targets)

    if mask is not None:
        t = ((mask == 1) | (mask == 2)).float()
        w = t.mean(3, True).mean(2,True)
        w[w==0] = 1
        loss = loss*(t/w)

    return loss.mean()

class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) # sigmoide获取概率
        nclass = predict.shape[1]
        target = F.one_hot(target.long(), nclass)
        target = target.permute(0, 3, 1, 2)
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

# Dice系数
def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


# Dice损失函数
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        nclass = predict.shape[1]
        target = F.one_hot(target.long(), nclass)
        target = target.permute(0, 3, 1, 2)
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.reshape(num, -1)

        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

# Copyright (c) 2019 BangguWu, Qilong Wang
# Modified by Bowen Xu, Jiakun Xu, Nan Xue and Gui-song Xia
class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        C = channel

        t = int(abs((log(C, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

        self.out_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        y = self.avg_pool(x1 + x2)
        y = self.conv(y.squeeze(-1).transpose(-1, -2).contiguous())   #
        y = y.transpose(-1,-2).contiguous().unsqueeze(-1)   #
        y = self.sigmoid(y)

        out = self.out_conv(x2 * y.expand_as(x2))
        return out

class CVNet(nn.Module):
    def __init__(self, cfg, test=False):
        super(CVNet, self).__init__()
        self.backbone = build_backbone(cfg)
        self.backbone_name = cfg.MODEL.NAME
        self.junc_loss_bce = nn.CrossEntropyLoss()   # weight=torch.tensor([1, 30]).float().to('cuda:0')
        self.junc_focal_loss = BCEFocalLoss()  # weight=torch.tensor([1, 30]).float().to('cuda:0')
        self.mask_loss_bce = nn.CrossEntropyLoss()

        if not test:
            from CVNet.encoder import Encoder
            self.encoder = Encoder(cfg)

        # add two junction decoder
        self.pred_height = cfg.DATASETS.TARGET.HEIGHT
        self.pred_width = cfg.DATASETS.TARGET.WIDTH
        self.origin_height = cfg.DATASETS.ORIGIN.HEIGHT
        self.origin_width = cfg.DATASETS.ORIGIN.WIDTH

        self.dim_in = cfg.MODEL.OUT_FEATURE_CHANNELS
        self.mask_head_CD = self._make_conv(self.dim_in, self.dim_in, self.dim_in)
        self.mask_head = self._make_conv(self.dim_in, self.dim_in, self.dim_in)
        self.jloc_head = self._make_conv(self.dim_in, self.dim_in, self.dim_in)


        self.mask_predictor_CD = self._make_predictor(self.dim_in, 2)
        self.mask_predictor = self._make_predictor(self.dim_in, 2)
        self.jloc_predictor = self._make_predictor(self.dim_in, 2)

        self.train_step = 0

        last_inp_channels = self.backbone.last_inp_channels
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0)
        )
        head_size = cfg.MODEL.HEAD_SIZE
        num_class = sum(sum(head_size, []))
        self.head = MultitaskHead(256, num_class, head_size=head_size)

    def forward(self, image_CD, annotations_CD):
        if self.training:
            return self.forward_train(image_CD, annotations_CD)
        else:
            return self.forward_test(image_CD, annotations_CD)

    def forward_test(self, images_CD, annotations_CD):
        device = images_CD.device
        image_A = images_CD[:, :3, :, :]
        image_B = images_CD[:, 3:6, :, :]

        features_A = self.backbone(image_A)  # (classes,H,W) (256,H,W)
        features_B = self.backbone(image_B)  # (classes,H,W) (256,H,W)

        # feature map subtract
        features = []
        for (f_A, f_B) in zip(features_A, features_B):
            features.append(torch.abs(f_A - f_B))

        x0_h, x0_w = features[0].size(2), features[0].size(3)
        x0 = features[0]
        x1 = F.interpolate(features[1], size=(x0_h, x0_w), mode='bilinear', align_corners=None)
        x2 = F.interpolate(features[2], size=(x0_h, x0_w), mode='bilinear', align_corners=None)
        x3 = F.interpolate(features[3], size=(x0_h, x0_w), mode='bilinear', align_corners=None)
        features_c = torch.cat([x0, x1, x2, x3], 1)
        change_features = self.last_layer(features_c)
        outputs = self.head(change_features)

        # two branch feature
        mask_feature = self.mask_head_CD(change_features)
        jloc_feature = self.jloc_head(change_features)

        # two branch output
        mask_pred = self.mask_predictor_CD(mask_feature)
        jloc_pred = self.jloc_predictor(jloc_feature)

        joff_pred = outputs[:, :].sigmoid() - 0.5
        mask_pred = mask_pred.softmax(1)[:, 1:]
        jloc_pred = jloc_pred.softmax(1)[:, 1:]

        scale_y = self.origin_height / self.pred_height
        scale_x = self.origin_width / self.pred_width

        batch_polygons = []
        batch_masks = []
        batch_scores = []
        batch_juncs = []

        for b in range(mask_pred.size(0)):
            mask_pred_per_im = cv2.resize(mask_pred[b][0].cpu().numpy(), (self.origin_width, self.origin_height))
            juncs_pred = get_pred_junctions(jloc_pred[b], joff_pred[b], mask_pred[b][0].cpu().numpy())

            juncs_pred[:,0] = juncs_pred[:,0] * scale_x
            juncs_pred[:,1] = juncs_pred[:,1] * scale_y

            polys, scores = [], []
            props = regionprops(label(mask_pred_per_im > 0.5))
            for prop in props:
                poly, juncs_sa, edges_sa, score, juncs_index = generate_polygon(prop, mask_pred_per_im, juncs_pred, 0, False)
                if juncs_sa.shape[0] == 0:
                    continue

                polys.append(poly)
                scores.append(score)
            batch_scores.append(scores)
            batch_polygons.append(polys)

            batch_masks.append(mask_pred_per_im)
            batch_juncs.append(juncs_pred)

        extra_info = {}
        output = {
            'polys_pred': batch_polygons,
            'mask_pred': batch_masks,
            'scores': batch_scores,
            'juncs_pred': batch_juncs
        }
        return output, extra_info

    def forward_train(self, image_CD, annotations_CD):
        self.train_step += 1

        device = image_CD.device
        image_A = image_CD[:, :3, :, :]
        image_B = image_CD[:, 3:6, :, :]
        image_BE = image_CD[:, 6:, :, :]

        targets_CD, metas = self.encoder(annotations_CD)

        features_A = self.backbone(image_A)
        features_B = self.backbone(image_B)
        features_BE = self.backbone(image_BE)

        """Change detection"""
        # change detection feature map subtract
        features_CD = []
        for (f_A, f_B) in zip(features_A, features_B):
            features_CD.append(torch.abs(f_A - f_B))

        # change features upsample
        x0_h, x0_w = features_CD[0].size(2), features_CD[0].size(3)
        x0 = features_CD[0]
        x1 = F.interpolate(features_CD[1], size=(x0_h, x0_w), mode='bilinear', align_corners=None)
        x2 = F.interpolate(features_CD[2], size=(x0_h, x0_w), mode='bilinear', align_corners=None)
        x3 = F.interpolate(features_CD[3], size=(x0_h, x0_w), mode='bilinear', align_corners=None)
        features_c = torch.cat([x0, x1, x2, x3], 1)
        change_features = self.last_layer(features_c)

        mask_feature_CD = self.mask_head_CD(change_features)
        mask_pred_CD = self.mask_predictor_CD(mask_feature_CD)

        """Building extraction"""
        # features upsample
        x0_h, x0_w = features_BE[0].size(2), features_BE[0].size(3)
        x0 = features_BE[0]
        x1 = F.interpolate(features_BE[1], size=(x0_h, x0_w), mode='bilinear', align_corners=None)
        x2 = F.interpolate(features_BE[2], size=(x0_h, x0_w), mode='bilinear', align_corners=None)
        x3 = F.interpolate(features_BE[3], size=(x0_h, x0_w), mode='bilinear', align_corners=None)
        features_c = torch.cat([x0, x1, x2, x3], 1)
        BE_features = self.last_layer(features_c)
        outputs = self.head(BE_features)

        mask_feature_BE = self.mask_head(BE_features)
        jloc_feature_BE = self.jloc_head(BE_features)

        # two branch outputs
        mask_pred_BE = self.mask_predictor(mask_feature_BE)
        jloc_pred_BE = self.jloc_predictor(jloc_feature_BE)

        loss_dict = {
            'loss_jloc': 0.0,
            'loss_joff': 0.0,
            'loss_mask_CD': 0.0,
            'loss_mask_BE': 0.0,
        }
        if targets_CD is not None:
            loss_dict['loss_jloc'] += self.junc_focal_loss(jloc_pred_BE, targets_CD['jloc'].squeeze(dim=1))
            loss_dict['loss_joff'] += sigmoid_l1_loss(outputs[:, :], targets_CD['joff'], -0.5, targets_CD['jloc'])
            loss_dict['loss_mask_BE'] += self.mask_loss_bce(mask_pred_BE, targets_CD['mask_BE'].squeeze(dim=1).long())
            loss_dict['loss_mask_CD'] += self.mask_loss_bce(mask_pred_CD, targets_CD['mask_CD'].squeeze(dim=1).long())
        extra_info = {}

        return loss_dict, extra_info
    
    def _make_conv(self, dim_in, dim_hid, dim_out):
        layer = nn.Sequential(
            nn.Conv2d(dim_in, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        return layer

    def _make_predictor(self, dim_in, dim_out):
        m = int(dim_in / 4)
        layer = nn.Sequential(
                    nn.Conv2d(dim_in, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, dim_out, kernel_size=1),
                )
        return layer

    def _make_decoder_conv(self,channel_list):
        conv_list = []
        for i in range(1,len(channel_list)):
            in_c = channel_list[-i] + channel_list[-i-1]
            out_c = channel_list[-i-1]
            conv_list.append(nn.Conv2d(in_c,out_c,kernel_size=1).to('cuda'))
        return conv_list






# def get_pretrained_model(cfg, dataset, device, pretrained=True):
#     PRETRAINED = {
#         'crowdai': 'https://github.com/XJKunnn/pretrained_model/releases/download/pretrained_model/crowdai_hrnet48_e100.pth',
#         'inria': 'https://github.com/XJKunnn/pretrained_model/releases/download/pretrained_model/inria_hrnet48_e5.pth'
#     }
#
#     model = New_BuildingDetector(cfg, test=True)
#     if pretrained:
#         url = PRETRAINED[dataset]
#         state_dict = torch.hub.load_state_dict_from_url(url, map_location=device, progress=True)
#         state_dict = {k[7:]:v for k,v in state_dict['model'].items() if k[0:7] == 'module.'}
#         model.load_state_dict(state_dict)
#         model = model.eval()
#         return model
#     return model
