import cv2
import random
import os.path as osp
import numpy as np

from skimage import io
from pycocotools.coco import COCO
from shapely.geometry import Polygon
from torch.utils.data import Dataset
import torch
from torch.utils.data.dataloader import default_collate


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_jmaps(jmap_cut_List):
    juncs_list = []
    new_jmap_list = []
    #new_joff_list = []
    for i in jmap_cut_List:
        i = i.squeeze(axis=-1)
        h,w = i.shape[0], i.shape[1]
        scale_h, scale_w = 150 / h, 150 / w
        juncs = np.argwhere(i==1)
        #print(len(juncs))

        juncs[:,0] = np.clip(juncs[:,0] * scale_h, a_min=0, a_max=150)
        juncs[:,1] = np.clip(juncs[:,1] * scale_w, a_min=0, a_max=150)

        jmap = torch.zeros((150, 150), dtype=torch.long)
        #joff = torch.zeros((2,150,150), dtype=torch.float32)
        points = torch.tensor(juncs[:], dtype=torch.float32)
        junc_tags = torch.ones(points.shape[0], dtype=torch.long)
        xint, yint = points[:, 0].long(), points[:, 1].long()
        # off_x = points[:, 0] - xint.float() - 0.5
        # off_y = points[:, 1] - yint.float() - 0.5
        #
        jmap[xint,yint] = junc_tags
        # joff[0, yint, xint] = off_x
        # joff[1, yint, xint] = off_y

        new_jmap_list.append(jmap.detach().numpy())
        #new_joff_list.append(joff.detach().numpy())

    return juncs_list, new_jmap_list

def filter_nonzero_pixels(img):
    # 创建一个布尔掩码，将所有不全为0的像素位置设为True
    mask = np.sum(img, axis=2) != 0
    # 使用掩码筛选出不全为0的像素，并存储到新的数组中
    nonzero_pixels = img[mask]
    return nonzero_pixels

def junc_augmentation(idx_, all_images_id, coco, imgs_path):
    idx_list = np.concatenate((np.array([idx_]),np.random.randint(0,len(all_images_id),size=7)),axis=-1)
    imgs_info = []
    for i in idx_list:
        imgs_id = all_images_id[i]
        imgs_info.append(*coco.loadImgs(imgs_id))
    # print(imgs_info)

    file_name = [i['file_name'] for i in imgs_info]
    width = imgs_info[0]['width']
    height = imgs_info[0]['height']
    #print(file_name)

    # load annotations
    points_list = []
    jmap_list = []
    mask_list = []
    for i in idx_list:
        imgs_id = all_images_id[i]
        ann_ids = coco.getAnnIds(imgIds=[imgs_id])
        ann_coco = coco.loadAnns(ids=ann_ids)
        #print(ann_coco)

        seg_mask = np.zeros([width, height])
        jmap = torch.zeros((height, width), dtype=torch.long)  #device=device,
        points_img = []
        for ann_per_ins in ann_coco:
            segmentations = ann_per_ins['segmentation']
            for i, segm in enumerate(segmentations):
                segm = np.array(segm).reshape(-1, 2)  # the shape of the segm is (N,2)
                segm[:, 0] = np.clip(segm[:, 0], 0, width - 1e-4)
                segm[:, 1] = np.clip(segm[:, 1], 0, height - 1e-4)

                points = torch.tensor(segm[:-1],dtype=torch.long)
                points_img.append(points)

                junc_tags = torch.ones(points.shape[0],dtype=torch.long)
                xint, yint = points[:,0].long(), points[:,1].long()
                jmap[yint, xint] = junc_tags
                seg_mask += coco.annToMask(ann_per_ins)
        seg_mask = np.clip(seg_mask, 0, 1)

        mask_list.append(seg_mask)
        points_list.append(points_img)   # [[]]
        jmap_list.append(jmap)   # [tensor]

    # load images
    edge_sum_list = []
    img_list = []
    jmap_cut_list = []
    mask_cut_list = []
    for i in range(len(idx_list)):
        # info
        img_name = file_name[i]
        image_A = io.imread(osp.join(imgs_path, 'A', img_name)).astype(float)[:, :, :3]   # (300,300,3)
        image_B = io.imread(osp.join(imgs_path, 'B', img_name)).astype(float)[:, :, :3]   # (300,300,3)
        junctions = torch.cat(points_list[i],dim=0)
        jmap = jmap_list[i]
        mask = mask_list[i]

        # cut area
        x_min, x_max, y_min, y_max = junctions[:,0].min(), junctions[:,0].max(), junctions[:,1].min(), junctions[:,1].max()
        x_min, x_max, y_min, y_max = torch.clamp(x_min-3,min=0,max=300), torch.clamp(x_max+3,min=0,max=300),\
                                     torch.clamp(y_min-3,min=0,max=300),torch.clamp(y_max+3,min=0,max=300)
        # select proper ratio
        y_l = y_max - y_min
        x_l = x_max - x_min
        ratio = min(y_l,x_l)/max(y_l,x_l)
        if ratio > 2.5:
            continue
        cut_A = image_A[y_min:y_max, x_min:x_max, :]
        cut_B = image_B[y_min:y_max, x_min:x_max, :]
        cut_jmap = jmap[y_min:y_max, x_min:x_max, np.newaxis]
        cut_mask = mask[y_min:y_max, x_min:x_max, np.newaxis]

        cut_mask_A = cut_A * cut_mask
        cut_mask_B = cut_B * cut_mask

        nonzero_A = filter_nonzero_pixels(cut_mask_A)
        nonzero_B = filter_nonzero_pixels(cut_mask_B)
        A_color_std = np.mean(nonzero_A.std(axis=0)/nonzero_A.mean(axis=0))
        B_color_std = np.mean(nonzero_B.std(axis=0)/nonzero_B.mean(axis=0))

        if 1.2*A_color_std > B_color_std:
            img_list.append(cut_B)
            jmap_cut_list.append(cut_jmap.detach().numpy())
            mask_cut_list.append(cut_mask)

        # judge building in which picture
        img_A = cut_A.astype(np.uint8)
        #print(img_A.shape)
        gray = cv2.cvtColor(img_A, cv2.COLOR_RGB2GRAY)
        edges_A = cv2.Canny(gray, 250, 500).sum(axis=None)
        #
        img_B = cut_B.astype(np.uint8)
        gray = cv2.cvtColor(img_B, cv2.COLOR_RGB2GRAY)
        edges_B = cv2.Canny(gray, 250, 500).sum(axis=None)

        if edges_B > edges_A:
            image = cut_B
        else:
            image = cut_A

        edge_sum_list.append(np.abs(edges_A/(edges_B+1) - 1))
        img_list.append(image)
        #print(cut_jmap.detach().numpy().sum())
        jmap_cut_list.append(cut_jmap.detach().numpy())
        mask_cut_list.append(cut_mask)

    # select top4
    top_threshod = np.where(np.array(edge_sum_list) > 0.2)[0]
    # top_l = len(top_threshod)
    # top4 = []
    # if top_l >= 4:
    #     top4 = np.array((edge_sum_list)).argsort()[-4:][::-1]
    # elif top_l > 1:
    #     top_x = np.array((edge_sum_list)).argsort()[-top_l:][::-1]
    #     top4 = np.concatenate([top_x, top_x[:(4-top_l)]], axis=-1)
    # else:
    #     max_ = np.argmax(np.array(edge_sum_list))
    #     top4 = np.full_like(np.arange(4),max_)
    max_ = np.argmax(np.array(edge_sum_list))
    top4 = np.full_like(np.arange(4),max_)


    img_list = [img_list[i] for i in top4]
    jmap_cut_list = [jmap_cut_list[i] for i in top4]
    mask_cut_list = [mask_cut_list[i] for i in top4]
    # resize four images and joint
    img_list = [cv2.resize(i,dsize=(128,128),interpolation=cv2.INTER_LINEAR) for i in img_list]
    mask_cut_list = [cv2.resize(i, dsize=(128, 128), interpolation=cv2.INTER_LINEAR) for i in mask_cut_list]
    juncs_list, jmap_cut_list = get_jmaps(jmap_cut_list)
    # big_img = np.zeros((300,300,3))


    # images joint
    img_1 = np.concatenate([img_list[0],img_list[1]],1)
    img_2 = np.concatenate([img_list[2],img_list[3]],1)
    big_img = np.vstack((img_1,img_2))

    jmap_1 = np.concatenate((jmap_cut_list[0],jmap_cut_list[1]),1)
    jmap_2 = np.concatenate((jmap_cut_list[2],jmap_cut_list[3]),1)
    big_jmap = np.vstack((jmap_1,jmap_2))

    mask_1 = np.concatenate([mask_cut_list[0],mask_cut_list[1]], axis=1)
    mask_2 = np.concatenate([mask_cut_list[2],mask_cut_list[3]], axis=1)
    big_mask = np.vstack((mask_1,mask_2))

    return big_img, big_jmap, big_mask

# def build_cut_aug(images_A, images_B, targets_CD):
#     ann_file_CD = '/home/isalab301/yyl/VecCD/data/WHU_VectorCD/train/annotation.json'
#     coco = COCO(ann_file_CD)
#
#     img_list = []
#     mask_list = []
#     jmap_list = []
#     joff_list = []
#     for i in range(len(targets_CD)):
#
#         idx_list = np.concatenate((np.array([idx_]), np.random.randint(0, len(all_images_id), size=7)), axis=-1)
#         imgs_info = []
#         for i in idx_list:
#             imgs_id = all_images_id[i]
#             imgs_info.append(*coco.loadImgs(imgs_id))
#         # print(imgs_info)
#
#         file_name = [i['file_name'] for i in imgs_info]
#         width = imgs_info[0]['width']
#         height = imgs_info[0]['height']
#         # print(file_name)
#
#         # load annotations
#         points_list = []
#         jmap_list = []
#         mask_list = []
#         for i in idx_list:
#             imgs_id = all_images_id[i]
#             ann_ids = coco.getAnnIds(imgIds=[imgs_id])
#             ann_coco = coco.loadAnns(ids=ann_ids)
#             # print(ann_coco)
#
#             seg_mask = np.zeros([width, height])
#             jmap = torch.zeros((height, width), dtype=torch.long)  # device=device,
#             points_img = []
#             for ann_per_ins in ann_coco:
#                 segmentations = ann_per_ins['segmentation']
#                 for i, segm in enumerate(segmentations):
#                     segm = np.array(segm).reshape(-1, 2)  # the shape of the segm is (N,2)
#                     segm[:, 0] = np.clip(segm[:, 0], 0, width - 1e-4)
#                     segm[:, 1] = np.clip(segm[:, 1], 0, height - 1e-4)
#
#                     points = torch.tensor(segm[:-1], dtype=torch.long)
#                     points_img.append(points)
#
#                     junc_tags = torch.ones(points.shape[0], dtype=torch.long)
#                     xint, yint = points[:, 0].long(), points[:, 1].long()
#                     jmap[yint, xint] = junc_tags
#                     seg_mask += coco.annToMask(ann_per_ins)
#                 seg_mask = np.clip(seg_mask, 0, 1)
#
#             mask_list.append(seg_mask)
#             points_list.append(points_img)  # [[]]
#             jmap_list.append(jmap)  # [tensor]
#
#         img_A = images_A[i,:,:,:].cpu().detach().numpy()   # (3,512,512)
#         img_A = img_A.transpose(1, 2, 0)   # (512, 512, 3)
#         img_B = images_B[i,:,:,:].cpu().detach().numpy()   # (3,512,512)
#         img_B = img_B.transpose(1, 2, 0)   # (512, 512, 3)
#         jloc = targets_CD['jloc'][i,:,:,:].cpu().detach().numpy()   # (1, 128, 128)
#         jloc = jloc.transpose(1, 2, 0)   # (128, 128)
#         junctions = torch.tensor(np.where(jloc.squeeze()==1),device='cuda')   # (N, 2)
#         mask = targets_CD['mask_CD'][i,:,:,:].cpu().detach().numpy()   # (1, 128, 128)
#         mask = mask.transpose(1, 2, 0)   # (128, 128, 1)
#
#         # cut area
#         x_min, x_max, y_min, y_max = junctions[:,0].min(), junctions[:,0].max(), junctions[:,1].min(), junctions[:,1].max()
#         x_min, x_max, y_min, y_max = torch.clamp(x_min-3,min=0,max=300), torch.clamp(x_max+3,min=0,max=300),\
#                                      torch.clamp(y_min-3,min=0,max=300), torch.clamp(y_max+3,min=0,max=300)
#
#         cut_A = img_A[y_min:y_max, x_min:x_max, :]
#         cut_B = img_B[y_min:y_max, x_min:x_max, :]
#         cut_jloc = jloc[y_min:y_max, x_min:x_max, :]
#         cut_mask = mask[y_min:y_max, x_min:x_max, :]
#         #cut_joff = joff[:,y_min:y_max, x_min:x_max]
#
#         # judge building in which picture
#         img_A = cut_A.astype(np.uint8)
#         gray = cv2.cvtColor(img_A, cv2.COLOR_RGB2GRAY)
#         edges_A = cv2.Canny(gray, 150, 400).sum(axis=None)
#         #
#         img_B = cut_B.astype(np.uint8)
#         gray = cv2.cvtColor(img_B, cv2.COLOR_RGB2GRAY)
#         edges_B = cv2.Canny(gray, 150, 400).sum(axis=None)
#
#         if edges_B > edges_A:
#             image = cut_B
#         else:
#             image = cut_A
#
#         # resize four images and joint
#         img_cut = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
#         mask_cut = cv2.resize(cut_mask, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)
#
#         juncs = np.argwhere(cut_jloc == 1)
#         cut_jloc = cut_jloc.squeeze(axis=-1)
#         h, w = cut_jloc.shape[0], cut_jloc.shape[1]
#         scale_h, scale_w = 64 / h, 64 / w
#
#         juncs[:, 0] = np.clip(juncs[:, 0] * scale_h, a_min=0, a_max=64)
#         juncs[:, 1] = np.clip(juncs[:, 1] * scale_w, a_min=0, a_max=64)
#
#         jmap_ = torch.zeros((64, 64), dtype=torch.long, device='cuda')
#         joff_ = torch.zeros((2, 64, 64), dtype=torch.float32, device='cuda')
#         points = torch.tensor(juncs[:], dtype=torch.float32, device='cuda')
#         junc_tags = torch.ones(points.shape[0], dtype=torch.long, device='cuda')
#         xint, yint = points[:, 0].long(), points[:, 1].long()
#         off_x = points[:, 0] - xint.float() - 0.5
#         off_y = points[:, 1] - yint.float() - 0.5
#
#         jmap_[xint, yint] = junc_tags
#         joff_[0, yint, xint] = off_x
#         joff_[1, yint, xint] = off_y
#
#         img_list.append(torch.tensor(img_cut,device='cuda').permute(2,0,1))   # (3, 256, 256)
#         mask_list.append(torch.tensor(mask_cut,device='cuda').unsqueeze(0))   # (1, 64, 64)
#         jmap_list.append(jmap_.unsqueeze(0))   # (1, 64, 64)
#         joff_list.append(joff_)   # (2, 64, 64)
#
#     img_out = torch.stack(img_list, dim=0)
#     mask_out = torch.stack(mask_list, dim=0)
#     jmap_out = torch.stack(jmap_list, dim=0)
#     joff_out = torch.stack(joff_list, dim=0)
#
#     target_ = {
#         'jloc': jmap_out,
#         'joff': joff_out,
#         'mask_BE': mask_out,
#     }
#
#     return img_out, target_

def build_cut(images_A, images_B, targets_CD):
    img_list = []
    mask_list = []
    jmap_list = []
    joff_list = []
    for i in range(len(targets_CD)):
        img_A = images_A[i,:,:,:].cpu().detach().numpy()   # (3,512,512)
        img_A = img_A.transpose(1, 2, 0)   # (512, 512, 3)
        img_B = images_B[i,:,:,:].cpu().detach().numpy()   # (3,512,512)
        img_B = img_B.transpose(1, 2, 0)   # (512, 512, 3)
        jloc = targets_CD['jloc'][i,:,:,:].cpu().detach().numpy()   # (1, 128, 128)
        jloc = jloc.transpose(1, 2, 0)   # (128, 128)
        junctions = torch.tensor(np.where(jloc.squeeze()==1),device='cuda')   # (N, 2)
        mask = targets_CD['mask_CD'][i,:,:,:].cpu().detach().numpy()   # (1, 128, 128)
        mask = mask.transpose(1, 2, 0)   # (128, 128, 1)

        # cut area
        x_min, x_max, y_min, y_max = junctions[:,0].min(), junctions[:,0].max(), junctions[:,1].min(), junctions[:,1].max()
        x_min, x_max, y_min, y_max = torch.clamp(x_min-3,min=0,max=300), torch.clamp(x_max+3,min=0,max=300),\
                                     torch.clamp(y_min-3,min=0,max=300), torch.clamp(y_max+3,min=0,max=300)

        cut_A = img_A[y_min:y_max, x_min:x_max, :]
        cut_B = img_B[y_min:y_max, x_min:x_max, :]
        cut_jloc = jloc[y_min:y_max, x_min:x_max, :]
        cut_mask = mask[y_min:y_max, x_min:x_max, :]
        #cut_joff = joff[:,y_min:y_max, x_min:x_max]

        # judge building in which picture
        img_A = cut_A.astype(np.uint8)
        gray = cv2.cvtColor(img_A, cv2.COLOR_RGB2GRAY)
        edges_A = cv2.Canny(gray, 150, 400).sum(axis=None)
        #
        img_B = cut_B.astype(np.uint8)
        gray = cv2.cvtColor(img_B, cv2.COLOR_RGB2GRAY)
        edges_B = cv2.Canny(gray, 150, 400).sum(axis=None)

        if edges_B > edges_A:
            image = cut_B
        else:
            image = cut_A

        # resize four images and joint
        img_cut = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        mask_cut = cv2.resize(cut_mask, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)

        juncs = np.argwhere(cut_jloc == 1)
        cut_jloc = cut_jloc.squeeze(axis=-1)
        h, w = cut_jloc.shape[0], cut_jloc.shape[1]
        scale_h, scale_w = 64 / h, 64 / w

        juncs[:, 0] = np.clip(juncs[:, 0] * scale_h, a_min=0, a_max=64)
        juncs[:, 1] = np.clip(juncs[:, 1] * scale_w, a_min=0, a_max=64)

        jmap_ = torch.zeros((64, 64), dtype=torch.long, device='cuda')
        joff_ = torch.zeros((2, 64, 64), dtype=torch.float32, device='cuda')
        points = torch.tensor(juncs[:], dtype=torch.float32, device='cuda')
        junc_tags = torch.ones(points.shape[0], dtype=torch.long, device='cuda')
        xint, yint = points[:, 0].long(), points[:, 1].long()
        off_x = points[:, 0] - xint.float() - 0.5
        off_y = points[:, 1] - yint.float() - 0.5

        jmap_[xint, yint] = junc_tags
        joff_[0, yint, xint] = off_x
        joff_[1, yint, xint] = off_y

        img_list.append(torch.tensor(img_cut,device='cuda').permute(2,0,1))   # (3, 256, 256)
        mask_list.append(torch.tensor(mask_cut,device='cuda').unsqueeze(0))   # (1, 64, 64)
        jmap_list.append(jmap_.unsqueeze(0))   # (1, 64, 64)
        joff_list.append(joff_)   # (2, 64, 64)

    img_out = torch.stack(img_list, dim=0)
    mask_out = torch.stack(mask_list, dim=0)
    jmap_out = torch.stack(jmap_list, dim=0)
    joff_out = torch.stack(joff_list, dim=0)

    target_ = {
        'jloc': jmap_out,
        'joff': joff_out,
        'mask_BE': mask_out,
    }

    return img_out, target_

class CD_TrainDataset(Dataset):
    def __init__(self, root_CD, ann_file_CD, transform=None, rotate_f=None):
        self.root_CD = root_CD

        self.coco_CD = COCO(ann_file_CD)
        images_id_CD = self.coco_CD.getImgIds()
        self.images_CD = images_id_CD.copy()
        self.num_samples = len(self.images_CD)

        self.transform = transform
        self.rotate_f = rotate_f

    def __getitem__(self, idx_):
        # build cut aug
        image_aug, jmap, mask = junc_augmentation(idx_, self.images_CD, self.coco_CD, self.root_CD)

        # basic information of CD
        img_id_CD = self.images_CD[idx_]
        img_info_CD = self.coco_CD.loadImgs(ids=[img_id_CD])[0]
        file_name = img_info_CD['file_name']
        width = img_info_CD['width']
        height = img_info_CD['height']

        # load annotations of CD
        ann_ids_CD = self.coco_CD.getAnnIds(imgIds=[img_id_CD])
        ann_coco_CD = self.coco_CD.loadAnns(ids=ann_ids_CD)

        ann = {
            'junctions': [],
            'juncs_index': [],
            'juncs_tag': [],
            'edges_positive': [],
            'bbox': [],
            'width': width,
            'height': height,
        }
        pid = 0
        instance_id = 0
        seg_mask_CD = np.zeros([width, height])
        for ann_per_ins in ann_coco_CD:
            juncs, tags = [], []
            segmentations = ann_per_ins['segmentation']
            for i, segm in enumerate(segmentations):
                segm = np.array(segm).reshape(-1, 2)  # the shape of the segm is (N,2)
                segm[:, 0] = np.clip(segm[:, 0], 0, width - 1e-4)
                segm[:, 1] = np.clip(segm[:, 1], 0, height - 1e-4)
                points = segm[:-1]
                junc_tags = np.ones(points.shape[0])
                if i == 0:  # outline
                    poly = Polygon(points)
                    if poly.area > 0:
                        convex_point = np.array(poly.convex_hull.exterior.coords)
                        convex_index = [(p == convex_point).all(1).any() for p in points]
                        juncs.extend(points.tolist())
                        # junc_tags[convex_index] = 2  # convex point label
                        tags.extend(junc_tags.tolist())
                        ann['bbox'].append(list(poly.bounds))
                        seg_mask_CD += self.coco_CD.annToMask(ann_per_ins)
                else:
                    juncs.extend(points.tolist())
                    tags.extend(junc_tags.tolist())
                    interior_contour = segm.reshape(-1, 1, 2)
                    cv2.drawContours(seg_mask_CD, [np.int0(interior_contour)], -1, color=0, thickness=-1)

            idxs = np.arange(len(juncs))
            edges = np.stack((idxs, np.roll(idxs, 1))).transpose(1, 0) + pid

            ann['juncs_index'].extend([instance_id] * len(juncs))
            ann['junctions'].extend(juncs)
            ann['juncs_tag'].extend(tags)
            ann['edges_positive'].extend(edges.tolist())
            if len(juncs) > 0:
                instance_id += 1
                pid += len(juncs)

        seg_mask_CD = np.clip(seg_mask_CD, 0, 1)

        # load image
        image_A = io.imread(osp.join(self.root_CD, 'A', file_name)).astype(float)[:, :, :3]
        image_B = io.imread(osp.join(self.root_CD, 'B', file_name)).astype(float)[:, :, :3]
        image_CD = np.concatenate((image_A, image_B),axis=2)

        for key, _type in (
                           ['junctions', np.float32],
                           ['edges_positive', np.long],
                           ['juncs_tag', np.long],
                           ['juncs_index', np.long],
                           ['bbox', np.float32],
                           ):
            ann[key] = np.array(ann[key], dtype=_type)

        # augmentation
        if self.rotate_f:
            reminder = random.randint(0, 5)
        else:
            reminder = random.randint(0, 3)
        ann['reminder'] = reminder

        if len(ann['junctions']) > 0:
            if reminder == 1:   # horizontal flip
                image_CD = image_CD[:, ::-1, :]
                ann['junctions'][:, 0] = width - ann['junctions'][:, 0]
                ann['bbox'] = ann['bbox'][:, [2, 1, 0, 3]]
                ann['bbox'][:, 0] = width - ann['bbox'][:, 0]
                ann['bbox'][:, 2] = width - ann['bbox'][:, 2]
                seg_mask_CD = np.fliplr(seg_mask_CD)
            elif reminder == 2: # vertical flip
                image_CD = image_CD[::-1, :, :]
                ann['junctions'][:, 1] = height - ann['junctions'][:, 1]
                ann['bbox'] = ann['bbox'][:, [0, 3, 2, 1]]
                ann['bbox'][:, 1] = height - ann['bbox'][:, 1]
                ann['bbox'][:, 3] = height - ann['bbox'][:, 3]
                seg_mask_CD = np.flipud(seg_mask_CD)
            elif reminder == 3: # horizontal and vertical flip
                image_CD = image_CD[::-1, ::-1, :]
                seg_mask_CD = np.fliplr(seg_mask_CD)
                seg_mask_CD = np.flipud(seg_mask_CD)
                ann['junctions'][:, 0] = width - ann['junctions'][:, 0]
                ann['junctions'][:, 1] = height - ann['junctions'][:, 1]
                ann['bbox'] = ann['bbox'][:, [2, 3, 0, 1]]
                ann['bbox'][:, 0] = width - ann['bbox'][:, 0]
                ann['bbox'][:, 2] = width - ann['bbox'][:, 2]
                ann['bbox'][:, 1] = height - ann['bbox'][:, 1]
                ann['bbox'][:, 3] = height - ann['bbox'][:, 3]
            elif reminder == 4: # rotate 90 degree
                rot_matrix = cv2.getRotationMatrix2D((int(width/2), (height/2)), 90, 1)
                image_CD = cv2.warpAffine(image_CD, rot_matrix, (width, height))
                seg_mask_CD = cv2.warpAffine(seg_mask_CD, rot_matrix, (width, height))
                ann['junctions'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['junctions']], dtype=np.float32)
                ann['bbox'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['bbox']], dtype=np.float32)
            elif reminder == 5: # rotate 270 degree
                rot_matrix = cv2.getRotationMatrix2D((int(width / 2), (height / 2)), 270, 1)
                image_CD = cv2.warpAffine(image_CD, rot_matrix, (width, height))
                seg_mask_CD = cv2.warpAffine(seg_mask_CD, rot_matrix, (width, height))
                ann['junctions'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['junctions']], dtype=np.float32)
                ann['bbox'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['bbox']], dtype=np.float32)
            else:
                pass
            ann['mask_CD'] = seg_mask_CD
        else:
            ann['mask_CD'] = np.zeros((height, width), dtype=np.float64)
            ann['junctions'] = np.asarray([[0, 0]])
            ann['bbox'] = np.asarray([[0,0,0,0]])
            ann['juncs_tag'] = np.asarray([0])
            ann['juncs_index'] = np.asarray([0])
        # add BE label
        ann['mask_BE'] = mask
        ann['junctions'] = np.argwhere(jmap == 1)

        if self.transform is not None:
            ann_A = ann.copy()
            ann_B = ann.copy()
            ann_BE = ann.copy()
            image_A, ann_A = self.transform(image_CD[:,:,:3], ann_A)
            image_B, ann_B = self.transform(image_CD[:,:,3:], ann_B)
            image_BE, ann_BE = self.transform(image_aug, ann_BE)
            image_CD = np.concatenate((image_A, image_B, image_BE), axis=0)

            return image_CD, ann_A
            #return self.transform(image, ann)
        return image_CD, ann

    def __len__(self):
        return self.num_samples


def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            [b[1] for b in batch])

class BE_TrainDataset(Dataset):
    def __init__(self, root_CD, ann_file_CD, transform=None, rotate_f=None):
        self.root_CD = root_CD

        self.coco_CD = COCO(ann_file_CD)
        images_id = self.coco_CD.getImgIds()
        self.images_CD = images_id.copy()
        self.num_samples = len(self.images_CD)

        self.transform = transform
        self.rotate_f = rotate_f

    def __getitem__(self, idx_):
        # junctions augmentation
        image_aug, jmap, mask = junc_augmentation(idx_, self.images_CD, self.coco_CD, self.root_CD)
        # basic information
        img_id = self.images_CD[idx_]
        img_info = self.coco_CD.loadImgs(ids=[img_id])[0]
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']

        ann = {
            'junctions': np.argwhere(jmap == 1),
            'mask_BE': mask,
            'width': width,
            'height': height,
        }

        if self.transform is not None:
            image, ann = self.transform(image_aug, ann)
            return image, ann
            # return self.transform(image, ann)
        return image_aug, ann

    def __len__(self):
        return self.num_samples

