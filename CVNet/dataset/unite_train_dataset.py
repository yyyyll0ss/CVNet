import cv2
import random
import os.path as osp
import numpy as np

from skimage import io
from pycocotools.coco import COCO
from shapely.geometry import Polygon
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torch

def get_jmaps(jmap_cut_List):
    juncs_list = []
    new_jmap_list = []
    for i in jmap_cut_List:
        i = i.squeeze(axis=-1)
        h,w = i.shape[0], i.shape[1]
        scale_h, scale_w = 300 / h, 300 / w
        juncs = np.argwhere(i==1)

        juncs[:,0] = np.clip(juncs[:,0] * scale_h, a_min=0, a_max=299)
        juncs[:,1] = np.clip(juncs[:,1] * scale_w, a_min=0, a_max=299)

        jmap = torch.zeros((300, 300), dtype=torch.long)
        points = torch.tensor(juncs[:], dtype=torch.float32)
        junc_tags = torch.ones(points.shape[0], dtype=torch.long)
        xint, yint = points[:, 0].long(), points[:, 1].long()

        jmap[xint,yint] = junc_tags

        new_jmap_list.append(jmap.detach().numpy())

    return juncs_list, new_jmap_list

def junc_augmentation_unite(idx_, all_images_id, coco, imgs_path):
    idx_list = np.array([idx_])
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
        image = io.imread(osp.join(imgs_path, 'images', img_name)).astype(float)[:, :, :3]   # (300,300,3)
        junctions = torch.cat(points_list[i],dim=0)
        jmap = jmap_list[i]
        mask = mask_list[i]

        # cut area
        x_min, x_max, y_min, y_max = junctions[:,0].min(), junctions[:,0].max(), junctions[:,1].min(), junctions[:,1].max()
        x_min, x_max, y_min, y_max = torch.clamp(x_min-3,min=0,max=300), torch.clamp(x_max+3,min=0,max=300),\
                                     torch.clamp(y_min-3,min=0,max=300),torch.clamp(y_max+3,min=0,max=300)

        cut_img = image[y_min:y_max, x_min:x_max, :]
        cut_jmap = jmap[y_min:y_max, x_min:x_max, np.newaxis]
        cut_mask = mask[y_min:y_max, x_min:x_max, np.newaxis]

        img_list.append(cut_img)
        jmap_cut_list.append(cut_jmap.detach().numpy())
        mask_cut_list.append(cut_mask)

    # resize four images and joint
    img_list = [cv2.resize(i,dsize=(300,300),interpolation=cv2.INTER_LINEAR) for i in img_list]
    mask_cut_list = [cv2.resize(i, dsize=(300, 300), interpolation=cv2.INTER_LINEAR) for i in mask_cut_list]
    juncs_list, jmap_cut_list = get_jmaps(jmap_cut_list)
    # big_img = np.zeros((300,300,3))

    return img_list[0], jmap_cut_list[0], mask_cut_list[0]

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

class CD_TrainDataset(Dataset):
    def __init__(self, root_CD, ann_file_CD, root_BE, ann_file_BE, transform=None, rotate_f=None):
        self.root_CD = root_CD

        self.coco_CD = COCO(ann_file_CD)
        images_id_CD = self.coco_CD.getImgIds()
        self.images_CD = images_id_CD.copy()
        self.num_samples = len(self.images_CD)

        self.transform = transform
        self.rotate_f = rotate_f

    def __getitem__(self, idx_):
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

        for key, _type in (['junctions', np.float32],
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

        if self.transform is not None:
            ann_A = ann.copy()
            ann_B = ann.copy()
            image_A, ann_A = self.transform(image_CD[:,:,:3], ann_A)
            image_B, ann_B = self.transform(image_CD[:,:,3:], ann_B)
            image_CD = np.concatenate((image_A, image_B), axis=0)
            return image_CD, ann_A
            #return self.transform(image, ann)
        return image_CD, ann

    def __len__(self):
        return self.num_samples


def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            [b[1] for b in batch])

class BE_TrainDataset(Dataset):
    def __init__(self, root_CD, ann_file_CD, root_BE, ann_file_BE, transform=None, rotate_f=None):
        self.root = root_BE

        self.coco = COCO(ann_file_BE)
        images_id = self.coco.getImgIds()
        self.images = images_id.copy()
        self.num_samples = len(self.images)

        self.transform = transform
        self.rotate_f = rotate_f

    def __getitem__(self, idx_):
        # # junction aug
        # image_aug, jmap, mask = junc_augmentation_unite(idx_, self.images, self.coco, self.root)
        # basic information
        img_id = self.images[idx_]
        img_info = self.coco.loadImgs(ids=[img_id])[0]
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']

        # load annotations
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_coco = self.coco.loadAnns(ids=ann_ids)

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
        seg_mask = np.zeros([width, height])
        for ann_per_ins in ann_coco:
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
                        #junc_tags[convex_index] = 2  # convex point label
                        tags.extend(junc_tags.tolist())
                        ann['bbox'].append(list(poly.bounds))
                        seg_mask += self.coco.annToMask(ann_per_ins)
                else:
                    juncs.extend(points.tolist())
                    tags.extend(junc_tags.tolist())
                    interior_contour = segm.reshape(-1, 1, 2)
                    cv2.drawContours(seg_mask, [np.int0(interior_contour)], -1, color=0, thickness=-1)

            idxs = np.arange(len(juncs))
            edges = np.stack((idxs, np.roll(idxs, 1))).transpose(1, 0) + pid

            ann['juncs_index'].extend([instance_id] * len(juncs))
            ann['junctions'].extend(juncs)
            ann['juncs_tag'].extend(tags)
            ann['edges_positive'].extend(edges.tolist())
            if len(juncs) > 0:
                instance_id += 1
                pid += len(juncs)

        seg_mask = np.clip(seg_mask, 0, 1)

        # load image
        image = io.imread(osp.join(self.root,'images', file_name)).astype(float)[:, :, :3]

        for key, _type in (['junctions', np.float32],
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

        # add BE junction

        if len(ann['junctions']) > 0:
            if reminder == 1:  # horizontal flip
                image = image[:, ::-1, :]
                ann['junctions'][:, 0] = width - ann['junctions'][:, 0]
                ann['bbox'] = ann['bbox'][:, [2, 1, 0, 3]]
                ann['bbox'][:, 0] = width - ann['bbox'][:, 0]
                ann['bbox'][:, 2] = width - ann['bbox'][:, 2]
                seg_mask = np.fliplr(seg_mask)
            elif reminder == 2:  # vertical flip
                image = image[::-1, :, :]
                ann['junctions'][:, 1] = height - ann['junctions'][:, 1]
                ann['bbox'] = ann['bbox'][:, [0, 3, 2, 1]]
                ann['bbox'][:, 1] = height - ann['bbox'][:, 1]
                ann['bbox'][:, 3] = height - ann['bbox'][:, 3]
                seg_mask = np.flipud(seg_mask)
            elif reminder == 3:  # horizontal and vertical flip
                image = image[::-1, ::-1, :]
                seg_mask = np.fliplr(seg_mask)
                seg_mask = np.flipud(seg_mask)
                ann['junctions'][:, 0] = width - ann['junctions'][:, 0]
                ann['junctions'][:, 1] = height - ann['junctions'][:, 1]
                ann['bbox'] = ann['bbox'][:, [2, 3, 0, 1]]
                ann['bbox'][:, 0] = width - ann['bbox'][:, 0]
                ann['bbox'][:, 2] = width - ann['bbox'][:, 2]
                ann['bbox'][:, 1] = height - ann['bbox'][:, 1]
                ann['bbox'][:, 3] = height - ann['bbox'][:, 3]
            elif reminder == 4:  # rotate 90 degree
                rot_matrix = cv2.getRotationMatrix2D((int(width / 2), (height / 2)), 90, 1)
                image = cv2.warpAffine(image, rot_matrix, (width, height))
                seg_mask = cv2.warpAffine(seg_mask, rot_matrix, (width, height))
                ann['junctions'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['junctions']],
                                              dtype=np.float32)
                ann['bbox'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['bbox']], dtype=np.float32)
            elif reminder == 5:  # rotate 270 degree
                rot_matrix = cv2.getRotationMatrix2D((int(width / 2), (height / 2)), 270, 1)
                image = cv2.warpAffine(image, rot_matrix, (width, height))
                seg_mask = cv2.warpAffine(seg_mask, rot_matrix, (width, height))
                ann['junctions'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['junctions']],
                                              dtype=np.float32)
                ann['bbox'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['bbox']], dtype=np.float32)
            else:
                pass
            ann['mask_BE'] = seg_mask
        else:
            ann['mask_BE'] = np.zeros((height, width), dtype=np.float64)
            ann['junctions'] = np.asarray([[0, 0]])
            ann['bbox'] = np.asarray([[0, 0, 0, 0]])
            ann['juncs_tag'] = np.asarray([0])
            ann['juncs_index'] = np.asarray([0])

        # add BE label
        # ann['mask_BE'] = mask
        # ann['junctions'] = np.argwhere(jmap == 1)


        if self.transform is not None:
            image, ann = self.transform(image, ann)
            return image, ann
            # return self.transform(image, ann)
        return image, ann

    def __len__(self):
        return self.num_samples

