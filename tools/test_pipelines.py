import os
import os.path as osp
import json
import torch
import logging
import numpy as np
import scipy
import scipy.ndimage

from PIL import Image
from tqdm import tqdm
from skimage import io
from tools.evaluation import coco_eval, boundary_eval, polis_eval
from CVNet.utils.comm import to_single_device
from CVNet.utils.polygon import generate_polygon
from CVNet.utils.visualizer import viz_inria, save_viz
from CVNet.dataset import build_test_dataset
from CVNet.dataset.build import build_transform
from CVNet.utils.polygon import juncs_in_bbox

from shapely.geometry import Polygon
from skimage.measure import label, regionprops
import imageio
import cv2
from math import floor

from pycocotools import mask as coco_mask

###############
import torch
import torch.nn.functional as F
import cv2

def feature_vis(feats, savedir, filename): # feaats形状: [b,c,h,w] tensor
     feats = feats.unsqueeze(0)
     output_shape = (300,300) # 输出形状
     channel_mean,_ = torch.max(feats,dim=1,keepdim=True) # channel_max,_ = torch.max(feats,dim=1,keepdim=True)
     channel_mean = F.interpolate(channel_mean, size=output_shape, mode='bilinear', align_corners=False)
     channel_mean = channel_mean.squeeze(0).squeeze(0).cpu().numpy() # 四维压缩为二维
     channel_mean = (((channel_mean - np.min(channel_mean))/(np.max(channel_mean)-np.min(channel_mean)))*255).astype(np.uint8)
     if not os.path.exists(osp.join(savedir,'max_feature_vis')):
         os.makedirs(osp.join(savedir,'max_feature_vis'))
     channel_mean = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET)
     cv2.imwrite(osp.join(savedir, 'max_feature_vis', filename),channel_mean)

##################

def poly_to_bbox(poly):
    """
    input: poly----2D array with points
    """
    lt_x = np.min(poly[:,0])
    lt_y = np.min(poly[:,1])
    w = np.max(poly[:,0]) - lt_x
    h = np.max(poly[:,1]) - lt_y
    return [float(lt_x), float(lt_y), float(w), float(h)]

def generate_coco_ann(polys, scores, img_id):
    sample_ann = []
    for i, polygon in enumerate(polys):
        if polygon.shape[0] < 3:
            continue

        vec_poly = polygon.ravel().tolist()
        poly_bbox = poly_to_bbox(polygon)
        ann_per_building = {
                'image_id': img_id,
                'category_id': 0,
                'segmentation': [vec_poly],
                'bbox': poly_bbox,
                'score': float(scores[i]),
            }
        sample_ann.append(ann_per_building)

    return sample_ann

def generate_coco_ann_DP(polys, img_id):
    sample_ann = []
    for i, polygon in enumerate(polys):
        if polygon.shape[0] < 3:
            continue

        vec_poly = polygon.ravel().tolist()
        poly_bbox = poly_to_bbox(polygon)
        ann_per_building = {
                'image_id': int(img_id.split('.')[0]),
                'category_id': 0,
                'segmentation': [vec_poly],
                'bbox': poly_bbox,
                'score': float(1),
            }
        sample_ann.append(ann_per_building)

    return sample_ann

def generate_coco_mask(mask, img_id):
    sample_ann = []
    props = regionprops(label(mask > 0.50))
    for prop in props:
        if ((prop.bbox[2] - prop.bbox[0]) > 0) & ((prop.bbox[3] - prop.bbox[1]) > 0):
            prop_mask = np.zeros_like(mask, dtype=np.uint8)
            prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1

            masked_instance = np.ma.masked_array(mask, mask=(prop_mask != 1))
            score = masked_instance.mean()
            encoded_region = coco_mask.encode(np.asfortranarray(prop_mask))
            ann_per_building = {
                'image_id': img_id,
                'category_id': 100,
                'segmentation': {
                    "size": encoded_region["size"],
                    "counts": encoded_region["counts"].decode()
                },
                'score': float(score),
            }
            sample_ann.append(ann_per_building)

    return sample_ann


class TestPipeline():
    def __init__(self, cfg, eval_type='coco_iou'):
        self.cfg = cfg
        self.device = cfg.MODEL.DEVICE
        self.output_dir = cfg.OUTPUT_DIR
        self.dataset_name = cfg.DATASETS.TEST[0]
        self.eval_type = eval_type
        
        self.gt_file = ''
        self.dt_file = ''
    
    def test(self, model):
        if 'LEVIR' in self.dataset_name or 'WHU' in self.dataset_name:
            self.test_on_VectorCD(model, self.dataset_name)

    def cd_test(self,model):
        logger = logging.getLogger("testing")
        logger.info('Testing on {} dataset'.format(self.dataset_name))

        results = []
        mask_results = []
        test_dataset, gt_file, img_path = build_test_dataset(self.cfg)
        for i, (images, annotations) in enumerate(tqdm(test_dataset)):
            with torch.no_grad():
                output, _ = model(images.to(self.device), to_single_device(annotations, self.device),[],[])
                output = to_single_device(output, 'cpu')

            batch_size = images.size(0)
            batch_scores = output['scores']
            batch_polygons = output['polys_pred']
            batch_masks = output['mask_pred']
            batch_junctions = output['juncs_pred']
            #batch_junctions_A = output['junctions_A']
            #batch_junctions_B = output['junctions_B']

            for b in range(batch_size):
                filename = annotations[b]['filename']
                img_id = int(filename[:-4])

                mask_pred = batch_masks[b]

                #output pred_mask
                file_path = osp.join(self.output_dir, 'mask_viz')
                if not osp.exists(file_path):
                    os.makedirs(file_path)
                impath = osp.join(file_path, filename).replace('tif','png')
                imageio.imwrite(impath, np.where(mask_pred > 0.5, 1, 0))   #*255

        #self.eval()


    def eval(self):
        logger = logging.getLogger("testing")
        logger.info('Evalutating on {}'.format(self.eval_type))
        if self.eval_type == 'coco_iou':
            coco_eval(self.gt_file, self.dt_file)
        elif self.eval_type == 'boundary_iou':
            boundary_eval(self.gt_file, self.dt_file)
        elif self.eval_type == 'polis':
            polis_eval(self.gt_file, self.dt_file)

    def test_on_crowdai(self, model, dataset_name):
        logger = logging.getLogger("testing")
        logger.info('Testing on {} dataset'.format(dataset_name))

        results = []
        mask_results = []
        test_dataset, gt_file, img_path = build_test_dataset(self.cfg)
        for i, (images, annotations) in enumerate(tqdm(test_dataset)):
            with torch.no_grad():
                output, _ = model(images.to(self.device), to_single_device(annotations, self.device))
                output = to_single_device(output, 'cpu')

            batch_size = images.size(0)
            batch_scores = output['scores']
            batch_polygons = output['polys_pred']
            batch_masks = output['mask_pred']
            batch_junctions = output['juncs_pred']

            for b in range(batch_size):
                filename = annotations[b]['filename']
                img_id = int(filename[:-4])

                scores = batch_scores[b]
                polys = batch_polygons[b]
                mask_pred = batch_masks[b]
                junctions = batch_junctions[b]

                # add result vis
                root_path = img_path
                img = io.imread(osp.join(root_path, 'images', filename))
                save_path = osp.join(self.output_dir, 'viz_BE')
                save_viz(image=img, polys=polys, save_path=save_path, filename=filename)

                # output pred_mask
                file_path = osp.join(self.output_dir, 'mask_viz_BE')
                if not osp.exists(file_path):
                    os.makedirs(file_path)
                impath = osp.join(file_path, filename).replace('tif', 'png')
                imageio.imwrite(impath, np.where(mask_pred > 0.5, 1, 0))

                # output pred_juncs
                file_path = osp.join(self.output_dir, 'juncs_viz_BE')
                if not osp.exists(file_path):
                    os.makedirs(file_path)
                impath = osp.join(file_path, filename).replace('tif', 'png')
                junc_img = np.zeros((mask_pred.shape[0], mask_pred.shape[1], 3), np.uint8)
                for i in junctions:
                    cv2.circle(junc_img, (floor(i[0]), floor(i[1])), 1, (255, 255, 255),
                               4)  # point_size,point_color,thickness
                imageio.imwrite(impath, junc_img)

                image_result = generate_coco_ann(polys, scores, img_id)
                if len(image_result) != 0:
                    results.extend(image_result)

                image_masks = generate_coco_mask(mask_pred, img_id)
                if len(image_masks) != 0:
                    mask_results.extend(image_masks)

        dt_file = osp.join(self.output_dir, '{}.json'.format(dataset_name))
        logger.info('Writing the results of the {} dataset into {}'.format(dataset_name,
                                                                           dt_file))
        with open(dt_file, 'w') as _out:
            json.dump(results, _out)

        self.gt_file = gt_file
        self.dt_file = dt_file
        self.eval()

        dt_file = osp.join(self.output_dir, '{}_mask.json'.format(dataset_name))
        logger.info('Writing the results of the {} dataset into {}'.format(dataset_name,
                                                                           dt_file))
        with open(dt_file, 'w') as _out:
            json.dump(mask_results, _out)

        self.gt_file = gt_file
        self.dt_file = dt_file
        self.eval()

    def test_on_inria(self, model, dataset_name):
        logger = logging.getLogger("testing")
        logger.info('Testing on {} dataset'.format(dataset_name))

        IM_PATH = './data/inria/raw/test/images/'
        if not os.path.exists(os.path.join(self.output_dir, 'seg')):
            os.makedirs(os.path.join(self.output_dir, 'seg'))
        transform = build_transform(self.cfg)
        test_imgs = os.listdir(IM_PATH)
        for image_name in tqdm(test_imgs, desc='Total processing'):
            file_name = image_name
            
            impath = osp.join(IM_PATH, file_name)
            image = io.imread(impath)
            
            # crop the original inria image(5000x5000) into small images(512x512)
            h_stride, w_stride = 400, 400
            h_crop, w_crop = 512, 512
            h_img, w_img, _ = image.shape
            h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
            w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
            pred_whole_img = np.zeros([h_img, w_img], dtype=np.float32)
            count_mat = np.zeros([h_img, w_img])
            # weight = np.zeros([h_img, w_img])
            juncs_whole_img = []
            
            patch_weight = np.ones((h_crop + 2, w_crop + 2))
            patch_weight[0,:] = 0
            patch_weight[-1,:] = 0
            patch_weight[:,0] = 0
            patch_weight[:,-1] = 0
            
            patch_weight = scipy.ndimage.distance_transform_edt(patch_weight)
            patch_weight = patch_weight[1:-1,1:-1]

            for h_idx in tqdm(range(h_grids), leave=False, desc='processing on per image'):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    
                    crop_img = image[y1:y2, x1:x2, :]
                    crop_img_tensor = transform(crop_img.astype(float))[None].to(self.device)
                    
                    meta = {
                        'filename': impath,
                        'height': crop_img.shape[0],
                        'width': crop_img.shape[1],
                        'pos': [x1, y1, x2, y2]
                    }

                    with torch.no_grad():
                        output, _ = model(crop_img_tensor, [meta])
                        output = to_single_device(output, 'cpu')

                    juncs_pred = output['juncs_pred'][0]
                    juncs_pred += [x1, y1]
                    juncs_whole_img.extend(juncs_pred.tolist())
                    mask_pred = output['mask_pred'][0]
                    mask_pred *= patch_weight
                    pred_whole_img += np.pad(mask_pred,
                                        ((int(y1), int(pred_whole_img.shape[0] - y2)),
                                        (int(x1), int(pred_whole_img.shape[1] - x2))))
                    count_mat[y1:y2, x1:x2] += patch_weight

            juncs_whole_img = np.array(juncs_whole_img)
            pred_whole_img = pred_whole_img / count_mat

            # match junction and seg results
            polygons = []
            props = regionprops(label(pred_whole_img > 0.5))
            for prop in tqdm(props, leave=False, desc='polygon generation'):
                y1, x1, y2, x2 = prop.bbox
                bbox = [x1, y1, x2, y2]
                select_juncs = juncs_in_bbox(bbox, juncs_whole_img, expand=8)
                poly, juncs_sa, _, _, juncs_index = generate_polygon(prop, pred_whole_img, \
                                                                          select_juncs, pid=0, test_inria=True)
                if juncs_sa.shape[0] == 0:
                    continue
                
                if len(juncs_index) == 1:
                    polygons.append(Polygon(poly))
                else:
                    poly_ = Polygon(poly[juncs_index[0]], \
                                    [poly[idx] for idx in juncs_index[1:]])
                    polygons.append(poly_)

            # visualize
            # viz_inria(image, polygons, self.cfg.OUTPUT_DIR, file_name)

            # save seg results
            #im = Image.fromarray(pred_whole_img)
            im = Image.fromarray(((pred_whole_img >0.5) * 255).astype(np.uint8), 'L')
            im.save(os.path.join(self.output_dir, 'seg', file_name))

    def test_on_VectorCD(self, model, dataset_name):
        logger = logging.getLogger("testing")
        logger.info('Testing on {} dataset'.format(dataset_name))

        results = []
        mask_results = []
        test_dataset, gt_file, img_path = build_test_dataset(self.cfg)
        for i, (images, annotations) in enumerate(tqdm(test_dataset)):
            with torch.no_grad():
                output, _ = model(images.to(self.device), to_single_device(annotations, self.device))
                output = to_single_device(output, 'cpu')

            batch_size = images.size(0)
            batch_scores = output['scores']
            batch_polygons = output['polys_pred']
            batch_masks = output['mask_pred']
            batch_junctions = output['juncs_pred']


            for b in range(batch_size):
                filename = annotations[b]['filename']
                img_id = int(filename[:-4])

                scores = batch_scores[b]
                polys = batch_polygons[b]
                mask_pred = batch_masks[b]
                junctions = batch_junctions[b]
                # feature_ = feature[b]

                # feature_vis(feature_, self.output_dir, filename)

                # output pred_mask
                root_path = img_path
                img = io.imread(osp.join(root_path,'B', filename))
                save_path = osp.join(self.output_dir, 'viz')
                save_viz(image=img, polys=polys, save_path=save_path, filename=filename)

                #output pred_mask
                file_path = osp.join(self.output_dir, 'mask_viz')
                if not osp.exists(file_path):
                    os.makedirs(file_path)
                impath = osp.join(file_path, filename).replace('tif','png')

                imageio.imwrite(impath, np.uint8(255*np.where(mask_pred > 0.5, 1, 0)))   #

                #output pred_juncs
                file_path = osp.join(self.output_dir, 'juncs_viz')
                if not osp.exists(file_path):
                    os.makedirs(file_path)
                impath = osp.join(file_path, filename).replace('tif','png')
                #junc_img = io.imread(osp.join(root_path, 'B', filename))
                junc_img = np.zeros((mask_pred.shape[0], mask_pred.shape[1], 3),np.uint8)   #
                #print(len(junctions))
                for i in junctions:
                    cv2.circle(junc_img, (floor(i[0]), floor(i[1])), 1, (255,255,255), 4)   #point_size,point_color,thickness
                imageio.imwrite(impath,junc_img)

                image_result = generate_coco_ann(polys, scores, img_id)
                if len(image_result) != 0:
                    results.extend(image_result)

                image_masks = generate_coco_mask(mask_pred, img_id)
                if len(image_masks) != 0:
                    mask_results.extend(image_masks)

        dt_file = osp.join(self.output_dir, '{}.json'.format(dataset_name))
        logger.info('Writing the results of the {} dataset into {}'.format(dataset_name,
                                                                           dt_file))
        with open(dt_file, 'w') as _out:
            json.dump(results, _out)

        self.gt_file = gt_file
        self.dt_file = dt_file
        self.eval()

        dt_file = osp.join(self.output_dir, '{}_mask.json'.format(dataset_name))
        logger.info('Writing the results of the {} dataset into {}'.format(dataset_name,
                                                                           dt_file))
        with open(dt_file, 'w') as _out:
            json.dump(mask_results, _out)

        self.gt_file = gt_file
        self.dt_file = dt_file
        self.eval()


    def test_on_VectorCD_new(self, model, dataset_name):
        logger = logging.getLogger("testing")
        logger.info('Testing on {} dataset'.format(dataset_name))

        results = []
        mask_results = []
        test_dataset, gt_file, img_path = build_test_dataset(self.cfg)
        for i, (images, annotations) in enumerate(tqdm(test_dataset)):
            with torch.no_grad():
                output, _ = model(images.to(self.device), to_single_device(annotations, self.device))
                output = to_single_device(output, 'cpu')

            batch_size = images.size(0)
            batch_scores = output['scores']
            batch_polygons = output['polys_pred']
            batch_masks = output['mask_pred']
            batch_junctions = output['juncs_pred']
            batch_junctions_A = output['junctions_A']
            batch_junctions_B = output['junctions_B']

            for b in range(batch_size):
                filename = annotations[b]['filename']
                img_id = int(filename[:-4])

                scores = batch_scores[b]
                polys = batch_polygons[b]
                mask_pred = batch_masks[b]
                junctions = batch_junctions[b]
                junctions_A = batch_junctions_A[b]   ###
                junctions_B = batch_junctions_B[b]   ###

                # output pred_mask
                root_path = img_path
                img = io.imread(osp.join(root_path,'B', filename))
                save_viz(image=img, polys=polys, save_path=self.output_dir, filename=filename)

                #output pred_mask
                file_path = osp.join(self.output_dir, 'mask_viz')
                if not osp.exists(file_path):
                    os.makedirs(file_path)
                impath = osp.join(file_path, filename).replace('tif','png')

                imageio.imwrite(impath, np.where(mask_pred > 0.5, 1, 0))   #*255

                #output pred_juncs
                file_path = osp.join(self.output_dir, 'juncs_viz')
                if not osp.exists(file_path):
                    os.makedirs(file_path)
                impath = osp.join(file_path, filename).replace('tif','png')
                junc_img = io.imread(osp.join(root_path, 'B', filename))
                #junc_img = np.zeros((512, 512, 3),np.uint8)   #mask_pred.shape[0], mask_pred.shape[1]
                #print(len(junctions))
                for i in junctions:
                    cv2.circle(junc_img, (floor(i[0]), floor(i[1])), 1, (255,0,0), 4)   #point_size,point_color,thickness
                imageio.imwrite(impath,junc_img)

                # output pred_juncs_A
                file_path = osp.join(self.output_dir, 'juncs_A_viz')
                if not osp.exists(file_path):
                    os.makedirs(file_path)
                impath = osp.join(file_path, filename).replace('tif', 'png')
                junc_img = io.imread(osp.join(root_path, 'A', filename))
                #junc_img = np.zeros((mask_pred.shape[0], mask_pred.shape[1], 3), np.uint8)
                for i in junctions_A:
                    cv2.circle(junc_img, (floor(i[0]), floor(i[1])), 1, (255, 255, 255),
                               4)  # point_size,point_color,thickness
                imageio.imwrite(impath, junc_img)

                # output pred_juncs_B
                file_path = osp.join(self.output_dir, 'juncs_B_viz')
                if not osp.exists(file_path):
                    os.makedirs(file_path)
                impath = osp.join(file_path, filename).replace('tif', 'png')
                junc_img = io.imread(osp.join(root_path, 'B', filename))
                #junc_img = np.zeros((mask_pred.shape[0], mask_pred.shape[1], 3), np.uint8)
                for i in junctions_B:
                    cv2.circle(junc_img, (floor(i[0]), floor(i[1])), 1, (255, 255, 255),
                               4)  # point_size,point_color,thickness
                imageio.imwrite(impath, junc_img)

                image_result = generate_coco_ann(polys, scores, img_id)
                if len(image_result) != 0:
                    results.extend(image_result)

                image_masks = generate_coco_mask(mask_pred, img_id)
                if len(image_masks) != 0:
                    mask_results.extend(image_masks)

        dt_file = osp.join(self.output_dir, '{}.json'.format(dataset_name))
        logger.info('Writing the results of the {} dataset into {}'.format(dataset_name,
                                                                           dt_file))
        with open(dt_file, 'w') as _out:
            json.dump(results, _out)

        self.gt_file = gt_file
        self.dt_file = dt_file
        self.eval()

        dt_file = osp.join(self.output_dir, '{}_mask.json'.format(dataset_name))
        logger.info('Writing the results of the {} dataset into {}'.format(dataset_name,
                                                                           dt_file))
        with open(dt_file, 'w') as _out:
            json.dump(mask_results, _out)

        self.gt_file = gt_file
        self.dt_file = dt_file
        self.eval()