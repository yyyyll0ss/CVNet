import argparse

from multiprocess import Pool
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from boundary_iou.coco_instance_api.coco import COCO as BCOCO
from boundary_iou.coco_instance_api.cocoeval import COCOeval as BCOCOeval
from CVNet.utils.metrics.polis import PolisEval
from CVNet.utils.metrics.angle_eval import ContourEval
from CVNet.utils.metrics.cIoU import compute_IoU_cIoU

def coco_eval(annFile, resFile):
    type=1
    annType = ['bbox', 'segm']
    print('Running demo for *%s* results.' % (annType[type]))

    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)

    imgIds = cocoGt.getImgIds()
    imgIds = imgIds[:]

    cocoEval = COCOeval(cocoGt, cocoDt, annType[type])
    cocoEval.params.imgIds = imgIds
    cocoEval.params.catIds = [0]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats, cocoEval.stats[1]

def boundary_eval(annFile, resFile):
    dilation_ratio = 0.02 # default settings 0.02
    cocoGt = BCOCO(annFile, get_boundary=True, dilation_ratio=dilation_ratio)
    cocoDt = cocoGt.loadRes(resFile)
    cocoEval = BCOCOeval(cocoGt, cocoDt, iouType="boundary", dilation_ratio=dilation_ratio)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats[1]

def polis_eval(annFile, resFile):
    gt_coco = COCO(annFile)
    dt_coco = gt_coco.loadRes(resFile)
    polisEval = PolisEval(gt_coco, dt_coco)
    polis = polisEval.evaluate()
    return polis

def max_angle_error_eval(annFile, resFile):
    gt_coco = COCO(annFile)
    dt_coco = gt_coco.loadRes(resFile)
    contour_eval = ContourEval(gt_coco, dt_coco)
    pool = Pool(processes=20)
    max_angle_diffs = contour_eval.evaluate(pool=pool)
    #print('Mean max tangent angle error(MTA): ', max_angle_diffs.mean())
    return max_angle_diffs.mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", default="/home/isalab301/yyl/VecCD/data/LEVIR_VectorCD/val/annotation.json")
    #parser.add_argument("--dt-file", default="/home/isalab301/yyl/VecCD/outputs/VectorCD_hrnet48_aug/VectorCD_test_aug.json")
    #parser.add_argument("--dt-file", default="/home/isalab301/yyl/VecCD/BiT_r101_infer/DP_output/VectorCD_DP.json")
    parser.add_argument("--dt-file", default="/home/isalab301/yyl/VecCD/outputs/LEVIR_VectorCD_hrnet48_aug/LEVIR_VectorCD_test_aug.json")
    parser.add_argument("--eval-type", default="boundary_iou", choices=["coco_iou", "boundary_iou", "polis", "angle", "ciou"])
    args = parser.parse_args()

    eval_type = args.eval_type
    gt_file = args.gt_file
    dt_file = args.dt_file

    _, ap50 = coco_eval(gt_file, dt_file)
    b_ap50 = boundary_eval(gt_file, dt_file)
    IoU, CIoU = compute_IoU_cIoU(dt_file, gt_file)
    polis = polis_eval(gt_file, dt_file)
    mta = max_angle_error_eval(gt_file, dt_file)

    print('coco_ap50:', ap50)
    print('boundary_ap50:', b_ap50)
    print('IoU:', IoU, 'CIOU:', CIoU)
    print('polis:', polis)
    print('MTA:',mta)

