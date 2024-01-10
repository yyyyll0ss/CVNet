import os
import argparse
import logging

from CVNet.config import cfg
#from CVNet.new_detector_multihead import New_BuildingDetector
from CVNet.new_detector_aug_v2 import BuildingDetector_aug_v2
from CVNet.utils.logger import setup_logger
from CVNet.utils.checkpoint import DetectronCheckpointer
from tools.test_pipelines import TestPipeline
from CVNet.new_detector_aug_v2 import BuildingDetector_aug_v2
import torch
from thop import profile
import torch
from calflops import calculate_flops

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')

    parser.add_argument("--config-file",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        default='config-files/LEVIR_VectorCD_hrnet48.yaml',
                        )

    parser.add_argument("--eval-type",
                        type=str,
                        help="evalutation type for the test results",
                        default="coco_iou",
                        choices=["coco_iou", "boundary_iou", "polis"]
                        )

    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )

    args = parser.parse_args()

    return args


def test(cfg, args):
    logger = logging.getLogger("testing")
    device = cfg.MODEL.DEVICE
    device = 'cuda:0'
    model = BuildingDetector_aug_v2(cfg, test=True)
    model = model.to(device)

    if args.config_file is not None:
        checkpointer = DetectronCheckpointer(cfg,
                                             model,
                                             save_dir=cfg.OUTPUT_DIR,
                                             save_to_disk=True,
                                             logger=logger)
        _ = checkpointer.load()
        model = model.eval()

    input = torch.randn(1,6,300,300).to(device)
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=(1,6,300,300),
                                          output_as_string=True,
                                          output_precision=4)
    print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))
    # flops, params = profile(model,(input,_))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    # compute run time
    iterations = 100  # 重复计算的轮次
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # GPU预热
    for _ in range(50):
        _ = model(input,_)

    # 测速
    times = torch.zeros(iterations)  # 存储每轮iteration的时间
    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            _ = model(input,_)
            ender.record()
            # 同步GPU时间
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # 计算时间
            times[iter] = curr_time
            # print(curr_time)

    mean_time = times.mean().item()
    print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))


if __name__ == "__main__":
    args = parse_args()
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    else:
        cfg.OUTPUT_DIR = 'outputs/default'
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    logger = setup_logger('testing', output_dir)
    logger.info(args)
    if args.config_file is not None:
        logger.info("Loaded configuration file {}".format(args.config_file))
    else:
        logger.info("Loaded the default configuration for testing")

    test(cfg, args)