import os
import os.path as osp

class DatasetCatalog(object):

    DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__),
                '..','..','data'))
    
    DATASETS = {
        'WHU_VectorCD_train': {
            'img_dir': 'WHU_VectorCD/train/',
            'ann_file': 'WHU_VectorCD/train/annotation.json',
        },
        'WHU_VectorCD_test': {
            'img_dir': 'WHU_VectorCD/val/',
            'ann_file': 'WHU_VectorCD/val/annotation.json',
        },
        'WHU_Unite_CD_train': {
            'img_dir_CD': 'WHU_VectorCD/train/',
            'ann_file_CD': 'WHU_VectorCD/train/annotation.json',
            'img_dir_BE': 'crowdai/train/',
            'ann_file_BE': 'crowdai/train/annotation-small.json',
        },
        'WHU_Unite_CD_test': {
            'img_dir': 'WHU_VectorCD/val/',
            'ann_file': 'WHU_VectorCD/val/annotation.json',
        },
        'LEVIR_VectorCD_train': {
            'img_dir': 'LEVIR_VectorCD/train/',
            'ann_file': 'LEVIR_VectorCD/train/annotation.json',
        },
        'LEVIR_VectorCD_test': {
            'img_dir': 'LEVIR_VectorCD/val/',
            'ann_file': 'LEVIR_VectorCD/val/annotation.json',
        },
        'LEVIR_Unite_CD_train': {
            'img_dir_CD': 'LEVIR_VectorCD/train/',
            'ann_file_CD': 'LEVIR_VectorCD/train/annotation.json',
            'img_dir_BE': 'crowdai/train/',
            'ann_file_BE': 'crowdai/train/annotation-small.json',
        },
        'LEVIR_Unite_CD_test': {
            'img_dir': 'LEVIR_VectorCD/val/',
            'ann_file': 'LEVIR_VectorCD/val/annotation.json',
        },
    }

    @staticmethod
    def get(name):
        assert name in DatasetCatalog.DATASETS
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]   # img and ann dir
        if 'train' in name:
            if 'Unite' in name:
                args = dict(
                    root_CD = osp.join(data_dir,attrs['img_dir_CD']),
                    ann_file_CD=osp.join(data_dir, attrs['ann_file_CD']),
                    root_BE = osp.join(data_dir, attrs['img_dir_BE']),
                    ann_file_BE = osp.join(data_dir, attrs['ann_file_BE']),
                )
            else:
                args = dict(
                    root_CD = osp.join(data_dir,attrs['img_dir']),
                    ann_file_CD=osp.join(data_dir, attrs['ann_file']),
                )

        else:
            args = dict(
                root=osp.join(data_dir, attrs['img_dir']),
                ann_file=osp.join(data_dir, attrs['ann_file']),
            )
        if 'train' in name:
            # return dict(factory="TrainDataset",args=args)
            # return dict(factory="Unite_TrainDataset", args=args)
            if 'Unite' in name:
                return dict(factory="CD_TrainDataset", args=args), dict(factory="BE_TrainDataset", args=args)
            else:
                return dict(factory="CD_TrainDataset", args=args), dict(factory="BE_TrainDataset", args=args)
        if 'test' in name and 'ann_file' in attrs:
            return dict(factory="TestDatasetWithAnnotations", args=args)
        raise NotImplementedError()
