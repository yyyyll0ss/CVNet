import torch
import functools

if torch.__version__.startswith('0'):
    from CVNet.utils.bn import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
    BatchNorm2d_class = InPlaceABNSync
    relu_inplace = False
else:
    BatchNorm2d_class = BatchNorm2d = torch.nn.BatchNorm2d
    relu_inplace = True