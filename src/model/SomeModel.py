from cli.utils_registry import Registry
import torch.nn as nn
import torch


@Registry.register("SomeModel")
class SomeModel(nn.Module):
    def __init__(self, **kwargs):
        super(SomeModel, self).__init__()
        
        print('SomeModel called')
        self.some_params = nn.parameter.Parameter(torch.randn(3, 3))
    