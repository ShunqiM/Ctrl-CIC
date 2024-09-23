
from torch.utils.data import Dataset as TorchDataset

from cli.utils_registry import Registry

@Registry.register("SomeDataset")
class SomeDataset(TorchDataset):
    def __init__(self,  **kwargs):
        self.split = kwargs.pop('split')
        print(f"SomeDataset-{self.split} called")

    def __len__(self):
        return 0
    
    @property
    def classname(self):
        return self.__class__.__name__
    
    def __repr__(self):
        return f"{self.classname}-{self.split}"
    