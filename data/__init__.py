from torch.utils.data import DataLoader
from .data import CounterfactualForecastDataset

datasets = {
    "CF": CounterfactualForecastDataset,
}

class MixDataset:
    def __init__(self, configs):
        self.configs = configs
        self.dataset = datasets[configs["name"]](**configs)

    def get_loader(self, split, batch_size, shuffle=True, num_workers=1, include_self=False):
        loader = DataLoader(
            dataset=self.dataset.get_split(split, include_self), 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers)
        return loader