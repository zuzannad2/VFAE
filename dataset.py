from torch.utils.data import Dataset

class DictionaryDataset(Dataset):
    def __init__(self, dict):
        self.dict = dict

    def __len__(self):
        return self.dict['x'].shape[0]

    def __getitem__(self, index):
        return {'x': self.dict['x'][index], 's': self.dict['s'][index], 'y': self.dict['y'][index]}