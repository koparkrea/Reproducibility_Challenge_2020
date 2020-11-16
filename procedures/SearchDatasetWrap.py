import torch, copy, random
import torch.utils.data as data

class SearchDataset(data.Dataset):
    def __init__(self, name, data, train_split, valid_split, check = True):
        self.datasetname = name
        
        self.mode_str = 'V1'
        self.data = data
        self.train_split = train_split.copy()
        self.valid_split = valid_split.copy()

        self.length = len(self.train_split)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        assert index >= 0 and index < self.length 
        train_index = self.train_split[index]
        valid_index = random.choice(self.valid_split)

        if self.mode_str == 'V1':
            train_image, train_label = self.data[train_index]
            valid_image, valid_label = self.data[valid_index]
        
        else:
            raise ValueError('invalid mode str = {:}'.format(self.mode_str))
        return train_image, train_label, valid_image, valid_label

        
