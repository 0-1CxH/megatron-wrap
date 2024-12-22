from torch.utils.data import Dataset

class BatchDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.start_index = 0
    
    def get_batch(self, batch_size):
        assert self.__len__() >= batch_size
        if self.start_index + batch_size > self.__len__():
            self.start_index = 0
        return [self[idx] for idx in range(self.start_index, self.start_index+batch_size)]


class EmptyDataset(BatchDataset):
    def __len__(self):
        return 2 ** 16
    
    def __getitem__(self, index):
        return [index]
    
    


