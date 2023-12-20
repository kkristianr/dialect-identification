import lightning as L
from torch.utils.data import DataLoader, random_split


class DataModule(L.LightningDataModule):
    def __init__(self, train, test, batch_size=32, shuffle_train=True):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.dataset = train
        self.test_dataset = test


        # Split the dataset into train, val, and test
        dataset_size = len(self.dataset)
        train_size = int(0.8 * dataset_size)
        val_size = (dataset_size - train_size)
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        '''Returns the DataLoader for the training set'''
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
        )

    def val_dataloader(self):
        '''Returns the DataLoader for the validation set'''
        return DataLoader(
            self.val_dataset,
            batch_size=256,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        '''Returns the DataLoader for the test set'''
        return DataLoader(
            self.test_dataset,
            batch_size=256,
            shuffle=False,
            drop_last=True,
        )
