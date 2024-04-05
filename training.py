import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import numpy as np
import lightning as L
import pyarrow.dataset as ds
import os

class MyModel(L.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int,
                 lr : float):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 3))

        self.model = nn.Sequential(*layers)
        self.norm = nn.Softmax()

        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss(weight = torch.Tensor([1, 2, 2]))

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.model(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        
        choice = torch.argmax(y_hat, dim=1)
        acc = (choice == y).sum() / len(y)
        choice_l = (choice == y)[y==0]
        acc_l = choice_l.sum() / len(choice_l)
        choice_c = (choice == y)[y==1]
        acc_c = choice_c.sum() / len(choice_c)
        choice_b = (choice == y)[y==2]
        acc_b = choice_b.sum() / len(choice_b)

        self.log('val_acc', acc)
        self.log('val_acc_l', acc_l)
        self.log('val_acc_c', acc_c)
        self.log('val_acc_b', acc_b)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        
        choice = torch.argmax(y_hat, dim=1)
        acc = (choice == y).sum() / len(y)
        choice_l = (choice == y)[y==0]
        acc_l = choice_l.sum() / len(choice_l)
        choice_c = (choice == y)[y==1]
        acc_c = choice_c.sum() / len(choice_c)
        choice_b = (choice == y)[y==2]
        acc_b = choice_b.sum() / len(choice_b)

        self.log('test_acc', acc)
        self.log('test_acc_l', acc_l)
        self.log('test_acc_c', acc_c)
        self.log('test_acc_b', acc_b)

        return loss

    def configure_optimizers(self):
        return self.optimizer

class MyData(Dataset):
    def __init__(self):
        self.df = ds.dataset("trainingdata", format='parquet').to_table().to_pandas()
        self.X = self.df[['pt', 'B', 'CvL', 'CvB']]
        self.X['pt'] = np.log(self.X['pt'])
        meanpt = self.X['pt'].mean()
        stdpt = self.X['pt'].std()
        self.X['pt'] = (self.X['pt'] - meanpt) / stdpt

        self.y = self.df['hadronFlavour'].replace([4,5], [1,2])

        class_counts = self.X

        class_counts = self.y.value_counts(normalize=True)
        class_weights = (1 / class_counts)
        self.class_reweight = class_weights[self.y]

        self.X = torch.tensor(self.X.values, dtype=torch.float32)
        self.y = torch.tensor(self.y.values, dtype=torch.int64)
        self.class_reweight = torch.tensor(self.class_reweight.values, 
                                           dtype=torch.float32)
        

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return X, y

torch.set_float32_matmul_precision('medium')

model = MyModel(4, 8, 3, lr=1e-2)

dataset = MyData()

train_set_size = int(len(dataset) * 0.8)
valid_set_size = len(dataset) - train_set_size

train_set, valid_set = random_split(dataset, 
                                    [train_set_size, valid_set_size])

train_weights = train_set.dataset.class_reweight[train_set.indices]
train_sampler = WeightedRandomSampler(weights=train_weights,
                                      num_samples=len(train_set),
                                      replacement=True)
train_loader = DataLoader(train_set, batch_size=64, 
                                     num_workers=1,
                                     shuffle=False,
                                     pin_memory=True,
                                     sampler=train_sampler)
valid_loader = DataLoader(valid_set, batch_size=10000, 
                                     num_workers=1,
                                     shuffle=False, pin_memory=True)

trainer = L.Trainer(limit_train_batches=10000, max_epochs=20,
                    limit_val_batches=50)
trainer.fit(model, train_loader, valid_loader)
trainer.test(model, valid_loader)
