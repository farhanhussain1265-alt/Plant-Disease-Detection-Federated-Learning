import torch

from torch.utils.data import DataLoader

from backend.clients.train import train_local

class FederatedClient:

    def __init__(
            self,
            client_id,
            dataset,
            model,
            config
    ):

        self.client_id = client_id

        self.dataset = dataset

        self.model = model

        self.config = config

        self.loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True
        )

    def train(self):

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE
        )

        criterion = (
            torch.nn.CrossEntropyLoss()
        )

        weights = train_local(
            self.model,
            self.loader,
            optimizer,
            criterion,
            self.config.DEVICE,
            self.config.LOCAL_EPOCHS
        )

        return weights