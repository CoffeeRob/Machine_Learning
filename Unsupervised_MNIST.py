import torch
from torch import nn
from torch import optim
from torch.utils.data import random_split,DataLoader
import torchvision
from torchvision import datasets,transforms
from torch import argmax
from torch import IntTensor

SUP_CLASSIFICATION=1
UNSUP_CONTROL=2
UNSUP_CLASSIFICATION=3

class Autoencoder(nn.Module):
    def __init__(self, encoder, latent_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(nn.Linear(latent_size, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 3 * 3 * 32),
                                     nn.ReLU(),
                                     nn.Unflatten(dim=1,
                                                  unflattened_size=(32, 3, 3)),
                                     nn.ConvTranspose2d(in_channels=32,
                                                        out_channels=16,
                                                        kernel_size=3, stride=2,
                                                        output_padding=0),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(in_channels=16,
                                                        out_channels=8,
                                                        kernel_size=3,
                                                        stride=2,padding=1,
                                                        output_padding=1),
                                     nn.BatchNorm2d(8),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(in_channels=8,
                                                        out_channels=1,
                                                        kernel_size=3,
                                                        stride=2,padding=1,
                                                        output_padding=1),
                                    )

    def forward(self,batch):
        batch = self.encoder(batch)
        batch = self.decoder(batch)
        batch = torch.sigmoid(batch)
        return batch

class Classifier(nn.Module):
    def __init__(self,encoder,latent_size):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(nn.Linear(latent_size,10))

    def forward(self,batch):
        batch = self.encoder(batch)
        batch = self.fc(batch)
        return batch

def autoencoder_train(autoencoder,train_loader,autoencoder_loss,
                      autoencoder_optimiser,batch_size):
    autoencoder.train()
    losses_train=list()
    for batch in train_loader:
        if len(batch[1])%batch_size==0:
            x,y=batch
            x_hat = autoencoder(x)
            J = autoencoder_loss(x_hat,x)
            autoencoder.zero_grad()
            J.backward()
            autoencoder_optimiser.step()
            losses_train.append(J.item())
    average_loss = torch.tensor(losses_train).mean()
    return average_loss

def autoencoder_val(autoencoder,val_loader,autoencoder_loss,batch_size):
    autoencoder.eval()
    losses_val=list()
    for batch in val_loader:
        if len(batch[1])%batch_size==0:
            x,y = batch
            x_hat = autoencoder(x)
            J = autoencoder_loss(x_hat,x)
            losses_val.append(J.item())
    average_loss = torch.tensor(losses_val).mean()
    return average_loss

def classifier_train(classifier,train_loader,classifier_loss,
                     classifier_optimiser,batch_size):
    classifier.train()
    losses_train=list()
    for batch in train_loader:
        if len(batch[1])%batch_size==0:
            x,y = batch
            y_hat = classifier(x)
            J = classifier_loss(y_hat,y)
            classifier.zero_grad()
            J.backward()
            classifier_optimiser.step()
            losses_train.append(J.item())
    average_loss = torch.tensor(losses_train).mean()
    return average_loss

def classifier_val(classifier,val_loader,classifier_loss,batch_size):
    classifier.eval()
    losses_val = list()
    accuracy = 0
    for batch in val_loader:
        if len(batch[1])%batch_size == 0:
            x,y = batch
            y_hat = classifier(x)
            try:
                accuracy += (y_hat.argmax(1) == y)
            except:
                print(y_hat.argmax(1).size(),y_hat.size(),y.size())
                raise RuntimeError
            J = classifier_loss(y_hat,y)
            losses_val.append(J.item())
    average_loss = torch.tensor(losses_val).mean()
    accuracy = (accuracy/len(val_loader)).mean().item()
    print(accuracy)
    return average_loss

def experiment(instruction):
    latent_size = 32
    encoder = nn.Sequential(
                            nn.Conv2d(in_channels=1,out_channels=32,
                                      kernel_size=3,stride=1),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=32,out_channels=64,
                                      kernel_size=3,stride=1),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=(2,2)),
                            nn.Flatten(),
                            nn.Dropout(),
                            nn.Linear(12*12*64,latent_size),
                            )
    autoencoder = Autoencoder(encoder,latent_size)
    classifier = Classifier(encoder,latent_size)
    autoencoder_loss = nn.MSELoss()
    classifier_loss = nn.CrossEntropyLoss()
    autoencoder_optimiser = optim.Adam(autoencoder.parameters(),lr=1e-3)
    nb_epochs_classifier = 10

    if instruction == SUP_CLASSIFICATION:
        nb_epochs_autoencoder = 0
        classifier_optimiser = optim.Adam(classifier.parameters(),lr=1e-3)

    elif instruction == UNSUP_CONTROL:
        nb_epochs_autoencoder = 0
        classifier_optimiser = optim.Adam(classifier.fc.parameters(),lr=1e-3)

    elif instruction == UNSUP_CLASSIFICATION:
        nb_epochs_autoencoder = 10
        classifier_optimiser = optim.Adam(classifier.fc.parameters(),lr=1e-3)

    train_data=datasets.MNIST('data',train=True,download=False,
                              transform=transforms.ToTensor())
    datasetsplit = random_split(train_data,[50000,5000,5000])
    training,val,test = datasetsplit
    batch_size = 32
    train_loader = DataLoader(training,batch_size=batch_size)
    val_loader = DataLoader(val,batch_size=batch_size)
    test_loader = DataLoader(test,batch_size=batch_size)
    start_loss = 784

    for epoch in range(nb_epochs_autoencoder):
        average_loss = autoencoder_train(autoencoder,train_loader,
                                         autoencoder_loss,autoencoder_optimiser,
                                         batch_size)
        print(f'Epoch {epoch+1},autoencoder train loss:{average_loss:.2f}')
        average_loss = autoencoder_val(autoencoder,val_loader,autoencoder_loss,
                                       batch_size)
        print(f'Epoch {epoch+1},autoencoder val loss:{average_loss:.2f}')
        if average_loss >= 0.975 * start_loss:
            break
        else:
            start_loss = average_loss

    start_loss = 784
    for epoch in range(nb_epochs_classifier):
        average_loss = classifier_train(classifier,train_loader,classifier_loss,
                                        classifier_optimiser,batch_size)
        print(f'Epoch {epoch+1},classifier train loss:{average_loss:.2f}')
        average_loss=classifier_val(classifier,val_loader,classifier_loss,
                                    batch_size)
        print(f'Epoch {epoch+1},classifier val loss:{average_loss:.2f}')
        if average_loss >= 0.975 * start_loss:
            break
        else:
            start_loss = average_loss
    average_loss = classifier_val(classifier,test_loader,classifier_loss,
                                  batch_size)
    return 'Completed'

def main():
    print(experiment(SUP_CLASSIFICATION),'supervised classification')
    print(experiment(UNSUP_CONTROL),'control for unsupervised')
    print(experiment(UNSUP_CLASSIFICATION),'unsupervised classification')


if __name__=="__main__":
    main()
