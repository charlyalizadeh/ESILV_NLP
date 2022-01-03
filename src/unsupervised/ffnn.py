import torch
import torch.nn as nn
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, RMSprop
from torch.nn.functional import one_hot
import matplotlib.pyplot as plt

optimizers = {
    'Adam': Adam,
    'RMSprop': RMSprop
}
losses = {
    'crossentropy': nn.CrossEntropyLoss
}


def get_tensordataset(X, y):
    return TensorDataset(tensor(X).float(), tensor(y).to(torch.int64))


def train(loader, model, criterion, optimizer):
    model.train()
    for i, (X, y) in enumerate(loader):
        print(f"    {i + 1} / {len(loader)}", end="\r")
        X, y = X.to('cuda'), y.to('cuda')
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print()


def score(loader, model, criterion):
    with torch.no_grad():
        model.eval()
        y_true, y_pred = predict(loader, model)
        return criterion(y_pred, y_true)


def predict(loader, model):
    with torch.no_grad():
        model.eval()
        y_true = tensor([]).cuda()
        y_pred = tensor([]).cuda()
        for X, y in loader:
            X, y = X.to('cuda'), y.to('cuda')
            out = model(X)
            y_pred = torch.cat((y_pred, out), 0)
            y_true = torch.cat((y_true, y), 0)
    return y_true.to(torch.int64), y_pred


def train_ffnn(X, y,
               X_val=None, y_val=None,
               optimizer='RMSprop',
               loss='crossentropy',
               lr=4e-5,
               epochs=100,
               **kwargs):
    model = nn.Sequential(
        nn.Linear(306, 128),
        nn.ReLU(),
        #nn.Linear(256, 128),
        #nn.ReLU(),
        #nn.Linear(128, 64),
        #nn.ReLU(),
        #nn.Dropout(0.5),
        nn.Linear(128, 5),
        nn.Softmax(dim=1),
    ).cuda()

    optimizer = optimizers[optimizer](model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_dataset = get_tensordataset(X, y)
    val_dataset = get_tensordataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs):
        train(train_loader, model, criterion, optimizer)
        with torch.no_grad():
            train_loss = score(train_loader, model, criterion)
            val_loss = score(val_loader, model, criterion)
            train_losses.append(train_loss.cpu())
            val_losses.append(val_loss.cpu())
            print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), './models/ffnn')
    plt.plot(list(range(len(train_losses))), train_losses)
    plt.plot(list(range(len(val_losses))), val_losses)
    plt.show()
    return model


def predict_ffnn(X, model):
    dataset = get_tensordataset(X, torch.randn(X.shape[0], 1))
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    y_true, y_pred = predict(loader, model)
    return torch.argmax(y_pred, dim=1).cpu()
