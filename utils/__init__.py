import os
from datetime import datetime
import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


class Timer:
    """Record multiple running times."""

    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


def evaluate_model(net, loss_fn, device, dataloader):
    """
    :param net: 模型
    :param loss_fn: 损失函数
    :param device: 设备
    :param dataloader: 使用dataloader来降低内存
    :return: 正确率， 损失
    """
    correct = 0.0
    loss = 0.0
    num = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            loss += loss_fn(y_hat, y).sum()
            y_hat = torch.argmax(y_hat, dim=1)
            correct += (y_hat == y.reshape(y_hat.shape, -1)).sum().float()
            num += y.numel()
    return correct / num, loss / num


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def train_model(net, loss_fn, optimizer, epochs, device, dataloader, test_dataloader, save_best=False, save_dir="",
                init=None,
                scheduler=None, log_num=1):
    if init is False:
        pass
    elif not init:
        net.apply(init_weights)
    else:
        net.apply(init)

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    net.to(device)
    timer = Timer()
    best_acc = 0
    best_net = None
    print("Training starting")
    for ep in range(1, epochs + 1):
        correct = 0.0
        loss = 0.0
        num = 0
        net.train()
        timer.start()
        for X, y in dataloader:
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            loss_ep = loss_fn(y_hat, y).sum()
            loss_ep.backward()
            optimizer.step()
            with torch.no_grad():
                y_hat = torch.argmax(y_hat, dim=1)
                correct += (y_hat == y.reshape(y_hat.shape, -1)).sum().float()
                num += y.numel()
                loss += loss_ep
        timer.stop()

        with torch.no_grad():
            acc = correct / num
            loss = loss / num

        if scheduler:
            scheduler.step()
        net.eval()
        test_acc_ep, test_ls = evaluate_model(net, loss_fn, device, test_dataloader)

        train_loss.append(loss.cpu())
        test_loss.append(test_ls.cpu())

        train_acc.append(acc.cpu())
        test_acc.append(test_acc_ep.cpu())

        if save_best and best_acc < test_acc[-1]:
            best_acc = test_acc[-1]
            best_net = net

        if ep % log_num == 0:
            print(f'Epoch: {ep}, train loss: {loss:.3f}, validation loss: {test_ls:.3f}, train acc: {acc:.3f}, '
                  f'validation acc: {test_acc_ep:.3f}')
            print(f'{num / timer.times[-1]:.1f} examples/sec '
                  f'on {str(device)} total training time:{timer.sum():.1f} sec')
    if best_net:
        t = datetime.now()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        best_path = os.path.join(save_dir, "Best_Model_" + str(t).replace(":", "-") + ".pt")
        try:
            torch.save(best_net, best_path)
            print("Best accuracy", best_acc, "\nSave model in", best_path)
        except:
            print("Failed to save model in", best_path)

    return train_loss, test_loss, train_acc, test_acc


def drawGraph(train_ls, test_ls, train_acc, test_acc):
    epochs = [i for i in range(1, 1 + len(train_ls))]
    plt.plot(epochs, train_ls, label='train loss')
    plt.plot(epochs, test_ls, label='validation loss')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.plot(epochs, train_acc, label='train accuracy')
    plt.plot(epochs, test_acc, label='validation accuracy')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def evaluate_rnn(net, loss_fn, device, dataloader):
    """
    :param net: 模型
    :param loss_fn: 损失函数
    :param device: 设备
    :param dataloader: 使用dataloader来降低内存
    :return: 正确率， 损失
    """
    correct = 0.0
    loss = 0.0
    num = 0
    with torch.no_grad():
        for X, y in dataloader:
            state = net.begin_state(batch_size=X.shape[0], device=device)
            X = X.to(device)
            y = y.to(device)
            y_hat, _ = net(X, state)
            loss += loss_fn(y_hat, y).sum()
            y_hat = torch.argmax(y_hat, dim=1)
            correct += (y_hat == y.reshape(y_hat.shape, -1)).sum().float()
            num += y.numel()
    return correct / num, loss / num


def train_rnn(net, loss_fn, optimizer, epochs, device, dataloader, test_dataloader, save_best=False, init=None,
              scheduler=None, theta=1):
    if init is False:
        pass
    elif not init:
        net.apply(init_weights)
    else:
        net.apply(init)

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    timer = Timer()
    best_acc = 0
    print("Training starting")
    for ep in range(1, epochs + 1):
        state = None
        correct = 0.0
        loss = 0.0
        num = 0
        net.train()
        timer.start()
        for X, y in dataloader:
            if state is None:
                state = net.begin_state(batch_size=X.shape[0], device=device)
            else:
                if isinstance(net, nn.Module) and not isinstance(state, tuple):
                    # nn.GRU
                    state.detach_()
                else:
                    # nn.LSTM
                    for s in state:
                        s.detach_()

            X = X.to(device)
            y = y.to(device)
            y_hat, state = net(X, state)
            loss_ep = loss_fn(y_hat, y).sum()
            optimizer.zero_grad()
            loss_ep.backward()
            grad_clipping(net, theta)
            optimizer.step()
            with torch.no_grad():
                y_hat = torch.argmax(y_hat, dim=1)
                correct += (y_hat == y.reshape(y_hat.shape, -1)).sum().float()
                num += y.numel()
                loss += loss_ep
        timer.stop()

        with torch.no_grad():
            acc = correct / num
            loss = loss / num

        if scheduler:
            scheduler.step()
        net.eval()
        test_acc_ep, test_ls = evaluate_rnn(net, loss_fn, device, test_dataloader)

        train_loss.append(loss.cpu())
        test_loss.append(test_ls.cpu())

        train_acc.append(acc.cpu())
        test_acc.append(test_acc_ep.cpu())

        if save_best and best_acc < test_acc[-1]:
            best_acc = test_acc[-1]
            torch.save(net.state_dict(), "Best_Param.pt")

        print(f'Epoch: {ep}, mean loss: {loss:.3f}, test loss: {test_ls:.3f}, mean acc: {acc:.3f}, '
              f'test acc: {test_acc_ep:.3f}')
        print(f'{num / timer.times[-1]:.1f} examples/sec '
              f'on {str(device)} total training time:{timer.sum():.1f} sec')

    return train_loss, test_loss, train_acc, test_acc


def grad_clipping(net, theta):
    params = [p for p in net.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def mean_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    if len(embeddings.shape) == 1:
        return embeddings
    return torch.mean(embeddings, dim=0)
