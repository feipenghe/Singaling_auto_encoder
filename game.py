import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np


class Game(nn.Module):
    def __init__(self, situation_size, message_size, prediction_size, func_size, hidden_size, transform=0):
        super().__init__()
        self.situation_size = situation_size
        self.message_size = message_size
        self.prediction_size = prediction_size
        self.func_size = func_size
        self.transform = transform
        self.criterion = nn.MSELoss()
        self.epoch = 0
        self.loss_list = []

        self.linear1_1 = nn.Linear(situation_size + prediction_size, hidden_size)
        self.linear1_2 = nn.Linear(hidden_size, hidden_size)
        self.linear1_3 = nn.Linear(hidden_size, message_size)

        self.linear2_1 = nn.Linear(situation_size + message_size, hidden_size)
        self.linear2_2 = nn.Linear(hidden_size, hidden_size)
        self.linear2_3 = nn.Linear(hidden_size, prediction_size)

        # self.batchnorm = nn.BatchNorm1d(hidden_size)

        self.functions = []
        functions1 = [nn.Linear(situation_size, prediction_size) for _ in range(func_size)]
        if transform:
            functions2 = [nn.Linear(situation_size, prediction_size) for _ in range(func_size)]
            for fc_i in range(func_size):
                functions2[fc_i].weight.data = transform * functions1[fc_i].weight.data
                functions2[fc_i].bias.data = transform * functions1[fc_i].bias.data
                self.functions.append(functions1[fc_i])
                self.functions.append(functions2[fc_i])
        else:
            self.functions = functions1
        for func in self.functions:
            for p in func.parameters():
                p.detach()
                p.requires_grad = False
        self.func_size = len(self.functions)

    def forward(self, situation, target):
        sender_input = torch.cat((situation, target), dim=1)
        message = F.relu(self.linear1_1(sender_input))
        message = F.relu(self.linear1_2(message))

        message = self.linear1_3(message)

        receiver_input = torch.cat((situation, message), dim=1)
        prediction = F.relu(self.linear2_1(receiver_input))
        prediction = F.relu(self.linear2_2(prediction))
        #        prediction = self.batchnorm(prediction)
        #        See more carefully whether batchnorm, dropout, or L1/L2regularization can help
        #        Especially with the later games, where the loss remains a bit high
        prediction = self.linear2_3(prediction)
        return prediction, message

    def target(self, situation, switch):
        A = torch.stack(tuple(func(situation) for func in self.functions), dim=1)
        A = A[range(len(switch)), switch, :]
        return A

    def message(self, situation, switch):
        with torch.no_grad():
            target = self.target(situation, switch)
            prediction, message = self.forward(situation, target)
            return message, torch.round(100 * message.view(-1, self.func_size, self.message_size).transpose(0, 1))

    def loss(self, situation, switch):
        with torch.no_grad():
            target = self.target(situation, switch)
        prediction, message = self.forward(situation, target)
        return self.criterion(prediction, target)

    def average_messages(self, n_examplars):
        self.av_messages = []
        with torch.no_grad():
            situations = torch.randn(n_examplars, self.situation_size)
            switch1 = torch.ones(n_examplars, dtype=torch.long)
            # MAYBE MAKE THE FOLLOWING LOOP IN ONE GO:
            # - repeat the situations
            # - vary the switches accordingly
            # - calculate the mean within groups
            for fc in range(len(self.functions)):
                switch = fc * switch1
                targets = self.target(situations, switch)
                p, messages = self.forward(situations, targets)
                self.av_messages.append(messages.mean(dim=0).squeeze())

    def discrete_forward(self, situations, switches, n_examplars_av=100, av_mess=True):
        if av_mess or not (self.av_messages):
            self.average_messages(n_examplars_av)
        messages = []  # self.av_messages
        for s in switches:
            messages.append(self.av_messages[s])
        messages = torch.stack(messages, dim=0)  # DOUBLE CHECK THIS
        targets = self.target(situations, switches)
        receiver_input = torch.cat((situations, messages), dim=1)
        prediction = F.relu(self.linear2_1(receiver_input))
        prediction = F.relu(self.linear2_2(prediction))
        prediction = self.linear2_3(prediction)
        return prediction


def playing_game(G: Game, epochs_n=1000, mini_batch_size=1000, learning_rate=0.001,
                 loss_every=None, func_out_training=None):
    OPTIM = optim.Adam(G.parameters(), lr=learning_rate)

    if func_out_training == None:
        func_out_training = list(range(G.func_size))
    else:
        func_out_training = list(set(range(G.func_size)) - set(func_out_training))

    for epoch in range(epochs_n):
        OPTIM.zero_grad()
        SITUATIONS = torch.randn(mini_batch_size, G.situation_size)
        SWITCH = torch.tensor(np.random.choice(func_out_training, mini_batch_size))
        #        torch.randint(high=G.func_size-func_left_out, size=(mini_batch_size,))
        loss = G.loss(SITUATIONS, SWITCH)
        loss.backward()
        OPTIM.step()

        G.epoch += 1

        try:
            if (epoch % loss_every == 0):
                G.loss_list.append((G.epoch, loss.item()))
        except:
            if epoch == 0:
                G.loss_list.append((G.epoch, loss.item()))
    G.loss_list.append((G.epoch, loss.item()))