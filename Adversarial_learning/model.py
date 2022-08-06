import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaNN(nn.Module):
    def __init__(self, n_feat, n_hid_1, n_hid_2, out_dim, dropout):
        super(VanillaNN, self).__init__()

        self.linear1 = nn.Linear(n_feat, n_hid_1)
        self.linear2 = nn.Linear(n_hid_1, n_hid_2)
        self.linear3 = nn.Linear(n_hid_2, out_dim)
        self.dropout = dropout

    def forward(
        self, x, adj=None
    ):  # add unused adj argument so same training functions can be used for now
        x = F.leaky_relu(self.linear1(x.transpose(0, 1)))
        F.dropout(x, self.dropout, inplace=True, training=True)
        x = F.leaky_relu(self.linear2(x))
        F.dropout(x, self.dropout, inplace=True, training=True)
        out = F.leaky_relu(self.linear3(x)[0][0])
        return out

    def initialise_weights_and_biases(self, seed: int = 0):
        torch.manual_seed(seed)
        for name, param in self.named_parameters():
            if name.endswith("weight"):
                torch.nn.init.xavier_uniform_(param)
            elif name.endswith("bias"):
                torch.nn.init.normal_(param)


class MICPredictor(VanillaNN):
    def forward(self, x):
        x = F.leaky_relu(self.linear1(x.transpose(0, 1)))
        F.dropout(x, self.dropout, inplace=True, training=True)
        x = F.leaky_relu(self.linear2(x))
        F.dropout(x, self.dropout, inplace=True, training=True)
        out = F.leaky_relu(self.linear3(x)[0][0])
        return out, x
