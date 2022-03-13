import torch
import torch.nn as nn
from torch import rand
from torch.nn import MSELoss
from torch.optim import Adam


class CRNNModel(nn.Module):

    def __init__(self,dataset_type="Strong"):
        super(CRNNModel, self).__init__()
        self.dataset_type=dataset_type
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 3), stride=(1, 1), padding='valid'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((1, 5), stride=(1, 5))
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding='valid'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), stride=(1, 2))
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding='valid'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), stride=(1, 2))
        )
        self.gru1 = nn.GRU(input_size=128, hidden_size=64, bidirectional=True, num_layers=2, batch_first=True,
                           dropout=0.1)
        self.activation1 = nn.Tanh()

        self.gru2 = nn.GRU(input_size=128, hidden_size=64, bidirectional=True, num_layers=2, batch_first=True,
                           dropout=0.1)
        self.activation2 = nn.Tanh()

        '''
        we use below network to converge all the 10 time frames to single dimension frame to predict the overall multilabel multiclass classification.
        '''
        if dataset_type=="Weak":
            self.weak_opt_layer=nn.Sequential(nn.Linear(in_features=10, out_features=1),nn.ReLU())

        self.linear1 = nn.Linear(in_features=128, out_features=64)
        self.dropout = nn.Dropout(0.2)
        self.activation3 = nn.ReLU()



        self.linear2 = nn.Linear(in_features=64, out_features=10)
        #self.activation4 = nn.Sigmoid()

    def forward(self, x_in):
        #print("inital size " + str(x_in.size()))
        x = x_in if x_in.ndimension() == 4 else x_in.unsqueeze(1)
        #print("inital size after channel " + str(x.size()))
        x = self.cnn1(x)
        #print("after cnn1 " + str(x.size()))
        x = self.cnn2(x)
        #print("after cnn2 " + str(x.size()))
        x = self.cnn3(x)
        #print("after cnn3 " + str(x.size()))

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x_in.size()[0], x.shape[1], -1)
        #print(x.shape)

        x, _ = self.gru1(x)
        x = self.activation1(x)
        #print('gru')
        #print(x.shape)
        x, _ = self.gru2(x)
        x = self.activation2(x)
        #print(x.shape)

        '''
        here we convert [batch,10,128] dim of feature to [batch,128] by performing non-linear nn operation in time dimension.
        '''
        if self.dataset_type=="Weak":
            x=x.permute(0,2,1)
            #print("shape 1 permute "+str(x.shape))
            x=self.weak_opt_layer(x)
            x = torch.squeeze(x, -1)
            #print("shape 2 permute " + str(x.shape))

            #x = torch.sum(x, dim=-2, keepdim=False)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.activation3(x)
        #print(x.shape)

        x = self.linear2(x)
        #print(x.shape)
        #x = self.activation4(x)
        return x

