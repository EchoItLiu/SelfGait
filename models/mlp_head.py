from torch import nn


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            #nn.Linear(in_channels, mlp_hidden_size),
            nn.Conv1d(in_channels, mlp_hidden_size,kernel_size=1),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            #nn.Linear(mlp_hidden_size, projection_size)
            nn.Conv1d(mlp_hidden_size, projection_size,kernel_size=1)
        )

    def forward(self, x):
        #print("projection in:",x.size())
        #x=[16,1,128]
        x=x.permute(1, 2, 0).contiguous()
        #x=[1,128,16]
        #print("projection per:",x.size())
        x=self.net(x)
        #print("projection out:",x.size())
        x=x.permute(2, 0, 1).contiguous()
        #x=[16,1,128]
        return x
