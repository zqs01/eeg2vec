import torch
import torch.nn  as nn


class CNNnet(nn.Module):
    def __init__(self, in_channel=64, out_channel=64, kernel_size=3):
        super(CNNnet, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.in_channel, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.LayerNorm([64, 640]),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.LayerNorm([64, 640]),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.LayerNorm([64, 640]),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.LayerNorm([64, 640]),
            nn.Conv1d(64, 1, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )


    def forward(self, x):
        x = x.squeeze(0)
        x = self.conv1(x)
        
        return x



if __name__ == "__main__":
    x = torch.randn(740,64,640)
    # conv1 = nn.Sequential(
    #         nn.Conv1d(64, 64, 3, padding=1),
    #         nn.ReLU(),
    #         nn.Conv1d(64, 1, 3, padding=1),
    #         nn.ReLU()
    #     )
    # out = conv1(x)
    # print(out.size())
    net = CNNnet()
    out = net(x)
    print(out.size())