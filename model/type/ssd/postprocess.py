import torch
import torch.nn as nn

class ssd_post(nn.Module):
    def __init__(self, ssd):
        super().__init__()

        self.ssd = ssd
        self.nc = ssd.nc
        self.activation = nn.Softmax(dim=-1)

    def forward(self, x):
        x, anchor = self.ssd(x)
        x = self.postprocess(x, anchor)

        return x

    def postprocess(self, x, anchor):
        cls, d_x, d_y, d_w, d_h = torch.split(x[0], [self.nc, 1, 1, 1, 1], dim=1)
        a_x, a_y, a_w, a_h = torch.split(anchor, [1, 1, 1, 1], dim=1)
        
        cx = (d_x * a_w / 10.0) + a_x
        cy = (d_y * a_h / 10.0) + a_y
        w = torch.exp(d_w / 5.0) * a_w
        h = torch.exp(d_h / 5.0) * a_h

        x1 = cx - (w / 2.0)
        x2 = cx + (w / 2.0)
        y1 = cy - (h / 2.0)
        y2 = cy + (h / 2.0)
        
        cls = self.activation(cls)

        return torch.concat([cls, x1, y1, x2, y2], dim=1)