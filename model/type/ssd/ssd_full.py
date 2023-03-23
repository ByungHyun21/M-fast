import torch
import torch.nn as nn

class ssd_full(nn.Module):
    def __init__(self, model, anchor):
        super().__init__()

        self.model = model
        self.anchor = anchor
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.postprocess(x, self.anchor)
        return x

    def postprocess(self, x):
        """
        x = [batch, anchor_n, 4+1+class_n], 4 = [delta_cx, delta_cy, delta_w, delta_h], 1 = background
        self.anchor = [anchor_n, 4], 4 = [cx, cy, w, h]
        """
        cls = x[:, :, 4:]
        d_x, d_y, d_w, d_h = torch.split(x[:, :, :4], [1, 1, 1, 1], dim=2)
        a_x, a_y, a_w, a_h = torch.split(self.anchor, [1, 1, 1, 1], dim=1)
        
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