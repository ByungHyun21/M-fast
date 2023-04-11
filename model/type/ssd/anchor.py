import numpy as np
import math

import torch 
"""
anchor generator

return anchor

anchor shape: (anchor_n, 4), 4: [cx, cy, w, h]
anchor order
 - increase cx
    - increase cy
        
example
    [0.25, 0.25, 0.4705, 1.0]
    [0.75, 0.25, 0.4705, 1.0]   # increase cx
    [0.25, 0.75, 0.4705, 1.0]   # increase cy
    [0.75, 0.75, 0.4705, 1.0]   # increase cx
    [0.5, 0.5, 0.95, 0.95]      # anchor 1
    [0.5, 0.5, 0.974, 0.974]    # anchor 2
    [0.5, 0.5, 1.0, 0.678]      # anchor 3
    [0.5, 0.5, 0.678, 1.0]      # anchor 4
    [0.5, 0.5, 1.0, 0.558]      # anchor 5
    [0.5, 0.5, 0.558, 1.0]      # anchor 6
"""

def anchor_generator(config):
    anchor_scale = config['ANCHOR_SCALE']
    anchor_grid_w = config['ANCHOR_GRID_W']
    anchor_grid_h = config['ANCHOR_GRID_H']
    anchor_n = config['ANCHOR_N']
    
    a1 = get_anchor(anchor_grid_h[0], anchor_grid_w[0], anchor_n[0], anchor_scale[0], anchor_scale[1])
    a2 = get_anchor(anchor_grid_h[1], anchor_grid_w[1], anchor_n[1], anchor_scale[1], anchor_scale[2])
    a3 = get_anchor(anchor_grid_h[2], anchor_grid_w[2], anchor_n[2], anchor_scale[2], anchor_scale[3])
    a4 = get_anchor(anchor_grid_h[3], anchor_grid_w[3], anchor_n[3], anchor_scale[3], anchor_scale[4])
    a5 = get_anchor(anchor_grid_h[4], anchor_grid_w[4], anchor_n[4], anchor_scale[4], anchor_scale[5])
    a6 = get_anchor(anchor_grid_h[5], anchor_grid_w[5], anchor_n[5], anchor_scale[5], 1.0)
    
    anchor = np.concatenate([a1, a2, a3, a4, a5, a6], axis=0)
    
    return anchor

def get_anchor(fh, fw, n_anchor, s1, s2):
        grid = np.linspace(0.0, 1.0, fw+1)
        grid = (grid[:-1] + grid[1:]) / 2.0
        grid_w = np.tile(np.expand_dims(grid, 0), (fh, 1))
        
        grid = np.linspace(0.0, 1.0, fh+1)
        grid = (grid[:-1] + grid[1:]) / 2.0
        grid_h = np.tile(np.expand_dims(grid, 1), (1, fw))
        
        cx = np.reshape(grid_w, (-1, 1))
        cy = np.reshape(grid_h, (-1, 1))
        
        na = fw * fh

        # 1: s1
        aw = s1
        ah = s1
        cw = np.tile(aw, (na, 1))
        ch = np.tile(ah, (na, 1))
        a1 = np.concatenate([cx, cy, cw, ch], axis=1)
        
        # 2: sqrt(s1, s2)
        aw = np.sqrt(s1 * s2)
        ah = np.sqrt(s1 * s2)
        cw = np.tile(aw, (na, 1))
        ch = np.tile(ah, (na, 1))
        a2 = np.concatenate([cx, cy, cw, ch], axis=1)
        
        # 3: 2x 1/2x
        # 1.4 1.61 1.82
        aw = s1 * np.sqrt(2.0)
        ah = s1 / np.sqrt(2.0)
        
        cw = np.tile(aw, (na, 1))
        ch = np.tile(ah, (na, 1))
        a3_1 = np.concatenate([cx, cy, cw, ch], axis=1)
        a3_2 = np.concatenate([cx, cy, ch, cw], axis=1)
        
        if n_anchor != 6:
            anchors = np.concatenate([a1, a2, a3_1, a3_2], axis=1)
            anchors = np.reshape(anchors, (-1, 4))
            anchors = np.clip(anchors, 0.0, 1.0)
            return anchors
        else: 
            # 4: 3x 1/3x
            # 1.7 1.95 2.2
            aw = s1 * np.sqrt(3.0)
            ah = s1 / np.sqrt(3.0)
            
            cw = np.tile(aw, (na, 1))
            ch = np.tile(ah, (na, 1))
            a4_1 = np.concatenate([cx, cy, cw, ch], axis=1)
            a4_2 = np.concatenate([cx, cy, ch, cw], axis=1)
            anchors = np.concatenate([a1, a2, a3_1, a3_2, a4_1, a4_2], axis=1)
            anchors = np.reshape(anchors, (-1, 4))
            anchors = np.clip(anchors, 0.0, 1.0)
            return anchors