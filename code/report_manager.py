import time, sys, os
import cv2
import numpy as np

import torch


class AverageMeter:
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        
    def get_data(self, data):
        self.cnt += 1
        self.sum += data
    
    def get_mean(self):
        return self.sum / self.cnt

    def reset(self):
        self.cnt = 0
        self.sum = 0

class report_manager():
    def __init__(self, cfg:dict, rank:int):
        self.loss_name = cfg['loss']['type']
        n_loss = len(cfg['loss']['type'])
        self.loss = []
        for i in range(n_loss):
            self.loss.append(AverageMeter())
        
    def reset(self):
        for i in range(len(self.loss)):
            self.loss[i].reset()
            
    def accumulate_loss(self, loss):
        for i in range(len(self.loss)):
            self.loss[i].get_data(loss[i].item())

    def loss_print(self):
        str_out = ''
        
        for i in range(len(self.loss)):
            str_out += f"{self.loss_name[i]}: {self.loss[i].get_mean():.2f}, "
        return str_out
    
    def get_loss_dict(self):
        dict_out = {}
        for i in range(len(self.loss)):
            dict_out[self.loss_name[i]] = self.loss[i].get_mean()
        return dict_out