
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
import torch.nn.functional as F
from geoopt.manifolds.stereographic.math import mobius_add, dist
from models.manifolds import PoincareBall
import random

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_children_np(data, starting=1023, kmin =100, kmax=500):
        #torch.seed(10)

        k = random.randint(kmin,kmax)
        idknn = knn(data[:,:3,:],k)
        #print(idknn.shape)
        pos_child = data[:,:,:k]

        for id in range(data.shape[0]):
               starting_point = random.randint(0,starting)
               pos_child[id,:,:] = data[id,:,idknn[id, starting_point ,:]]

        mar = 1000./k
        return mar,pos_child, k-1

def hype_triplet_losses(parent_mu , pos_child_mu, hier_margin=0.2, contr_margin=4, ball_dim = 256, one_child = False, opposite_hier=False):

        if one_child:
               neg_child_mu = torch.flip(parent_mu,[0])
        else:
               neg_child_mu = torch.flip(pos_child_mu,[0])

        ball = PoincareBall(c=1.0,dim=ball_dim)
        parent_norm = torch.norm(parent_mu,dim=1)       #dist(parent_mu, parent_mu*0, keepdim=True).pow(2).sum(1)
        pos_norm = torch.norm(pos_child_mu ,dim=1)

        par_norm = ball.dist0(parent_mu)
        p_norm = ball.dist0(pos_child_mu)       #.pow(2).sum(1)

        distance_positive = ball.dist(parent_mu, pos_child_mu)  #, keepdim=True).pow(2).sum(1)
        distance_negative = ball.dist(parent_mu, neg_child_mu)  #, keepdim=True).pow(2).sum(1)


        distance = distance_positive - distance_negative + contr_margin
        #distance = -distance_positive + distance_negative + mar_dist
        if opposite_hier:
               norm = par_norm-p_norm + hier_margin
        else:
               norm = -par_norm+p_norm + hier_margin            #Standard version wich improves the baseline
        #norm = par_norm-p_norm + margin                        #Possible change: parent becomes the mean of its children --EXperiment 07--

        triplet = torch.mean(torch.max(distance, torch.zeros_like(distance)))                   #F.relu(distance_positive - distance_negative + 0.5 )
        hierarch =torch.mean(torch.max(norm, torch.zeros_like(norm)))                   #F.relu(-par_norm+p_norm + margin)

        return parent_norm.mean(), pos_norm.mean()  ,distance_positive.mean(),distance_negative.mean(),triplet, hierarch

def euc_triplet_losses(parent_mu , pos_child_mu, hier_margin=2, contr_margin=4, one_child = False, opposite_hier=False):

        if one_child:
               neg_child_mu = torch.flip(parent_mu,[0])
        else:
               neg_child_mu = torch.flip(pos_child_mu,[0])

        parent_norm = torch.norm(parent_mu,dim=1)       #dist(parent_mu, parent_mu*0, keepdim=True).pow(2).sum(1)
        pos_norm = torch.norm(pos_child_mu ,dim=1)

        distance_positive = F.pairwise_distance(parent_mu, pos_child_mu)  #, keepdim=True).pow(2).sum(1)
        distance_negative = F.pairwise_distance(parent_mu, neg_child_mu)  #, keepdim=True).pow(2).sum(1)

        distance = distance_positive - distance_negative + contr_margin

        if opposite_hier:
               norm = parent_norm-pos_norm + hier_margin
        else:
               norm = -parent_norm+pos_norm + hier_margin


        triplet = torch.mean(torch.max(distance, torch.zeros_like(distance)))                   #F.relu(distance_positive - distance_negative + 0.5 )
        hierarch =torch.mean(torch.max(norm, torch.zeros_like(norm)))                   #F.relu(-par_norm+p_norm + margin)

        #triplet = triplet_losses.mean()
        #hierarch = hierarch_losses.mean()

        return parent_norm.mean(), pos_norm.mean()  ,distance_positive.mean(),distance_negative.mean(),triplet, hierarch



def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


