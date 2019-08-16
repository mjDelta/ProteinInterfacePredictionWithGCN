#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-15 22:17:05
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

import torch.nn as nn
import torch.nn.functional as F
from layers import GC4Protein
import torch

class GCN4Protein(nn.Module):
	def __init__(self,feature_dim,hid_dim,drop_prob):
		super(GCN4Protein,self).__init__()

		self.gc1=GC4Protein(feature_dim,hid_dim)
		self.gc2=GC4Protein(2*hid_dim,hid_dim)

		self.fc1=nn.Sequential(
			nn.Linear(4*hid_dim,hid_dim),
			nn.ReLU(),
			nn.Dropout(drop_prob)
			)
		self.fc2=nn.Sequential(
			nn.Linear(hid_dim,2),
			nn.Sigmoid(),
			)
		self.drop_prob=drop_prob
	def forward(self,l_vertex,l_adj_distance,l_adj_angle,r_vertex,r_adj_distance,r_adj_angle,ligands,receptors):
		l_out_distance,l_out_angle,r_out_distance,r_out_angle=self.gc1(l_vertex,l_adj_distance,l_adj_angle,r_vertex,r_adj_distance,r_adj_angle)
		l_out_distance=F.dropout(F.relu(l_out_distance),self.drop_prob)
		l_out_angle=F.dropout(F.relu(l_out_angle),self.drop_prob)
		r_out_distance=F.dropout(F.relu(r_out_distance),self.drop_prob)
		r_out_angle=F.dropout(F.relu(r_out_angle),self.drop_prob)

		l_outs=torch.cat([l_out_distance,l_out_angle],dim=1)
		r_outs=torch.cat([r_out_distance,r_out_angle],dim=1)

		l_out_distance,l_out_angle,r_out_distance,r_out_angle=self.gc2(l_outs,l_adj_distance,l_adj_angle,r_outs,r_adj_distance,r_adj_angle)
		l_out_distance=F.dropout(F.relu(l_out_distance),self.drop_prob)
		l_out_angle=F.dropout(F.relu(l_out_angle),self.drop_prob)
		r_out_distance=F.dropout(F.relu(r_out_distance),self.drop_prob)
		r_out_angle=F.dropout(F.relu(r_out_angle),self.drop_prob)

		l_outs=torch.cat([l_out_distance,l_out_angle],dim=1)
		r_outs=torch.cat([r_out_distance,r_out_angle],dim=1)

		l_embs=l_outs[ligands]
		r_embs=r_outs[receptors]

		embs=torch.cat([l_embs,r_embs],dim=1)
		fc1=self.fc1(embs)
		fc2=self.fc2(fc1)
		return fc2





