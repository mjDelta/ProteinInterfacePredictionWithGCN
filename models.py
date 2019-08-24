#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-15 22:17:05
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

import torch.nn as nn
import torch.nn.functional as F
from layers import GC4Protein,GCLayer,RGCLayer
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

class GCN4ProteinV2(nn.Module):
	def __init__(self,in_feature_dim,hid_dim,drop_prob):
		super(GCN4ProteinV2,self).__init__()

		self.pre_adj=nn.Sequential(
			nn.Conv1d(2,hid_dim,1),
			nn.BatchNorm1d(hid_dim),
			nn.LeakyReLU())


		self.gc1=GCLayer(in_feature_dim,hid_dim)
		# self.gc2=GCLayer(hid_dim,hid_dim)
		# self.gc3=GCLayer(hid_dim,hid_dim)

		self.fc1=nn.Sequential(
			nn.Linear(2*hid_dim,hid_dim),
			nn.ReLU(),
			nn.Dropout(drop_prob))
		self.fc2=nn.Sequential(
			nn.Linear(hid_dim,2),
			nn.Sigmoid())
		self.drop_prob=drop_prob
	def forward(self,l_vertex,l_adj_distance,l_adj_angle,r_vertex,r_adj_distance,r_adj_angle,ligands,receptors):
		l_adj=torch.stack([l_adj_distance,l_adj_angle],dim=1)
		l_adj=self.pre_adj(l_adj)
		l_adj=torch.mean(l_adj,dim=1)

		l_gc1=F.dropout(F.relu(self.gc1(l_vertex,l_adj)),self.drop_prob)
		# l_gc2=F.dropout(F.relu(self.gc2(l_gc1,l_adj)),self.drop_prob)
		# l_gc3=F.dropout(F.relu(self.gc2(l_gc2,l_adj)),self.drop_prob)

		l_embs=l_gc1[ligands]

		r_adj=torch.stack([r_adj_distance,r_adj_angle],dim=1)
		r_adj=self.pre_adj(r_adj)
		r_adj=torch.mean(r_adj,dim=1)

		r_gc1=F.dropout(F.relu(self.gc1(r_vertex,r_adj)),self.drop_prob)
		# r_gc2=F.dropout(F.relu(self.gc2(r_gc1,r_adj)),self.drop_prob)
		# r_gc3=F.dropout(F.relu(self.gc2(r_gc2,r_adj)),self.drop_prob)

		r_embs=r_gc1[receptors]

		embs=torch.cat([l_embs,r_embs],dim=1)
		fc1=self.fc1(embs)
		out=self.fc2(fc1)
		return out

class GCN4ProteinV3(nn.Module):
	"""docstring for GCN4ProteinV3"""
	def __init__(self, in_dim,h_dim,drop_prob):
		super(GCN4ProteinV3, self).__init__()
		self.drop_prob=drop_prob

		self.gc1=RGCLayer(in_dim,h_dim,2)
		self.gc2=RGCLayer(h_dim,h_dim,2)

		self.fc1=nn.Sequential(
			nn.Linear(2*h_dim,h_dim),
			nn.ReLU(),
			nn.Dropout(drop_prob))
		self.fc2=nn.Sequential(
			nn.Linear(h_dim,2),
			nn.Sigmoid())
	def forward(self,l_vertex,l_adj_distance,l_adj_angle,r_vertex,r_adj_distance,r_adj_angle,ligands,receptors):
		l_adj=[l_adj_distance,l_adj_angle]
		l_gc1=F.dropout(F.relu(self.gc1(l_vertex,l_adj)),self.drop_prob)
		l_gc2=F.dropout(F.relu(self.gc2(l_gc1,l_adj)),self.drop_prob)

		l_embs=l_gc2[ligands]

		r_adj=[r_adj_distance,r_adj_angle]
		r_gc1=F.dropout(F.relu(self.gc1(r_vertex,r_adj)),self.drop_prob)
		r_gc2=F.dropout(F.relu(self.gc2(r_gc1,r_adj)),self.drop_prob)

		r_embs=r_gc2[receptors]

		embs=torch.cat([l_embs,r_embs],dim=1)

		# print(embs)

		fc1=self.fc1(embs)
		fc2=self.fc2(fc1)
		return fc2





