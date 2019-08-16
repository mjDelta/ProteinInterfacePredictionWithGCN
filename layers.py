#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-15 22:02:07
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

import torch 
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math

class GC4Protein(Module):
	"""
	modify from https://github.com/tkipf/pygcn
	"""
	def __init__(self,in_features,out_features):
		super(GC4Protein,self).__init__()
		self.in_features=in_features
		self.out_features=out_features

		self.weights_distance=Parameter(torch.FloatTensor(in_features,out_features))
		self.bias_distance=Parameter(torch.FloatTensor(out_features))

		self.weights_angle=Parameter(torch.FloatTensor(in_features,out_features))
		self.bias_angle=Parameter(torch.FloatTensor(out_features))

		self.reset_parameters()

	def reset_parameters(self):
		stdv=1/math.sqrt(self.weights_distance.size(1))
		self.weights_angle.data.uniform_(-stdv,stdv)
		self.bias_angle.data.uniform_(-stdv,stdv)

		self.weights_distance.data.uniform_(-stdv,stdv)
		self.bias_distance.data.uniform_(-stdv,stdv)

	def forward(self,l_vertex,l_adj_distance,l_adj_angle,r_vertex,r_adj_distance,r_adj_angle):
		l_support_distance=torch.mm(l_vertex,self.weights_distance)
		l_support_angle=torch.mm(l_vertex,self.weights_angle)

		l_out_distance=torch.mm(l_adj_distance,l_support_distance)
		l_out_angle=torch.mm(l_adj_angle,l_support_angle)

		l_out_distance+=self.bias_distance
		l_out_angle+=self.bias_angle

		r_support_distance=torch.mm(r_vertex,self.weights_distance)
		r_support_angle=torch.mm(r_vertex,self.weights_angle)

		r_out_distance=torch.mm(r_adj_distance,r_support_distance)
		r_out_angle=torch.mm(r_adj_angle,r_support_angle)

		r_out_distance+=self.bias_distance
		r_out_angle+=self.bias_angle


		return l_out_distance,l_out_angle,r_out_distance,r_out_angle


