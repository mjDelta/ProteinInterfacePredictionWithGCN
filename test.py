#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-16 13:40:07
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

import os
from utils import load_data,compute_accuracy
from models import GCN4Protein,GCN4ProteinV2,GCN4ProteinV3
from torch import nn
from torch import optim
import torch
import math
import numpy as np
from sklearn.metrics import roc_auc_score
def mkdirs(path):
	if not os.path.exists(path):
		os.makedirs(path)


USE_CUDA=torch.cuda.is_available()
device=torch.device("cuda" if USE_CUDA else "cpu")

data_path="E:/proteins/test.cpkl.gz"
load_model_path="E:/proteins/saved_models/model_100.tar"

hidden_dim=200
train_rate=0.
drop_prob=0.5
graphs=load_data(data_path)
vertex_features_dim=graphs[0]["ligand"]["vertex"].shape[1]

train_graphs=graphs[:int(train_rate*len(graphs))]
val_graphs=graphs[int(train_rate*len(graphs)):]
# model=GCN4Protein(vertex_features_dim,hidden_dim,drop_prob)
# model=GCN4ProteinV2(vertex_features_dim,hidden_dim,drop_prob)
model=GCN4ProteinV3(vertex_features_dim,hidden_dim,drop_prob)
model_sd=torch.load(load_model_path)
model.load_state_dict(model_sd["model"])
model.to(device)



val_accs=[]

all_preds=[]
all_ys=[]
length=0
for g in val_graphs:

	l_graph=g["ligand"]
	r_graph=g["receptor"]
	l_indices=g["ligand_indices"]
	r_indices=g["receptor_indices"]
	labels=g["label"]

	l_vertex=l_graph["vertex"];l_adj_distance=l_graph["adj_distance"];l_adj_angle=l_graph["adj_angle"]
	r_vertex=r_graph["vertex"];r_adj_distance=r_graph["adj_distance"];r_adj_angle=r_graph["adj_angle"]

	l_vertex=torch.FloatTensor(l_vertex).to(device)
	l_adj_distance=torch.FloatTensor(l_adj_distance).to(device)
	l_adj_angle=torch.FloatTensor(l_adj_angle).to(device)
	r_vertex=torch.FloatTensor(r_vertex).to(device)
	r_adj_distance=torch.FloatTensor(r_adj_distance).to(device)
	r_adj_angle=torch.FloatTensor(r_adj_angle).to(device)
	l_indices=torch.FloatTensor(l_indices).to(device)
	r_indices=torch.FloatTensor(r_indices).to(device)
	l_indices=l_indices.to(torch.int64)
	r_indices=r_indices.to(torch.int64)
	model.eval()
	preds=model(l_vertex,l_adj_distance,l_adj_angle,r_vertex,r_adj_distance,r_adj_angle,l_indices,r_indices)
	acc=compute_accuracy(preds.detach().cpu().numpy(),labels)
	all_preds.append(preds.detach().cpu().numpy())
	all_ys.append(labels)
	val_accs.append(acc)
	length+=len(labels)
	print("Protein:{} ACC:{} Positive:{}".format(g["complex_code"],acc,np.sum(np.argmax(labels,axis=1))/len(labels)))
	# break

all_preds_arr=np.zeros(shape=(length,))
all_ys_arr=np.zeros(shape=(length,))
cnter=0
for preds,ys in zip(all_preds,all_ys):
	all_preds_arr[cnter:cnter+len(preds)]=preds[:,1]
	all_ys_arr[cnter:cnter+len(preds)]=ys[:,1]
	cnter+=len(preds)
auc=roc_auc_score(all_ys_arr,all_preds_arr)
# print(all_ys_arr)
# print(all_preds_arr)
print("AUC:{}\t ACC:{}".format(auc,np.array(val_accs).mean()))
