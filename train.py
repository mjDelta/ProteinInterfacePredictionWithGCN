#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-15 22:39:47
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

import os
from utils import load_data,compute_accuracy
from models import GCN4Protein,GCN4ProteinV2
from torch import nn
from torch import optim
import torch
import math

def mkdirs(path):
	if not os.path.exists(path):
		os.makedirs(path)


USE_CUDA=torch.cuda.is_available()
device=torch.device("cuda" if USE_CUDA else "cpu")

train_path="E:/proteins/train.cpkl.gz"
saved_models="E:/proteins/saved_models";mkdirs(saved_models)
epochs=30000
hidden_dim=100
train_rate=0.8
drop_prob=0.5
lr=0.01
weight_decay=5e-4
batch_size=1024
best_loss=9999
graphs=load_data(train_path)
vertex_features_dim=graphs[0]["ligand"]["vertex"].shape[1]

train_graphs=graphs[:int(train_rate*len(graphs))]
val_graphs=graphs[int(train_rate*len(graphs)):]
# model=GCN4Protein(vertex_features_dim,hidden_dim,drop_prob)
model=GCN4ProteinV2(vertex_features_dim,hidden_dim,drop_prob)
model.to(device)

optimizer=optim.SGD(model.parameters(),lr=lr,weight_decay=weight_decay)
criterion=nn.BCELoss()

iter_losses=[]
train_losses=[]
val_losses=[]
val_accs=[]
for e in range(epochs):
	e_loss=0.
	for g in train_graphs:

		l_graph=g["ligand"]
		r_graph=g["receptor"]
		l_indices=g["ligand_indices"]
		r_indices=g["receptor_indices"]
		labels=g["label"]

		label_size=len(labels)
		iters=math.ceil(label_size/batch_size)

		l_vertex=l_graph["vertex"];l_adj_distance=l_graph["adj_distance"];l_adj_angle=l_graph["adj_angle"]
		r_vertex=r_graph["vertex"];r_adj_distance=r_graph["adj_distance"];r_adj_angle=r_graph["adj_angle"]

		labels=torch.FloatTensor(labels).to(device)
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
		g_loss=0.
		for it in range(iters):
			optimizer.zero_grad()
			start=it*batch_size
			end=start+batch_size
			if end>label_size:end=label_size
			batch_preds=model(l_vertex,l_adj_distance,l_adj_angle,r_vertex,r_adj_distance,r_adj_angle,l_indices[start:end],r_indices[start:end])
			# criterion.weight=weights[start:end]
			batch_loss=criterion(batch_preds,labels[start:end])

			it_loss=batch_loss.item()
			iter_losses.append(it_loss)
			g_loss+=it_loss
			batch_loss.backward()
			optimizer.step()

		g_loss/=iters
		e_loss+=g_loss
	e_loss/=len(train_graphs)
	train_losses.append(e_loss)
	e_loss=0.
	val_acc=0.
	for g in val_graphs:

		l_graph=g["ligand"]
		r_graph=g["receptor"]
		l_indices=g["ligand_indices"]
		r_indices=g["receptor_indices"]
		labels=g["label"]

		label_size=len(labels)
		iters=math.ceil(label_size/batch_size)

		l_vertex=l_graph["vertex"];l_adj_distance=l_graph["adj_distance"];l_adj_angle=l_graph["adj_angle"]
		r_vertex=r_graph["vertex"];r_adj_distance=r_graph["adj_distance"];r_adj_angle=r_graph["adj_angle"]

		labels=torch.FloatTensor(labels).to(device)
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
		g_loss=0.
		g_acc=0.
		for it in range(iters):
			start=it*batch_size
			end=start+batch_size
			if end>label_size:end=label_size
			batch_preds=model(l_vertex,l_adj_distance,l_adj_angle,r_vertex,r_adj_distance,r_adj_angle,l_indices[start:end],r_indices[start:end])
			batch_loss=criterion(batch_preds,labels[start:end])

			it_loss=batch_loss.item()
			g_loss+=it_loss

			acc=compute_accuracy(batch_preds.detach().cpu().numpy(),labels[start:end].cpu().numpy())
			g_acc+=acc
		g_loss/=iters
		g_acc/=iters
		e_loss+=g_loss
		val_acc+=g_acc
	e_loss/=len(val_graphs)
	val_acc/=len(val_graphs)
	val_losses.append(e_loss)
	val_accs.append(val_acc)
	if best_loss>val_losses[-1]:
		torch.save({
			"model":model.state_dict(),
			"optimizer":optimizer.state_dict()
			},os.path.join(saved_models,"model_{}.tar".format(e)))
		best_loss=val_losses[-1]
	if e%10==0:
		print("Epoch {}: train loss {}\tval loss {}\tval acc{}".format(e,train_losses[-1],val_losses[-1],val_accs[-1]))


			








