#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-16 21:38:39
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)
from matplotlib import pyplot as plt
import numpy as np

loss_path="E:/proteins/saved_models/epoch_loss.txt"
out_path="E:/proteins/saved_models/loss_plot.png"

train_losses=[]
val_losses=[]
val_accs=[]
with open(loss_path,"r") as f:
	for line in f.readlines():
		line=line.strip()
		if "train" not in line:
			break
		begin_idx=line.index("train loss ")+len("train loss ")
		end_idx=line.index("val acc")
		v=line[begin_idx:end_idx]
		v=v.strip()
		train_losses.append(float(v.split(" ")[0]))
		val_losses.append(float(v.split(" ")[-1]))
		val_accs.append(float(line.split(" ")[-1][3:]))

epochs=np.arange(0,len(train_losses)*10,10)

fig=plt.figure(figsize=(10,8),dpi=300)
plt.plot(epochs,train_losses,label="train loss",c="b")
plt.plot(epochs,val_losses,label="val loss",c="r")
plt.legend(loc="upper right")
plt.savefig(out_path,bbox_tight=True)
