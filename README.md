# ProteinInterfacePredictionWithGCN
This repo recorded applying graph convolutional operation (defined by Kipf et al.). But Kipf's method cannot be applied directly on **Protein Interface Prediction**. Because its adjacent matrix's dim is [n,n], where n represents the vertex numbers. While in this task, the adjacent matrix's dim is [n,n,2], including distance and angle among *amino acids* in one protein.
### Difficulties and solution
* In Kipf's defination, adjacent matrix represents whether the vertex is neighbors or not, where they are distance and angle numbers here. So it needs some transformation before aggregation, which can transfer the distance and angle to aggregation weights.
* The dim confict in equations, [n,n] and [n,n,2].
And we proposed to solved it by adding a convolutional operation before graph convolution.
### Results
Graph convolution networks, defined as below:</br>
<a href="https://www.codecogs.com/eqnedit.php?latex=Z=f(X,A)=sigmoid(\widehat{A}XW^{(0)})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z=f(X,A)=sigmoid(\widehat{A}XW^{(0)})" title="Z=f(X,A)=sigmoid(\widehat{A}XW^{(0)})" /></a>

| hidden_dim | GC layer number | auc | acc |
| :--: | :--: | :--: | :--: |
| 50 | 2 | 0.75 | 0.69 |
| 100 | 2 | 0.74 | 0.68 |
| 128 | 2 | 0.74 | 0.68 |
| 50 | 1 | 0.77 | 0.70 |

Relation graph convolution networks, defined as below:</br>
<a href="https://www.codecogs.com/eqnedit.php?latex=Z=f(X,A)=sigmoid((\sum_{r\subseteq{R}}&space;{\widehat{A}_{r}X})W^{(0)})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z=f(X,A)=sigmoid((\sum_{r\subseteq{R}}&space;{\widehat{A}_{r}X})W^{(0)})" title="Z=f(X,A)=sigmoid((\sum_{r\subseteq{R}} {\widehat{A}_{r}X})W^{(0)})" /></a>

| hidden_dim | GC layer number | auc | acc |
| :--: | :--: | :--: | :--: |
| 50 | 2 | 0.67 | 0.64 |
| 100 | 2 | 0.73 | 0.66 |
| 200 | 2 | 0.72 | 0.66 |
| 50 | 1 | 0.75 | 0.68 |
| 100 | 1 | 0.75 | 0.68 |
| 200 | 1 | 0.76 | 0.69 |
