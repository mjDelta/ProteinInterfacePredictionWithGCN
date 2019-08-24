# ProteinInterfacePredictionWithGCN
This repo recorded applying graph convolutional operation (defined by Kipf et al.). But Kipf's method cannot be applied directly on **Protein Interface Prediction**. Because its adjacent matrix's dim is [n,n], where n represents the vertex numbers. While in this task, the adjacent matrix's dim is [n,n,2], including distance and angle among *amino acids* in one protein.
### Difficulties and solution
* In Kipf's defination, adjacent matrix represents whether the vertex is neighbors or not, where they are distance and angle numbers here. So it needs some transformation before aggregation, which can transfer the distance and angle to aggregation weights.
* The dim confict in equations, [n,n] and [n,n,2].
And we proposed to solved it by adding a convolutional operation before graph convolution.
### Results
Graph convolution networks, defined as below:
[GCN](https://latex.codecogs.com/gif.latex?Z%3Df%28X%2CA%29%3Dsoftmax%28%5Cwidehat%7BA%7D%20ReLU%28%5Cwidehat%20%7BA%7DX%5E%7B%280%29%7DW%5E%7B%280%29%7D%29W%5E%7B%281%29%7D%29)
| hidden_dim | GC layer number | auc | acc |
| :--: | :--: | :--: | :--: |
| 50 | 2 | 0.75 | 0.69 |
| 100 | 2 | 0.74 | 0.68 |
| 128 | 2 | 0.74 | 0.68 |
| 50 | 1 | 0.77 | 0.70 |

Relation graph convolution networks, defined as below:

