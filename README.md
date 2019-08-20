# ProteinInterfacePredictionWithGCN
This repo recorded applying graph convolutional operation (defined by Kipf et al.). But Kipf's method cannot be applied directly on **Protein Interface Prediction**. Because its adjacent matrix's dim is [n,n], where n represents the vertex numbers. While in this task, the adjacent matrix's dim is [n,n,2], including distance and angle among *amino acids* in one protein.
### Difficulties and solution
* In Kipf's defination, adjacent matrix represents whether the vertex is neighbors or not, where they are distance and angle numbers here. So it needs some transformation before aggregation, which can transfer the distance and angle to aggregation weights.
* The dim confict in equations, [n,n] and [n,n,2].
And we proposed to solved it by adding a convolutional operation before graph convolution.
### Results

