# Welcome to scTM

scTM is a package for single cell topic modelling for transcriptomics data. Currently, we only have 1 module which supports spatial data (STAMP).

Topic models are powerful algorithms used in natural language processing to uncover hidden themes or topics within a collection of documents. The most widely used model, Latent Dirichlet Allocation (LDA), assumes that each document is a mixture of various topics, and each word is generated from one of those topics. In single cell, we assume that each document is a cell and each topic is a gene module. This leads to a intuitive interpretation to single cell that each cell is a mixture of gene modules.

### STAMP
In STAMP, we have more assumptions on the data generating process. The main two ones are
1. Spatial Information is important, therefore we need a way to incorporate spatial information in the model. Here, we use a graph neural network for the inference network to incorporate spatial information
2. The gene modules are sparse leading to interpretabiity and more robustness. Therefore, we suggest the use of horseshoe priors
STAMP is solved with black box inference, which leads to efficient optimization.
