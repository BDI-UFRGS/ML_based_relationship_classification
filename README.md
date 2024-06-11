## Datasets
This repository contains 3 datasets in csv format: a dataset that only contains direct relations, a dataset that contains only indirect relations, and a complete dataset. 
## Matrices
This repository includes two matrices that can be used to generate unrelated instances one for holonyms and the other for hypernyms. Each rown (n) and columns (m) in the matrix represents a synset.
If the index [n,m] has a 1, it means synset[n] has a relationship with synset[m]. 
This relation is a holonyms relation in the holonyms matric´x and a hypernym relation for the hypernym matrix. 
Notice that this matrices might be used for other purpuses and to generate different kinds of datasets based on the Wordnet relations.

## Code
The code snippet serves to generate the unreated unstances using the holonyms and hypernym matrices.

