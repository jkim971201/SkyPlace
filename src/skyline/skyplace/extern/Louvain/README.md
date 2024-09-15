# Overview
Goal of this project was to implement Community detection algorithm (using Louvain method) 
based on a paper: [Community Detection on the GPU](https://bora.uib.no/bora-xmlui/bitstream/handle/1956/16753/PaperIII.pdf?sequence=1&isAllowed=y).

# Usage
Flag `-f` specifies input file, `-g` gain threshold, `-v` represents verbose mode
```bash
make
./gpulouvain -f mtx-matrix-file -g min-gain [-v]
```

# Algorithm
Obviously, first step of the algorithm is to read data from input file. <br/>
I assume that given graph is undirected. <br/>
Actual algorithm is split into 2 parts:
  - modularity optimisation - in this step we find currently optimal community (group of vertices)
  - community aggregation - in this we merge vertices within a single community into one new vertex (keeping edges between communities)

These 2 steps are repeated as long as modularity gain is bigger than a threshold (provided by user).
In the end final modularity is printed. In verbose mode `original vertex -> final community` assignment is additionally printed. 

# Optimisation
During both phases vertices are divided into buckets based on degrees. This way, we dedicate more resources on a vertex with greater degree.
Buckets containing vertices with smaller degrees use only shared memory, this way we utilise slow global memory only when it is necessary.
