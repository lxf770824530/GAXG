# GAXG

  GAXG, a novel GNN explanation framework that utilizes global and self-adaptive optimization at the model level.
## Requirement

Python 3.8

Pytorch 1.9.0

Pytorch-Geometric 2.0.0

network 2.5.1

## Data

Our method is evaluated on following datasets. Is_Acyclic and MUTAG datasets can be found in an open-source library [DIG](https://github.com/divelab/DIG/tree/main/dig/xgraph/datasets), which can be directly used to reproduce results of existing GNN explanation methods, develop new algorithms, and conduct evaluations for explanation results.
BA-2motif and Twitch Egos are available on [pyG](https://pytorch-geometric.readthedocs.io/en/latest/).

| Dataset    | Task                  | Data class     |
|------------|-----------------------|----------------|
| Is_Acyclic | Graph classification  | Synthetic data |
| BA-2motif  | Graph classification  | Synthetic data |
| Twitch Egos| Graph classification  | Real-word data |
| MUTAG      | Graph classification  | Real-word data |


## How to use

Our method can be used to explain graph classification. For each task, you just need to enter the corresponding folder to find the main.py.

For example, when we run the method for explaining the graph classification model which is trained on the Is_Acyclic dataset, we run the following command in the console.

First, we need train a simple GNN model which will be explained:
```
python Is_acyclic\main.py --mode train
```
If needed, you can change other configuration parameters in the source file 'main.py'.

Before to run the explanation process, you should also change the configuration parameters in the source file 'main.py'. 

Then, running this command to execute the explanation process:

```
python Is_acyclic\MCTS.py
```
You can also change other configuration parameters shown in the source file 'main.py'.
The explanation results will be saved in the folder '/Img'.  


