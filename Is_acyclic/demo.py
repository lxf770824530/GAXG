from dateset import IsAcyclicDataset
import torch
from torch.distributions.categorical import Categorical
from torch_geometric.utils import degree
import time
import psutil
import os
from torch_geometric.datasets import TUDataset
import tqdm


star_time = time.time()


data = IsAcyclicDataset('data', name='Is_Acyclic')
# data = TUDataset(root='data/TUDataset',name='twitch_egos')
absence_edge_ratio_sum = 0.0
for i in data:
    single_absence_edge_ratio = 1 - len(i.edge_index[0]) / ((i.edge_index.max().item()+1) * degree(i.edge_index[0]).max())
    absence_edge_ratio_sum += single_absence_edge_ratio
absence_edge_ratio = absence_edge_ratio_sum / len(data)
probs = torch.FloatTensor([absence_edge_ratio, 1 - absence_edge_ratio])
edge_category_distribution = Categorical(probs)
edge_categories = []

end_time = time.time()
print('run-time',end_time-star_time)
print('内存使用：%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))