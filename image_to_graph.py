import torch

import open3d as o3d
import numpy as np
from tqdm import tqdm

from torch_geometric.data import Data
import torch_geometric.utils as py_utils

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from torchvision import datasets
from torchvision import transforms

import networkx as nx

class ImageToGraph:
    def __init__(self):
        pass

    def _get_k_nearest(self, point_source, points, k):
        dists = []
        dists = [np.sqrt(np.sum(np.square(point_source-point_target))) for point_target in points]
        nearest_k_idx = np.argsort(dists)[:k + 1] # adding 1 becsuse the first one will be the point itself (self-loop)
        return nearest_k_idx

    def get_edges(self, points, k):
        edge_idx = []
        for source_idx, point_source in tqdm(enumerate(points)):
            nearest_k_idx = self._get_k_nearest(point_source, points, k)
            for target_idx in nearest_k_idx:
                pair = [source_idx, target_idx]
                edge_idx.append(pair)
        return np.array(edge_idx)

    def convert(self, img, k=3, plot=True, device='cuda'):
        plt.figure(figsize=(10, 5))
        img_for_graph = np.rot90(img,3)

        # data
        pixel_locs = {}
        idx_locs = {}
        id=0
        for i in range(img_for_graph.shape[1]):
            for j in range(img_for_graph.shape[0]):
                pixel_locs[(i, j)] = img_for_graph[i][j] # {(x,y): intensity}
                idx_locs[id] = (i, j) # {id: (x,y)} for graph plotting
                id+=1

        # get elements of graph
        pixels = np.array([pixel for pixel in pixel_locs.values()])
        locs =  np.array([loc for loc in pixel_locs.keys()])

        # create edges
        edge_idx = self.get_edges(locs, k)

        # plot graph
        graph = Data(x=torch.Tensor(pixels), edge_index=torch.Tensor(edge_idx).t().contiguous())
        graph=graph.to(device)

        if plot == True:
            plt.subplot(121)
            plt.imshow(img) # plot original image
            plt.subplot(122)
            g = py_utils.to_networkx(graph, to_undirected=True, remove_self_loops= True)
            nx.draw(g, idx_locs, node_color='red', node_size=[v / 10 for v in pixels])
            plt.show()

# Download the MNIST Dataset
dataset = datasets.MNIST(root = "./data",
                         train = True,
                         download = True,
                         transform = transforms.ToTensor())

# extract distinct numbers [0,1,...,9]
images = []
num = 0
for i in range(len(dataset.data)):
    if num < 10:
        if dataset.targets[i].item() == num:
            images.append(dataset.data[i].numpy())
            num += 1
    else:
        break
images = np.array(images)
for img in images:
    itg = ImageToGraph().convert(img, k=8, plot=True)
