from typing import List

import torch
import torch.nn as nn
from torch import Tensor

import torch_geometric.nn as gnn

from src.datatypes.features.geometric import SinusoidalPosEmb
from src.models.architectures.gnn.rgcn import RGCN

from src.models.mifh.degree_utils import create_fake_edge_index_batch


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, dropout):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(n_layers - 1)],
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)



class GraphAdaptiveClassifier(nn.Module):
    """Graph Classifier used for computing a set of values for each node in the graph.
    The values computed don't have anything to do with the node in the same position,
    just the number of values returned per graph is equal to the number of nodes.
    This can be useful for computing a value for each degree, where a graph with
    N nodes and without self-loops will have N degrees, but each specific degree
    is not linked with the node at its position.
    """


    def __init__(self, in_dim, y_dim, mlp_cfg, static_classes=0, out_channels=1):
        super().__init__()

        self.aggr_fn = gnn.MeanAggregation()

        #self.pos_encoder = SinusoidalPosEmb(in_dim+y_dim)
        self.pos_encoder = SinusoidalPosEmb(in_dim)

        self.mlp = MLP(in_dim=2*in_dim+y_dim, out_dim=out_channels, **mlp_cfg)

        self.static_classes = static_classes



    def forward(self, encoded_x, global_y, batch, num_nodes_per_sample, ptr, return_new_ptr_batch=False):

        # aggregate and concatenate global_y
        x = self.aggr_fn(encoded_x, batch)
        x = torch.cat([x, global_y], dim=-1)

        num_out_classes = num_nodes_per_sample + self.static_classes
        tot_num_nodes = encoded_x.size(0) + self.static_classes * num_nodes_per_sample.size(0)
        ptr = ptr + torch.arange(ptr.size(0), device=ptr.device) * self.static_classes

        # repeat x and ptr to match the number of nodes
        x = x.repeat_interleave(num_out_classes, dim=0)
        ptr_rep = ptr[:-1].repeat_interleave(num_out_classes, dim=0)

        # produce aranges sample-wise for positional encoding
        # batch =   (0, 0, 0, 1, 1, 2, 2)
        # ptr_rep = (0, 0, 0, 3, 3, 5, 5)
        # arange =  (0, 1, 2, 3, 4, 5, 6)
        # pos =     (0, 1, 2, 0, 1, 0, 1)
        pos = torch.arange(tot_num_nodes, device=x.device)
        pos = pos - ptr_rep
        posenc = self.pos_encoder(pos)

        # add positional encoding to x
        #x = x + posenc
        x = torch.cat([x, posenc], dim=-1)

        # apply MLP to compute values, out_channels for each node
        x = self.mlp(x)
        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        if return_new_ptr_batch:
            batch = torch.arange(num_nodes_per_sample.size(0), device=batch.device).repeat_interleave(num_out_classes)
            return x, ptr, batch
        else:
            return x


class GraphNodeClassifierWithAdjacency(nn.Module):


    def __init__(self, in_dim, out_dim, rgcn_cfg=None):
        super().__init__()

        self.initial_emb = nn.Parameter(torch.randn(in_dim))
        if rgcn_cfg is None:
            rgcn_cfg = {}

        self.rgcn = RGCN(
            in_channels=in_dim,
            out_channels=out_dim,
            num_relations=2,
            **rgcn_cfg
        )
        

    def forward(self, encoded_x, adj_vector, batch, batch_size, fake_edge_index=None, undirected=True):

        additional_nodes = self.initial_emb.repeat(batch_size, 1)

        # augment nodes with new dummy nodes which serve to relay all messages
        # from each node to all other nodes
        encoded_x = torch.cat([additional_nodes, encoded_x], dim=0)

        # if fake edges are not provided, create them
        if fake_edge_index is None:
            # create edge_index for the fake graph
            fake_edge_index = create_fake_edge_index_batch(batch, batch_size, undirected=undirected)
        
        # if undirected, replicate labels
        if undirected:
            adj_vector = torch.cat([adj_vector, adj_vector], dim=0)

        # use the rgcn to compute predictions
        x = self.rgcn(
            x=encoded_x,
            edge_index=fake_edge_index,
            edge_attr=adj_vector.to(torch.int64),
        )
        
        # remove the additional fake nodes
        return x[batch_size:]
    

class GraphAdaptiveClassifierWithAdjacency(nn.Module):

    def __init__(self, in_dim, y_dim, rgcn_cfg, mlp_cfg, out_channels=1):
        super().__init__()

        self.initial_emb = nn.Parameter(torch.randn(in_dim))

        self.adj_encoder = GraphNodeClassifierWithAdjacency(
            in_dim=in_dim,
            rgcn_cfg=rgcn_cfg
        )

        self.classifier = GraphAdaptiveClassifier(
            in_dim=in_dim,
            y_dim=y_dim,
            mlp_cfg=mlp_cfg,
            out_channels=out_channels
        )

    def forward(self, encoded_x, global_y, adj_vector, batch, num_nodes_per_sample, ptr, return_new_ptr_batch=False):
            
        # compute adjacency values
        adj_values = self.adj_encoder(encoded_x, adj_vector, batch, num_nodes_per_sample.size(0))

        # compute values for each node
        return self.classifier(encoded_x, global_y, batch, num_nodes_per_sample, ptr, return_new_ptr_batch=return_new_ptr_batch), adj_values


import math

class MultiLinear(nn.Module):
    """MultiLinear is used to compute multiple linear transformations in parallel.
    It is useful when we need to compute multiple linear transformations with the same input.
    Computes the operation x @ W + b for each head. W is a tensor of shape (n_heads, in_features, out_features),
    and b is a tensor of shape (n_heads, out_features). x is expected to have shape (n_heads, *, in_features).
    """

    def __init__(self, n_heads, in_features, out_features, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((n_heads, in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((n_heads, out_features)))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    # def forward(self, x):
    #     # this is a more general implementation, probably not needed and slower
    #     assert x.shape[-1] == self.in_features, 'Input tensor has incorrect number of features'
    #     assert x.shape[0] == self.n_heads, 'Input tensor has incorrect number of heads'
    #     x = torch.matmul(x, self.weight.view([self.n_heads] + [1] * (len(x.shape) - 3) + [self.in_features, self.out_features]))
    #     x = x + self.bias.view([self.n_heads] + [1] * (len(x.shape) - 2) + [self.out_features])
    #     return x
    
    def forward(self, x):
        assert x.shape[-1] == self.in_features, 'Input tensor has incorrect number of features'
        assert x.shape[0] == self.n_heads, 'Input tensor has incorrect number of heads'
        x = torch.bmm(x, self.weight)
        x = x + self.bias.unsqueeze(1)
        return x
    
class MultiMLP(nn.Module):
    """MultiMLP is used to compute multiple MLPs in parallel. This is like MLP, but with multiple heads."""

    def __init__(self, n_heads, in_dim, out_dim, hidden_dim=256, n_layers=2, dropout=0.1):
        super().__init__()

        self.mlp = nn.Sequential(
            MultiLinear(n_heads, in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            *[nn.Sequential(
                MultiLinear(n_heads, hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(n_layers - 1)],
            MultiLinear(n_heads, hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)
    

class GraphAdaptiveMultiClassifier(nn.Module):
    """Graph Classifier used for computing a set of values for each node in the graph.
    The values computed don't have anything to do with the node in the same position,
    just the number of values returned per graph is equal to the number of nodes.
    This can be useful for computing a value for each degree, where a graph with
    N nodes and without self-loops will have N degrees, but each specific degree
    is not linked with the node at its position.
    """

    def __init__(
            self,
            n_properties,
            in_dim,
            y_dim,
            emb_dim=128,
            mlp_cfg=None,
            out_channels=1,
            add_nodes_for_final_prop=True,
            aggregation='mean'
        ):
        super().__init__()

        mlp_cfg = {} if mlp_cfg is None else mlp_cfg

        self.aggr_fn = getattr(gnn, f'{aggregation.capitalize()}Aggregation')()
        self.add_nodes_for_final_prop = add_nodes_for_final_prop
        self.in_dim = in_dim

        self.n_properties = n_properties
        self.pos_encoder = SinusoidalPosEmb(emb_dim)
        self.property_encoder = MultiLinear(n_properties, emb_dim, in_dim+y_dim)

        self.out_channels = out_channels
        self.mlp = MultiMLP(n_heads=n_properties, in_dim=in_dim+y_dim, out_dim=out_channels, **mlp_cfg)


    def forward(self, encoded_x, global_y, batch, num_nodes_per_sample, properties: List|Tensor):

        ##########################  PROPERTIES BRANCH  #########################
        assert len(properties) == self.n_properties, f'Number of properties given ' \
            f'({len(properties)}) does not match the expected number ({self.n_properties})'
        # stack properties if they are given as a list
        if isinstance(properties, list):
            properties = torch.stack(properties, dim=0)         # (n_props, N)

        # encode properties with different linear transformations
        prop_emb = self.pos_encoder(properties)                 # (n_props, N, emb_dim)
        prop_emb = self.property_encoder(prop_emb)              # (n_props, N, in_dim+y_dim)

        # compute cumsum of properties -> this is done to give an order to the properties
        prop_emb = prop_emb.cumsum(dim=0)

        ############################  NODES BRANCH  ############################
        # aggregate and concatenate global_y
        x = self.aggr_fn(encoded_x, batch)                      # (bs, in_dim)
        x = torch.cat([x, global_y], dim=-1)                    # (bs, in_dim+y_dim)

        # repeat x and ptr to match the number of nodes
        x = x.repeat_interleave(num_nodes_per_sample, dim=0)    # (N, in_dim+y_dim)
        x = x.unsqueeze(0)                                      # (1, N, in_dim+y_dim)

        ##########################  PROCESSING BRANCH  #########################
        # add positional encoding to x
        x = x + prop_emb                                        # (n_props, N, in_dim+y_dim)
        if self.add_nodes_for_final_prop:
            # add the encoded nodes to the last property vectors
            # on the nodes position (not on global components)
            x[-1, :, :self.in_dim] = x[-1, :, :self.in_dim] + encoded_x
        
        # apply MLP to compute values for each property and node pair
        x = self.mlp(x)                                         # (n_props, N, out_channels)

        # if out_channels is 1, remove last dimension
        if self.out_channels == 1:
            x = x.squeeze(-1)
        
        return x    # weight/weights for each property and node pair


from src.models.mifh.bayes import compute_probs, compute_logprobs, global_logsumexp_pool
from torch_geometric.utils import softmax

class BeliefUpdateClassifier(nn.Module):
    """BeliefUpdateClassifier is used to compute the belief update for each node in the graph.
    First, nodes encoded using only the node features are passed through another encoder
    incorporating the adjacency information. The output of this encoder is then used to compute
    """

    def __init__(self, n_properties, in_dim, y_dim, adjenc_cfg=None, posenc_dim=128, wnn_cfg=None, aggregation='mean'):
        super().__init__()

        adjenc_cfg = {} if adjenc_cfg is None else adjenc_cfg
        wnn_cfg = {} if wnn_cfg is None else wnn_cfg

        # the adj_encoder will incorporate the nodes with information
        # about the already connected nodes
        # the output is the same size as the input
        self.adj_encoder = GraphNodeClassifierWithAdjacency(
            in_dim=in_dim,
            out_dim=in_dim,
            rgcn_cfg=adjenc_cfg
        )

        self.weight_network = GraphAdaptiveMultiClassifier(
            n_properties=n_properties,
            in_dim=in_dim,
            y_dim=y_dim,
            out_channels=1,
            emb_dim=posenc_dim,
            mlp_cfg=wnn_cfg,
            aggregation=aggregation
        )

        self.weight_positive_fn = nn.Softplus()

        self.n_properties = n_properties


    def forward(
            self,
            encoded_x,
            global_y,
            adj_vector,
            batch,
            num_nodes_per_sample,
            properties: List|Tensor,
            undirected=True,
            return_weights=False,
            use_logprobs=True,
            return_logprobs=False
        ):

        # compute adjacency encoding
        adj_encoding = self.adj_encoder(encoded_x, adj_vector, batch, num_nodes_per_sample.size(0), undirected=undirected)
        # compute a residual connection
        encoded_x = encoded_x + adj_encoding

        # compute weights for each property and node pair
        weights = self.weight_network(encoded_x, global_y, batch, num_nodes_per_sample, properties)

        # remove the weights of already linked nodes
        nadj = ~adj_vector.bool()
        weights_nadj = weights[:, nadj]
        properties = properties[:, nadj]
        batch = batch[nadj]

        # compute the probabilities for each node
        if use_logprobs:
            #logprobs_nadj = compute_logprobs(weights_nadj, properties, batch)
            logprobs_nadj = weights_nadj.sum(dim=0)
            #logprobs_nadj = torch.log(softmax(logprobs_nadj, batch) + 1e-8)
            logprobs_nadj = logprobs_nadj - global_logsumexp_pool(logprobs_nadj, batch)[batch]

            if return_logprobs:
                logprobs = torch.full_like(nadj, -float('inf'), dtype=torch.float)
                logprobs[nadj] = logprobs_nadj
                return logprobs

            probs_nadj = torch.exp(logprobs_nadj)
        else:
            weights = self.weight_positive_fn(weights)
            probs_nadj = compute_probs(weights_nadj, properties, batch)
        # probabilities for already linked nodes are set to 0
        probs = torch.zeros_like(nadj, dtype=torch.float)
        probs[nadj] = probs_nadj

        if return_weights:
            return probs, weights
        else:
            return probs
    