from typing import Tuple, Dict, Optional, List, Union

import numpy as np

import torch
from torch import Tensor, IntTensor, BoolTensor


from src.datatypes.dense import (
    DenseGraph,
    DenseEdges,
    get_bipartite_edge_mask_dense,
    get_edge_mask_dense
)

from src.noise import reg_diffusion
from src.noise.core import NoiseSchedule, NoiseProcess

from src.noise.schedules import DiffusionProcessException, CosineDiffusionSchedule


from src.noise.discrete_diffusion import (
    UniformDiscreteDiffusionProcess,
    MarginalDiscreteDiffusionProcess
)
from src.noise.multimodal_diffusion import StructuredMultimodalDiffusionProcess

from src.datatypes.dense import dense_to_undirected

class GraphDiffusionProcess(StructuredMultimodalDiffusionProcess):

    def __init__(
            self,
            diff_x: NoiseProcess,
            diff_e: NoiseProcess,
            undirected: bool=True,
            **kwargs
        ):

        super().__init__({
            'x': diff_x,
            'e': diff_e
        })

        self.undirected = undirected


    def map_datapoint_to_dict(self, datapoint: Union[DenseGraph, DenseEdges]) -> Dict[str, Tensor]:
        if isinstance(datapoint, DenseEdges):
            return dict(e=datapoint.edge_adjmat)
        elif isinstance(datapoint, DenseGraph):
            return dict(x=datapoint.x, e=datapoint.edge_adjmat)
        elif datapoint is None:
            return {}
        else:
            raise DiffusionProcessException(f'Could not map datapoint to dict: {datapoint}')
    

    def compose_back(self, datapoint: Dict[str, Tensor], other_datapoint: DenseGraph) -> DenseGraph:

        if len(datapoint) == 0:
            return None
        
        edge_adjmat = datapoint['e']

        if 'x' in datapoint:
            if self.undirected:
                edge_adjmat = dense_to_undirected(edge_adjmat)
            
            graph = DenseGraph(
                x=datapoint['x'],
                edge_adjmat=edge_adjmat,
                y=other_datapoint.y,
                node_mask=other_datapoint.node_mask,
                edge_mask=other_datapoint.edge_mask
            ).apply_mask()

        elif 'e' in datapoint:
            graph = DenseEdges(
                edge_adjmat=edge_adjmat,
                edge_mask=other_datapoint.edge_mask
            ).apply_mask()

        return graph
    
    
    def kwargs_per_data_from_datapoint(self, datapoint: Union[DenseGraph, DenseEdges]) -> Dict[str, Dict[str, Tensor]]:
        if isinstance(datapoint, DenseEdges):
            return dict(mask=dict(e=datapoint.edge_mask))
        elif isinstance(datapoint, DenseGraph):
            return dict(mask=dict(x=datapoint.node_mask, e=datapoint.edge_mask))
        elif datapoint is None:
            return {}
        else:
            raise DiffusionProcessException(f'Could not map datapoint to dict: {datapoint}')
        

    ############################################################################
    #                     STATIONARY DISTRIBUTION (t->+inf)                    #
    ############################################################################

    def sample_stationary(
            self,
            num_new_nodes: IntTensor,
            ext_node_mask: Optional[BoolTensor]=None,
            device: torch.device=None,
            generate_edges: bool=False
        ) -> Tuple[DenseGraph, DenseEdges]:
        # num new nodes has shape (bs,), and each element
        # is the number of nodes the graph should have
        bs = len(num_new_nodes)
        max_num_nodes = num_new_nodes.max().item()

        # get current device
        device = num_new_nodes.device if device is None else device

        shape_x = (bs, max_num_nodes)
        shape_e = (bs, max_num_nodes, max_num_nodes)

        # compute current node mask
        node_mask = torch.arange(max_num_nodes, device=device) < num_new_nodes.unsqueeze(-1)

        if not generate_edges:
            # sample from stationary distributions
            x_and_e = super().sample_stationary(dict(x=dict(shape=shape_x), e=dict(shape=shape_e)), device=device)
            x, edge_adjmat = x_and_e['x'], x_and_e['e']

            # compute edge mask
            edge_mask = get_edge_mask_dense(node_mask)

            # transform to undirected graph
            if self.undirected:
                edge_adjmat = dense_to_undirected(edge_adjmat)

            # compose graph
            graph = DenseGraph(
                x =				x,
                edge_adjmat =	edge_adjmat,
                y =				None,
                node_mask =		node_mask,
                edge_mask =     edge_mask
            ).apply_mask()

        else:
            # generate uniform external edge adjmat
            max_ext_nodes = ext_node_mask.shape[1]
            ext_edge_adjmat = super().sample_stationary(dict(e=dict(shape=(bs, max_num_nodes, max_ext_nodes))), device=device)['e']
            # mask out fake nodes
            ext_edge_mask = get_bipartite_edge_mask_dense(node_mask, ext_node_mask)
            
            graph = DenseEdges(
                edge_adjmat =   ext_edge_adjmat,
                edge_mask =     ext_edge_mask
            ).apply_mask()


        return graph

@reg_diffusion.register('graph_uniform')
class UniformGraphDiffusionProcess(GraphDiffusionProcess):

    def __init__(
            self,
            schedule : NoiseSchedule,
            num_cls_x: int,
            num_cls_e: int,
            undirected: bool=True,
            **kwargs
        ):
        """
        Parameters
        ----------
        schedule : DiffusionSchedule
            gives the parameter values for next, sample_t, posterior
        """

        super().__init__(
            diff_x=UniformDiscreteDiffusionProcess(schedule, num_cls=num_cls_x),
            diff_e=UniformDiscreteDiffusionProcess(schedule, num_cls=num_cls_e),
            undirected=undirected
        )


@reg_diffusion.register('graph_marginal')
class MarginalGraphDiffusionProcess(GraphDiffusionProcess):

    def __init__(
            self,
            schedule : NoiseSchedule,
            num_cls_x: int,
            num_cls_e: int,
            minimum_number_updates: int=100,
            undirected: bool=True,
        ):
        """
        Parameters
        ----------
        schedule : DiffusionSchedule
            gives the parameter values for next, sample_t, posterior
        """
        # call super for the NoiseProcess
        super().__init__(
            diff_x=MarginalDiscreteDiffusionProcess(schedule, num_cls=num_cls_x, minimum_number_updates=minimum_number_updates),
            diff_e=MarginalDiscreteDiffusionProcess(schedule, num_cls=num_cls_e, minimum_number_updates=minimum_number_updates),
            undirected=undirected
        )
    
    
    def update(self, x_labels=None, e_labels=None):
        if x_labels is not None:
            self.diffusion_procs_per_data['x'].update(x_labels)
        if e_labels is not None:
            self.diffusion_procs_per_data['e'].update(e_labels)


    def stop_updating(self):
        self.diffusion_procs_per_data['x'].stop_updating()
        self.diffusion_procs_per_data['e'].stop_updating()


        
################################################################################
#                            RESOLVE OBJECT BY NAME                            #
################################################################################

DIFFUSION_SCHEDULE_COSINE = 'cosine'

DIFFUSION_PROCESS_GRAPH_UNIFORM = 'discrete_uniform'
DIFFUSION_PROCESS_GRAPH_MARGINAL = 'discrete_marginal'

def resolve_graph_diffusion_schedule(name: str) -> type:
    if name == DIFFUSION_SCHEDULE_COSINE:
        return CosineDiffusionSchedule
    else:
        raise DiffusionProcessException(f'Could not resolve diffusion schedule name: {name}')

def resolve_graph_diffusion_process(name: str) -> type:
    if name == DIFFUSION_PROCESS_GRAPH_UNIFORM:
        return UniformGraphDiffusionProcess
    elif name == DIFFUSION_PROCESS_GRAPH_MARGINAL:
        return MarginalGraphDiffusionProcess
    else:
        raise DiffusionProcessException(f'Could not resolve diffusion process name: {name}')