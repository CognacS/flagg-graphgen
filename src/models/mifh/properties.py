
import torch
from torch import Tensor

from src.datatypes.sparse import SparseGraph
from src.models.mifh.distance_utils import compute_distances

from src.utils.decorators import ClassRegister

reg_mifh_properties = ClassRegister('Model')


class MIFHProperty:

    def __init__(self, **kwargs):
        pass

    def __call__(self, graph: SparseGraph, adj: Tensor, **kwargs):
        raise NotImplementedError
    
@reg_mifh_properties.register('degree')
class DegreeProperty(MIFHProperty):

    def __call__(self, graph: SparseGraph, adj: Tensor, **kwargs):
        return graph.indegree

@reg_mifh_properties.register('distance')
class DistanceProperty(MIFHProperty):

    def __init__(self, threshold=4):
        super().__init__()
        self.threshold = threshold
    
    def __call__(self, graph: SparseGraph, adj: Tensor, **kwargs):
        # compute distances
        distances = compute_distances(
            graph.edge_index, adj, graph.num_nodes_per_sample,
            threshold=self.threshold
        )
        # replace -1 with 0
        distances[distances == -1] = 0
        # distances=0 entries will (must) be masked by adj
        return distances