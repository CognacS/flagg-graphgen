from torch_geometric.transforms import BaseTransform
from src.datatypes.sparse import SparseGraph
from src.data.transforms.core import TransformAdapter
from src.data.datasets.core import DataResources

from src.data.transforms import reg_transforms


class ToOneHot(BaseTransform):
    def __init__(self, num_classes_node, num_classes_edge, **kwargs):
        self.num_classes_node = num_classes_node
        self.num_classes_edge = num_classes_edge

    def forward(self, data: SparseGraph):
        return data.to_onehot(self.num_classes_node, self.num_classes_edge)

    def __repr__(self):
        return '{}(num_classes_node={}, num_classes_edge={})'.format(
            self.__class__.__name__, self.num_classes_node, self.num_classes_edge
        )


@reg_transforms.register('to_onehot')
class ToOneHotAdapter(TransformAdapter):

    def instantiate(self, data_resources: DataResources, **kwargs) -> BaseTransform:
        info = data_resources.info_total

        tr = ToOneHot(
            num_classes_node=info[self.map['num_classes_node']],
            num_classes_edge=info[self.map['num_classes_edge']]
        )

        return tr