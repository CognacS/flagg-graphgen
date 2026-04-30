

# take resources and model as input
# align the frequencies if marginal is available
# needs to instantiate! Pre-compute alignment
# if marginal is not available, just pass through the data
# need to modify the torch-to-mol adapter?
# probably, to allow encoding and decoding! Not only encoding

import torch
from torch_geometric.transforms import BaseTransform

from src.data.datasets.core import DataResources
from src.data.simple_transforms.molecular import GraphToMoleculeConverter
from src.models.generator import Generator
from src.data.transforms.core import TransformAdapter
from src.data.transforms import reg_transforms


def match_histograms(model_hist, data_hist, data_classes):
    # finds ordering of model_hist that best matches data_hist
    idx_model_hist = torch.argsort(model_hist.cpu())
    idx_data_hist = torch.argsort(data_hist.cpu())
    idx_map = torch.zeros_like(idx_model_hist)
    idx_map[idx_model_hist] = idx_data_hist
    mapped_classes = [data_classes[idx_map[i].item()] for i in range(len(data_classes))]
    mapped_data_hist = data_hist[idx_map]
    changed = not torch.all(idx_model_hist == idx_data_hist)
    return mapped_classes, idx_map, mapped_data_hist, changed
    
class AlignLabels(BaseTransform):

    def __init__(self, x_cls_perm=None, e_cls_perm=None, **kwargs):
        super().__init__()
        self.x_cls_perm = x_cls_perm
        self.e_cls_perm = e_cls_perm

    def forward(self, data):
        if self.x_cls_perm is not None:
            data.x = data.x[:, self.x_cls_perm]

        if self.e_cls_perm is not None and data.edge_attr is not None:
            data.edge_attr = data.edge_attr[:, self.e_cls_perm]

        return data
    

@reg_transforms.register('align_labels')
class AlignLabelsAdapter(TransformAdapter):

    def instantiate(
        self,
        data_resources: DataResources,
        model: Generator,
        context,
        **kwargs
    ) -> BaseTransform:
        
        # gather dataset classes map
        decoder = data_resources.get('decoder')
        
        if isinstance(decoder, GraphToMoleculeConverter):
            
            context.load_checkpoint('best')
            
            x_classes = sorted(decoder.atom_decoder.items(), key=lambda x: x[0])
            x_classes = [x[1] for x in x_classes]
            e_classes = sorted(decoder.bond_decoder.items(), key=lambda x: x[0])
            e_classes = [x[1] for x in e_classes]
            
            # gather training dataset marginals
            info = data_resources.get('info', 'train')
            marginals = info.get('marginals', None)
            
            if marginals is None:
                raise ValueError('Marginals must be provided in the data resources info to use AlignLabelsAdapter')
            
            model_x_marginal = context.model.filler_model.diffusion_process.diffusion_procs_per_data.x.marginal
            model_e_marginal = context.model.filler_model.diffusion_process.diffusion_procs_per_data.e.marginal[1:]
            
            data_x_marginal = torch.tensor(marginals['x'], device=model_x_marginal.device)
            data_edge_attr_marginal = torch.tensor(marginals['edge_attr'][1:], device=model_e_marginal.device)
            
            
            new_x_classes, x_cls_perm, new_x_marginal, x_changed = match_histograms(model_x_marginal, data_x_marginal, x_classes)
            new_e_classes, e_cls_perm, new_e_marginal, e_changed = match_histograms(model_e_marginal, data_edge_attr_marginal, e_classes)
            print(f'AlignLabelsAdapter: x_changed={x_changed}, e_changed={e_changed}')
            
            change_decoder = not hasattr(data_resources.decoder, '_modified') or not data_resources.decoder._modified
            
            if not x_changed:
                x_cls_perm = None
            else:
                # modify the decoder to reflect the new ordering
                if change_decoder:
                    print(f'Reordering: {new_x_classes}')
                    data_resources.decoder.atom_decoder = {i: new_x_classes[i] for i in range(len(new_x_classes))}
                    data_resources.decoder._modified = True
                else:
                    print(f'Keeping original ordering: {x_classes}')
                                
            if not e_changed:
                e_cls_perm = None
            else:
                # modify the decoder to reflect the new ordering
                if change_decoder:
                    print(f'Reordering: {new_e_classes}')
                    data_resources.decoder.bond_decoder = {i: new_e_classes[i] for i in range(len(new_e_classes))}
                    data_resources.decoder._modified = True
                else:
                    print(f'Keeping original ordering: {e_classes}')

        else:
            x_cls_perm = None
            e_cls_perm = None

            
        tr = AlignLabels(x_cls_perm=x_cls_perm, e_cls_perm=e_cls_perm)

        return tr