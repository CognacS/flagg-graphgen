# runtime transforms
from torch_geometric.transforms import BaseTransform

from src.models.generator import Generator
from src.data.transforms.core import TransformAdapter
from src.data.transforms import reg_transforms

from src.noise import (
    reg_diffusion,
    reg_schedule,
)
from src.noise.core import NoiseProcess
from copy import deepcopy
    
class IFHPrepareTransform(BaseTransform):

    def __init__(self, removal, **kwargs):

        if isinstance(removal, NoiseProcess):
            self.removal_process = deepcopy(removal).cpu()

        else:
            self.removal_process = reg_diffusion.get_instance_from_cfg(
                removal.process,
                schedule = reg_schedule.get_instance_from_cfg(
                    removal.schedule
                )
            )

    def forward(self, data):
        self.removal_process.prepare_data(datapoint=data)
        return data
    

@reg_transforms.register('ifh')
class IFHTPrepareAdapter(TransformAdapter):

    def instantiate(self, model: Generator, **kwargs) -> BaseTransform:

        assert hasattr(model, 'removal_process'), 'Model must have a removal process to use IFHPrepareTransform'

        tr = IFHPrepareTransform(
            removal=model.removal_process
        )

        return tr