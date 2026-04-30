
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pathlib import Path

import src.callbacks as clb


PREFIX_MODULE = 'model_'

@clb.reg_checkpoints.register('ModularModelCheckpoint')
def modular_model_checkpoint(dirpath, filename, module_monitors, **kwargs):

    checkpoint_callback = []

    for module_name, cfg_module in module_monitors.items():
        if cfg_module is not None:
            # create a checkpoint callback for the module
            curr_callback = ModularModelCheckpoint(
                dirpath =       dirpath,
                filename =      filename,
                **cfg_module,
                **kwargs
            )
            # include the module in the callback
            curr_callback.include_module(module_name)
            # append the callback to the list
            checkpoint_callback.append(curr_callback)

    return checkpoint_callback




class ModularModelCheckpoint(ModelCheckpoint):


    def include_module(self, module_name: str) -> None:
        """Include a module in the checkpointing process.
        """
        self._target_module = module_name


    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:

        # hijack the filepath to get the directory
        directory = Path(filepath).parent

        # generate new filepath for the single module
        new_filepath = directory / f'{PREFIX_MODULE}{self._target_module}.pt'

        # extract the module
        module = trainer.model.get_module(self._target_module)

        # save the module
        torch.save(module.state_dict(), new_filepath)