from typing import Dict, Tuple, Union, Optional, List, Any

import time
from copy import copy, deepcopy

from logging import Logger

################  TORCH IMPORTS  #################
import torch
from torch import Tensor, IntTensor
import torch.nn as nn

import pytorch_lightning as pl

from torch_geometric.data import Data, Batch

##############  DATATYPES IMPORTS  ###############
from src.models.utils.empirical_blocks import EmpiricalBlockSampler
from src.datatypes import (
    sparse
)
from src.datatypes.dense import DenseGraph, DenseEdges
from src.datatypes.sparse import SparseGraph, SparseEdges
import src.datatypes.split as split

################  NOISE IMPORTS  #################
from src.noise.timesample import (
    resolve_timesampler
)
from src.noise.removal import (
    resolve_removal_process,
    resolve_removal_schedule
)
from src.noise.config_support import build_noise_process
from src.noise.batch_transform.sequence_sampler import sample_sequences

###############  METRICS IMPORTS  ################
from src.models.ifh.losses.train_insertion import (
    RegressionInsertionLoss,
    DistributionInsertionLoss,
    CategoricalInsertionLoss
)
from src.models.ifh.losses.train_halting import HaltingLoss

from src.models.generator import Generator, GeneratorWithEvaluation
from src.datatypes.features import get_features_list
from src.datatypes.features.core import Feature, increase_dims
from src.models import reg_models, reg_architectures
from src.evaluation.assignment.core import Assignment
from src.datatypes.features.core import increase_dims_list

from torchmetrics.aggregation import (
    MeanMetric
)

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryRecall
)
from src.models.ifh.metrics.test import (
    HaltingPriorEMD,
    OpenClassesAccuracy
)
import src.models.ifh.labels as labels

from src.datatypes.features.posenc import SinusoidalPosEmb

from src.models.utils.batch_ops import (
    assert_is_onehot
)
from src.models.utils.diffusion import (
    append_time_to_graph_globals,
    change_time_in_graph_globals
)

from src.noise import (
    reg_diffusion,
    reg_schedule,
    reg_timesampler
)

KEY_INSERTION = 'insertion'
KEY_HALTING = 'halting'
KEY_FILLER = 'filler'


KEY_TRAIN = 'TRAIN'
KEY_VALID = 'VALID'
KEY_TEST = 'TEST'


@reg_models.register()
class InsertFillHaltModel(GeneratorWithEvaluation):

    def __init__(
            self,

            ########### configurations ###########
            # model configurations
            insertion: Dict,
            filler: Dict,
            halting: Dict,
            removal: Dict,

            training: Dict,

            # optimizer configuration
            optimizer: Dict,
            
            # features configurations
            features: Dict = None,

            # generation configuration
            # e.g., conditional, batch size
            generation: Dict = None,

            # validation config
            validation: Dict = None,
            
            discard_conditioning: bool = True,
            check_metrics: bool = False, # check metrics for debugging purposes

            ######## passed by configurator ######
            dataset_info: Dict = None,
            test_assignment: Assignment = None,
            console_logger: Logger = None
        ):
        
        super().__init__(
            validation=validation,
            dataset_info=dataset_info,
            test_assignment=test_assignment,
            console_logger=console_logger
        )

        ############################  CONFIGS SETUP  ###########################

        # setup console logger
        self.console_logger = console_logger

        # setup config on how to build the model and noise processes
        self.insertion_config = insertion
        self.filler_config = filler
        self.halting_config = halting
        self.removal_config = removal

        self.training_config = training
        self.time_enc_dim = 16
        self.positional_embedding = SinusoidalPosEmb(self.time_enc_dim)

        # setup optimizer configuration
        self.optimizer_config = optimizer

        # setup additional features
        self.additional_features: List[Feature] = get_features_list(features) if features else []

        # setup generation
        self.generation_config = generation

        self.check_metrics = check_metrics
        self.discard_conditioning = discard_conditioning

        #######################  GRAPHS DIMENSIONS SETUP  ######################
        # setup model input and output dimensions (based on the dataset)
        self.data_dims = {
            'x': dataset_info['num_cls_nodes'],
            'e': dataset_info['num_cls_edges'],
            'y': 0 if discard_conditioning else dataset_info['dim_targets']
        }

        intermediate_dims = increase_dims(self.data_dims, {
            'y': self.time_enc_dim if self.using_pos_emb() else 1
            # account for removal time as a global y feature
        })

        # increase dimensions based on additional features (creates a copy)
        self.augmented_dims = increase_dims_list(intermediate_dims, self.additional_features)

        self.console_logger.info(f'{self.__class__.__name__} dimensions:')
        self.console_logger.info(f"Size of input features: {self.augmented_dims}")
        self.console_logger.info(f"Size of output features: {self.data_dims}")


        #########################  BUILD NOISE PROCESS  ########################

        self.removal_timesampler = reg_timesampler.get_instance_from_cfg(
            self.removal_config.timesampler
        )

        self.removal_process = reg_diffusion.get_instance_from_cfg(
            self.removal_config.process,
            schedule = reg_schedule.get_instance_from_cfg(
                self.removal_config.schedule
            )
        )

        if hasattr(self.removal_process.schedule, 'apply_dataset_info'):
            self.removal_process.schedule.apply_dataset_info(self.dataset_info)


        ############################  BUILD MODELS  ############################

        ###########  INSERTION MODEL  ############

        additional_kwargs = {}
        if hasattr(self.removal_process.schedule, 'categories') and \
            hasattr(self.insertion_config.architecture.params, 'ffn_config'):
            self.insertion_config.architecture.params.ffn_config['ffn_out_dim'] = \
                len(self.removal_process.schedule.categories)

        self.insertion_model = reg_architectures.get_instance(
            name =              self.insertion_config.architecture.name,
            params =            self.insertion_config.architecture.params,
            input_dims =        self.augmented_dims,
            dataset_info =      self.dataset_info,
            device =            self.device,
            **additional_kwargs
        )


        ############  HALTING MODEL  #############

        self.halting_model = reg_architectures.get_instance(
            name =              self.halting_config.architecture.name,
            params =            self.halting_config.architecture.params,
            input_dims =        self.augmented_dims,
            device =            self.device
        )


        #############  FILLER MODEL  #############
        # recall: the filler model is the sampler
        # the filler encoder encodes the past graph
        # in order to use it as conditioning for the
        # filler model

        self.filler_model: pl.LightningModule
        self.filler_model = reg_models.get_instance_from_dict(
            config =            self.filler_config.model,
            received_dims =     intermediate_dims,
            dataset_info =      self.dataset_info,
            test_assignment =   None,
            console_logger =    self.console_logger
        )

        self.filler_encoder = reg_architectures.get_instance(
            name =                  self.filler_config.encoder.name,
            params =                self.filler_config.encoder.params,
            input_dims =            self.augmented_dims,
            encoder_out_channels =  self.filler_model.get_external_nodes_dim()
        )

        # set the filler model logger as the same as the current logger
        self.filler_model.log = self.log

        ########################################################################
        #                                LOSSES                                #
        ########################################################################
            
        self.training_enabled = {
            KEY_INSERTION:  True,
            KEY_HALTING:    True,
            KEY_FILLER:     True
        }

        self.evaluating_enabled = deepcopy(self.training_enabled)

        self.losses = nn.ModuleDict()

        ###########################  TRAINING LOSSES  ##########################
        # save training loss
        self.losses = nn.ModuleDict()

        self.insertion_loss_label = ''

        # decide insertion loss depending on the model chosen
        if self.insertion_model.output_type == 'classifier':
            if hasattr(self.removal_process.schedule, 'get_posterior_distribution'):
                insertion_loss = DistributionInsertionLoss
                self.insertion_loss_label = labels.INSERTION_LOSS_KL
            else:
                insertion_loss = CategoricalInsertionLoss
                self.insertion_loss_label = labels.INSERTION_LOSS_CE
                
            self.node_regressive = False

        elif self.insertion_model.output_type == 'regressor':
            insertion_loss = RegressionInsertionLoss
            self.node_regressive = True
            self.insertion_loss_label = labels.INSERTION_LOSS_MSE

        self.losses[KEY_INSERTION] = insertion_loss(**self.insertion_config.loss)
        self.losses[KEY_HALTING] = HaltingLoss(**self.halting_config.loss)


        ################  ADDING CUMULATIVE EVALUATION METRICS  ################

        # the following code adds a set of cumulative evaluation metrics
        # for each enabled modules. The metrics to be added
        # are defined in the RUNTIME_METRICS_EVALUATION dictionary
        # the final structure will be:
        # self.losses[<phase>][<module_name>][<metric_name>]

        metrics = nn.ModuleDict({
            KEY_INSERTION: nn.ModuleDict({
                self.insertion_loss_label: MeanMetric(),
                labels.INSERTION_ACCURACY: OpenClassesAccuracy()
            }),
            KEY_HALTING: nn.ModuleDict({
                labels.HALTING_LOSS: MeanMetric(),
                labels.HALTING_ACCURACY: BinaryAccuracy(),
                labels.HALTING_RECALL: BinaryRecall(),
                **({
                    labels.HALTING_PRIOR_EMD: HaltingPriorEMD()
                } if self.halting_model.output_type != 'regressor' and False else {})
                
            })
        })

        self.metrics = nn.ModuleDict({
            KEY_TRAIN: deepcopy(metrics),
            KEY_VALID: deepcopy(metrics),
            KEY_TEST: deepcopy(metrics)
        })

        # save hyperaparameters (but those not in the Generator ignored list)
        self.save_hyperparameters(ignore=Generator.IGNORED_HPARAMS)


    def is_conditional(self):
        return self.generation_config['conditional']
    
    def is_node_empirical(self):
        return self.generation_config.get('node_empirical', False)
    

    def get_module(self, module_name: str) -> nn.Module:
        if module_name == KEY_INSERTION:
            return self.insertion_model
        elif module_name == KEY_HALTING:
            return self.halting_model
        elif module_name == KEY_FILLER:
            return self.denoising_model
        else:
            raise ValueError(f'Invalid module name {module_name}')
        

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint['training_enabled'] = self.training_enabled
        checkpoint['evaluating_enabled'] = self.evaluating_enabled

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        self.training_enabled = checkpoint['training_enabled']
        self.evaluating_enabled = checkpoint['evaluating_enabled']


    ############################################################################
    #                 SHORTHANDS FOR TRAINING/VALIDATION STEPS                 #
    ############################################################################


    def prepare_batch(self, batch: Union[Data, Dict[str, Data]]) -> Tuple[SparseGraph, SparseGraph, SparseGraph, SparseEdges]:

        seqs, max_seq_len = sample_sequences(
            batch = batch,
            removal_process = self.removal_process,
            num_subsamples = self.training_config.num_subsamples if hasattr(self.training_config, 'num_subsamples') else -1,
            return_first = self.training_config.first_subsampl if hasattr(self.training_config, 'first_subsampl') else True,
            need_preparation=False
        )

        batch = seqs['batch']
        surv_batch = seqs['surv_batch']
        remv_batch = seqs['remv_batch']
        remv_edges_ba = seqs['remv_edges_ba']

        if not self.is_conditional():
            batch.y = None
            surv_batch.y = None
            remv_batch.y = None

        self.append_time(
            batch,
            time = batch.global_rev_t
        )
        self.append_time(
            surv_batch,
            time = surv_batch.global_rev_t
        )
        self.append_time(
            remv_batch,
            time = remv_batch.global_rev_t
        )

        ###########  FORMAT BEFORE BRANCHING INTO THE TWO TRAININGS  ###########
        self.add_additional_features(batch)
        self.add_additional_features(surv_batch)

        return batch, surv_batch, remv_batch, remv_edges_ba, max_seq_len
    
    
    def compute_true_pred_insertion(
            self,
            batch: SparseGraph,
            remv_batch: SparseGraph,
        ) -> Tuple[List[Tensor], List[Tensor]]:
        """Generate the true and predicted properties of the insertion process:
        - true properties are generated from the removal process's posterior
        distribution.
        - predicted properties are generated by the insertion model from the
        survived batch graph.

        Parameters
        ----------
        batch : SparseGraph
            batch of graphs
        remv_batch : SparseGraph
            batch of removed subgraphs

        Returns
        -------
        true_props : List[Tensor]
            list of true properties
        pred_props : List[Tensor]
            list of predicted properties
        """
        ############################  CHECK INPUT  #############################
        # batch should be onehot
        assert batch.x.ndim == 2, f'Nodes are dimension {batch.x.ndim}, should be 2'
        assert batch.edge_attr.ndim == 2, f'Edges are dimension {batch.edge_attr.ndim}, should be 2'

        # generate the true probabilities
        # true_prob = self.removal_process.get_params_posterior(
        #     t = 		batch.global_t,
        #     max_time =	batch.global_n0
        # )

        # predict reverse process properties
        pred_props: Tensor = self.insertion_model(
            x =				batch.x,
            edge_index =	batch.edge_index,
            edge_attr =		batch.edge_attr,
            batch =			batch.batch,
            batch_size =	batch.num_graphs,
            y =				batch.y
        )

        loss_func = self.losses[KEY_INSERTION]

        if isinstance(loss_func, RegressionInsertionLoss):

            # get missing nodes to insert
            true_missing_nodes = (batch.global_n0 - batch.global_nt).float()

            # property to predict: number of missing nodes from true graph
            true_props = true_missing_nodes

        elif isinstance(loss_func, DistributionInsertionLoss):

            true_dist = self.removal_process.schedule.get_posterior_distribution(
                n0 = batch.global_n0,
                nt = batch.global_nt,
                t = batch.global_t
            )

            # property to predict: distribution on the moves to make to insert nodes
            true_props = true_dist

        elif isinstance(loss_func, CategoricalInsertionLoss):
                
            # get missing nodes to insert
            true_category = remv_batch.global_nt.unsqueeze(-1) == self.removal_process.schedule.categories.unsqueeze(0)
            true_category = true_category.nonzero()[:, 1]

            # property to predict: category of the missing nodes
            true_props = true_category

        else:
            raise ValueError(
                f'Invalid loss function {type(loss_func)} ' +\
                f'for insertion process {type(self.removal_process)} with '+\
                f'scheduler {type(self.removal_process.schedule)}, and model '+\
                f'with {self.insertion_model.out_properties} properties'
            )

        return true_props, pred_props
    

    def compute_true_pred_halting(
            self,
            batch: SparseGraph
        ) -> Tuple[List[Tensor], List[Tensor]]:
        """Generate the true and predicted properties of the halting part of removal:
        - true properties are generated as the halt signal at time t=0.
        - predicted properties are generated by the halting model from the current
        batch graph

        Parameters
        ----------
        batch : SparseGraph
            batch of graphs

        Returns
        -------
        true_props : List[Tensor]
            list of true properties
        pred_props : List[Tensor]
            list of predicted properties
        """
        ############################  CHECK INPUT  #############################
        # batch should be onehot
        assert batch.x.ndim == 2, f'Nodes are dimension {batch.x.ndim}, should be 2'
        assert batch.edge_attr.ndim == 2, f'Edges are dimension {batch.edge_attr.ndim}, should be 2'

        # predict reverse process properties
        pred_props: Tensor = self.halting_model(
            x =				batch.x,
            edge_index =	batch.edge_index,
            edge_attr =		batch.edge_attr,
            batch =			batch.batch,
            batch_size =	batch.num_graphs,
            y =				batch.y
        )

        # the halt signal indicates that the generator
        # should stop right at this point
        # during training: when time is t=0
        true_halt = (batch.global_t == 0).int()

        true_props = true_halt

        return true_props, pred_props
    

    @torch.no_grad()
    def compute_insertion_metrics(self, loss: Tensor, pred_values: Tensor, true_values: Tensor, split: str):

        pred_values = pred_values.detach()
        if true_values.ndim == 2:
            true_values = true_values.argmax(-1)

        # compute the metrics
        metrics = self.metrics[split][KEY_INSERTION]
        metrics[self.insertion_loss_label](loss[self.insertion_loss_label])
        metrics[labels.INSERTION_ACCURACY](pred_values, true_values)

        return metrics
    

    @torch.no_grad()
    def compute_halting_metrics(
            self, loss: Tensor, pred_values: Tensor, true_values: Tensor,
            batch_idx: Tensor, batch_size: int, max_seq_len: int,
            split: str
        ):

        pred_values = pred_values.detach()

        # compute the metrics
        metrics = self.metrics[split][KEY_HALTING]
        metrics[labels.HALTING_LOSS](loss[labels.HALTING_LOSS])
        metrics[labels.HALTING_ACCURACY](pred_values, true_values)
        metrics[labels.HALTING_RECALL](pred_values, true_values)
        if labels.HALTING_PRIOR_EMD in metrics:
            metrics[labels.HALTING_PRIOR_EMD](pred_values, true_values, batch_idx, batch_size, max_seq_len)

        return metrics


    def prepare_filler_batch(
            self,
            remv_batch: SparseGraph,
            surv_batch: SparseGraph,
            remv_edges_ba: SparseEdges
        ) -> Dict[str, Union[SparseGraph, SparseEdges]]:

        surv_batch_encoded = surv_batch

        surv_batch_encoded.x = self.filler_encoder(
            x =				surv_batch.x,
            edge_index =	surv_batch.edge_index,
            edge_attr =		surv_batch.edge_attr,
            batch =			surv_batch.batch,
            batch_size =	surv_batch.num_graphs,
            y =				surv_batch.y,
            num_nodes =     surv_batch.num_nodes_per_sample
        )

        batch = {
            'curr': remv_batch,
            'ext': surv_batch_encoded,
            'edges_curr_ext': remv_edges_ba
        }

        return batch


    ############################################################################
    #                          TRAINING PHASE SECTION                          #
    ############################################################################


    def on_train_epoch_start(self) -> None:
        self.filler_model.on_train_epoch_start()
        self.start_time = time.time()

    def on_train_epoch_end(self) -> None:
        """"Recall that this method is called AFTER the validation epoch, if there is any!"""

        self.filler_model.on_train_epoch_end()
        
        insertion_logs = self.apply_prefix(
            metrics = self.metrics[KEY_TRAIN][KEY_INSERTION],
            prefix = f'train_{KEY_INSERTION}'
        )
        halting_logs = self.apply_prefix(
            metrics = self.metrics[KEY_TRAIN][KEY_HALTING],
            prefix = f'train_{KEY_HALTING}'
        )
        self.log_dict({**insertion_logs, **halting_logs})

        self.total_elapsed_time += time.time() - self.start_time
        self.max_memory_reserved = max(torch.cuda.max_memory_reserved(0), self.max_memory_reserved)



    def training_step(self, batch: SparseGraph, batch_idx: int):

        ###########################  INITIAL SETUP  ############################
        true_bs = batch.num_graphs
        start_batch = batch
        batch, surv_batch, remv_batch, remv_edges_ba, max_seq_len = self.prepare_batch(batch)

        train_loss = []
        logs = {}

        ######################  TRAIN REINSERTION MODEL  #######################
        if self.training_enabled[KEY_INSERTION]:

            # FLOW DEFINITION
            # survived graph -> predict reverse process params -> match against true params

            # compute true and predicted params for the insertion process
            true_params, pred_params = self.compute_true_pred_insertion(
                batch = surv_batch,
                remv_batch = remv_batch
            )

            # compute insertion loss
            insertion_loss, insertion_logs = self.losses[KEY_INSERTION](
                pred_params,
                true_params,
                ret_log=True
            )

            # compute metrics
            self.compute_insertion_metrics(
                insertion_logs,
                pred_params,
                true_params,
                split=KEY_TRAIN
            )

            # apply prefix to logs
            insertion_logs = self.apply_prefix(
                metrics = self.metrics[KEY_TRAIN][KEY_INSERTION],
                prefix = f'train_{KEY_INSERTION}'
            )

            logs.update(insertion_logs)
            train_loss.append(insertion_loss)


        #######################  TRAIN HALTING MODEL  ##########################
        if self.training_enabled[KEY_HALTING]:

            # FLOW DEFINITION
            # batch -> predict halting signal -> match against true halting signal (i.e. t=0)

            # use true and predicted halting signals from the batch
            true_halting, pred_halting = self.compute_true_pred_halting(
                batch = batch
            )

            # compute halting loss
            halting_loss, halting_logs = self.losses[KEY_HALTING](
                pred_halting,
                true_halting,
                #dist = (batch.global_t + 1),
                ret_log=True
            )

            # compute metrics
            self.compute_halting_metrics(
                halting_logs, pred_halting, true_halting,
                batch.global_batch_idx, true_bs, max_seq_len,
                split=KEY_TRAIN
            )

            # apply prefix to logs
            halting_logs = self.apply_prefix(
                metrics = self.metrics[KEY_TRAIN][KEY_HALTING],
                prefix = f'train_{KEY_HALTING}'
            )

            logs.update(halting_logs)
            train_loss.append(halting_loss)


        ########################  TRAIN FILLER MODEL  ##########################
        if self.training_enabled[KEY_FILLER]:

            filler_batch = self.prepare_filler_batch(
                remv_batch = remv_batch,
                surv_batch = surv_batch,
                remv_edges_ba = remv_edges_ba
            )

            # training of filler model is delegated
            # to the filler model itself
            filler_logs = self.filler_model.training_step(
                batch = filler_batch,
                batch_idx = batch_idx
            )

            filler_loss = filler_logs['loss']
            train_loss.append(filler_loss)


        def check_metrics():
            apply_check_to = [
                f'train_{KEY_HALTING}/{labels.HALTING_ACCURACY}',
                f'train_{KEY_HALTING}/{labels.HALTING_RECALL}',
                f'train_{KEY_INSERTION}/{labels.INSERTION_ACCURACY}'
            ]
            check = {key: False for key in apply_check_to}
            any_check = False
            for key in apply_check_to:
                check[key] = logs[key].compute().cpu().item() < 0.2
                any_check = any_check or check[key]

            if any_check:

                # recompute losses without aggregation
                insertion_loss, insertion_logs = self.losses[KEY_INSERTION](
                    pred_params,
                    true_params,
                    reduce=False,
                    ret_log=True
                )
                halting_loss, halting_logs = self.losses[KEY_HALTING](
                    pred_halting,
                    true_halting,
                    reduce=False,
                    ret_log=True
                )

                self.console_logger.warning('PRINTING REPORT FOR DEBUGGING METRICS SPIKES')
                infos = dict(
                    batch_idx = batch_idx,
                    batch_size = true_bs,
                    check=check,
                    logs=logs,
                    start_batch=start_batch,
                    batch=batch,
                    surv_batch=surv_batch,
                    remv_batch=remv_batch,
                    remv_edges_ba=remv_edges_ba,
                    max_seq_len=max_seq_len,
                    true_params=true_params,
                    pred_params=pred_params,
                    true_halting=true_halting,
                    pred_halting=pred_halting,
                    insertion_loss=insertion_loss,
                    halting_loss=halting_loss
                )
                self.console_logger.warning(str(infos))
                # torch save with step index
                torch.save(infos, f'./metrics_spikes/spikes_{self.global_step}.pt')
                self.console_logger.warning('SAVED REPORT FOR DEBUGGING METRICS SPIKES')

        if self.check_metrics:
            check_metrics()

        self.log_dict(logs)

        return {'loss': sum(train_loss)}



    def configure_optimizers(self):

        # gather parameters for the optimizer
        ins_halt_params = list(self.insertion_model.parameters()) \
            + list(self.halting_model.parameters()) \
            + list(self.filler_encoder.parameters())
        filler_params = self.filler_model.parameters()

        return torch.optim.AdamW(
            params=[
                {'params': ins_halt_params, **self.optimizer_config},
                {'params': filler_params, **self.filler_model.optimizer_config}
            ]
        )
    
    ############################################################################
    #                         VALID/TEST PHASE SECTION                         #
    ############################################################################

    @torch.no_grad()
    def on_evaluation_epoch_start(self, which=KEY_VALID) -> None:

        if which == KEY_VALID:
            self.filler_model.on_validation_epoch_start()
        else:
            self.filler_model.on_test_epoch_start()

        # part used for gathering conditioning
        # attributes from the validation or test set
        # to be used for generation
        self.conditioning_y = None
        if self.is_conditional():
            self.conditioning_y = []
            self.num_cond_y = 0


    @torch.no_grad()
    def evaluation_step(self, batch: SparseGraph, batch_idx: int, which=KEY_VALID) -> None:

        #############  SAVE PROPERTIES FOR CONDITIONAL GENERATION  #############
        # save some target properties if needed for conditional generation
        if self.is_conditional():

            # get how many will be sampled
            sampling_metrics = self.losses['sampling']
            if which in sampling_metrics:
                sampling_metrics = sampling_metrics[which]

            num_to_sample = sampling_metrics.generation_cfg['num_samples']

            # get the conditioning attributes from the batch
            if self.num_cond_y < num_to_sample:
                to_grab = min(num_to_sample - self.num_cond_y, batch.num_graphs)
                self.conditioning_y.append(batch.y[:to_grab, -2:].float())
                self.num_cond_y += to_grab

        ###########################  INITIAL SETUP  ############################
        true_bs = batch.num_graphs
        batch, surv_batch, remv_batch, remv_edges_ba, max_seq_len = self.prepare_batch(batch)

        eval_loss = []

        ######################  TRAIN REINSERTION MODEL  #######################
        if self.evaluating_enabled[KEY_INSERTION]:

            # FLOW DEFINITION
            # survived graph -> predict reverse process params -> match against true params

            # compute true and predicted params for the insertion process
            true_params, pred_params = self.compute_true_pred_insertion(
                batch = surv_batch,
                remv_batch = remv_batch
            )

            # compute insertion loss
            insertion_loss, insertion_logs = self.losses[KEY_INSERTION](
                pred_params,
                true_params,
                reduce=False,
                ret_log=True
            )

            # compute metrics
            self.compute_insertion_metrics(
                insertion_logs,
                pred_params,
                true_params,
                split=which
            )

            # update metrics
            eval_loss.append(insertion_loss.mean())


        #######################  TRAIN HALTING MODEL  ##########################
        if self.evaluating_enabled[KEY_HALTING]:

            # FLOW DEFINITION
            # batch -> predict halting signal -> match against true halting signal (i.e. t=0)

            # use true and predicted halting signals from the batch
            true_halting, pred_halting = self.compute_true_pred_halting(
                batch = batch
            )

            # compute halting loss
            halting_loss, halting_logs = self.losses[KEY_HALTING](
                pred_halting,
                true_halting,
                reduce=False,
                ret_log=True
            )

            # compute metrics
            self.compute_halting_metrics(
                halting_logs, pred_halting, true_halting,
                batch.global_batch_idx, true_bs, max_seq_len,
                split=which
            )

            # update metrics
            eval_loss.append(halting_loss.mean())


        #######################  TRAIN DENOISING MODEL  ########################
        if self.evaluating_enabled[KEY_FILLER]:

            filler_batch = self.prepare_filler_batch(
                remv_batch = remv_batch,
                surv_batch = surv_batch,
                remv_edges_ba = remv_edges_ba
            )

            if which == KEY_VALID:
                filler_model_step = self.filler_model.validation_step
            else:
                filler_model_step = self.filler_model.test_step

            # run filler model evaluation
            filler_logs = filler_model_step(
                batch = filler_batch,
                batch_idx = batch_idx
            )

            filler_loss = filler_logs['loss']
            eval_loss.append(filler_loss)


        return {'loss': sum(eval_loss)}


    @torch.no_grad()
    def on_evaluation_epoch_end(self, which=KEY_VALID) -> None:

        if which == KEY_VALID:
            self.filler_model.on_validation_epoch_end()
            assignment = self.valid_assignment
        else:
            self.filler_model.on_test_epoch_end()
            assignment = self.test_assignment

        # start with already computed metrics (during evaluation epochs)
        metrics = {
            **self.metrics[which][KEY_INSERTION],
            **self.metrics[which][KEY_HALTING]
        }

        if assignment is not None:

            batch_size = self.generation_config['batch_size']
            save_chains = self.generation_config.get('save_chains', 0)
        
            # compute sampling metrics
            assignment_results, hists, *others = self.perform_assignment(
                assignment=assignment, other_metrics=metrics,
                sampling_kwargs={'batch_size': batch_size, 'save_chains': save_chains},
                return_samples=(which == KEY_TEST)
            )

            # add the assignment results to the metrics
            metrics.update(assignment_results)

            # log histograms with wandb if available (check is inside)
            self.log_wandb_histograms(self.apply_prefix(hists, f'{which}/sampling'))

        # output the metrics for reading purposes
        self.console_logger.info(str(self.get_metrics_values(metrics)))

        # add prefix to logs
        to_log = self.apply_prefix(
            metrics = metrics,
            prefix = f'{which}'
        )

        self.log_dict(to_log)
                
        if which == KEY_TEST:
            from pathlib import Path
            ckp_path = Path(self.trainer.log_dir)
            # extract the path, without the checkpoint name
            if ckp_path != '':
                ckp_path = ckp_path.parent.parent
                from src.configurator import store_graphs
                store_graphs(
                    graphs = others[0],
                    path = ckp_path
                )


    ############################################################################
    #           VALIDATION PHASE SECTION (executed during validation)          #
    ############################################################################

    def on_validation_epoch_start(self):
        self.on_evaluation_epoch_start(which=KEY_VALID)

    def validation_step(self, batch: SparseGraph, batch_idx: int):
        return self.evaluation_step(batch, batch_idx, which=KEY_VALID)

    def on_validation_epoch_end(self):
        return self.on_evaluation_epoch_end(which=KEY_VALID)

    ############################################################################
    #               TEST PHASE SECTION (executed during testing)               #
    ############################################################################

    def on_test_epoch_start(self):
        self.on_evaluation_epoch_start(which=KEY_TEST)

    def test_step(self, batch: SparseGraph, batch_idx: int):
        return self.evaluation_step(batch, batch_idx, which=KEY_TEST)

    def on_test_epoch_end(self):
        return self.on_evaluation_epoch_end(which=KEY_TEST)

    ############################################################################
    #                           CHECKPOINT FUNCTIONS                           #
    ############################################################################
    
    def get_training_status(self) -> str:
        st = [
            f"{k}: {'[X]' if v else '[ ]'}"
            for k, v in self.training_enabled.items()
        ]
        return 'Modules enabled for training: ' + ', '.join(st)


    ############################################################################
    #                           MODEL CALL FUNCTIONS                           #
    ############################################################################

    @torch.no_grad()
    def forward_insertion(
            self,
            graph: SparseGraph,
            reversed_insertion_time: IntTensor
        ) -> IntTensor:

        assert_is_onehot(graph)

        """Forward pass of the insertion model."""
        # predict final number of nodes from
        # current graph

        pred_props: Tensor = self.insertion_model(
            x =				graph.x,
            edge_index =	graph.edge_index,
            edge_attr =		graph.edge_attr,
            batch =			graph.batch if hasattr(graph, 'batch') else None,
            batch_size =	graph.num_graphs if hasattr(graph, 'num_graphs') else None,
            y =				graph.y
        )

        if self.node_regressive:
            sampled_num_new_nodes, new_time, insertion_time = self._sample_new_nodes_regressive(
                pred_props =                pred_props,
                graph =                     graph,
                reversed_insertion_time = reversed_insertion_time
            )

        else:
            sampled_num_new_nodes, new_time, insertion_time = self._sample_new_nodes_distribution(
                pred_props =                pred_props,
                graph =                     graph,
                reversed_insertion_time = reversed_insertion_time
            )


        return sampled_num_new_nodes, new_time, insertion_time, pred_props
    

    def _sample_new_nodes_regressive(self, pred_props: Tensor, graph: SparseGraph, reversed_insertion_time: IntTensor):
        # transform the regression property to discrete prediction
        pred_num_remaining_nodes = torch.round(torch.relu(pred_props)).int()

        # compute number nodes currently (nt)
        num_nodes = graph.ptr[1:] - graph.ptr[:-1]
        pred_n0 = pred_num_remaining_nodes + num_nodes

        # compute the correct insertion time by reversing the reversed time
        insertion_time = self.removal_process.get_schedule().reverse_step(
            t =		reversed_insertion_time,
            n0 =	pred_n0
        )

        # sample number of nodes to add
        # using posterior of removal process
        sampled_num_new_nodes = self.removal_process.sample_noise_posterior(
            original_datapoint =    pred_n0,
            current_datapoint =     graph,
            t =                     insertion_time,
            return_quantity =       True
        )

        # get new time
        new_time = self.removal_process.normalize_time(
            t = reversed_insertion_time+1,
            n0 = pred_n0
        )

        return sampled_num_new_nodes, new_time, insertion_time
    

    def _sample_new_nodes_distribution(self, pred_props: Tensor, graph: SparseGraph, reversed_insertion_time: IntTensor):
        
        # transform the logits property to a distribution
        pred_nodes_logits = pred_props

        if pred_nodes_logits.ndim == 1:
            pred_nodes_logits = pred_nodes_logits.unsqueeze(-1)

        # these would be the probabilities from the logits
        #pred_nodes_probs = torch.softmax(pred_nodes_logits, dim=-1)

        # sample from the removal process
        # actually a categorical with a final mapping to the right
        # number of nodes
        sampled_num_new_nodes = self.removal_process.schedule.sample_nodes_from_dist(
            logits = pred_nodes_logits
        )

        #print('sampled_num_new_nodes', sampled_num_new_nodes)

        # get new time
        new_time = self.removal_process.normalize_time(
            t = reversed_insertion_time+1
        )

        return sampled_num_new_nodes, new_time, None
    

    def forward_halting(
            self,
            graph: SparseGraph
        ) -> IntTensor:

        assert_is_onehot(graph)

        """Forward pass of the halting model."""
        # predict final number of nodes from
        # current graph
        #print(graph)
        #print(graph.y)
        pred_halt_logits: Tensor = self.halting_model(
            x =				graph.x,
            edge_index =	graph.edge_index,
            edge_attr =		graph.edge_attr,
            batch =			graph.batch if hasattr(graph, 'batch') else None,
            batch_size =	graph.num_graphs if hasattr(graph, 'num_graphs') else None,
            y =				graph.y
        )

        if self.halting_model.output_type == 'regressor':
            # the output of the network is the halting signal
            halt_signal = pred_halt_logits
        else:
            # the output of the network is the logits for the probability of halting
            halt_signal = torch.distributions.Bernoulli(
                logits=pred_halt_logits
            ).sample()


        return halt_signal, pred_halt_logits
        

    def is_halting_disabled(self):
        return hasattr(self.training_config, 'fix_num_nodes') and self.training_config.fix_num_nodes
    
    @torch.no_grad()
    def sample_batch(
        self,
        batch_size: int,
        conditioning_y: Optional[Tensor]=None,
        maximum_insertion_steps: int=1000,
        maximum_number_nodes: int=None,
        return_directed: bool=True,
        chains: List[Dict[str, Any]] = None,
    ):
        ########################################################################
        #                        INITIAL SAMPLING SETUP                        #
        ########################################################################

        if self.is_halting_disabled():
            maximum_number_nodes = self.dataset_info['num_nodes_max']

        # TODO: implement the generation chain saving
        do_save_chains = chains is not None

        ########################################################################
        #                SAMPLE THE STARTING GRAPH TO GENERATE                 #
        ########################################################################

        ############  SAMPLE THE STARTING GRAPHS (AS EMPTY GRAPHS)  ############
        initialization = {}
        if conditioning_y is not None:
            initialization['y'] = conditioning_y
        else:
            initialization['y'] = torch.empty((batch_size, 0), dtype=torch.float, device=self.device)

        # generate the starting batch of graphs
        graph: SparseGraph
        graph = self.removal_process.sample_stationary(
            batch_size = batch_size,
            initialization = initialization,
            device = self.device
        ).to_onehot(
            num_classes_x =	self.data_dims['x'],
            num_classes_e =	self.data_dims['e']
        )

        # initialize the global time
        if conditioning_y is not None:
            graph.y = conditioning_y
        else:
            graph.y = None

        ############  INITIALIZE INSERTION TIME AS 0 (REVERSED!)  ############

        insertion_time = torch.zeros(batch_size, device=self.device)

        self.append_time(
            graph = graph,
            time = insertion_time
        )

        ########################################################################
        #                            INSERTION LOOP                            #
        ########################################################################

        # generated is equal to the number of nodes predicted by the insertion
        # model
        t = 0
        remaining_graphs_idx = torch.arange(batch_size, dtype=torch.long, device=self.device)

        # initialize batch of generated graphs
        output_batch = [None] * batch_size

        # initialize number of remaining graphs
        remaining_graphs_num = batch_size
        
        empirical_block_sampler = None
        if self.is_node_empirical():
            empirical_block_sampler = EmpiricalBlockSampler(
                cat_remv_process=self.removal_process,
                dataset_info=self.dataset_info,
                device=self.device
            )
            empirical_block_sampler.initialize(batch_size=batch_size)
        
        
        if do_save_chains:
            chain_info = {
                'time': t,
                'graph': graph.clone().cpu(),
                'remaining_graphs_idx': remaining_graphs_idx.clone().cpu(),
                'nodes_inserted': None,
                'pred_insert': None,
                'halt_signal': None,
                'pred_halt': None
            }
            chains.append(chain_info)

        while t < maximum_insertion_steps and remaining_graphs_num > 0:

            ####################################################################
            #                SAMPLE THE NUMBER OF NODES TO ADD                 #
            ####################################################################

            #prev_graph = graph.clone()
            prev_graph = copy(graph)
            self.add_additional_features(prev_graph)

            # sample the number of nodes to remove
            nodes_to_insert: IntTensor
            
            reversed_insertion_time = torch.full((remaining_graphs_num,), t, dtype=torch.int, device=self.device)
            
            if self.is_node_empirical():
                nodes_to_insert = empirical_block_sampler.sample_block()
                new_time = empirical_block_sampler.get_new_time(
                    reversed_insertion_time = reversed_insertion_time
                )
                true_insertion_time = None
                pred_props = None
            else:
                nodes_to_insert, new_time, true_insertion_time, pred_props = self.forward_insertion(
                    graph =                     prev_graph,
                    reversed_insertion_time =   reversed_insertion_time
                )
            

            if nodes_to_insert.sum() > 0:
                ################################################################
                #            SAMPLE SUBGRAPH FROM THE FILLER MODEL             #
                ################################################################

                ########  PRE-ENCODE THE PREVIOUS GRAPH ONTO ITS NODES  ########
                encoded_ext_x = self.filler_encoder(
                    x =				prev_graph.x,
                    edge_index =	prev_graph.edge_index,
                    edge_attr =		prev_graph.edge_attr,
                    batch =			prev_graph.batch,
                    batch_size =	prev_graph.num_graphs,
                    y =				prev_graph.y,
                    num_nodes =     prev_graph.num_nodes_per_sample
                )

                ##########  SAMPLE THE SUBGRAPH FROM THE FILLER MODEL  #########
                # graph gets new <nodes_to_insert> nodes

                new_subgraph, new_ext_edges = self.filler_model.sample_batch(
                    batch_size =        remaining_graphs_num,
                    ext_graph =         graph,
                    encoded_ext_x =     encoded_ext_x,
                    number_of_nodes =   nodes_to_insert,
                    save_chains =       0
                )

                ##############  MERGE THE OLD AND NEW SUBGRAPHS  ###############

                graph = merge_graphs(
                    ext_subgraph =      graph,
                    new_subgraph =      new_subgraph,
                    new_edges_ba =      new_ext_edges
                )


            ######  COMPUTE HALTING SIGNAL  ######

            # update insertion time (in-place), insertion go up!
            self.change_time(graph, new_time)
            t += 1
            
            if true_insertion_time is not None:
                halt_signal = true_insertion_time <= 1
            elif self.halting_model is not None and not self.is_halting_disabled():
                curr_graph = graph.clone()
                #curr_graph = copy(graph)
                self.add_additional_features(curr_graph)

                if self.is_node_empirical():
                    halt_signal = empirical_block_sampler.should_halt(curr_graph)
                    pred_halt_logits = None
                else:
                    halt_signal, pred_halt_logits = self.forward_halting(
                        graph = curr_graph
                    )
            else:
                halt_signal = torch.zeros_like(remaining_graphs_idx, dtype=torch.bool)


            ####################################################################
            #                      CHECK COMPLETED GRAPHS                      #
            ####################################################################
            # check if any of the graphs is completed
            if maximum_number_nodes is not None:
                halt_signal = torch.logical_or(
                    halt_signal,
                    graph.num_nodes_per_sample >= maximum_number_nodes
                )

            if t == maximum_insertion_steps:
                completed_graphs_mask = torch.ones_like(halt_signal, dtype=torch.bool)
            else:
                completed_graphs_mask = halt_signal.bool()
                
            
            if do_save_chains:
                chain_info = {
                    'time': t,
                    'graph': graph.clone().cpu(),
                    'remaining_graphs_idx': remaining_graphs_idx.clone().cpu(),
                    'nodes_inserted': nodes_to_insert.clone().cpu() if nodes_to_insert is not None else None,
                    'pred_insert': pred_props.clone().cpu() if pred_props is not None else None,
                    'halt_signal': halt_signal.clone().cpu() if halt_signal is not None else None,
                    'pred_halt': pred_halt_logits.clone().cpu() if pred_halt_logits is not None else None
                }
                chains.append(chain_info)
            

            ###########  IF SOME GRAPHS ARE COMPLETED, REMOVE THEM  ############
            # TODO: the following might be costly, check it!
            completed_graphs_num = completed_graphs_mask.sum().item()
            if completed_graphs_num > 0:

                graph_list = graph.to_data_list()

                # compute completed and remaining graphs indices
                remaining_graphs_mask = ~completed_graphs_mask
                completed_graphs_idx = remaining_graphs_idx[completed_graphs_mask]
                remaining_graphs_idx = remaining_graphs_idx[remaining_graphs_mask]
                remaining_graphs_num = remaining_graphs_idx.shape[0]
                new_time = new_time[remaining_graphs_mask]

                # get completed and remaining graphs
                remaining_graphs = [graph_list[i] for i in torch.nonzero(remaining_graphs_mask).squeeze(-1)]
                completed_graphs = [graph_list[i] for i in torch.nonzero(completed_graphs_mask).squeeze(-1)]

                # insert finished graphs into the output batch
                for i, g in zip(completed_graphs_idx, completed_graphs):
                    if return_directed:
                        g.edge_index, g.edge_attr = sparse.to_directed(g.edge_index, g.edge_attr)

                    if conditioning_y is not None:
                        g.y = conditioning_y[i]
                    output_batch[i] = g.collapse()

                # if remaining_graphs_num == 0:
                #     break
                # resume remaining batch
                if remaining_graphs_num > 0:
                    graph = Batch.from_data_list(remaining_graphs)

                    if graph.y.ndim == 1:
                        graph.y = graph.y.unsqueeze(-1)
                
                
            if self.is_node_empirical():
                empirical_block_sampler.update(
                    graphs = graph,
                    mask = ~completed_graphs_mask
                )

        ########################  END OF INSERTION LOOP  #######################

        ########################################################################
        #                                RETURN                                #
        ########################################################################

        # insert remaining graphs into the output batch
        for i, g in zip(remaining_graphs_idx, remaining_graphs):
            if return_directed:
                g.edge_index, g.edge_attr = sparse.to_directed(g.edge_index, g.edge_attr)
            output_batch[i] = g.collapse()

        # store the output batch to cpu
        for i, g in enumerate(output_batch):
            output_batch[i] = g.cpu()

        # replace globals with starting variables, removing time
        if conditioning_y is None:
            for i in range(batch_size):
                output_batch[i].y = None
        else:
            for i in range(batch_size):
                output_batch[i].y = conditioning_y[i]
        
        return output_batch


    @torch.no_grad()
    def sample(
            self,
            num_samples: int,
            condition: Optional[Dict]=None,
            batch_size: Optional[int]=None,
            log_images: int = 10,
            save_chains: int = 0
        ):

        if batch_size is None:
            batch_size = self.generation_config['batch_size']

        samples_left_to_generate = num_samples
        batch_idx = 0
        samples = []
        chains = []

        while samples_left_to_generate > 0:
            to_generate = min(samples_left_to_generate, batch_size)
            self.console_logger.info(f'Generating {to_generate} graphs...')

            new_chains = []

            graph_batch = self.sample_batch(
                batch_size=to_generate,
                conditioning_y=condition[batch_idx] if condition is not None else None,
                maximum_number_nodes=self.dataset_info['num_nodes_max'] * 2,
                chains=new_chains if save_chains > 0 else None,
            )

            samples.extend(graph_batch)
            if save_chains > 0:
                chains.append(new_chains)

            samples_left_to_generate -= to_generate
            save_chains = max(0, save_chains - to_generate)
            batch_idx += 1
            self.console_logger.info(f'Generated {len(samples)}/{num_samples} graphs')

        self.log_sampled_graphs(samples, how_many=log_images)
        
        if len(chains) > 0:
            from pathlib import Path
            ckp_path = Path(self.trainer.log_dir)
            # extract the path, without the checkpoint name
            if ckp_path != '':
                ckp_path = ckp_path.parent.parent
                from src.configurator import store_file
                store_file(
                    content = chains,
                    path = ckp_path,
                    filename = 'chains.pkl',
                    append_datetime=True
                )

        return samples
    

    ############################################################################
    #                         UTILITY MODULE FUNCTIONS                         #
    ############################################################################


    def add_additional_features(self, graph: SparseGraph|DenseGraph|Tuple[DenseGraph, DenseEdges]) -> Tensor:

        for feature in self.additional_features:
            feature(graph)

        return graph


    def using_pos_emb(self):
        return hasattr(self.training_config, 'embed_time') and self.training_config.embed_time


    def append_time(self, graph, time):
        if self.using_pos_emb():
            emb = self.positional_embedding
        else:
            emb = None

        append_time_to_graph_globals(graph, time, emb)


    def change_time(self, graph, time):
        if self.using_pos_emb():
            emb = self.positional_embedding
        else:
            emb = None

        change_time_in_graph_globals(graph, time, emb)



def merge_graphs(
        ext_subgraph: SparseGraph,
        new_subgraph: SparseGraph,
        new_edges_ba: SparseEdges
    ) -> SparseGraph:

    if new_edges_ba is not None:

        # get both directions of the new edges
        new_edges_ab = new_edges_ba.clone().transpose()

        # merge the sparse graph with the sparsified dense graph
        merged_graph = split.merge_subgraphs(
            graph_a =	ext_subgraph,
            graph_b =	new_subgraph,
            edges_ab =	new_edges_ab,
            edges_ba =	new_edges_ba
        )

    else:
        merged_graph = new_subgraph

    return merged_graph