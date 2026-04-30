from typing import Dict, Tuple, Union, Optional, List, Callable, Any

import time
import os
from copy import copy, deepcopy

from logging import Logger
import wandb

import numpy as np

# utils for debugging
import sys

################  TORCH IMPORTS  #################
import torch
from torch import Tensor, IntTensor, LongTensor
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from torch_geometric.data import Data, Batch
import wandb.plot

##############  DATATYPES IMPORTS  ###############
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

###############  METRICS IMPORTS  ################
from src.models.ifh.losses.train_halting import HaltingLoss
from pytorch_lightning.loggers import WandbLogger

from src.models.generator import Generator, GeneratorWithEvaluation
from src.datatypes.features import get_features_list
from src.datatypes.features.core import Feature, increase_dims
from src.models import reg_models, reg_architectures
from src.evaluation.assignment.core import Assignment
from src.datatypes.features.core import increase_dims_list
from torch_geometric.utils import to_networkx

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

from src.datatypes.features.geometric import SinusoidalPosEmb

from src.models.mifh.batch_utils import (
    batch_softmax,
    batch_softmax_simple,
    batch_softmax_md,
    get_incr_tensor,
    batch_idx_to_hist,
    compute_hist
)
from src.models.mifh.degree_utils import (
    batch_edges_to_adj,
    batch_sample_sub_adj,
    batch_sample_sub_adj_avg_dist,
    create_fake_edge_index_batch
)
from src.models.mifh.distance_utils import (
    compute_distances
)
from src.models.mifh.modules import (
    GraphAdaptiveClassifier,
    BeliefUpdateClassifier
)
from src.models.mifh.properties import reg_mifh_properties

from src.models.mifh.losses.digamma_loss import DigammaWithLogitsLoss
from src.models.mifh.losses.dynamic_ce_loss import DynamicCrossEntropyLoss
from src.models.mifh.losses.link_loss import LinkLoss, LinkLossWithLogprobs

from src.models.mifh.sequence_sampler import sample_sequences

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

try:
    import plotly.graph_objects as go
    _GO_IMPORTED = True
except ImportError:
    _GO_IMPORTED = False

KEY_TRAIN = 'TRAIN'
KEY_VALID = 'VALID'
KEY_TEST = 'TEST'


@reg_models.register()
class ModularInsertFillHaltModel(GeneratorWithEvaluation):

    def __init__(
            self,

            ########### configurations ###########
            # model configurations
            encoder: Dict,
            new_degree: Dict,
            properties_network: Dict,
            properties: Dict,
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
        self.encoder_config = encoder
        self.new_degree_config = new_degree
        self.properties_network_config = properties_network
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

        self.encoded_dim = self.encoder_config.out_channels

        #############  ENCODER MODEL  ############

        # the encoder model encodes the input graphs nodes
        self.encoder = reg_architectures.get_instance(
            name =                  self.encoder_config.name,
            params =                self.encoder_config.params,
            input_dims =            self.augmented_dims,
            encoder_out_channels =  self.encoded_dim
        )

        ###########  NEW DEGREE MODEL  ###########

        # the new degree model predicts the degree of the new nodes
        self.new_degree_model = GraphAdaptiveClassifier(
            in_dim = self.encoded_dim,
            y_dim = self.augmented_dims['y'],
            mlp_cfg = self.new_degree_config.params,
            static_classes = 2 # account for halting probability and all nodes selected
        )

        ############  PROPERTIES MODEL  ##########

        self.properties_model = BeliefUpdateClassifier(
            n_properties=len(properties)+1,
            in_dim = self.encoded_dim,
            y_dim = self.augmented_dims['y'],
            **self.properties_network_config.params
        )

        self.properties_fn = [
            reg_mifh_properties.get_instance_from_dict(prop)
            for prop in properties
        ]


        ########################################################################
        #                                LOSSES                                #
        ########################################################################

        ###########################  TRAINING LOSSES  ##########################
        # save training loss
        self.losses = nn.ModuleDict()

        self.losses['new_deg'] = DynamicCrossEntropyLoss(**self.new_degree_config.loss)
        #self.losses['preferences'] = DigammaWithLogitsLoss(**self.preferences_config.loss)
        self.losses['properties'] = LinkLossWithLogprobs(**self.properties_network_config.loss)

        ###############################  METRICS  ##############################
        metrics = nn.ModuleDict({
            'new_deg': nn.ModuleDict({
                'new_deg_loss': MeanMetric(),
            }),
            'properties': nn.ModuleDict({
                'properties_loss': MeanMetric(),
            })
        })

        self.metrics = nn.ModuleDict({
            KEY_TRAIN: deepcopy(metrics),
            KEY_VALID: deepcopy(metrics),
            KEY_TEST: deepcopy(metrics)
        })

        # save hyperaparameters (but those not in the Generator ignored list)
        self.save_hyperparameters(ignore=Generator.IGNORED_HPARAMS)


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
        remv_edges_ba = seqs['remv_edges_ba']

        batch.y = None

        self.append_time(
            batch,
            time = batch.global_rev_t
        )

        ###########  FORMAT BEFORE BRANCHING INTO THE TWO TRAININGS  ###########
        self.add_additional_features(batch)

        return batch, remv_edges_ba, max_seq_len
    

    def encode_batch(self, batch: SparseGraph) -> SparseGraph:

        encoded_x = self.encoder(
            x =				batch.x,
            edge_index =	batch.edge_index,
            edge_attr =		batch.edge_attr,
            batch =			batch.batch,
            batch_size =	batch.num_graphs,
            y =				batch.y,
            num_nodes =     batch.num_nodes_per_sample
        )

        return encoded_x


    ############################################################################
    #                          TRAINING PHASE SECTION                          #
    ############################################################################


    def on_train_epoch_start(self) -> None:
        self.start_time = time.time()

    def on_train_epoch_end(self) -> None:
        """"Recall that this method is called AFTER the validation epoch, if there is any!"""
        metrics = {
            **self.metrics[KEY_TRAIN]['new_deg'],
            **self.metrics[KEY_TRAIN]['properties']
        }
        self.apply_prefix(metrics, 'train')
        self.log_dict(metrics)

        self.total_elapsed_time += time.time() - self.start_time
        self.max_memory_reserved = max(torch.cuda.max_memory_reserved(0), self.max_memory_reserved)



    def training_step(self, batch: SparseGraph, batch_idx: int):

        ###########################  INITIAL SETUP  ############################
        batch, remv_edges_ba, max_seq_len = self.prepare_batch(batch)

        train_loss = []
        logs = {}


        ############################  ENCODE NODES  ############################
        #TODO: check types of edge_attr
        encoded_x = self.encode_batch(batch)

        #######################  TRAIN NEW DEGREE MODEL  #######################

        new_degree_pred, new_ptr, new_batch = self.new_degree_model(
            encoded_x,
            batch.y,
            batch.batch,
            batch.num_nodes_per_sample,
            batch.ptr,
            return_new_ptr_batch=True
        )
        new_degree_true = batch.global_new_degree

        new_degree_loss = self.losses['new_deg'](
            input = new_degree_pred,
            target = new_degree_true,
            batch = new_batch,
            ptr = new_ptr
        )

        train_loss.append(new_degree_loss)
        logs['new_degree_loss'] = new_degree_loss
        self.metrics[KEY_TRAIN]['new_deg']['new_deg_loss'](new_degree_loss)


        ########################  REMOVE RANDOM EDGES  #########################
        edges_adj = batch_edges_to_adj(batch, remv_edges_ba) # (N,)
        linker_true = edges_adj
        # mask part of it with a uniform number of 0s per sample
        new_degree_corrected = torch.clip(batch.global_new_degree-1, 0) # remove the additional class
        linker_masked_adj, linker_masked_adj_next = batch_sample_sub_adj_avg_dist(batch, remv_edges_ba.edge_index[1], new_degree_corrected)

        # get links that were cancelled as ground truth
        nadj = linker_true - linker_masked_adj

        ##################  TRAIN PROPERTIES PREFERENCE MODEL  #################

        # compute properties
        properties = []
        for prop_fn in self.properties_fn:
            prop_value = prop_fn(batch, linker_masked_adj)
            properties.append(prop_value)

        properties.append(torch.zeros_like(linker_true).long()) # add a dummy property

        properties = torch.stack(properties, dim=0) # (n_props, N)

        # classify using properties
        properties_w = self.properties_model(
            encoded_x = encoded_x,
            global_y = batch.y,
            adj_vector = linker_masked_adj,
            batch = batch.batch,
            num_nodes_per_sample = batch.num_nodes_per_sample,
            properties = properties,
            return_logprobs=True
        )

        properties_loss = self.losses['properties'](
            w_nod = properties_w,
            #target = nadj,
            target = linker_masked_adj_next,
            batch = batch.batch,
            num_nodes = batch.num_nodes_per_sample
        )

        train_loss.append(properties_loss)
        logs['properties_loss'] = properties_loss
        self.metrics[KEY_TRAIN]['properties']['properties_loss'](properties_loss)

        logs = self.apply_prefix(logs, 'train_losses')

        self.log_dict(logs)

        return {'loss': sum(train_loss)}



    def configure_optimizers(self):

        return torch.optim.AdamW(
            params=self.parameters(),
            **self.optimizer_config
        )
    
    ############################################################################
    #                         VALID/TEST PHASE SECTION                         #
    ############################################################################

    @torch.no_grad()
    def on_evaluation_epoch_start(self, which=KEY_VALID) -> None:
        pass


    @torch.no_grad()
    def evaluation_step(self, batch: SparseGraph, batch_idx: int, which=KEY_VALID) -> None:
        
        ###########################  INITIAL SETUP  ############################
        batch, remv_edges_ba, max_seq_len = self.prepare_batch(batch)
        logs = {}


        ############################  ENCODE NODES  ############################
        #TODO: check types of edge_attr
        encoded_x = self.encode_batch(batch)

        #######################  TRAIN NEW DEGREE MODEL  #######################

        new_degree_pred, new_ptr, new_batch = self.new_degree_model(
            encoded_x,
            batch.y,
            batch.batch,
            batch.num_nodes_per_sample,
            batch.ptr,
            return_new_ptr_batch=True
        )
        new_degree_true = batch.global_new_degree

        new_degree_loss = self.losses['new_deg'](
            input = new_degree_pred,
            target = new_degree_true,
            batch = new_batch,
            ptr = new_ptr,
            reduce=False
        )

        self.metrics[which]['new_deg']['new_deg_loss'](new_degree_loss)

        ########################  REMOVE RANDOM EDGES  #########################
        edges_adj = batch_edges_to_adj(batch, remv_edges_ba) # (N,)
        linker_true = edges_adj
        # mask part of it with a uniform number of 0s per sample
        new_degree_corrected = torch.clip(batch.global_new_degree-1, 0) # remove the additional class
        linker_masked_adj, linker_masked_adj_next = batch_sample_sub_adj_avg_dist(batch, remv_edges_ba.edge_index[1], new_degree_corrected)

        # get links that were cancelled as ground truth
        nadj = linker_true - linker_masked_adj

        ##################  TRAIN PROPERTIES PREFERENCE MODEL  #################

        # compute properties
        properties = []
        for prop_fn in self.properties_fn:
            prop_value = prop_fn(batch, linker_masked_adj)
            properties.append(prop_value)

        properties.append(torch.zeros_like(linker_true).long()) # add a dummy property

        properties = torch.stack(properties, dim=0) # (n_props, N)

        # classify using properties
        properties_w = self.properties_model(
            encoded_x = encoded_x,
            global_y = batch.y,
            adj_vector = linker_masked_adj,
            batch = batch.batch,
            num_nodes_per_sample = batch.num_nodes_per_sample,
            properties = properties,
            return_logprobs=True
        )

        properties_loss = self.losses['properties'](
            w_nod = properties_w,
            #target = nadj,
            target = linker_masked_adj_next,
            batch = batch.batch,
            num_nodes = batch.num_nodes_per_sample,
            reduce=False
        )

        self.metrics[which]['properties']['properties_loss'](properties_loss)


    @torch.no_grad()
    def on_evaluation_epoch_end(self, which=KEY_VALID) -> None:

        if which == KEY_VALID:
            assignment = self.valid_assignment
        else:
            assignment = self.test_assignment

        # start with already computed metrics (during evaluation epochs)
        metrics = {
            **self.metrics[which]['new_deg'],
            **self.metrics[which]['properties']
        }

        if assignment is not None:

            batch_size = self.generation_config['batch_size']
        
            # compute sampling metrics
            assignment_results, hists, *others = self.perform_assignment(
                assignment=assignment, other_metrics=metrics,
                sampling_kwargs={'batch_size': batch_size},
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
    #                           MODEL CALL FUNCTIONS                           #
    ############################################################################

    @torch.no_grad()
    def forward_new_degree(
            self,
            batch: SparseGraph,
            encoded_x: Tensor,
            num_nodes: Tensor,
            batch_size: int,
            same_size_assumption: bool=False,
            return_logits: bool=False
        ) -> Tensor:

        """Forward pass of the new degree model."""
        # probs_new_degree has shape (N,) where N is the number of nodes
        logits_new_degree: Tensor = self.new_degree_model(
            encoded_x,
            batch.y,
            batch.batch,
            batch.num_nodes_per_sample,
            batch.ptr
        )

        # torch.set_printoptions(profile="full", linewidth=100000, precision=1)
        # self.console_logger.info(f'{logits_new_degree}')
        # torch.set_printoptions(profile="default")

        new_degrees = self.sample_dynamic_categorical(
            num_nodes = num_nodes,
            batch_size = batch_size,
            logits = logits_new_degree,
            same_size_assumption = same_size_assumption
        )
        if return_logits:
            return new_degrees, logits_new_degree
        else:
            return new_degrees
    

    @torch.no_grad()
    def sample_dynamic_categorical(
            self,
            num_nodes: Tensor,
            batch_size: int,
            probs: Tensor=None,
            logits: Tensor=None,
            same_size_assumption: bool=False,
            alive: Optional[Tensor]=None
        ) -> Tensor:
        
        using_logits = logits is not None

        if using_logits:
            x = logits
        else:
            x = probs

        if same_size_assumption:
            x = x.reshape(batch_size, -1) # shape (batch_size, N)
            if using_logits:
                x = F.softmax(x, dim=-1)

            if alive is not None:
                x[~alive] = 1 / x.shape[-1]

            # torch.set_printoptions(profile="full", linewidth=100000)
            # self.console_logger.info(f'{x.sum(dim=-1)}')
            # torch.set_printoptions(profile="default")

            sampled = torch.multinomial(x, 1).squeeze(-1)

        else:
            # split probs
            x = torch.split(x, num_nodes.tolist())

            if using_logits:
                x = [F.softmax(l, dim=-1) for l in x]

            # sample exactly batch_size new degrees from the distribution
            sampled = [torch.multinomial(p, 1).squeeze(-1) for p in x]
            sampled = torch.cat(sampled, dim=0)

        return sampled
    

    @torch.no_grad()
    def forward_preferences(
            self,
            encoded_x: Tensor,
            incr_degrees: Tensor,
            batch: SparseGraph
        ) -> Tensor:

        """Forward pass of the preferences model."""
        preferences: Tensor = self.preferences_model(encoded_x, batch.y, batch.batch, batch.num_nodes_per_sample, batch.ptr)

        # compute the weights for the degrees
        w_deg = self.compute_degree_weights_normalized_by_example(
            w_deg = preferences,
            batch = batch,
            hist = batch_idx_to_hist(batch, incr_degrees)
        )

        return w_deg
        

    def is_halting_disabled(self):
        return hasattr(self.training_config, 'fix_num_nodes') and self.training_config.fix_num_nodes



    # def make_histogram(self, values):
    #     labels = np.arange(len(values)+1)
    #     hist = wandb.Histogram(np_histogram=(values, labels))
    #     return hist

    def make_histogram(self, values):
        if not _GO_IMPORTED:
            return wandb.Html('Plotly not available, will not be able to log histograms')
        
        labels = np.arange(len(values)+1)
        # make a horizontal bar chart
        fig = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation='h'
        ))
        # remove padding/ margin
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            #xaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),
            #yaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),
            showlegend=False
        )
        # transform it to html
        html = fig.to_html(
            full_html=False,
            include_plotlyjs='cdn',
            default_height='100%',
            default_width='100%'
        )
        return wandb.Html(html)
    
    
    @torch.no_grad()
    def sample_batch(
        self,
        batch_size: int,
        maximum_insertion_steps: int=1000,
        maximum_number_nodes: int=None,
        return_directed: bool=True,
        save_chain: bool=False
    ):
        ########################################################################
        #                        INITIAL SAMPLING SETUP                        #
        ########################################################################

        if self.is_halting_disabled():
            maximum_number_nodes = self.dataset_info['num_nodes_max']

        ########################################################################
        #                SAMPLE THE STARTING GRAPH TO GENERATE                 #
        ########################################################################

        ############  SAMPLE THE STARTING GRAPHS (AS 1-NODE GRAPHS)  ###########

        # make collection of graphs with empty nodes and edges
        batch = [
            SparseGraph(
                x = 			torch.ones(1, 1, dtype=torch.long, device=self.device),
                edge_index = 	torch.empty((2, 0), dtype=torch.long, device=self.device),
                edge_attr = 	torch.empty(0, dtype=torch.long, device=self.device),
                y = None
            ) for _ in range(batch_size)
        ]

        # collate graphs
        batch = Batch.from_data_list(batch)

        ############  INITIALIZE INSERTION TIME AS 1 (REVERSED!)  ############

        insertion_time = torch.ones(batch_size, device=self.device)

        self.append_time(
            graph = batch,
            time = insertion_time
        )

        ########################################################################
        #                            INSERTION LOOP                            #
        ########################################################################

        # generated is equal to the number of nodes predicted by the insertion
        # model
        #for t in range(self.removal_process.get_max_time().max().item()):
        t = 1
        remaining_graphs_idx = torch.arange(batch_size, dtype=torch.long, device=self.device)

        # initialize batch of generated graphs
        output_batch = [None] * batch_size

        # initialize number of remaining graphs
        remaining_graphs_num = batch_size

        # initialize table
        if save_chain:
            props_col = [f'property {i}' for i in range(len(self.properties_fn)+1)]
            weights_col = [f'weight {i}' for i in range(len(self.properties_fn)+1)]
            interleaved = [val for pair in zip(props_col, weights_col) for val in pair]
            wb_degree_table = wandb.Table(columns=["time", "new_degree"], allow_mixed_types=True)
            wb_weights_table = wandb.Table(columns=["time", "num_edge", "adj", "probs"] + interleaved, allow_mixed_types=True)


        while t < maximum_insertion_steps and remaining_graphs_num > 0:

            ####################################################################
            #                    ENCODE THE CURRENT GRAPH                      #
            ####################################################################

            #prev_graph = graph.clone()
            prev_batch = copy(batch)
            starting_y = batch.y
            self.add_additional_features(prev_batch)

            # encode the current graph
            encoded_x = self.encode_batch(prev_batch)

            ##############  SAMPLE THE NEW DEGREE FOR THE NODES  ###############

            # compute probabilities and sample new degrees
            # recall, if a new_degree is 0, that is the signal for halting
            new_degrees, deg_logits = self.forward_new_degree(
                batch = prev_batch,
                encoded_x = encoded_x,
                num_nodes = prev_batch.num_nodes_per_sample,
                batch_size = remaining_graphs_num,
                same_size_assumption = True,
                return_logits = True
            )

            # if save_chain and output_batch[0] is None:
            #     deg_hist = self.make_histogram(deg_logits[:t+2].cpu().numpy())
            #     wb_degree_table.add_data(t, deg_hist)

            # torch.set_printoptions(profile="full", linewidth=100000, precision=1)
            # self.console_logger.info(f'{new_degrees}')
            # torch.set_printoptions(profile="default")

            # compute halting signal
            halt_signal = new_degrees == 0
            halt_nodes = halt_signal.repeat_interleave(prev_batch.num_nodes_per_sample, dim=0)

            stored_y = prev_batch.y
            
            ####################################################################
            #                      CHECK COMPLETED GRAPHS                      #
            ####################################################################
            # check if any of the graphs is completed
            if maximum_number_nodes is not None:
                halt_signal = torch.logical_or(
                    halt_signal,
                    batch.num_nodes_per_sample >= maximum_number_nodes
                )

            if t == maximum_insertion_steps:
                completed_graphs_mask = torch.ones_like(halt_signal, dtype=torch.bool)
            else:
                completed_graphs_mask = halt_signal.bool()
            

            ###########  IF SOME GRAPHS ARE COMPLETED, REMOVE THEM  ############
            # TODO: the following might be costly, check it!
            completed_graphs_num = completed_graphs_mask.sum().item()
            if completed_graphs_num > 0:

                graph_list = batch.to_data_list()

                # compute completed and remaining graphs indices
                remaining_graphs_mask = ~completed_graphs_mask
                completed_graphs_idx = remaining_graphs_idx[completed_graphs_mask]
                remaining_graphs_idx = remaining_graphs_idx[remaining_graphs_mask]
                remaining_graphs_num = remaining_graphs_idx.shape[0]

                # get completed and remaining graphs
                remaining_graphs = [graph_list[i] for i in torch.nonzero(remaining_graphs_mask).squeeze(-1)]
                completed_graphs = [graph_list[i] for i in torch.nonzero(completed_graphs_mask).squeeze(-1)]

                # insert finished graphs into the output batch
                for i, g in zip(completed_graphs_idx, completed_graphs):
                    if return_directed:
                        g.edge_index, g.edge_attr = sparse.to_directed(g.edge_index, g.edge_attr)
                    output_batch[i] = g

                if remaining_graphs_num == 0:
                    break
                # resume remaining batch
                batch = Batch.from_data_list(remaining_graphs)

            prev_batch = copy(batch)

            # lower degrees by 1
            # this is done because new_degree==0 is reserved for halting
            # while degree 0 is actually new_degree==1
            new_degrees = new_degrees[~halt_signal] - 1
            encoded_x = encoded_x[~halt_nodes]
            prev_batch.y = stored_y[~halt_signal]
            starting_y = starting_y[~halt_signal]

            num_nodes_per_sample = prev_batch.num_nodes_per_sample

            ####################  CREATE NEW EDGES TO NEW  #####################

            adj = torch.zeros((prev_batch.num_nodes), dtype=torch.float, device=self.device)

            # create new edges
            for s in range(new_degrees.max().item()):

                alive_edge_gen = new_degrees > s

                # compute properties
                properties = []
                for prop_fn in self.properties_fn:
                    prop_value = prop_fn(prev_batch, adj)
                    properties.append(prop_value)

                properties.append(torch.zeros(prev_batch.num_nodes, dtype=torch.long, device=self.device)) # add a dummy property

                properties = torch.stack(properties, dim=0) # (n_props, N)

                # classify using properties
                p_nod, logw = self.properties_model(
                    encoded_x = encoded_x,
                    global_y = prev_batch.y,
                    adj_vector = adj,
                    batch = prev_batch.batch,
                    num_nodes_per_sample = num_nodes_per_sample,
                    properties = properties,
                    return_weights=True
                )

                # torch.set_printoptions(profile="full", linewidth=100000, precision=1)
                # fill = torch.ones_like(p_nod)
                # self.console_logger.info(f'{torch.stack([fill, p_nod, w_nod[0], w_nod[1], w_nod[2], adj.float(), prev_batch.batch], dim=0)[:20]}')
                # from torch_geometric.nn import global_add_pool
                # self.console_logger.info(f'{global_add_pool(p_nod, prev_batch.batch)}')
                # torch.set_printoptions(profile="default")

                # sample the new edges
                new_edges = self.sample_dynamic_categorical(
                    num_nodes = num_nodes_per_sample,
                    batch_size = remaining_graphs_num,
                    probs = p_nod,
                    same_size_assumption = True,
                    alive = alive_edge_gen
                )

                # mask out the edges for graphs which stopped inserting edges
                new_edges = new_edges[alive_edge_gen]

                # increment edges by ptr
                new_edges += prev_batch.ptr[:-1][alive_edge_gen]

                # update with new edges
                adj[new_edges] = 1.

                if save_chain and output_batch[0] is None and alive_edge_gen[0]:
                    adj_hist = self.make_histogram(adj[:t].cpu().numpy())
                    prop_hists = [
                        self.make_histogram(properties[i, :t].cpu().numpy())
                        for i in range(properties.size(0))
                    ]
                    weights_hists = [
                        self.make_histogram(logw[i, :t].cpu().numpy())
                        for i in range(logw.size(0))
                    ]
                    p_hist = self.make_histogram(p_nod[:t].cpu().numpy())
                    interleave_pw = [val for pair in zip(prop_hists, weights_hists) for val in pair]
                    wb_weights_table.add_data(t, s, adj_hist, p_hist, *interleave_pw)


                assert torch.all(adj.reshape(remaining_graphs_num, -1).sum(-1) <= new_degrees), f'new_edges: {new_edges}, adj: {adj}, new_degrees: {new_degrees}'

            assert torch.all(adj.reshape(remaining_graphs_num, -1).sum(-1) == new_degrees), f'adj: {adj}, new_degrees: {new_degrees}'

            ################  MERGE THE OLD AND NEW SUBGRAPHS  #################

            # create new subgraph
            new_subgraph = SparseGraph(
                x = torch.ones(remaining_graphs_idx.size(0), 1, dtype=torch.long, device=self.device),
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device),
                edge_attr = torch.empty(0, dtype=torch.long, device=self.device),
                y = None,
                batch = torch.arange(remaining_graphs_num, dtype=torch.long, device=self.device),
            )

            # build edges from new to old
            nodes_idx = adj.nonzero().squeeze(-1)
            new_edge_index = torch.stack([
                prev_batch.batch[nodes_idx],
                nodes_idx
            ])
            new_ext_edges = SparseEdges(
                edge_index = new_edge_index,
                edge_attr = torch.zeros(nodes_idx.size(0), dtype=torch.long, device=self.device),
                #num_nodes_s = torch.ones_like(remaining_graphs_idx, dtype=torch.long),
                #num_nodes_t = prev_batch.num_nodes_per_sample,
                #num_nodes = prev_batch.num_nodes + remaining_graphs_num
            )

            # merge old graph with new subgraph
            batch = merge_graphs(
                ext_subgraph =      batch,
                new_subgraph =      new_subgraph,
                new_edges_ba =      new_ext_edges
            )
            batch.y = starting_y

            # update insertion time (in-place), insertion go up!
            t += 1
            self.change_time(batch, torch.full((remaining_graphs_num,), t, device=self.device))


        ########################  END OF INSERTION LOOP  #######################

        ########################################################################
        #                                RETURN                                #
        ########################################################################

        # insert remaining graphs into the output batch
        for i, g in zip(remaining_graphs_idx, remaining_graphs):
            if return_directed:
                g.edge_index, g.edge_attr = sparse.to_directed(g.edge_index, g.edge_attr)
            output_batch[i] = g

        # store the output batch to cpu
        for i, g in enumerate(output_batch):
            output_batch[i] = g.cpu()

        if save_chain:
            wandb.log({"gentables/degree_table": wb_degree_table})
            wandb.log({"gentables/weights_table": wb_weights_table})
        
        return output_batch


    @torch.no_grad()
    def sample(
            self,
            num_samples: int,
            condition: Optional[Dict]=None,
            batch_size: Optional[int]=None,
            log_images: int=10
        ):

        if batch_size is None:
            batch_size = self.generation_config['batch_size']

        samples_left_to_generate = num_samples
        batch_idx = 0
        samples = []

        while samples_left_to_generate > 0:
            to_generate = min(samples_left_to_generate, batch_size)
            self.console_logger.info(f'Generating {to_generate} graphs...')

            graph_batch = self.sample_batch(
                batch_size=to_generate,
                maximum_number_nodes=self.dataset_info['num_nodes_max'] * 2
            )

            samples.extend(graph_batch)

            samples_left_to_generate -= to_generate
            batch_idx += 1
            self.console_logger.info(f'Generated {len(samples)}/{num_samples} graphs')

        self.log_sampled_graphs(samples, how_many=log_images)

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
