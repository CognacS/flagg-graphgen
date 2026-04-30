from typing import Dict, Tuple, Union, Optional, List, Any

import time
from copy import copy, deepcopy

from logging import Logger

################  TORCH IMPORTS  #################
import torch
from torch import Tensor, IntTensor, LongTensor
import torch.nn as nn


from torch_geometric.data import Data, Batch

##############  DATATYPES IMPORTS  ###############
from src.datatypes import (
    sparse
)

from src.datatypes.sparse import SparseGraph, SparseEdges

################  NOISE IMPORTS  #################
from src.noise.batch_transform.sequence_sampler import sample_sequences
from src.noise.batch_transform.sequence_sampler_sparser import sample_sequences_sparser

###############  METRICS IMPORTS  ################

from src.models.generator import Generator
from src.models import reg_models, reg_architectures
from src.evaluation.assignment.core import Assignment

from torchmetrics.aggregation import (
    MeanMetric
)

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryRecall,
    BinaryPrecision
)

from src.datatypes.batch import compute_cum_elems

from src.models.ifh.model import *
import src.models.ifh_sparser.labels as labels


KEY_SELECTION = 'selection'




@reg_models.register()
class InsertFillHaltModelSparser(InsertFillHaltModel):

    def __init__(
            self,

            ########### configurations ###########
            # model configurations
            insertion: Dict,
            selector: Dict,
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
            insertion, filler, halting, removal, training, optimizer,
            features, generation, validation, discard_conditioning, check_metrics,
            dataset_info, test_assignment, console_logger
        )

        self.selector_config = selector

        self.selector_encoder = reg_architectures.get_instance(
            name =                  self.filler_config.encoder.name,
            params =                self.filler_config.encoder.params,
            input_dims =            self.augmented_dims,
            encoder_out_channels =  self.filler_model.get_external_nodes_dim()
        )

        sel_model_inp_dim = self.filler_model.get_external_nodes_dim()
        sel_model_inp_dim += self.insertion_model.ffn_out_dim # add information about nodes to insert

        self.selector_model = nn.Sequential(
            nn.Linear(sel_model_inp_dim, self.selector_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.selector_config.hidden_dim, 1)
        )
        

        if hasattr(training, 'pos_weight'):
            pos_weight = training.pos_weight
        else:
            pos_weight = 4 # set to 4 for now
        self.losses[KEY_SELECTION] = nn.BCEWithLogitsLoss(
            pos_weight=torch.full((1,), pos_weight)
        )

        metrics = nn.ModuleDict({
            labels.SELECTION_LOSS: MeanMetric(),
            labels.SELECTION_ACCURACY: BinaryAccuracy(),
            labels.SELECTION_RECALL: BinaryRecall(),
            labels.SELECTION_PRECISION: BinaryPrecision()
        })

        for split in [KEY_TRAIN, KEY_VALID, KEY_TEST]:
            self.metrics[split][KEY_SELECTION] = deepcopy(metrics)

        # save hyperaparameters (but those not in the Generator ignored list)
        self.save_hyperparameters(ignore=Generator.IGNORED_HPARAMS)


    ############################################################################
    #                 SHORTHANDS FOR TRAINING/VALIDATION STEPS                 #
    ############################################################################


    def prepare_batch(self, batch: Union[Data, Dict[str, Data]]) -> Tuple[SparseGraph, SparseGraph, SparseGraph, SparseEdges]:

        if hasattr(self.training_config, 'perm_select') and self.training_config.perm_select:
            sample_seq = sample_sequences_sparser # will also return the node selection for permanent selection
        else:
            sample_seq = sample_sequences

        seqs, max_seq_len = sample_seq(
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

    

    @torch.no_grad()
    def compute_selection_metrics(
            self, loss: Tensor, pred_values: Tensor, true_values: Tensor,
            split: str
        ):

        pred_values = pred_values.detach()

        # compute the metrics
        metrics = self.metrics[split][KEY_SELECTION]
        metrics[labels.SELECTION_LOSS](loss.detach())
        metrics[labels.SELECTION_ACCURACY](pred_values, true_values)
        metrics[labels.SELECTION_RECALL](pred_values, true_values)
        metrics[labels.SELECTION_PRECISION](pred_values, true_values)

        return metrics


    def prepare_filler_batch(
            self,
            remv_batch: SparseGraph,
            surv_batch: SparseGraph,
            remv_edges_ba: SparseEdges
        ) -> Dict[str, Union[SparseGraph, SparseEdges]]:

        surv_batch_encoded = copy(surv_batch)

        surv_x = self.filler_encoder(
            x =				surv_batch.x,
            edge_index =	surv_batch.edge_index,
            edge_attr =		surv_batch.edge_attr,
            batch =			surv_batch.batch,
            batch_size =	surv_batch.num_graphs,
            y =				surv_batch.y,
            num_nodes =     surv_batch.num_nodes_per_sample
        )

        ######################  PREDICT NODES TO SELECT  #######################
        enc_select_x = self.selector_encoder(
            x =				surv_batch.x,
            edge_index =	surv_batch.edge_index,
            edge_attr =		surv_batch.edge_attr,
            batch =			surv_batch.batch,
            batch_size =	surv_batch.num_graphs,
            y =				surv_batch.y,
            num_nodes =     surv_batch.num_nodes_per_sample
        )

        # repeat y for each node in the examples
        if not self.node_regressive:
            # encode the numbers of nodes which were removed as onehot vectors
            num_nodes_emb = self.removal_process.schedule.coin_change_dist.get_onehot_coin(
                remv_batch.num_nodes_per_sample
            )
        else:
            num_nodes_emb = remv_batch.num_nodes_per_sample.unsqueeze(-1)
        
        num_nodes_emb = num_nodes_emb.repeat_interleave(surv_batch.num_nodes_per_sample, dim=0)
        enc_select_x = torch.cat([enc_select_x, num_nodes_emb], dim=-1)

        # make predictions and true values
        pred_selected = self.selector_model(enc_select_x).squeeze(-1)

        if hasattr(self.training_config, 'perm_select') and self.training_config.perm_select:

            # select nodes as those which are missing links
            idx_selected = torch.nonzero(surv_batch.node_select).squeeze(-1)
            mapping_new_edge_index = torch.cat([
                torch.zeros(1, device=self.device),
                torch.cumsum(surv_batch.node_select, dim=0)
            ])
            new_edge_index_1 = mapping_new_edge_index[remv_edges_ba.edge_index[1]]

            true_selected = surv_batch.node_select

        else:
            # create index of true selected nodes from remv_edges_ba
            # also return new indices
            idx_selected_true = torch.unique(remv_edges_ba.edge_index[1], return_inverse=False)
            # oversampling of selected nodes
            tot_nodes_num = surv_batch_encoded.x.shape[0]
            selected_nodes_num = idx_selected_true.shape[0]
            added_nodes_num = int(selected_nodes_num * 0.2) + 1 # every 10 selected there are 2 added

            true_selected = torch.zeros(tot_nodes_num, device=self.device)
            true_selected[idx_selected_true] = 1.

            missing_nodes = torch.arange(
                end=tot_nodes_num, device=self.device
            )[~true_selected.bool()]

            perm = torch.randperm(tot_nodes_num - selected_nodes_num)
            added_nodes = missing_nodes[perm[:added_nodes_num]]

            oversampled_nodes = torch.cat([remv_edges_ba.edge_index[1], added_nodes])

            # repeat procedure
            idx_selected, new_edge_index_1 = torch.unique(oversampled_nodes, return_inverse=True)

            # remove fake indices
            new_edge_index_1 = new_edge_index_1[:-added_nodes_num]


            true_selected = torch.zeros_like(pred_selected)
            true_selected[idx_selected_true] = 1.

        # select the nodes that are actually linked
        surv_batch_encoded.x = surv_x[idx_selected]
        surv_batch_encoded.batch = surv_batch.batch[idx_selected]
        #self.console_logger.info(f'{surv_batch_encoded.node_depth[idx_selected]}')
        surv_batch_encoded.node_indegree = surv_batch.indegree[idx_selected]
        surv_batch_encoded.global_num_nodes = surv_batch.num_nodes_per_sample
        # map the selected nodes to the new indices
        remv_edges_ba.edge_index[1] = new_edge_index_1


        batch = {
            'curr': remv_batch,
            'ext': surv_batch_encoded,
            'edges_curr_ext': remv_edges_ba
        }

        #self.console_logger.info(f'edge_attr: {remv_batch.edge_attr}')

        return batch, true_selected, pred_selected


    ############################################################################
    #                          TRAINING PHASE SECTION                          #
    ############################################################################



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
        selection_logs = self.apply_prefix(
            metrics = self.metrics[KEY_TRAIN][KEY_SELECTION],
            prefix = f'train_{KEY_SELECTION}'
        )
        self.log_dict({**insertion_logs, **halting_logs, **selection_logs})

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

            filler_batch, true_selected, pred_selected = self.prepare_filler_batch(
                remv_batch = remv_batch,
                surv_batch = surv_batch,
                remv_edges_ba = remv_edges_ba
            )

            #self.console_logger.info(f'{pred_selected}')

            selection_loss = self.losses[KEY_SELECTION](
                pred_selected,
                true_selected
            )


            selection_logs = self.compute_selection_metrics(
                selection_loss, pred_selected, true_selected,
                split=KEY_TRAIN
            )

            #self.console_logger.info(f'{pred_selected}')

            selection_logs = self.apply_prefix(
                metrics = selection_logs,
                prefix = f'train_{KEY_SELECTION}'
            )

            logs.update(selection_logs)
            train_loss.append(selection_loss)

            # training of filler model is delegated
            # to the filler model itself
            filler_logs = self.filler_model.training_step(
                batch = filler_batch,
                batch_idx = batch_idx
            )

            filler_loss = filler_logs['loss']
            train_loss.append(filler_loss)

        self.log_dict(logs)

        return {'loss': sum(train_loss)}



    def configure_optimizers(self):
    
        # gather parameters for the optimizer
        ins_halt_params = list(self.insertion_model.parameters()) \
            + list(self.halting_model.parameters()) \
            + list(self.filler_encoder.parameters()) \
            + list(self.selector_encoder.parameters()) \
            + list(self.selector_model.parameters())
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

            filler_batch, true_selected, pred_selected = self.prepare_filler_batch(
                remv_batch = remv_batch,
                surv_batch = surv_batch,
                remv_edges_ba = remv_edges_ba
            )

            selection_loss = self.losses[KEY_SELECTION](
                pred_selected,
                true_selected
            )

            self.compute_selection_metrics(
                selection_loss, pred_selected, true_selected,
                split=which
            )

            eval_loss.append(selection_loss)


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
            **self.metrics[which][KEY_HALTING],
            **self.metrics[which][KEY_SELECTION]
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
    #                           MODEL CALL FUNCTIONS                           #
    ############################################################################

    @torch.no_grad()
    def sample_batch(
        self,
        batch_size: int,
        conditioning_y: Optional[Tensor]=None,
        maximum_insertion_steps: int=1000,
        maximum_number_nodes: int=None,
        return_directed: bool=True,
        save_chains: int=0
    ):

        if self.is_halting_disabled():
            maximum_number_nodes = self.dataset_info['num_nodes_max']

        sampler = None
        if hasattr(self.training_config, 'sample_mode'):
            if self.training_config.sample_mode == 'simple':
                sampler = self.sample_batch_simple
                self.console_logger.info(f'Sampling with simple mode')
        
        if sampler is None:
            sampler = self.sample_batch_reconsider_nodes
            self.console_logger.info(f'Sampling with reconsider nodes mode')

        return sampler(
            batch_size, conditioning_y, maximum_insertion_steps, maximum_number_nodes,
            return_directed, save_chains
        )


    @torch.no_grad()
    def sample_batch_simple(
        self,
        batch_size: int,
        conditioning_y: Optional[Tensor]=None,
        maximum_insertion_steps: int=1000,
        maximum_number_nodes: int=None,
        return_directed: bool=True,
        save_chains: int=0
    ):
        ########################################################################
        #                        INITIAL SAMPLING SETUP                        #
        ########################################################################

        # TODO: implement the generation chain saving
        do_save_chains = save_chains > 0

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
        #for t in range(self.removal_process.get_max_time().max().item()):
        t = 0
        remaining_graphs_idx = torch.arange(batch_size, dtype=torch.long, device=self.device)

        # initialize batch of generated graphs
        output_batch = [None] * batch_size

        # initialize number of remaining graphs
        remaining_graphs_num = batch_size

        while t < maximum_insertion_steps and remaining_graphs_num > 0:

            ####################################################################
            #                SAMPLE THE NUMBER OF NODES TO ADD                 #
            ####################################################################

            prev_graph = copy(graph)
            self.add_additional_features(prev_graph)

            # sample the number of nodes to remove
            nodes_to_insert: IntTensor
            nodes_to_insert, new_time, true_insertion_time = self.forward_insertion(
                graph =                     prev_graph,
                reversed_insertion_time =   torch.full((remaining_graphs_num,), t, dtype=torch.int, device=self.device)			
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

                ###############  SELECT NODES TO GENERATE FROM  ################
                enc_select_x = self.selector_encoder(
                    x =				prev_graph.x,
                    edge_index =	prev_graph.edge_index,
                    edge_attr =		prev_graph.edge_attr,
                    batch =			prev_graph.batch,
                    batch_size =	prev_graph.num_graphs,
                    y =				prev_graph.y,
                    num_nodes =     prev_graph.num_nodes_per_sample
                )

                
                if not self.node_regressive:
                    # encode the numbers of nodes which were removed as onehot vectors
                    num_nodes_emb = self.removal_process.schedule.coin_change_dist.get_onehot_coin(
                        nodes_to_insert
                    )
                else:
                    num_nodes_emb = nodes_to_insert.unsqueeze(-1)
                    
                # repeat the num nodes embedding for each node
                num_nodes_emb = num_nodes_emb.repeat_interleave(prev_graph.num_nodes_per_sample, dim=0)
                enc_select_x = torch.cat([enc_select_x, num_nodes_emb], dim=-1)

                selected_nodes_logits = self.selector_model(enc_select_x).squeeze(-1) # (N,)
                selected_nodes_mask = torch.distributions.Bernoulli(
                    logits=selected_nodes_logits
                ).sample() # (N,)


                #self.console_logger.info(f'{t}: {selected_nodes_mask.sum()}')

                # get the indices of the selected nodes
                selected_nodes_idx = torch.nonzero(selected_nodes_mask).squeeze(-1) # (N',)

                # get the selected nodes
                encoded_ext_x = encoded_ext_x[selected_nodes_idx] # (N, D) -> (N', D)
                ext_graph = copy(graph)
                ext_graph.batch = ext_graph.batch[selected_nodes_idx] # (N,) -> (N',)
                ext_graph.node_indegree = graph.indegree[selected_nodes_idx]
                _, ext_graph.ptr = compute_cum_elems(ext_graph.batch, remaining_graphs_num)


                ##########  SAMPLE THE SUBGRAPH FROM THE FILLER MODEL  #########
                # generate a new subgraph and new edges linking the new nodes
                # to the old nodes (from the former to the latter)
                # these are both sparse
                
                new_subgraph, new_ext_edges = self.filler_model.sample_batch(
                    batch_size =        remaining_graphs_num,
                    ext_graph =         ext_graph,
                    encoded_ext_x =     encoded_ext_x,
                    number_of_nodes =   nodes_to_insert,
                    save_chains =       save_chains
                )

                ###############  REINDEX THE INTERMEDIATE EDGES  ###############
                # remap back the edges from the selected nodes to the old nodes
                new_ext_edges.edge_index[1] = selected_nodes_idx[new_ext_edges.edge_index[1]]

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
                self.add_additional_features(curr_graph)

                halt_signal = self.forward_halting(
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

                if remaining_graphs_num == 0:
                    break
                # resume remaining batch
                graph = Batch.from_data_list(remaining_graphs)

                if graph.y.ndim == 1:
                    graph.y = graph.y.unsqueeze(-1)

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


BIG_LONG = 1e10

def compute_bfs_mask(g, num_nodes):
    # select nodes (with respect to batch) which are
    # at max depth or 1 before
    maxes = torch.zeros(g.num_graphs, dtype=torch.long, device=g.x.device)
    maxes.scatter_reduce_(0, g.batch, g.node_depth, reduce='max')
    maxes = torch.repeat_interleave(maxes, num_nodes) # expand to correct size
    return (g.node_depth == maxes) + (g.node_depth == maxes-1)


def compute_depths(g, num_nodes, max_num_nodes=None, ext_g=None, replace_big_long=True):
    # fills out missing values of g.node_depth

    new_depth = g.node_depth.clone()

    if max_num_nodes is None:
        max_num_nodes = num_nodes.max().item()

    for _ in range(max_num_nodes):
        # for each node, propagate the depth from its neighbors
        # to itself
        new_depth = new_depth.scatter_reduce(
            0, g.edge_index[0], new_depth[g.edge_index[1]]+1, reduce='min', include_self=True
        )

    # make sure we don't have any BIG_LONG values by replacing with highest depth
    if replace_big_long:
        # compute max batch-wise
        mask_ok = new_depth < BIG_LONG
        old_maxes = torch.zeros(g.num_graphs, dtype=torch.long, device=g.x.device)
        if ext_g is not None:
            old_maxes = old_maxes.scatter_reduce(0, ext_g.batch, ext_g.node_depth, reduce='max')
        maxes = old_maxes.scatter_reduce(0, g.batch, new_depth * mask_ok, reduce='max')
        maxes = torch.repeat_interleave(maxes, num_nodes) # expand to correct size
        # propagate back
        new_depth = new_depth.where(mask_ok, other=maxes)

    return new_depth


def initialize_bfs(g):
    node_depth = g.node_depth.clone()
    # select the first node of each graph
    node_depth[g.ptr[:-1]] = 0
    return node_depth


def compute_bfs_update(
        ext_subgraph: SparseGraph,
        new_subgraph: SparseGraph,
        new_edges_ba: SparseEdges,
        num_new_nodes: LongTensor,
        max_num_new_nodes: int
    ) -> SparseGraph:
    # warning: still not included case where new_subgraph has 0 nodes in the first iteration

    # initialize all new depths as +INF    
    new_subgraph.node_depth = torch.full((new_subgraph.x.shape[0],), BIG_LONG, dtype=torch.long, device=ext_subgraph.x.device)

    # if first iteration, initialize the seed nodes
    if ext_subgraph.num_nodes == 0:
        new_subgraph.node_depth = initialize_bfs(new_subgraph)
    else: # if intermediate iteration
        # propagate a BFS step from ext_graph to new_graph
        new_subgraph.node_depth = new_subgraph.node_depth.scatter_reduce(
            0, new_edges_ba.edge_index[0], ext_subgraph.node_depth[new_edges_ba.edge_index[1]]+1, reduce='min', include_self=False
        )
    
    # after first propagation, take many steps in the new block
    new_subgraph.node_depth = compute_depths(
        new_subgraph,
        num_new_nodes,
        max_num_new_nodes,
        ext_g = ext_subgraph
    )