from typing import Dict, List, Tuple, Any, Optional, Union, Callable

import numpy as np
import torch
from torch import nn, Tensor
import re
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from omegaconf import OmegaConf, DictConfig
import networkx as nx

from src.data.simple_transforms.molecular import mol2smiles, GraphToMoleculeConverter

from src.evaluation.metrics.core import Metric

import src.evaluation.metrics as m_list
import src.evaluation.utils.molecular as molecular
import src.evaluation.utils.synth as synth
import src.evaluation.utils.graphgdp_metrics.evaluator as graphgdp_metrics

from src.evaluation import reg_metrics


class BaseSamplingMetric(Metric):
    pass


def is_sampling_metric(metric: Callable) -> bool:
    return isinstance(metric, BaseSamplingMetric)

def contains_sampling_metrics(metrics: Dict[str, Callable]) -> bool:
    return any([is_sampling_metric(metric) for metric in metrics.values()])
 

################################################################################
#                          MOLECULAR SAMPLING METRICS                          #
################################################################################

@reg_metrics.register(m_list.KEY_MOLECULAR_VALIDITY)
class ValidMoleculeMetric(BaseSamplingMetric):
    
    def __init__(self, ret_conn_comps: bool=False):
        super().__init__()

        self.ret_conn_comps = ret_conn_comps


    def compute_validity(
            self,
            molecules: List[Chem.Mol]

        ) -> Tuple[List[str], float, np.ndarray, List[str]]:
        """ generated: list of couples (positions, atom_types)"""
        
        valid_smiles = []
        num_components = []
        all_smiles = []
        
        for mol in molecules:

            # RDKit molecule to string (SMILES)
            smiles = mol2smiles(mol)

            try:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                num_components.append(len(mol_frags))
            except:
                pass
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    all_smiles.append(smiles)
                    if smiles != '' and smiles is not None:
                        valid_smiles.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                    all_smiles.append(None)
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
                    all_smiles.append(None)
            else:
                all_smiles.append(None)

        return valid_smiles, len(valid_smiles) / len(molecules), np.array(num_components), all_smiles
    

    def __call__(self, molecules: List[Chem.Mol]) -> Dict:
        valid_smiles, validity, num_components, all_smiles = self.compute_validity(molecules)
        ret = {
            m_list.KEY_MOLECULAR_VALIDITY: validity,
            'valid_smiles': valid_smiles
        }

        if self.ret_conn_comps:
            ret[m_list.KEY_MOLECULAR_CONN_COMP] = dict(
                min=num_components.min(),
                max=num_components.max(),
                mean=num_components.mean()
            )

        return ret


@reg_metrics.register(m_list.KEY_MOLECULAR_UNIQUENESS)
class UniqueMoleculeMetric(BaseSamplingMetric):
    
    def compute_uniqueness(
        self,
        valid_smiles: List[str]
    ) -> Tuple[List[str], float]:
        if len(valid_smiles) == 0: return [], 0

        return list(set(valid_smiles)), len(set(valid_smiles)) / len(valid_smiles)


    def __call__(self, smiles: List[str]):
        unique_smiles, uniqueness = self.compute_uniqueness(smiles)

        ret = {
            m_list.KEY_MOLECULAR_UNIQUENESS: uniqueness,
            'unique_smiles': unique_smiles
        }
        
        return ret


@reg_metrics.register(m_list.KEY_MOLECULAR_NOVELTY)
class NovelMoleculeMetric(BaseSamplingMetric):
            
    def __init__(self, ref_smiles: List[str] = None):
        super().__init__()

        self.ref_smiles = ref_smiles

    
    def compute_novelty(
        self,
        unique_smiles: List[str],
        ref_smiles: Optional[List[str]]=None

    ) -> Tuple[List[str], float]:

        if len(unique_smiles) == 0: return [], 0
        num_novel = 0
        novel = []
        if ref_smiles is None:
            print("Dataset smiles is None, novelty computation skipped")
            return [], 1
        for smiles in unique_smiles:
            if smiles not in ref_smiles:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique_smiles)


    def __call__(
            self,
            smiles: List[str],
        ):
        novel_smiles, novelty = self.compute_novelty(smiles, self.ref_smiles)

        ret = {
            m_list.KEY_MOLECULAR_NOVELTY: novelty,
            'novel_smiles': novel_smiles
        }
        
        return ret



@reg_metrics.register(m_list.KEY_FCD)
class FCDMetric(BaseSamplingMetric):
        
    def __init__(self, ref_smiles: List[str], n_jobs: int = 1, device: Optional[str]=None, batch_size: int=512):
        super().__init__()

        self.n_jobs = n_jobs
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.batch_size = batch_size

        self.fcd = molecular.get_FCDMetric(
            ref_smiles = ref_smiles,
            n_jobs = self.n_jobs,
            device = self.device,
            batch_size = self.batch_size
        )

    def __call__(
            self,
            smiles: List[str],
        ):
        
        return {m_list.KEY_FCD: self.fcd(smiles)}

    

@reg_metrics.register(m_list.KEY_NSPDK)
class NSPDKMetric(BaseSamplingMetric):
            
        def __init__(self, ref_nx = None, n_jobs: int = 10):
            super().__init__()
    
            self.n_jobs = n_jobs
            self.ref_nx = ref_nx
    
    
        def __call__(
                self,
                graphs: List[nx.Graph]
            ):

            nspdk = molecular.compute_nspdk_mmd(graphs, self.ref_nx, metric='nspdk', is_hist=False, n_jobs=self.n_jobs)
    
            return {m_list.KEY_NSPDK: nspdk}



################################################################################
#                      GRAPH GENERATION SAMPLING METRICS                       #
################################################################################

from networkx import number_connected_components

@reg_metrics.register(m_list.KEY_GRAPH_CONN_COMP)
class GraphConnCompMetric(BaseSamplingMetric):

    def __init__(self):
        super().__init__()


    def __call__(self, generated_graphs: List[nx.Graph]) -> Dict:
        
        conn_comps = [number_connected_components(g) for g in generated_graphs]

        conn_comps = dict(
            min =   np.min(conn_comps),
            max =   np.max(conn_comps),
            mean =  np.mean(conn_comps)
        )

        return {m_list.KEY_GRAPH_CONN_COMP: conn_comps}
        


class GraphMMDMetric(BaseSamplingMetric):

    def __init__(
            self,
            metric: Callable,
            name: str,
            test_graphs: List[nx.Graph] = None,
            compute_emd: bool = True,
            **kwargs
        ):
        super().__init__()

        self.test_graphs = test_graphs
        self.metric = metric
        self.name = name
        self.compute_emd = compute_emd
        self.kwargs = kwargs

    def __call__(self, generated_graphs: List) -> Dict:
        value = self.metric(
            self.test_graphs,
            generated_graphs,
            compute_emd=self.compute_emd,
        )
        return {self.name: value}

        
@reg_metrics.register(m_list.KEY_GRAPH_DEGREE)
class DegreeMetric(GraphMMDMetric):
    
    def __init__(self, test_graphs: List[nx.Graph] = None, compute_emd=True):
        super().__init__(
            test_graphs = test_graphs,
            metric = synth.degree_stats,
            name = m_list.KEY_GRAPH_DEGREE,
            compute_emd = compute_emd,
            is_parallel=True
        )

@reg_metrics.register(m_list.KEY_GRAPH_SPECTRE)
class SpectreMetric(GraphMMDMetric):
        
    def __init__(self, test_graphs: List[nx.Graph] = None, compute_emd=True):
        super().__init__(
            test_graphs = test_graphs,
            metric = synth.spectral_stats,
            name = m_list.KEY_GRAPH_SPECTRE,
            compute_emd = compute_emd,
            is_parallel=True,
            n_eigvals=-1
        )

@reg_metrics.register(m_list.KEY_GRAPH_CLUSTERING)
class ClusteringMetric(GraphMMDMetric):
            
    def __init__(self, test_graphs: List[nx.Graph] = None, compute_emd=True):
        super().__init__(
            test_graphs = test_graphs,
            metric = synth.clustering_stats,
            name = m_list.KEY_GRAPH_CLUSTERING,
            compute_emd = compute_emd,
            is_parallel=True,
            bins=100
        )

@reg_metrics.register(m_list.KEY_GRAPH_ORBIT)
class OrbitMetric(GraphMMDMetric):
                
    def __init__(self, test_graphs: List[nx.Graph] = None, compute_emd=True):
        super().__init__(
            test_graphs = test_graphs,
            metric = synth.orbit_stats_all,
            name = m_list.KEY_GRAPH_ORBIT,
            compute_emd = compute_emd
        )

@reg_metrics.register(m_list.KEY_GRAPH_NODES)
class NodesMetric(GraphMMDMetric):
    
    def __init__(self, test_graphs: List[nx.Graph] = None, compute_emd=True):
        super().__init__(
            test_graphs = test_graphs,
            metric = synth.nodes_stats,
            name = m_list.KEY_GRAPH_NODES,
            compute_emd = compute_emd,
            is_parallel=True
        )

@reg_metrics.register(m_list.KEY_GRAPH_ECCENTRICITY)
class EccentricityMetric(GraphMMDMetric):
    
    def __init__(self, test_graphs: List[nx.Graph] = None, compute_emd=True):
        super().__init__(
            test_graphs = test_graphs,
            metric = synth.radius_stats,
            name = m_list.KEY_GRAPH_ECCENTRICITY,
            compute_emd = compute_emd,
            is_parallel=True
        )


@reg_metrics.register(m_list.KEY_GRAPH_GIN)
class GraphGinMetric(BaseSamplingMetric):
                    
    def __init__(
            self,
            test_graphs: List[nx.Graph] = None,
            cfg = None
        ):
        super().__init__()

        if cfg is None:
            cfg = OmegaConf.create({})

        self.fn = graphgdp_metrics.get_nn_eval(cfg)
        self.test_graphs = test_graphs
        

    def __call__(self, generated_graphs: List) -> Dict:
        values = self.fn(
            test_dataset = self.test_graphs,
            pred_graph_list = generated_graphs
        )
        value = values['gin_MMD_RBF_mean']
        return {m_list.KEY_GRAPH_GIN: value}
    

@reg_metrics.register(m_list.KEY_GRAPH_CDGS)
class GraphCDGSMetric(BaseSamplingMetric):

    def __init__(
            self,
            test_graphs: List[nx.Graph] = None,
            cfg = None
        ):
        super().__init__()

        if cfg is None:
            cfg = OmegaConf.create({})

        self.fn = graphgdp_metrics.get_stats_eval(cfg)
        self.test_graphs = test_graphs



    def __call__(self, generated_graphs: List) -> Dict:
        values = self.fn(
            test_dataset = self.test_graphs,
            pred_graph_list = generated_graphs
        )

        # get_value gets the value of which key contains the sub-key s
        # e.g., s='degree', ls={'degree_rbf': 0.05, 'cluster_rbf': 0.04}
        # get_value(ls, k)=0.05
        get_value = lambda ls, s: [v for k, v in ls.items() if s in k][0]

        ret_values = {
            m_list.KEY_GRAPH_DEGREE: get_value(values, 'degree'),
            m_list.KEY_GRAPH_CLUSTERING: get_value(values, 'cluster'),
            m_list.KEY_GRAPH_SPECTRE: get_value(values, 'spectre')
        }

        return {m_list.KEY_GRAPH_CDGS: ret_values}


@reg_metrics.register(m_list.KEY_GRAPH_VUN)
class VUNGraphMetric(BaseSamplingMetric):
    
    def __init__(self, train_graphs: List[nx.Graph], type_of_graph: str):
        super().__init__()

        self.train_graphs = train_graphs
        self.type_of_graph = type_of_graph

    def __call__(self, generated_graphs: List[nx.Graph]) -> Dict:
        unique, novel, vun = synth.eval_vun(
            generated_graphs, self.train_graphs, validity_func=synth.VALIDITY_FUNCTIONS[self.type_of_graph]
        )
        return {m_list.KEY_GRAPH_UNIQUE: unique, m_list.KEY_GRAPH_UNIQUE_NOVEL: novel, m_list.KEY_GRAPH_VUN: vun}