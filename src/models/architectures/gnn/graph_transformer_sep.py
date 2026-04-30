##########################################################################################################
#
# adapted from https://github.com/cvignac/DiGress/blob/main/dgd/models/transformer_model.py
#
##########################################################################################################

from typing import Optional, Dict, Tuple

import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

from torch.nn.modules.linear import Linear

from src.datatypes.dense import DenseGraph, DenseEdges, get_bipartite_edge_mask_dense, get_edge_mask_dense


class SeparatedXEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(
            self,
            dx: int,
            de: int,
            dy: int,
            heads: int = 8,
            dim_ffX: int = 2048,
            dim_ffE: int = 128,
            dim_ffy: int = 2048,
            dropout: float = 0.1,
            layer_norm_eps: float = 1e-5,
            act_fn = nn.ReLU
        ):
        """Builds graph transformer layer
        Parameters
        ----------
        dx : int
            number of node features
        de : int
            number of edge features
        dy : Optional[int]
            number of global features. Optional for allowing the absence of global features
        n_head : int
            number of attention heads. Must be a divisor of dx
        dim_ffX : int, optional
            size of intermediate features in nodes FFN, by default 2048
        dim_ffE : int, optional
            size of intermediate features in edges FFN, by default 128
        dim_ffy : int, optional
            size of intermediate features in global FFN, by default 2048
        dropout : float, optional
            dropout probability, by default 0.1
        layer_norm_eps : float, optional
            layer normalization parameter epsilon, by default 1e-5
        """
        
        super().__init__()

        self.self_attn = XEyBlockAttention(dx, de, dy, heads)

        self.normX1 = LayerNorm(dx, eps=layer_norm_eps)
        self.normXext1 = LayerNorm(dx, eps=layer_norm_eps)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps)
        self.normEext1 = LayerNorm(de, eps=layer_norm_eps)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutXext1 = Dropout(dropout)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutEext1 = Dropout(dropout)
        self.dropout_y1 = Dropout(dropout)

        # nodes FFN
        self.ffnX = nn.Sequential(
            Linear(dx, dim_ffX),
            act_fn(),
            Dropout(dropout),
            Linear(dim_ffX, dx),
            Dropout(dropout)
        )
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps)

        # nodes external FFN
        self.ffnXext = nn.Sequential(
            Linear(dx, dim_ffX),
            act_fn(),
            Dropout(dropout),
            Linear(dim_ffX, dx),
            Dropout(dropout)
        )
        self.normXext2 = LayerNorm(dx, eps=layer_norm_eps)

        # edges FFN
        self.ffnE = nn.Sequential(
            Linear(de, dim_ffE),
            act_fn(),
            Dropout(dropout),
            Linear(dim_ffE, de),
            Dropout(dropout)
        )
        self.normE2 = LayerNorm(de, eps=layer_norm_eps)

        # external edges FFN
        self.ffnEext = nn.Sequential(
            Linear(de, dim_ffE),
            act_fn(),
            Dropout(dropout),
            Linear(dim_ffE, de),
            Dropout(dropout)
        )
        self.normEext2 = LayerNorm(de, eps=layer_norm_eps)

        # global FFN
        self.ffny = nn.Sequential(
            Linear(dy, dim_ffy),
            act_fn(),
            Dropout(dropout),
            Linear(dim_ffy, dy),
            Dropout(dropout)
        )
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps)


    def forward(
            self,
            X_b: Tensor,
            X_a: Tensor,
            E_bb: Tensor,
            E_ba: Tensor,
            y: Tensor,
            node_mask_b: Tensor,    # only new
            node_mask_a: Tensor,    # only old
            edge_mask_bb: Optional[Tensor]=None,
            edge_mask_ba: Optional[Tensor]=None
        ) -> tuple[Tensor, Tensor, Tensor]:


        ####################  SELF-ATTENTION-RESIDUAL BLOCK  ###################
        # self-attention + cross-attention
        newX, newXext, newE, newEext, new_y = self.self_attn(
            X_b=X_b,
            X_a=X_a,
            node_mask_b=node_mask_b,
            node_mask_a=node_mask_a,
            E_bb=E_bb,
            E_ba=E_ba,
            y=y,
            edge_mask_bb=edge_mask_bb,
            edge_mask_ba=edge_mask_ba 
        )

        # residual on nodes
        newX = self.dropoutX1(newX)
        X_b = self.normX1(X_b + newX)

        # residual on external nodes
        newXext = self.dropoutXext1(newXext)
        X_a = self.normXext1(X_a + newXext)

        # residual on edges
        newE = self.dropoutE1(newE)
        E_bb = self.normE1(E_bb + newE)

        # residual on external edges
        newEext = self.dropoutEext1(newEext)
        E_ba = self.normEext1(E_ba + newEext)

        # residual on global
        new_y = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y)

        #########################  FFN-RESIDUAL BLOCK  #########################
        # X = norm(X + FFN(X))
        newX = self.ffnX(X_b)
        X_b = self.normX2(X_b + newX)

        # Xext = norm(Xext + FFN(Xext))
        newXext = self.ffnXext(X_a)
        X_a = self.normXext2(X_a + newXext)

        # E = norm(E + FFN(E))
        newE = self.ffnE(E_bb)
        E_bb = self.normE2(E_bb + newE)

        # Eext = norm(Eext + FFN(Eext))
        newEext = self.ffnEext(E_ba)
        E_ba = self.normEext2(E_ba + newEext)

        # y = norm(y + FFN(y))
        new_y = self.ffny(y)
        y = self.norm_y2(y + new_y)

        return X_b, X_a, E_bb, E_ba, y


class MaskedSoftmax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x, mask):
        x = x.masked_fill(~mask, -float('inf'))
        x = torch.softmax(x, dim=self.dim)
        return x.masked_fill(~mask, 0.0)



class OuterAttention(nn.Module):

    def __init__(
            self,
            dx: int,
            de: int,
            n_head: int
        ):

        super().__init__()

        assert dx % n_head == 0, f"Cannot divide nodes features size by number of heads: dx: {dx} -- nhead: {n_head}"

        self.dx = dx
        self.de = de

        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q_proj = Linear(dx, dx)
        self.k_proj = Linear(dx, dx)
        self.v_proj = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        self.masked_softmax = MaskedSoftmax(dim=2)


    def forward(
            self,
            Xq: Tensor,
            Xk: Tensor,
            E: Tensor,
            node_mask_q: Tensor,
            node_mask_k: Tensor,
            edge_mask: Optional[Tensor]=None
        ) -> tuple[Tensor, Tensor, Tensor]:

        bs, nq, nk, _ = E.shape

        #######################  FAKE NODES MASKS SETUP  #######################

        # unsqueeze to enable masking (dot product only if same number of dims)
        xk_mask = node_mask_k                   # (bs, nk, 1)
        ek_mask = xk_mask.unsqueeze(1)          # (bs, 1, nk, 1)

        ####################  QUERIES, KEYS, VALUES SETUP  #####################

        Q = self.q_proj(Xq)			# (bs, nq, dx)
        K = self.k_proj(Xk)			# (bs, nk, dx)
        V = self.v_proj(Xk)			# (bs, nk, dx)


        # Reshape to (bs, n, n_head, df) with dx = n_head * df
        Q = Q.reshape((*Q.shape[:2], self.n_head, self.df))
        K = K.reshape((*K.shape[:2], self.n_head, self.df))
        V = V.reshape((*V.shape[:2], self.n_head, self.df))

        # setup dimensions for outer product
        Q = Q.unsqueeze(2)			# (bs, nq, 1, n_head, df)
        K = K.unsqueeze(1)			# (bs, 1, nk, n head, df)
        V = V.unsqueeze(1)			# (bs, 1, nk, n_head, df)

        ####################  OUTER PRODUCT SELF-ATTENTION  ####################

        # Compute unnormalized attentions. A is (bs, nq, nk, n_head, df)
        A = Q * K					# outer product
        A = A / math.sqrt(self.df)	# scaling by sqrt(df)

        # ----> node informed attention matrix A of shape (bs, nq, nk, n_head, df)

        #################  ATTENTION = FILM(EDGES, ATTENTION)  #################
        E1 = self.e_add(E)				# bs, nq, nk, dx
        E1 = E1.reshape((*E.shape[:3], self.n_head, self.df))

        E2 = self.e_mul(E)				# bs, nq, nk, dx
        E2 = E2.reshape((*E.shape[:3], self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        A = E1 + (E2 + 1) * A                   # (bs, nq, nk, n_head, df)

        # ----> node/edge informed attention matrix A of shape (bs, n, n, n_head, df)

        ####################  ATTENTION: AGGREGATE VALUES  #####################

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        # use masked softmax to avoid attention on non-existing nodes
        softmax_mask = ek_mask.expand(-1, nq, -1, self.n_head)	# bs, nq, nk, n_head
        attn_weights = A.sum(-1)                                # bs, nq, nk, n_head
        attn = self.masked_softmax(attn_weights, softmax_mask)  # bs, nq, nk, n_head

        # Compute weighted values
        weighted_V = attn.unsqueeze(-1) * V				# (bs, nq, nk, n_head, df)
        weighted_V = weighted_V.sum(dim=2)				# (bs, nq, n_head, df)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=-2)	# (bs, nq, dx)

        # ----> self-attention output, aggregated node values (bs, nq, dx)

        return weighted_V, A.flatten(start_dim=-2)



class XEyBlockAttention(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """

    def __init__(
            self,
            dx: int,
            de: int,
            dy: int,
            n_head: int
        ):
        """Self-attention block for computing new edges, nodes and global features
        Parameters
        ----------
        dx : int
            number of node features
        de : int
            number of edge features
        dy : int
            number of global features
        n_head : int
            number of attention heads. Must be a divisor of dx
        """
        super().__init__()

        assert dx % n_head == 0, f"Cannot divide nodes features size by number of heads: dx: {dx} -- nhead: {n_head}"

        self.dx = dx
        self.de = de
        self.dy = dy

        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        # B: new block, A: old block
        self.bb_attn = OuterAttention(dx, de, n_head) # B to B (self-attention)
        self.ba_attn = OuterAttention(dx, de, n_head) # B to A (cross-attention)
        self.ab_attn = OuterAttention(dx, de, n_head) # A to B (cross-attention)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_proj = Linear(dy, dy)
        self.reduce_x = Xtoy(dx, dy)	# projection of (mean, std, min, max)
        self.reduce_e = Etoy(de, dy)	# projection of (mean, std, min, max)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.e_ext_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(5 * dy, dy), nn.ReLU(), nn.Linear(dy, dy))


    def forward(
            self,
            X_b: Tensor,           # new
            X_a: Tensor,           # old
            node_mask_b: Tensor,    # only new
            node_mask_a: Tensor,    # only old
            E_bb: Tensor,         # new
            E_ba: Tensor,         # old
            y: Tensor,
            edge_mask_bb: Optional[Tensor]=None,    # new
            edge_mask_ba: Optional[Tensor]=None     # old
        ) -> tuple[Tensor, Tensor, Tensor]:
        """Updates the nodes, edges and global representations
        Parameters
        ----------
        X : Tensor
            node features of shape (bs, n, d)
        E : Tensor
            edge features of shape (bs, nq, nk, d)
        y : Tensor
            global features of shape (bs, dy)
        node_mask : Tensor
            node masks for non-existing nodes (due to padding in dense representation) of shape (bs, n)
        Returns
        -------
        newX : Tensor
            new node features of shape (bs, n, d)
        newE : Tensor
            new edge features of shape (bs, n, n, d)
        new_y : Tensor
            new global features of shape (bs, dy)
        """

        # E_ba is rectangular, where the first dimension is that of the new block
        # while the second is that of the old block
        bs, nb, na, _ = E_ba.shape

        #######################  FAKE NODES MASKS SETUP  #######################

        # unsqueeze to enable masking (dot product only if same number of dims)
        xb_mask = node_mask_b             # (bs, nb, 1)
        xa_mask = node_mask_a             # (bs, na, 1)
        ebb_mask = edge_mask_bb           # (bs, nb, nb, 1)
        eba_mask = edge_mask_ba           # (bs, nb, na, 1)

        # get interlinking edges
        E_ab = E_ba.transpose(1, 2)                     # (bs, nk, nq, de)
        edge_mask_ab = edge_mask_ba.transpose(1, 2)	    # (bs, nk, nq, 1)

        # B pays attention to itself
        V_b: Tensor     # (bs, nb, dx)
        A_bb: Tensor  # (bs, nb, nb, de)
        V_b, A_bb = self.bb_attn(
            Xq=X_b, Xk=X_b, E=E_bb,
            node_mask_q=node_mask_b, node_mask_k=node_mask_b, edge_mask=edge_mask_bb
        )

        V_ba: Tensor     # (bs, nb, dx)
        A_ba: Tensor    # (bs, nb, na, de)
        V_ba, A_ba = self.ba_attn(
            Xq=X_b, Xk=X_a, E=E_ba,
            node_mask_q=node_mask_b, node_mask_k=node_mask_a, edge_mask=edge_mask_ba
        )
        
        # A pays attention only to B
        V_ab: Tensor     # (bs, na, dx)
        A_ab: Tensor    # (bs, na, nb, de)
        V_ab, A_ab = self.ab_attn(
            Xq=X_a, Xk=X_b, E=E_ab,
            node_mask_q=node_mask_a, node_mask_k=node_mask_b, edge_mask=edge_mask_ab
        )

        # make symmetric
        A_self = (A_bb + A_bb.transpose(1, 2)) / 2      # (bs, nb, nb, de)
        A_cross = (A_ba + A_ab.transpose(1, 2)) / 2     # (bs, nb, na, de)

        # aggregate attention
        V_b = V_b+V_ba
        V_a = V_ab
        A_bb = A_self
        A_ba = A_cross

        ###############  OUT_EDGES = LIN(FILM(GLOBAL, EDGES))  #################

        # Incorporate y to E
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)     # (bs, 1, 1, dx)
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)     # (bs, 1, 1, dx)
        newE = ye1 + (ye2 + 1) * A_bb
        newEext = ye1 + (ye2 + 1) * A_ba

        # Output E
        newE = self.e_out(newE) * ebb_mask					# (bs, nb, nb, de)
        newEext = self.e_ext_out(newEext) * eba_mask		# (bs, nb, ba, de)

        # ----> END OF EDGES BRANCH, new edges (bs, nb, nb+na, de)

        ###############  OUT_NODES = LIN(FILM(GLOBAL, NODES))  #################

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)      # (bs, 1, dx)
        yx2 = self.y_x_mul(y).unsqueeze(1)      # (bs, 1, dx)
        newX = yx1 + (yx2 + 1) * V_b
        newXext = yx1 + (yx2 + 1) * V_a

        # Output X
        newX = self.x_out(newX) * xb_mask      # (bs, nb, dx)
        newXext = self.x_out(newXext) * xa_mask  # (bs, na, dx)

        # ----> END OF NODES BRANCH, new nodes (bs, n, dx)

        ###############  GLOBAL BRANCH  #################

        # Process y based on X and E
        y = self.y_proj(y)			# (bs, dy)
        e_y = self.reduce_e(E_bb, ebb_mask)		# (bs, dy)
        eext_y = self.reduce_e(E_ba, eba_mask)	# (bs, dy)
        x_y = self.reduce_x(X_b, xb_mask)		# (bs, dy)
        xext_y = self.reduce_x(X_a, xa_mask)	# (bs, dy)

        new_y = torch.cat([y, x_y, xext_y, e_y, eext_y], dim=-1)	# concat everything
        new_y = self.y_out(new_y)   # (bs, dy)

        return newX, newXext, newE, newEext, new_y


#############  TRANSFORMER OPTIONS  ##############

DIM_X = 'x'
DIM_E = 'e'
DIM_Y = 'y'

if False:
    POS_INF = 1e9
    NEG_INF = -1e9
else:
    POS_INF = float('inf')
    NEG_INF = float('-inf')


from src.models import reg_architectures

@reg_architectures.register()
class GraphTransformerSeparated(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(
            self,
            input_dims: Dict,
            output_dims: Dict,
            num_layers: int,
            encdec_hidden_dims: Dict,
            transf_inout_dims: Dict,
            transf_ffn_dims: Dict,
            transf_hparams: Dict,
            use_residuals_inout: bool = True,
            act_fn = 'relu',
            simpler: bool = False,
            **kwargs
        ):

        super().__init__()

        if act_fn == 'relu':
            self.act_fn = nn.ReLU
        elif act_fn == 'silu':
            self.act_fn = nn.SiLU
        else:
            raise ValueError(f"Activation function {act_fn} not recognized")

        self.num_layers = num_layers
        self.use_residuals_inout = use_residuals_inout
        self.simpler = simpler

        self.in_dim_x = input_dims[DIM_X]
        self.in_dim_e = input_dims[DIM_E]
        self.in_dim_y = input_dims[DIM_Y]

        self.encdec_hidden_dims = encdec_hidden_dims
        self.transf_inout_dims = transf_inout_dims
        self.transf_ffn_dims = transf_ffn_dims

        self.using_y = self.in_dim_y is not None

        self.out_dim_x = output_dims[DIM_X]
        self.out_dim_e = output_dims[DIM_E]
        self.out_dim_y = output_dims[DIM_Y]

        ###########################  INPUT ENCODERS  ###########################
        # nodes encoder
        self.mlp_in_X = nn.Sequential(
            nn.Linear(self.in_dim_x, encdec_hidden_dims[DIM_X]),
            self.act_fn(),
            nn.Linear(encdec_hidden_dims[DIM_X], transf_inout_dims[DIM_X]),
            nn.LayerNorm(transf_inout_dims[DIM_X])
        )

        # edges encoder
        self.mlp_in_E = nn.Sequential(
            nn.Linear(self.in_dim_e, encdec_hidden_dims[DIM_E]),
            self.act_fn(),
            nn.Linear(encdec_hidden_dims[DIM_E], transf_inout_dims[DIM_E])
        )

        if self.using_y:
            # global encoder
            self.mlp_in_y = nn.Sequential(
                nn.Linear(self.in_dim_y, encdec_hidden_dims[DIM_Y]),
                self.act_fn(),
                nn.Linear(encdec_hidden_dims[DIM_Y], transf_inout_dims[DIM_Y])
            )
        else:
            self.fixed_y = nn.Parameter(torch.randn(transf_inout_dims[DIM_Y]))


        self.mlp_in_ext_E = nn.Sequential(
            nn.Linear(self.in_dim_e, encdec_hidden_dims[DIM_E]),
            self.act_fn(),
            nn.Linear(encdec_hidden_dims[DIM_E], transf_inout_dims[DIM_E])
        )


        #######################  MAIN BODY: TRANSFORMER  #######################

        self.tf_layers = nn.ModuleList([
            SeparatedXEyTransformerLayer(
                dx=transf_inout_dims[DIM_X],
                de=transf_inout_dims[DIM_E],
                dy=transf_inout_dims[DIM_Y],
                dim_ffX=transf_ffn_dims[DIM_X],
                dim_ffE=transf_ffn_dims[DIM_E],
                dim_ffy=transf_ffn_dims[DIM_Y],
                act_fn=self.act_fn,
                **transf_hparams
            )
            for _ in range(num_layers)
        ])

        ##########################  OUTPUT DECODERS  ###########################

        # nodes decoder
        self.mlp_out_X = nn.Sequential(
            nn.Linear(transf_inout_dims[DIM_X], encdec_hidden_dims[DIM_X]),
            self.act_fn(),
            nn.Linear(encdec_hidden_dims[DIM_X], self.out_dim_x)
        )

        # edges decoder
        self.mlp_out_E = nn.Sequential(
            nn.Linear(transf_inout_dims[DIM_E], encdec_hidden_dims[DIM_E]),
            self.act_fn(),
            nn.Linear(encdec_hidden_dims[DIM_E], self.out_dim_e)
        )

        if self.using_y:
            # global decoder
            self.mlp_out_y = nn.Sequential(
                nn.Linear(transf_inout_dims[DIM_Y], encdec_hidden_dims[DIM_Y]),
                self.act_fn(),
                nn.Linear(encdec_hidden_dims[DIM_Y], self.out_dim_y)
            )


        self.mlp_out_ext_E = nn.Sequential(
            nn.Linear(transf_inout_dims[DIM_E], encdec_hidden_dims[DIM_E]),
            self.act_fn(),
            nn.Linear(encdec_hidden_dims[DIM_E], self.out_dim_e)
        )


    def get_external_nodes_dim(self):
        return self.transf_inout_dims[DIM_X]

    def forward(
            self,
            graph: DenseGraph,
            ext_X: Optional[Tensor]=None,
            ext_node_mask: Optional[Tensor]=None,
            ext_edges: Optional[DenseEdges]=None
        ) -> Tuple[DenseGraph, Optional[DenseEdges]]:

        ########################  ASSERTIONS ON INPUT  #########################
        X, E, y = graph.x, graph.edge_adjmat, graph.y

        using_ext = ext_X is not None
        assert using_ext, "External nodes and edges are required for this unfrozen model"

        assert X.shape[-1] == self.in_dim_x, \
            f"X.shape[-1] = {X.shape[-1]}, self.in_dim_x = {self.in_dim_x}"
        assert E.shape[-1] == self.in_dim_e, \
            f"E.shape[-1] = {E.shape[-1]}, self.in_dim_e = {self.in_dim_e}"
        assert y is None or y.shape[-1] == self.in_dim_y, \
            f"y.shape[-1] = {y.shape[-1]}, self.in_dim_y = {self.in_dim_y}"
        if using_ext:
            ext_E = ext_edges.edge_adjmat
            assert ext_X is None or ext_X.shape[-1] == self.transf_inout_dims[DIM_X], \
                f"ext_X.shape[-1] = {ext_X.shape[-1]}, self.transf_inout_dims[DIM_X] = {self.transf_inout_dims[DIM_X]}"
            assert ext_E is None or ext_E.shape[-1] == self.in_dim_e, \
                f"ext_E.shape[-1] = {ext_E.shape[-1]}, self.in_dim_e = {self.in_dim_e}"
            assert ext_node_mask is None or ext_node_mask.shape[1] == ext_X.shape[1], \
                f"ext_node_mask.shape[1] = {ext_node_mask.shape[1]}, ext_X.shape[1] = {ext_X.shape[1]}"
        else:
            ext_E = None



        bs, nq = X.shape[0], X.shape[1]

        ###############  SETUP SELFLOOP REMOVAL (DIAGONAL) MASK  ###############
        """diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)"""

        node_mask = graph.node_mask.unsqueeze(-1)
        edge_mask = graph.edge_mask.unsqueeze(-1)
        triang_mask = get_edge_mask_dense(edge_mask=graph.edge_mask, only_triangular=True).unsqueeze(-1)

        def mask_everything(X, E, ext_E=None):

            X = X * node_mask
            E = E * edge_mask
            if ext_E is not None:
                ext_E = ext_E * ext_edges.edge_mask.unsqueeze(-1)
            
            return X, E, ext_E

        ######################  SAVE RESIDUAL FOR LATER  #######################
        if self.use_residuals_inout:
            X_to_out = X[..., :self.out_dim_x]
            E_to_out = E[..., :self.out_dim_e]
            if self.using_y:
                y_to_out = y[..., :self.out_dim_y]
            if using_ext:
                ext_E_to_out = ext_E[..., :self.out_dim_e]

        ###########################  ENCODE INPUTS  ############################
        # special treatment for edges (to make it symmetric (shouldn't this already be?))
        X = self.mlp_in_X(X)
        E = self.mlp_in_E(E)
        E = (E + E.transpose(1, 2)) / 2   # new_E should already be symmetric if E is symmetric!!!
        if self.using_y:
            y = self.mlp_in_y(y)
        else:
            y = self.fixed_y.clone().expand(bs, -1)

        if using_ext:
            ext_E = self.mlp_in_ext_E(ext_E)

        # mask everything before feeding to transformer
        X, E, ext_E = mask_everything(X, E, ext_E)

        #######################  MAIN BODY: TRANSFORMER  #######################

        # if using cross attention
        if using_ext:

            if ext_X.ndim == 4: # if ext_X is actually a stack of encoded nodes
                # create a list from dimension 2 (the layers dimension)
                ext_X = torch.unbind(ext_X, dim=2)

            bs, nq, nk = ext_E.shape[:3]
            ext_node_mask = ext_node_mask.unsqueeze(-1)

            edge_mask_ba = ext_edges.edge_mask.unsqueeze(-1)

            # for each layer use the only provided set of external encoded nodes
            for layer in self.tf_layers:
                X, ext_X, E, ext_E, y = layer(
                    X, ext_X, E, ext_E, y,
                    node_mask_b=node_mask, node_mask_a=ext_node_mask,
                    edge_mask_bb=edge_mask, edge_mask_ba=edge_mask_ba
                )
        
        # if no external set of nodes and edges are provided
        else:
            raise NotImplementedError("This model requires external nodes and edges")


        ###########################  DECODE OUTPUT  ############################
        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        if self.using_y:
            y = self.mlp_out_y(y)
        if using_ext:
            ext_E = self.mlp_out_ext_E(ext_E)

        ###########################  FINAL RESIDUAL  ###########################
        if self.use_residuals_inout:
            X = X + X_to_out
            E = E + E_to_out
            
        # remove selfloop and make symmetric
        E = E * triang_mask
        #E = (E + torch.transpose(E, 1, 2)) / 2 # here it's ok!
        E = (E + torch.transpose(E, 1, 2))
        
        if self.use_residuals_inout:
            if self.using_y:
                y = y + y_to_out
            if using_ext:
                ext_E = ext_E + ext_E_to_out
        
        # mask everything before returning
        out_graph = DenseGraph(X, E, y, graph.node_mask, graph.edge_mask).apply_mask()
        if using_ext:
            out_ext_edges = DenseEdges(ext_E, ext_edges.edge_mask).apply_mask()
        else:
            out_ext_edges = None

        ###############################  RETURN  ###############################

        return out_graph, out_ext_edges
        


##########################################################################################################
#
# FROM https://github.com/cvignac/DiGress/blob/main/dgd/models/layers.py
#
##########################################################################################################

def compute_masked_mean(X: Tensor, mask: Tensor, dim: int=1) -> Tensor:
    """ Computes the mean of X along dimension dim, ignoring masked values. """
    denominator = mask.sum(dim=dim)
    denominator.masked_fill_(denominator == 0, 1.)
    return X.sum(dim=dim) / denominator


def compute_masked_std(X: Tensor, mean_X: Tensor, mask: Tensor, dim: int|Tuple[int]=1) -> Tensor:
    """ Computes the standard deviation of X along dimension dim, ignoring masked values. """
    diff_X = X - mean_X.unsqueeze(dim=dim)
    std = (compute_masked_mean(diff_X * diff_X, mask, dim) + 1e-10).sqrt()
    std = std.masked_fill(mask.sum(dim=dim) < 2, 0.)
    return std


def compute_masked_min(X: Tensor, mask: Tensor, dim: int=1) -> Tensor:
    """ Computes the min of X along dimension dim, ignoring masked values. """
    X = X.masked_fill(~mask, float('inf'))
    X = X.min(dim=dim)[0]
    return torch.nan_to_num(X, posinf=0)


def compute_masked_max(X: Tensor, mask: Tensor, dim: int=1) -> Tensor:
    """ Computes the max of X along dimension dim, ignoring masked values. """
    X = X.masked_fill(~mask, float('-inf'))
    X = X.max(dim=dim)[0]
    return torch.nan_to_num(X, neginf=0)


def compute_masked_mean_std_min_max(X: Tensor, mask: Tensor, dim: Tuple[int, int]=(1,1)) -> Tensor:

    if X.numel() == 0:
        return torch.zeros((X.shape[0], 4 * X.shape[-1]), device=X.device)

    else:
        X = X.flatten(start_dim=dim[0], end_dim=dim[1])
        mask = mask.flatten(start_dim=dim[0], end_dim=dim[1])
        dim = dim[0]

        X = X * mask

        mean_X = compute_masked_mean(X, mask, dim)
        std_X = compute_masked_std(X, mean_X, mask, dim)
        min_X = compute_masked_min(X, mask, dim)
        max_X = compute_masked_max(X, mask, dim)

        X.masked_fill_(~mask, 0.)

        return torch.hstack((mean_X, std_X, min_X, max_X))
    



class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X: Tensor, node_mask: Tensor):
        """ X: bs, n, dx. """
        z = compute_masked_mean_std_min_max(X, node_mask, dim=(1,1))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E: Tensor, edge_mask: Tensor):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """
        z = compute_masked_mean_std_min_max(E, edge_mask, dim=(1,2))
        out = self.lin(z)
        return out
    
def assert_correctly_masked(variable, node_mask):
    if variable.numel() == 0:
        return
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'