from typing import Tuple, Dict

import torch
from torch import Tensor, IntTensor

from src.noise import reg_diffusion
from src.noise.core import NoiseSchedule, NoiseProcess

from src.noise.schedules import DiffusionProcessException, CosineDiffusionSchedule


################################################################################
#                             DIFFUSION PROCESSES                              #
################################################################################
@reg_diffusion.register('gaussian')
class GaussianDiffusionProcess(NoiseProcess):

    """
    This class implements the continuous diffusion process as described in the paper
    "Denoise Diffusion Probabilistic Models" by Jonathan Ho, Ajay Jain, Pieter Abbeel.
    https://arxiv.org/pdf/2006.11239
    """


    def __init__(
            self,
            schedule : NoiseSchedule,
            **kwargs
        ):
        """
        Parameters
        ----------
        schedule : DiffusionSchedule
            gives the parameter values for next, sample_t, posterior
        """
        # call super for the NoiseProcess
        super().__init__(schedule=schedule)


    ############################################################################
    #                     STATIONARY DISTRIBUTION (t->+inf)                    #
    ############################################################################

    def sample_stationary(
            self,
            shape,
            device: torch.device=None,
        ) -> Tensor:

        # sample from standard normal
        datapoint = torch.randn(*shape, device=device)
        
        return datapoint


    ############################################################################
    #                      NEXT TRANSITION (from t-1 to t)                     #
    ############################################################################

    def sample_noise_next(self, current_datapoint: Tensor, t: IntTensor, **kwargs):

        return torch.randn_like(current_datapoint)


    def apply_noise_next(
            self,
            current_datapoint: Tensor,
            noise: Tensor,
            t: IntTensor,
            **kwargs
        ) -> Tensor:

        beta_t = self.get_params_next(t, **kwargs)
        for _ in range(len(current_datapoint.shape) - 1):
            beta_t = beta_t.unsqueeze(-1)

        next_datapoint = torch.sqrt(1 - beta_t) * current_datapoint + torch.sqrt(beta_t) * noise

        return next_datapoint


    ############################################################################
    #                  TRANSITION FROM ORIGINAL (from 0 to t)                  #
    ############################################################################

    def sample_noise_from_original(
            self,
            original_datapoint: Tensor,
            t: IntTensor,
            **kwargs
        ):
        
        return torch.randn_like(original_datapoint)


    def apply_noise_from_original(
            self,
            original_datapoint: Tensor,
            noise: Tensor,
            t: IntTensor,
            **kwargs
        ) -> Tensor:

        alpha_t = self.get_params_from_original(t, **kwargs)
        for _ in range(len(original_datapoint.shape) - 1):
            alpha_t = alpha_t.unsqueeze(-1)

        noisy_datapoint = torch.sqrt(alpha_t) * original_datapoint + torch.sqrt(1 - alpha_t) * noise

        return noisy_datapoint
    
    ############################################################################
    #             POSTERIOR TRANSITION (from t to t-1 knowing t=0)             #
    ############################################################################

    def sample_noise_posterior(
            self,
            original_datapoint: Tensor,
            current_datapoint: Tensor,
            t: IntTensor,
            **kwargs
        ) -> Tensor:

        return torch.randn_like(original_datapoint)


    def apply_noise_posterior(
            self,
            original_datapoint: Tensor,
            current_datapoint: Tensor,
            noise: Tensor,
            t: IntTensor,
            **kwargs
        ) -> Tensor:

        alpha_t = self.get_params_from_original(t, **kwargs)
        alpha_t_1 = self.get_params_from_original(t-1, **kwargs)
        beta_t = self.get_params_next(t, **kwargs)
        for _ in range(len(original_datapoint.shape) - 1):
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_t_1 = alpha_t_1.unsqueeze(-1)
            beta_t = beta_t.unsqueeze(-1)


        x0_mult = torch.sqrt(alpha_t_1) * beta_t / (1 - alpha_t)
        xt_mult = torch.sqrt(1-beta_t) * (1-alpha_t_1) / (1 - alpha_t)
        beta_tilde_t = (1 - alpha_t_1) * beta_t / (1 - alpha_t)

        noisy_datapoint = x0_mult * original_datapoint + xt_mult * current_datapoint + torch.sqrt(beta_tilde_t) * noise

        return noisy_datapoint
    

    def sample_backward(
            self,
            current_datapoint: Tensor,
            noise: Tensor,
            t: IntTensor,
            **kwargs
        ) -> Tensor:

        alpha_t = self.get_params_from_original(t, **kwargs)
        alpha_t_1 = self.get_params_from_original(t-1, **kwargs)
        beta_t = self.get_params_next(t, **kwargs)
        for _ in range(len(current_datapoint.shape) - 1):
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_t_1 = alpha_t_1.unsqueeze(-1)
            beta_t = beta_t.unsqueeze(-1)

        epsilon_mult = beta_t / torch.sqrt(1 - alpha_t)
        beta_tilde_t = (1 - alpha_t_1) * beta_t / (1 - alpha_t)

        new_noise = torch.randn_like(noise)

        datapoint_t_1 = (current_datapoint - epsilon_mult * noise) / torch.sqrt(1-beta_t) + torch.sqrt(beta_tilde_t) * new_noise

        return datapoint_t_1

        
################################################################################
#                            RESOLVE OBJECT BY NAME                            #
################################################################################

DIFFUSION_SCHEDULE_COSINE = 'cosine'

DIFFUSION_PROCESS_CONTINUOUS = 'continuous'

def resolve_cont_diffusion_schedule(name: str) -> type:
    if name == DIFFUSION_SCHEDULE_COSINE:
        return CosineDiffusionSchedule
    else:
        raise DiffusionProcessException(f'Could not resolve diffusion schedule name: {name}')

def resolve_cont_diffusion_process(name: str) -> type:
    if name == DIFFUSION_PROCESS_CONTINUOUS:
        return GaussianDiffusionProcess
    else:
        raise DiffusionProcessException(f'Could not resolve diffusion process name: {name}')