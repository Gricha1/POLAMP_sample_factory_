import torch
from torch import nn
from torch.distributions import Normal, Independent
import numpy as np

def apply_squashing_func(action_distribution_params, actions, log_prob_actions):
    """
    Squash the output of the Gaussian distribution
    and account for that in the log probability
    The squashed mean is also returned for using
    deterministic actions.
    :param mu_: (tf.Tensor) Mean of the gaussian
    :param pi_: (tf.Tensor) Output of the policy before squashing
    :param logp_pi: (tf.Tensor) Log probability before squashing
    :return: ([tf.Tensor])
    """
    EPS = 1e-6
    mu_, log_std = torch.chunk(action_distribution_params, 2, dim=-1)
    pi_ = actions
    logp_pi = log_prob_actions
    # Squash the output
    deterministic_policy = torch.tanh(mu_)
    policy = torch.tanh(pi_)
    # OpenAI Variation:
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    # logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - policy ** 2, lower=0, upper=1) + EPS), axis=1)
    # Squash correction (from original implementation)
    logp_pi -= torch.sum(torch.log(1 - policy ** 2 + EPS), dim=-1)
    
    action_distribution_params = torch.cat((deterministic_policy, 
                                        log_std), dim=-1)

    return action_distribution_params, actions, logp_pi