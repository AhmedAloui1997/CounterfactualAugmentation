#perf.py contains the metrics for measuring the performance of different causal learners.

import torch


def perf_epehe_e_ate(mu_0,mu_1,ite_est):
    """
    Estimating the error in ATE and the precision of estimating heteregenous treatmente effects

    Arguments
    -------------
    mu_0: is the true conditional potential outcome under t=0 (E[Y_0|X])
    mu_1: is the true conditional potential outcome under t=1 (E[Y_1|X])
    ite_est: the estimated value of the ITE
    
    """
    
    e_pehe = torch.sqrt(torch.mean((mu_1-mu_0-ite_est)**2))
    e_ate = torch.abs(torch.mean(mu_1-mu_0) - torch.mean(ite_est))
    return e_pehe,e_ate