import equinox as eqx 

from .schedules import Schedule, GeomSchedule

class SLIPSParams(eqx.Module):
    """
    Parameters for the SLIPS algorithm.
    Object to be passed to the slips function, which contains all
    the parameters for the algorithm.

    Parameters
    ----------
    sigma: float
        The standard deviation of the base gaussian distribution.
    schedule: Schedule
        Schedule object containing the functions alpha(t) and g(t)
    target_accept: float
        Target acceptance rate of the MALA chains. The step size is tuned automatically to reach this acceptance rate.
    step_min: float
        Minimum step size in the MALA chains.
    step_max: float
        Maximum step size in the MALA chains.
    learning_rate: float
        Rate at which the step size is tuned.
    n_mcmc_steps: int
        Number of MCMC steps to take in each iteration of the SLIPS algorithm.
    n_chains: int
        Number of MALA chains to run in parallel.
    n_init_steps: int
        Number of MCMC steps to take in the initialisation phase of the SLIPS algorithm.
    burn_in_ratio: float
        Ratio of the MCMC steps to discard as burn-in in the SLIPS algorithm.
    return_history: bool
        Whether to return the history of the MCMC steps.
    """
    # Dynamic params
    sigma: float
    
    # Dynamic named params
    schedule: Schedule = GeomSchedule() 
    target_accept: float = 0.75 
    step_min: float = 1e-2
    step_max: float = 1e2
    learning_rate: float = 2.0 

    # Static named params
    n_mcmc_steps:   int   = eqx.field(default=32, static=True)
    n_chains:       int   = eqx.field(default=4, static=True)          
    n_init_steps:   int   = eqx.field(default=32, static=True)
    burn_in_ratio:  float = eqx.field(default=0.5, static=True)
    return_history: bool  = eqx.field(default=False, static=True)
