import jax.numpy as jnp
from jaxtyping import Array, Float
import equinox as eqx 

class Schedule(eqx.Module):
    """
    Base class for SLIPS time schedules, defining the alpha(t) and g(t) functions.
    Subclass this to implement specific schedules by overriding the g(t) function.
    Schedule objects are passed to the SLIPSParams object, 
    which is then passed to the slips function.
    """
    def g(self, t: Float[Array, ""]) -> Float[Array, ""]:
        raise NotImplementedError
        
    def alpha(self, t: Float[Array, ""]) -> Float[Array, ""]:
        return jnp.sqrt(t) * self.g(t)
        
    def validate_grid(self, time_grid: Float[Array, "steps"]) -> Float[Array, "steps"]:
        """Override to add specific bounds checks."""
        return eqx.error_if(time_grid, time_grid[0] <= 0, "time_grid[0] must be strictly > 0")

class StandardSchedule(Schedule):
    """
    Asymptotic geometric schedule (Standard SLIPS).

    Parameters
    ----------
    alpha_1: float
        alpha_1 parameter of the schedule, as defined in section 3.2 a) of the SLIPS paper.
    """
    alpha_1: float = 1.0
    
    def g(self, t: Float[Array, ""]) -> Float[Array, ""]:
        return t ** (self.alpha_1 / 2.0)

class GeomSchedule(Schedule):
    """
    Non-asymptotic geometric schedule.

    Parameters
    ----------
    alpha_1: float
        alpha_1 parameter of the schedule, as defined in section 3.2 b) of the SLIPS paper.
    alpha_2: float
        alpha_2 parameter of the schedule, as defined in section 3.2 b) of the SLIPS paper.
    """
    alpha_1: float = 1.0
    alpha_2: float = 1.0
    
    def g(self, t: Float[Array, ""]) -> Float[Array, ""]:
        return (t ** (self.alpha_1 / 2.0)) * ((1 - t) ** (-self.alpha_2 / 2.0))
        
    def validate_grid(self, time_grid: Float[Array, "steps"]) -> Float[Array, "steps"]:
        time_grid = super().validate_grid(time_grid)
        return eqx.error_if(time_grid, time_grid[-1] >= 1, "time_grid[-1] must be strictly < 1 for Geom schedule")
