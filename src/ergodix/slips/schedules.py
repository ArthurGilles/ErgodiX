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
    
    def log_snr(self, t: Float[Array, "..."]) -> Float[Array, "..."]:
        """
        Computes log SNR(t) = 2 * log(g(t)) to be used to compute the
        snr time grid. The missing constant term (appearing in the paper)
        cancels out in the get_snr_grid function.
        """
        return 2.0 * jnp.log(self.g(t))

    def get_snr_grid(self, 
                     t_0: float, 
                     t_end: float, 
                     steps: int,
                     dense_resolution: int = 10_000) -> Float[Array, "steps"]:
        """
        Generates an SNR-adapted time grid using JAX interpolation.
        This provides a robust, JIT-compilable fallback for any arbitrary schedule.
        
        Parameters
        ----------
        t_0: float
            Starting time for the grid (must be > 0).
        t_end: float
            Ending time for the grid (must be < 1).
        steps: int
            Number of steps in the grid.
        dense_resolution: int
            Resolution of the dense evaluation for interpolation (default: 10,000).
        Returns
        -------
        Float[Array, "steps"]
            The computed SNR-adapted time grid.
        """
        log_snr_0 = self.log_snr(jnp.array(t_0))
        log_snr_end = self.log_snr(jnp.array(t_end))
        target_log_snr = jnp.linspace(log_snr_0, log_snr_end, steps)
        
        # Dense evaluation for robust interpolation
        dense_t = jnp.linspace(t_0, t_end, dense_resolution)
        dense_log_snr = self.log_snr(dense_t)
        
        # g(t) is strictly increasing
        return jnp.interp(target_log_snr, dense_log_snr, dense_t)
        
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
    
    def get_snr_grid(self, 
                     t_0: float, 
                     t_end: float, 
                     steps: int,
                     dense_resolution: int = 10_000) -> Float[Array, "steps"]:
        """
        Generates an SNR-adapted time grid using JAX interpolation.
        Uses exact analytical inverse for known alpha_1.

        Parameters
        ----------
        t_0: float
            Starting time for the grid (must be > 0).
        t_end: float
            Ending time for the grid (must be < 1).
        steps: int
            Number of steps in the grid.
        dense_resolution: int
            Not needed for this schedule, but kept for compatibility with the base class.
        Returns
        -------
        Float[Array, "steps"]
            The computed SNR-adapted time grid.
        """
        log_snr_0 = self.log_snr(jnp.array(t_0))
        log_snr_end = self.log_snr(jnp.array(t_end))
        
        target_log_snr = jnp.linspace(log_snr_0, log_snr_end, steps)
        return jnp.exp(target_log_snr / self.alpha_1)

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
    
    def get_snr_grid(self, 
                     t_0: float, 
                     t_end: float, 
                     steps: int,
                     dense_resolution: int = 10_000) -> Float[Array, "steps"]:
        """
        Generates an SNR-adapted time grid using JAX interpolation.
        Uses exact analytical inverse for known alpha_1 and alpha_2,
        otherwise falls back to the robust interpolation from the base class.
        
        Parameters
        ----------
        t_0: float
            Starting time for the grid (must be > 0).
        t_end: float
            Ending time for the grid (must be < 1).
        steps: int
            Number of steps in the grid.
        dense_resolution: int
            Not needed for this schedule, but kept for compatibility with the base class.
        Returns
        -------
        Float[Array, "steps"]
            The computed SNR-adapted time grid.
        """
        log_snr_0 = self.log_snr(jnp.array(t_0))
        log_snr_end = self.log_snr(jnp.array(t_end))
        target_log_snr = jnp.linspace(log_snr_0, log_snr_end, steps)
        
        # Map target log-SNR back to target g values
        target_g = jnp.exp(target_log_snr / 2.0)
        
        # Analytical overrides for paper-specific parameters
        if self.alpha_1 == 1.0 and self.alpha_2 == 1.0:
            g_sq = target_g ** 2
            return g_sq / (1.0 + g_sq)
            
        elif self.alpha_1 == 2.0 and self.alpha_2 == 1.0:
            g_sq = target_g ** 2
            return (jnp.sqrt(g_sq**2 + 4 * g_sq) - g_sq) / 2.0
            
        # Fallback to dense interpolation for custom parameters
        else:
            return super().get_snr_grid(t_0, t_end, steps, dense_resolution)