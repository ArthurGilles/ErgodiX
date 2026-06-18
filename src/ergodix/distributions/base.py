import equinox as eqx
from jaxtyping import Array, Float

class TargetDistribution(eqx.Module):
    """Base class for all target distributions."""
    
    def __call__(self, x: Float[Array, "dim"]) -> Float[Array, ""]:
        raise NotImplementedError