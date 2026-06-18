from jaxtyping import Array, Float

from .base import TargetDistribution

class Rosenbrock(TargetDistribution):
    """
    The classic 2D Rosenbrock density.
    Improper distribution
    """
    a: float = 1.0
    b: float = 100.0
    
    def __call__(self, x: Float[Array, "2"]) -> Float[Array, ""]:        
        return -(self.a - x[0])**2 - self.b * (x[1] - x[0]**2)**2