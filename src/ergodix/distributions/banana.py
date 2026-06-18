from jaxtyping import Array, Float

from .base import TargetDistribution


class Banana(TargetDistribution):
    """
    The Twisted 2D Gaussian (Banana) Distribution.
    Unnormalized
    """
    
    curvature: float = 0.03
    sigma_x: float = 10.0
    sigma_y: float = 1.0

    def __call__(self, x: Float[Array, "2"]) -> Float[Array, ""]:
        x1, x2 = x[0], x[1]
        
        twist = x2 - self.curvature * (x1**2 - self.sigma_x**2)
        log_prob = -0.5 * ((x1 / self.sigma_x)**2 + (twist / self.sigma_y)**2)
        return log_prob






