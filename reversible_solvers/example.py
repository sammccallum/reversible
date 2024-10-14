import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from reversible_solvers.reversible import Reversible
from reversible_solvers.solver_step import Dopri5
from reversible_solvers.vector_field import AbstractVectorField


# Simple neural vector field
class VectorField(AbstractVectorField):
    layers: list

    def __init__(self, key):
        key1, key2 = jr.split(key, 2)
        self.layers = [
            eqx.nn.Linear(1, 10, use_bias=True, key=key1),
            jnp.tanh,
            eqx.nn.Linear(10, 1, use_bias=True, key=key2),
        ]

    def __call__(self, t, y):
        for layer in self.layers:
            y = layer(y)
        return y


# Setup vector field
key = jr.PRNGKey(0)
vf = VectorField(key)

# Reversible Dopri5
solver = Reversible(l=0.999, solver=Dopri5())

# Solve
h = 0.01
T = 1
y0 = jnp.asarray(1.0)[None]  # shape (1,)
y1 = solver.solve_forward(vf, y0, h, T)
