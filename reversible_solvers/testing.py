import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from reversible_solvers.reversible import Reversible
from reversible_solvers.solver_step import Midpoint


@eqx.filter_value_and_grad
def grad_loss(vf, solver, x, h, T):
    state = (x, x)
    y = jax.vmap(solver.solve_backward, in_axes=(None, 0, None, None))(vf, state, h, T)
    return jnp.sum(y**2)


class VectorField(eqx.Module):
    """Vector field for Neural ODE.

    We take a point in y and return f_\theta(y).
    """

    layers: list

    def __init__(self, key):
        key1, key2 = jr.split(key, 2)

        self.layers = [
            eqx.nn.Linear(1, 200, use_bias=True, key=key1),
            jnp.tanh,
            eqx.nn.Linear(200, 1, use_bias=True, key=key2),
        ]

    def __call__(self, t, y):
        for layer in self.layers:
            y = layer(y)
        return y


if __name__ == "__main__":
    y0 = jnp.linspace(-2, 2, 2000).reshape(-1, 1)

    key = jr.PRNGKey(2)
    vf = VectorField(key)
    rev_mid = Reversible(l=0.99, solver=Midpoint())

    h = 0.01
    T = 1

    loss, grads = grad_loss(vf, rev_mid, y0, h, T)
    print(grads.layers[0].weight)
