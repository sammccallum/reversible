import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from reversible.reversible_solver import Reversible
from reversible.solver_step import Midpoint
from reversible.standard_solver import Solver
from reversible.tracking.memory import MemoryTracker
from reversible.vector_field import AbstractVectorField

jax.config.update("jax_enable_x64", True)


def get_data(n):
    f = lambda x: x**3 + x
    x = jnp.linspace(-2, 2, num=n)
    y = f(x)
    return x, y


class VectorField(AbstractVectorField):
    layers: list

    def __init__(self, key):
        key1, key2 = jr.split(key, 2)

        self.layers = [
            eqx.nn.Linear("scalar", 50, use_bias=True, key=key1),
            jnp.tanh,
            eqx.nn.Linear(50, "scalar", use_bias=True, key=key2),
        ]

    def __call__(self, t, y):
        for layer in self.layers:
            y = layer(y)
        return y


class NeuralODE(eqx.Module):
    vf: VectorField
    solver: eqx.Module
    h: float
    T: float

    def __init__(self, vf, solver, h, T):
        self.vf = vf
        self.solver = solver
        self.h = h
        self.T = T

    def __call__(self, x):
        y = self.solver.solve_backward(self.vf, x, self.h, self.T)
        return y


@eqx.filter_value_and_grad
def grad_loss(model, x, y):
    y_pred = jax.vmap(model)(x)
    return jnp.mean((y_pred - y) ** 2)


@eqx.filter_jit
def make_step(model, x, y, optim, opt_state):
    loss, grads = grad_loss(model, x, y)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


if __name__ == "__main__":
    x, y = get_data(n=1000)

    n = 9
    n_runs = 10
    h = 0.01
    Ts = jnp.linspace(1, 5, num=n)

    method = "rev"

    if method == "rev":
        solver = Reversible(l=0.99, solver=Midpoint())
    elif method == "dto":
        solver = Solver(Midpoint(), continuous_adjoint=False)

    with open(f"results/{method}_mem_usage.txt", "w") as file:
        file.write("T, Memory Usage (MB)\n")
        for i in range(n):
            T = float(Ts[i])
            print(f"T={T}")
            file.write(f"{T}")
            for j in range(n_runs):
                mem_track = MemoryTracker()
                mem_track.start()

                key = jr.PRNGKey(j)
                vf = VectorField(key)
                model = NeuralODE(vf, solver, h, T)

                loss, grad = grad_loss(model, x, y)

                peak_memory = mem_track.end()
                file.write(f", {peak_memory}")
            file.write("\n")
