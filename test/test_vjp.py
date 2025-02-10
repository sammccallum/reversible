import functools as ft
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from reversible.reversible_solver import Reversible
from reversible.solver_step import Euler
from reversible.vector_field import AbstractVectorField

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
jax.config.update("jax_enable_x64", True)


class VectorField(AbstractVectorField):
    mlp: eqx.nn.MLP

    def __init__(self, y_dim, width_size, depth, key):
        self.mlp = eqx.nn.MLP(
            y_dim, y_dim, width_size, depth, activation=jnp.tanh, key=key
        )

    def __call__(self, t, y):
        return self.mlp(y)


@eqx.filter_jit
@eqx.filter_value_and_grad
def grad_loss(model, y0, h, T):
    y1 = solver.solve_forward(model, y0, h, T)
    return jnp.mean(y1**2)


def measure_runtime(model, y0, h, T):
    tic = time.time()
    loss, grads = grad_loss(model, y0, h, T)
    toc = time.time()
    print(f"Compile time: {(toc - tic):.5f}")

    repeats = 10
    tic = time.time()
    for i in range(repeats):
        loss, grads = jax.block_until_ready(grad_loss(model, y0, h, T))
    toc = time.time()
    runtime = (toc - tic) / repeats
    print(f"Runtime: {runtime:.5f}")
    return runtime


@eqx.filter_jit
def grad_step(self, vf, t1, y1, z1, adj_y1, adj_z1, adj_theta1):
    step_vjp_fn = lambda vf, h, t, y: eqx.filter_vjp(self.solver.step, vf, h, t, y)

    step_y1, grad_step_y1_fun = step_vjp_fn(vf, -h, t1, y1)
    t0 = t1 - h
    z0 = z1 + step_y1

    step_z0, grad_step_z0_fun = step_vjp_fn(vf, h, t0, z0)
    y0 = (1 / self.l) * y1 + (1 - (1 / self.l)) * z0 - (1 / self.l) * step_z0

    grad_step_y1 = grad_step_y1_fun(adj_z1)
    adj_y1 = adj_y1 - grad_step_y1[3]

    grad_step_z0 = grad_step_z0_fun(adj_y1)
    adj_y0 = self.l * adj_y1
    adj_z0 = adj_z1 + (1 - self.l) * adj_y1 + grad_step_z0[3]

    adj_theta0 = jax.tree_map(lambda x, y: y - x, grad_step_y1[0], grad_step_z0[0])
    adj_theta0 = eqx.apply_updates(adj_theta1, adj_theta0)
    return (t0, y0, z0, adj_y0, adj_z0, adj_theta0)


if __name__ == "__main__":
    y_dim = 100
    width_size = 100
    depth = 4
    model = VectorField(y_dim, width_size, depth, key=jr.PRNGKey(10))
    solver = Reversible(l=0.999, solver=Euler())
    h = 0.01
    t = jnp.array(0.05)[None]
    T = 5
    y0 = jnp.linspace(1.0, 10.0, num=y_dim)

    adj_y1 = jnp.ones_like(y0)
    adj_z1 = jnp.zeros_like(y0)
    adj_theta1 = eqx.filter(model, eqx.is_inexact_array)
    adj_theta1 = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), adj_theta1)

    print(
        eqx.filter_make_jaxpr(grad_step)(
            solver, model, t, y0, y0, adj_y1, adj_z1, adj_theta1
        )
    )
