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


if __name__ == "__main__":
    y_dim = 1000
    width_size = 1000
    depth = 4
    model = VectorField(y_dim, width_size, depth, key=jr.PRNGKey(10))
    solver = Reversible(l=0.999, solver=Euler())
    h = 0.01
    T = 5
    y0 = jnp.linspace(1.0, 10.0, num=y_dim)
    measure_runtime(model, y0, h, T)

    t0 = 0.5

    # step, vjp_fun = eqx.filter_vjp(solver.solver.step, model, h, t0, y0)
    # print(eqx.filter_make_jaxpr(vjp_fun))
    # print(eqx.filter_jit(vjp_fun).lower(y0).as_text())

    # loss, grads = grad_loss(model, y0, h, T)
    # print(grads)
    # print(grads.mlp.layers[0].weight)

    # n = 20
    # y_dims = jnp.linspace(10, 1000, num=n, dtype=jnp.int64)
    # runtimes = np.zeros(n)
    # for i in range(len(y_dims)):
    #     y0 = jnp.linspace(1.0, 10.0, num=y_dims[i])
    #     model = VectorField(y_dims[i], y_dims[i], depth, key=jr.PRNGKey(10))
    #     runtimes[i] = measure_runtime(model, y0, h, T)

    # plt.plot(y_dims, runtimes, ".-")
    # plt.savefig("runtimes_no_update.png", dpi=300)
