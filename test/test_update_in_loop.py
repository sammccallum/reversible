import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from reversible.vector_field import AbstractVectorField

jax.config.update("jax_enable_x64", True)


class VectorField(AbstractVectorField):
    mlp: eqx.nn.MLP

    def __init__(self, y_dim, width_size, depth, key):
        self.mlp = eqx.nn.MLP(y_dim, y_dim, width_size, depth, key=key)

    def __call__(self, t, y):
        return self.mlp(y)


def update_step(state):
    i, pytree = state
    # pytree = eqx.apply_updates(
    #     pytree,
    #     jax.tree_util.tree_map(lambda x, y: y + x, pytree, pytree),
    # )
    pytree = jax.tree_util.tree_map(lambda x, y: y + x, pytree, pytree)
    i += 1
    return i, pytree


def cond_fun(state):
    return state[0] < 500


@eqx.filter_jit
def run_loop(pytree):
    state = (0, pytree)
    state = eqx.internal.while_loop(cond_fun, update_step, state, kind="lax")
    i, pytree = state
    return pytree


def update_step_array(state):
    i, array = state
    array_copy = array.copy()
    new_array = array + array_copy
    i += 1
    return i, new_array


@eqx.filter_jit
def run_loop_array(array):
    state = (0, array)
    state = eqx.internal.while_loop(cond_fun, update_step_array, state, kind="lax")
    i, array = state
    return array


def measure_runtime(pytree, loop):
    tic = time.time()
    pytree = loop(pytree)
    toc = time.time()
    print(f"Compile time: {(toc - tic):.5f}")

    repeats = 10
    tic = time.time()
    for i in range(repeats):
        pytree = jax.block_until_ready(loop(pytree))
    toc = time.time()
    runtime = (toc - tic) / repeats
    print(f"Runtime: {runtime:.5f}")
    return runtime


class Array(eqx.Module):
    array: jax.Array

    def __init__(self, y_dim):
        self.array = jnp.linspace(1.0, 10.0, num=y_dim)


if __name__ == "__main__":
    y_dim = 10
    width_size = 10
    depth = 4
    pytree = VectorField(y_dim, width_size, depth, key=jr.PRNGKey(10))

    n = 20
    y_dims = jnp.linspace(10, 1000, num=n, dtype=jnp.int64)
    runtimes = np.zeros(n)
    for i in range(len(y_dims)):
        pytree = VectorField(y_dims[i], y_dims[i], depth, key=jr.PRNGKey(10))
        pytree = eqx.filter(pytree, eqx.is_inexact_array)
        runtimes[i] = measure_runtime(pytree, run_loop)

    plt.plot(y_dims, runtimes, ".-")
    plt.savefig("runtimes.png", dpi=300)

    # pytree = Array(y_dim)
    # array = jnp.linspace(1.0, 10.0, num=y_dim)
    # measure_runtime(array, run_loop_array)
