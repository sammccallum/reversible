import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.profiler
import jax.random as jr
import optax
from sklearn.datasets import make_circles, make_moons

from reversible.cnf_solver import CNFSolver
from reversible.solver_step import RK4, Dopri5, Midpoint
from reversible.standard_solver import Solver
from reversible.tracking.memory import MemoryTracker
from reversible.vector_field import AbstractVectorField


def generate_data_moons(n):
    X, _ = make_moons(n_samples=n, noise=0.05, random_state=0)
    X = (X - X.mean()) / X.std()
    return jnp.array(X)


def generate_data_circles(n):
    X, _ = make_circles(n_samples=n, noise=0.05, factor=0.5, random_state=0)
    X = (X - X.mean()) / X.std()
    return jnp.array(X)


class ConcatSquash(eqx.Module):
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    lin3: eqx.nn.Linear

    def __init__(self, in_size, out_size, key):
        key1, key2, key3 = jr.split(key, 3)
        self.lin1 = eqx.nn.Linear(in_size, out_size, key=key1)
        self.lin2 = eqx.nn.Linear(1, out_size, key=key2)
        self.lin3 = eqx.nn.Linear(1, out_size, use_bias=False, key=key3)

    def __call__(self, t, y):
        return self.lin1(y) * jax.nn.sigmoid(self.lin2(t)) + self.lin3(t)


class VectorField(AbstractVectorField):
    layers: list

    def __init__(self, data_size, width_size, key):
        keys = jr.split(key, 4)
        self.layers = [
            ConcatSquash(in_size=data_size, out_size=width_size, key=keys[0]),
            ConcatSquash(in_size=width_size, out_size=width_size, key=keys[1]),
            ConcatSquash(in_size=width_size, out_size=width_size, key=keys[2]),
            ConcatSquash(in_size=width_size, out_size=data_size, key=keys[3]),
        ]

    def __call__(self, t, y):
        for layer in self.layers[:-1]:
            y = layer(t, y)
            y = jax.nn.tanh(y)
        y = self.layers[-1](t, y)

        return y


class ConcatVectorField(AbstractVectorField):
    vf: AbstractVectorField
    data_size: float

    def __init__(self, vf, data_size):
        self.vf = vf
        self.data_size = data_size

    def calculate_vectorfields(self, t, y):
        fn = lambda y: self.vf(t, y)
        f, vjp_fn = jax.vjp(fn, y)
        eye = jnp.eye(self.data_size)
        (dfdy,) = jax.vmap(vjp_fn)(eye)
        trJ = jnp.trace(dfdy)[None]
        return jnp.concatenate((f, -trJ))

    def __call__(self, t, y):
        state = y[: self.data_size]
        return self.calculate_vectorfields(t, state)


class CNF(eqx.Module):
    vf: AbstractVectorField
    solver: Solver
    T: float
    h: float
    data_size: int

    def __init__(self, vf, solver, T, h, data_size):
        self.vf = vf
        self.solver = solver
        self.T = T
        self.h = h
        self.data_size = data_size

    def train(self, y):
        X_and_integral1 = jnp.concatenate((y, jnp.zeros(1)))
        X_and_integral0 = self.solver.solve_backward(
            self.vf, X_and_integral1, self.h, self.T
        )

        y0 = X_and_integral0[:data_size]
        integral0 = X_and_integral0[data_size]
        return log_normal(y0) - integral0


class CNFReversible(eqx.Module):
    vf: AbstractVectorField
    solver: CNFSolver
    T: float
    h: float

    def __init__(self, vf, solver, T, h):
        self.vf = vf
        self.solver = solver
        self.T = T
        self.h = h

    def train(self, y):
        y0, I0 = self.solver.solve_backward(self.vf, y, self.h, self.T)
        return log_normal(y0) - I0


def log_normal(y):
    return -0.5 * (2 * jnp.log(2 * jnp.pi * jnp.ones(y.shape)) + jnp.sum(y**2))


@eqx.filter_value_and_grad
def grad_loss(model, X):
    logpT = jax.vmap(model.train)(X)
    return -jnp.mean(logpT)


@eqx.filter_jit
def make_step(model, X, optim, opt_state):
    loss, grads = grad_loss(model, X)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


if __name__ == "__main__":
    X = generate_data_moons(n=1000)
    _, data_size = X.shape
    width_size = 128

    h = 0.01
    T = 1
    n_steps = 5000
    lr = 1e-3

    method = "dto"
    solver_steps = [Midpoint(), RK4(), Dopri5()]
    solver_names = ["midpoint", "rk4", "dopri5"]

    for i in range(len(solver_steps)):
        if method == "rev":
            solver = CNFSolver(l=0.999, solver=solver_steps[i])
        elif method == "dto":
            solver = Solver(solver_steps[i], continuous_adjoint=False)

        mem_usages = []
        losses = []
        n_runs = 10
        for j in range(n_runs):
            mem_track = MemoryTracker()
            mem_track.start()

            key = jr.PRNGKey(j)
            vf = VectorField(data_size, width_size, key)
            if method == "rev":
                model = CNFReversible(vf, solver, T, h)
            else:
                concat_vf = ConcatVectorField(vf, data_size)
                model = CNF(concat_vf, solver, T, h, data_size)

            optim = optax.adamw(lr)
            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

            best_loss = 100
            for step in range(n_steps):
                loss, model, opt_state = make_step(model, X, optim, opt_state)
                if step % 100 == 0 or step == n_steps - 1:
                    print(f"Step: {step}, Loss: {loss}")

                if loss < best_loss:
                    best_loss = loss

            mem_usage = mem_track.end()
            mem_usages.append(int(mem_usage))
            losses.append(best_loss)

        with open(f"results/{method}/{solver_names[i]}.txt", "w") as file:
            file.write("Memory Usage (MB), Best Loss \n")
            for k in range(n_runs):
                file.write(f"{mem_usages[k]}, {losses[k]} \n")
