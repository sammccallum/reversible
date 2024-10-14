import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from reversible.reversible_solver import Reversible
from reversible.solver_step import RK4, Dopri5, Midpoint
from reversible.standard_solver import Solver
from reversible.tracking.memory import MemoryTracker
from reversible.vector_field import AbstractVectorField


def load_data(permutation_key, split):
    X = np.load("data/character_trajectories.npy")
    y = np.load("data/labels.npy")
    dataset_size, ts_length, _ = X.shape
    ts = np.broadcast_to(np.linspace(0, 1, num=ts_length), (dataset_size, ts_length))
    Xs = np.concatenate([ts[:, :, None], X], axis=-1)

    Xs = jnp.array(Xs)
    y = jnp.array(y, dtype="int32") - 1  # [1 ... 20] to [0 ... 19]

    Xs = jr.permutation(permutation_key, Xs)
    y = jr.permutation(permutation_key, y)

    X_train, X_test = Xs[:split], Xs[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test


class BaseVectorField(AbstractVectorField):
    data_size: int
    hidden_size: int
    mlp: eqx.nn.MLP

    def __init__(self, data_size, hidden_size, width_size, depth, key):
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * data_size,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.relu,
            final_activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t, y):
        return self.mlp(y).reshape(self.hidden_size, self.data_size)


class VectorField(AbstractVectorField):
    base: BaseVectorField
    control: diffrax.CubicInterpolation

    def __init__(self, base, control):
        self.base = base
        self.control = control

    def __call__(self, t, y):
        x = self.control.derivative(t)
        return jnp.dot(self.base(t, y), x.T).flatten()


def solve(y0, xs, base: BaseVectorField, solver: Reversible):
    ts = xs[:, 0]
    coeffs = diffrax.backward_hermite_coefficients(ts, xs)
    control = diffrax.CubicInterpolation(ts, coeffs)
    vf = VectorField(base, control)

    h = 0.01
    T = 1
    yN = solver.solve_forward(vf, y0, h, T)
    return yN


class NeuralCDE(eqx.Module):
    initial: eqx.nn.Linear
    base: BaseVectorField
    final: eqx.nn.Linear

    def __init__(self, base, key):
        key1, key2 = jr.split(key)
        self.initial = eqx.nn.Linear(base.data_size, base.hidden_size, key=key1)
        self.base = base
        self.final = eqx.nn.Linear(base.hidden_size, 20, key=key2)

    def __call__(self, xs, solver):
        y0 = self.initial(xs[0])
        yN = solve(y0, xs, self.base, solver)
        prediction = jax.nn.softmax(self.final(yN))
        return prediction


def dataloader(X, y, batch_size, key):
    dataset_size = X.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield (X[batch_perm], y[batch_perm])
            start = end
            end = start + batch_size


@eqx.filter_jit
def calculate_pred_and_accuracy(model, xs, ys, solver):
    pred = jax.vmap(model, in_axes=(0, None))(xs, solver)
    pred_argmax = jnp.argmax(pred, axis=1)
    acc = jnp.mean(ys == pred_argmax)
    return pred, acc


@eqx.filter_value_and_grad(has_aux=True)
def grad_loss(model, xs, ys, solver):
    pred, acc = calculate_pred_and_accuracy(model, xs, ys, solver)
    pred = jnp.take_along_axis(pred, jnp.expand_dims(ys, 1), axis=1)
    return -jnp.mean(pred), acc


@eqx.filter_jit
def make_step(model, xs, ys, solver, optim, opt_state):
    (loss, acc), grads = grad_loss(model, xs, ys, solver)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, acc, model, opt_state


if __name__ == "__main__":
    key = jr.PRNGKey(100)
    batch_size = 32
    X_train, X_test, y_train, y_test = load_data(key, split=70 * batch_size)

    n_steps = 1000
    lr = 1e-3
    batch_size = 70 * batch_size  # one batch for dataloader

    method = "dto"
    solver_steps = [Midpoint(), RK4(), Dopri5()]
    solver_names = ["midpoint", "rk4", "dopri5"]

    for i in range(len(solver_steps)):
        if method == "rev":
            solver = Reversible(l=0.999, solver=solver_steps[i])
        elif method == "dto":
            solver = Solver(solver_steps[i], continuous_adjoint=False)

        mem_usages = []
        accs = []
        n_runs = 10
        for j in range(n_runs):
            mem_track = MemoryTracker()
            mem_track.start()

            key = jr.PRNGKey(j)
            key1, key2, key3 = jr.split(key, 3)

            base = BaseVectorField(
                data_size=4, hidden_size=32, width_size=32, depth=4, key=key1
            )
            model = NeuralCDE(base, key2)

            optim = optax.adamw(lr)
            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

            best_acc = 0
            for step, (Xs, ys) in zip(
                range(n_steps), dataloader(X_train, y_train, batch_size, key3)
            ):
                loss, acc, model, opt_state = make_step(
                    model, Xs, ys, solver, optim, opt_state
                )
                _, test_acc = calculate_pred_and_accuracy(model, X_test, y_test, solver)
                if step % 100 == 0 or step == n_steps - 1:
                    print(
                        f"Step: {step}, Loss: {loss}, Train accuracy: {acc}, Test accuracy: {test_acc}"
                    )
                if test_acc > best_acc:
                    best_acc = test_acc

            mem_usage = mem_track.end()
            mem_usages.append(int(mem_usage))
            accs.append(best_acc)

        with open(f"results/{method}/{solver_names[i]}.txt", "w") as file:
            file.write("Memory Usage (MB), Best Accuracy \n")
            for k in range(n_runs):
                file.write(f"{mem_usages[k]}, {accs[k]} \n")
