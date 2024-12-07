# Efficient, Accurate and Stable Gradients for Neural ODEs

## Overview
This repository contains a JAX implementation of the Reversible Solver method introduced [here](https://arxiv.org/abs/2410.11648).

We present a general class of algebraically reversible solvers that allows any explicit numerical solver to be made reversible. This class of reversible solvers produce exact, memory-efficient gradients and are:
- high-order,
- numerically stable,
- and naturally extend to Neural CDEs and SDEs.

NOTE: we now have an improved implementation of reversible solvers in [diffrax](https://github.com/sammccallum/diffrax/tree/reversible). This has all the bells and whistles of diffrax + reversible backpropagation! See [this PR](https://github.com/patrick-kidger/diffrax/pull/528) for more information.

## Example
Simple Neural ODE example. We wrap the Dormand-Prince 5/4 (Dopri5) solver in a Reversible class.

If the `solve_forward` function appears in any `jax.grad` region, the memory-efficient backpropagation algorithm through the solve is automatically used.

```python
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from reversible.reversible_solver import Reversible
from reversible.solver_step import Dopri5
from reversible.vector_field import AbstractVectorField


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

# Solve over [0, T]
h = 0.01
T = 1
y0 = jnp.asarray(1.0)[None]  # shape (1,)
y1 = solver.solve_forward(vf, y0, h, T)

```

## Experiments
All code to reproduce the experiments presented in the paper can be found in the `experiments` folder.

## Installation
To install the reversible package, clone the repository and run:
```bash
pip install -e reversible
```
