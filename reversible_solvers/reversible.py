from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.lax import fori_loop
from jaxtyping import Array, Float

from reversible_solvers.solver_step import AbstractSolverStep
from reversible_solvers.vector_field import AbstractVectorField

State = Float[Array, " d"]
ReversibleState = Tuple[State, State]


class Reversible(eqx.Module):
    """
    Reversible Solver.

    Wraps an AbstractSolverStep to create a reversible version of that solver.

    Backpropagation through the forward and backward solves are implemented in constant-memory w.r.t integration time. See _solve_forward and _solve_backward internal functions for details.
    """

    l: float
    solver: AbstractSolverStep

    def __init__(self, l: float, solver: AbstractSolverStep):
        """
        **Arguments:**
        - l: coupling parameter
        - solver: explicit solver to be wrapped
        """

        self.l = l
        self.solver = solver

    def solve_forward(
        self,
        vf: AbstractVectorField,
        y0: State,
        h: float,
        T: float,
    ) -> State:
        """
        Solve the ODE forward in time - over [0, T].

        **Arguments:**
        - vf: vector field
        - y0: initial state
        - h: step size
        - T: terminal time

        **Returns:**
        - state: state at t=T
        """
        return _solve_forward((vf, y0), h, T, self)

    def solve_backward(
        self,
        vf: AbstractVectorField,
        y0: ReversibleState,
        h: float,
        T: float,
    ) -> ReversibleState:
        """
        Solve the ODE backward in time - over [T, 0].

        **Arguments:**
        - y0: initial state
        - h: step size
        - T: initial time
        - vf: vector field

        **Returns:**
        - state: state at t=0
        """

        return _solve_backward((vf, y0), h, T, self)


# ================================================================================
# Custom vector-Jacobian product (vjp) rules for backpropagating the forward solve
# There are three functions:
# - _solve_forward: helper function
# - _solve_forward_fwd: forward pass
# - _solve_forward_bwd: custom backpropagation
# ================================================================================


@eqx.filter_custom_vjp
def _solve_forward(vjp_arg, h, T, self):
    """
    Helper forward solve function to allow custom vjp rules.
    """

    def forward_step(i, t_and_state):
        t0, y0, z0 = t_and_state
        t1 = t0 + h
        y1 = self.l * y0 + (1 - self.l) * z0 + self.solver.step(vf, h, t0, z0)
        z1 = z0 - self.solver.step(vf, -h, t1, y1)
        return (t1, y1, z1)

    vf, y0 = vjp_arg
    N = int(T / h)
    t0 = jnp.asarray(0.0)[None]
    t_and_state = (t0, y0, y0)
    t_and_state = fori_loop(0, N, forward_step, t_and_state)
    _, yN, zN = t_and_state

    return yN


@_solve_forward.def_fwd
def _solve_forward_fwd(perturbed, vjp_arg, h, T, self):
    """
    Forward solve for vjp rule.

    **Arguments:**
    - perturbed: True/False PyTree used by Equinox to determine which elements require gradients (see docs)

    **Returns:**
    - yN: state at t=T
    - t_and_state: final solver state for vjp residual
    """

    def forward_step(i, t_and_state):
        t0, y0, z0 = t_and_state
        t1 = t0 + h
        y1 = self.l * y0 + (1 - self.l) * z0 + self.solver.step(vf, h, t0, z0)
        z1 = z0 - self.solver.step(vf, -h, t1, y1)
        return (t1, y1, z1)

    vf, y0 = vjp_arg
    N = int(T / h)
    t0 = jnp.asarray(0.0)[None]
    t_and_state = (t0, y0, y0)
    t_and_state = fori_loop(0, N, forward_step, t_and_state)
    _, yN, zN = t_and_state

    return yN, t_and_state


@_solve_forward.def_bwd
def _solve_forward_bwd(t_and_state, grad_obj, perturbed, vjp_arg, h, T, self):
    """
    Backpropagatation through forward solve.

    Implements the vjp rules for backpropagating the forward solve of the reversible method.

    **Arguments:**
    - t_and_state: residuals from forward solve
    - grad_obj: adjoint state for output of forward solve (yN)
    - perturbed: True/False PyTree used by Equinox to determine which elements require gradients (see docs)

    **Returns:**
    - adj_theta: gradients w.r.t. parameters of vf
    - adj_y0: gradients w.r.t state y
    """

    def backward_step(i, t_and_state):
        t1, y1, z1 = t_and_state
        t0 = t1 - h
        z0 = z1 + self.solver.step(vf, -h, t1, y1)
        y0 = (
            (1 / self.l) * y1
            + (1 - (1 / self.l)) * z0
            - (1 / self.l) * self.solver.step(vf, h, t0, z0)
        )
        return (t0, y0, z0)

    def grad_step(i, args):
        t_and_state1, adj_y1, adj_z1, adj_theta = args
        t1, *state1 = t_and_state1

        t_and_state0 = backward_step(i, t_and_state1)
        t0, *state0 = t_and_state0

        _, grad_step_y1_fun = eqx.filter_vjp(self.solver.step, vf, -h, t1, state1[0])
        _, grad_step_z0_fun = eqx.filter_vjp(self.solver.step, vf, h, t0, state0[1])

        grad_step_y1 = grad_step_y1_fun(adj_z1)
        adj_y1 = adj_y1 - grad_step_y1[3]

        grad_step_z0 = grad_step_z0_fun(adj_y1)
        adj_y0 = self.l * adj_y1
        adj_z0 = adj_z1 + (1 - self.l) * adj_y1 + grad_step_z0[3]

        grad_step_y1 = grad_step_y1_fun(adj_z1)
        adj_theta = eqx.apply_updates(
            adj_theta, jax.tree_util.tree_map(lambda x: -x, grad_step_y1[0])
        )
        adj_theta = eqx.apply_updates(adj_theta, grad_step_z0[0])

        return t_and_state0, adj_y0, adj_z0, adj_theta

    vf, y0 = vjp_arg
    adj_y1 = grad_obj
    adj_z1 = jnp.zeros_like(t_and_state[2])
    adj_theta = eqx.filter(vf, eqx.is_inexact_array)
    adj_theta = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), adj_theta)
    t_and_state1 = t_and_state

    N = int(T / h)
    args = t_and_state1, adj_y1, adj_z1, adj_theta
    args = fori_loop(0, N, grad_step, args)
    t_and_state0, adj_y0, adj_z0, adj_theta = args

    return adj_theta, adj_y0


# =================================================================================
# Custom vector-Jacobian product (vjp) rules for backpropagating the backward solve
# There are three functions:
# - _solve_backward: helper function
# - _solve_backward_fwd: backward pass
# - _solve_backward_bwd: custom backpropagation
# =================================================================================


@eqx.filter_custom_vjp
def _solve_backward(vjp_arg, h, T, self):
    """
    Helper backward solve function to allow custom vjp rules.
    """

    def backward_step(i, t_and_state):
        t1, y1, z1 = t_and_state
        t0 = t1 - h
        z0 = z1 + self.solver.step(vf, -h, t1, y1)
        y0 = (
            (1 / self.l) * y1
            + (1 - (1 / self.l)) * z0
            - (1 / self.l) * self.solver.step(vf, h, t0, z0)
        )
        return (t0, y0, z0)

    vf, y0 = vjp_arg
    N = int(T / h)
    t1 = jnp.asarray(T)[None]
    t_and_state = (t1, y0, y0)
    t_and_state = fori_loop(0, N, backward_step, t_and_state)
    _, y0, z0 = t_and_state

    return y0


@_solve_backward.def_fwd
def _solve_backward_fwd(perturbed, vjp_arg, h, T, self):
    """
    Backward solve for vjp rule.

    **Arguments:**
    - perturbed: True/False PyTree used by Equinox to determine which elements require gradients (see docs)

    **Returns:**
    - y0: state at t=0
    - t_and_state: final solver state for vjp residual
    """

    def backward_step(i, t_and_state):
        t1, y1, z1 = t_and_state
        t0 = t1 - h
        z0 = z1 + self.solver.step(vf, -h, t1, y1)
        y0 = (
            (1 / self.l) * y1
            + (1 - (1 / self.l)) * z0
            - (1 / self.l) * self.solver.step(vf, h, t0, z0)
        )
        return (t0, y0, z0)

    vf, y0 = vjp_arg
    N = int(T / h)
    t1 = jnp.asarray(T)[None]
    t_and_state = (t1, y0, y0)
    t_and_state = fori_loop(0, N, backward_step, t_and_state)
    _, y0, z0 = t_and_state

    return y0, t_and_state


@_solve_backward.def_bwd
def _solve_backward_bwd(t_and_state, grad_obj, perturbed, vjp_arg, h, T, self):
    """
    Backpropagatation through backward solve.

    Implements the vjp rules for backpropagating the backward solve of the reversible method.

    **Arguments:**
    - t_and_state: residuals from backward solve
    - grad_obj: adjoint state for output of backward solve (y0)
    - perturbed: True/False PyTree used by Equinox to determine which elements require gradients (see docs)

    **Returns:**
    - adj_theta: gradients w.r.t. parameters of vf
    - adj_y1: gradients w.r.t state y
    """

    def forward_step(i, t_and_state):
        t0, y0, z0 = t_and_state
        t1 = t0 + h
        y1 = self.l * y0 + (1 - self.l) * z0 + self.solver.step(vf, h, t0, z0)
        z1 = z0 - self.solver.step(vf, -h, t1, y1)

        return (t1, y1, z1)

    def grad_step(i, args):
        t_and_state0, adj_y0, adj_z0, adj_theta = args
        t0, *state0 = t_and_state0

        t_and_state1 = forward_step(i, t_and_state0)
        t1, *state1 = t_and_state1

        _, grad_step_z0_fun = eqx.filter_vjp(self.solver.step, vf, h, t0, state0[1])
        _, grad_step_y1_fun = eqx.filter_vjp(self.solver.step, vf, -h, t1, state1[0])

        grad_step_z0 = grad_step_z0_fun(adj_y0)
        adj_z0 = adj_z0 + (1 - (1 / self.l)) * adj_y0 - (1 / self.l) * grad_step_z0[3]
        adj_z1 = adj_z0

        grad_step_y1 = grad_step_y1_fun(adj_z0)
        adj_y1 = grad_step_y1[3] + (1 / self.l) * adj_y0

        adj_theta = eqx.apply_updates(adj_theta, grad_step_y1[0])
        adj_theta = eqx.apply_updates(
            adj_theta,
            jax.tree_util.tree_map(lambda x: -(1 / self.l) * x, grad_step_z0[0]),
        )

        return t_and_state1, adj_y1, adj_z1, adj_theta

    vf, y0 = vjp_arg
    adj_y0 = grad_obj
    adj_z0 = jnp.zeros_like(t_and_state[2])
    adj_theta = eqx.filter(vf, eqx.is_inexact_array)
    adj_theta = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), adj_theta)
    t_and_state0 = t_and_state

    N = int(T / h)
    args = t_and_state0, adj_y0, adj_z0, adj_theta
    args = fori_loop(0, N, grad_step, args)
    t_and_state1, adj_y1, adj_z1, adj_theta = args

    return adj_theta, adj_y1
