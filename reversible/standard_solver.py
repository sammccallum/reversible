from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.lax import fori_loop
from jaxtyping import Array, Float

from reversible.solver_step import AbstractSolverStep
from reversible.vector_field import AbstractVectorField

State = Float[Array, " d"]


class Solver(eqx.Module):
    """
    Standard Solver.

    Wraps an AbstractSolverStep to create a full solver.
    """

    solver: AbstractSolverStep
    adjoint: bool

    def __init__(self, solver: AbstractSolverStep, continuous_adjoint: bool = False):
        """
        **Arguments:**
        - step: explicit solver step
        """

        self.solver = solver
        self.adjoint = continuous_adjoint

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
        if self.adjoint:
            return _solve_forward_adjoint((vf, y0), h, T, self)
        else:
            return _solve_forward(vf, y0, h, T, self)

    def solve_backward(
        self,
        vf: AbstractVectorField,
        y1: State,
        h: float,
        T: float,
    ) -> State:
        """
        Solve the ODE backward in time - over [T, 0].

        **Arguments:**
        - y1: initial state
        - h: step size
        - T: initial time
        - vf: vector field

        **Returns:**
        - state: state at t=0
        """

        if self.adjoint:
            return _solve_backward_adjoint((vf, y1), h, T, self)
        else:
            return _solve_backward(vf, y1, h, T, self)


class AdjointVectorField(eqx.Module):
    """
    Adjoint equation vector fields.
    """

    vf: AbstractVectorField
    f: Callable

    def __init__(self, vf: AbstractVectorField):
        """
        **Arguments:**
        - vf: Neural ODE vector field
        """
        self.vf = vf
        self.f = lambda vf, t, y: vf(t, y)

    def __call__(self, t: Float[Array, " 1"], y: State) -> State:
        """
        **Arguments:**
        - t: time
        - y: adjoint equations state

        **Returns:**
        - adjoint vector fields
        """
        y, adj_y, adj_theta = y
        vf, vjp_fun = eqx.filter_vjp(self.f, self.vf, t, y)
        adj_vfs = vjp_fun(adj_y)
        return vf, -adj_vfs[2], jax.tree_util.tree_map(lambda x: -x, adj_vfs[0])


def _solve_forward(vf, y0, h, T, self):
    """
    Forward solve for discretise-then-optimise backpropagation.
    """

    def forward_step(i, t_and_state):
        t0, y0 = t_and_state
        t1 = t0 + h
        y1 = y0 + self.solver.step(vf, h, t0, y0)
        return (t1, y1)

    N = int(T / h)
    t0 = jnp.asarray(0.0)[None]
    t_and_state = (t0, y0)
    t_and_state = fori_loop(0, N, forward_step, t_and_state)
    t1, y1 = t_and_state
    return y1


def _solve_backward(vf, y1, h, T, self):
    """
    Backward solve for discretise-then-optimise backpropagation.
    """

    def backward_step(i, t_and_state):
        t1, y1 = t_and_state
        t0 = t1 - h
        y0 = y1 + self.solver.step(vf, -h, t1, y1)
        return (t0, y0)

    N = int(T / h)
    t1 = jnp.asarray(T)[None]
    t_and_state = (t1, y1)
    t_and_state = fori_loop(0, N, backward_step, t_and_state)
    t0, y0 = t_and_state
    return y0


# ================================================================================
# Custom vector-Jacobian product (vjp) rules for backpropagating the forward solve
# There are three functions:
# - _solve_forward: helper function
# - _solve_forward_fwd: forward pass
# - _solve_forward_bwd: continuous adjoint method
# ================================================================================


@eqx.filter_custom_vjp
def _solve_forward_adjoint(vjp_arg, h, T, self):
    """
    Helper forward solve function to allow continuous adjoint vjp rules.
    """

    vf, y0 = vjp_arg
    return _solve_forward(vf, y0, h, T, self)


@_solve_forward_adjoint.def_fwd
def _solve_forward_adjoint_fwd(perturbed, vjp_arg, h, T, self):
    """
    Forward solve for vjp rule.

    **Arguments:**
    - perturbed: True/False PyTree used by Equinox to determine which elements require gradients (see docs)

    **Returns:**
    - yN: state at t=T
    - t_and_state: final solver state for vjp residual
    """

    vf, y0 = vjp_arg
    y1 = _solve_forward(vf, y0, h, T, self)
    t_and_state = (jnp.asarray(T)[None], y1)
    return y1, t_and_state


@_solve_forward_adjoint.def_bwd
def _solve_forward_adjoint_bwd(t_and_state, grad_obj, perturbed, vjp_arg, h, T, self):
    """
    Backpropagatation through forward solve.

    Implements the vjp rules for the continuous adjoint method.

    **Arguments:**
    - t_and_state: residuals from forward solve
    - grad_obj: adjoint state for output of forward solve (yN)
    - perturbed: True/False PyTree used by Equinox to determine which elements require gradients (see docs)

    **Returns:**
    - adj_theta: gradients w.r.t. parameters of vf
    - adj_y0: gradients w.r.t state y
    """

    def grad_step(i, args):
        t_and_state1, adj_y1, adj_theta = args
        t1, y1 = t_and_state1
        step = self.solver.step(adj_vf, -h, t1, (y1, adj_y1, adj_theta))
        y0 = y1 + step[0]
        adj_y0 = adj_y1 + step[1]
        adj_theta = eqx.apply_updates(adj_theta, step[2])
        t0 = t1 - h
        return (t0, y0), adj_y0, adj_theta

    vf, y0 = vjp_arg
    adj_y1 = grad_obj
    adj_theta = eqx.filter(vf, eqx.is_inexact_array)
    adj_theta = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), adj_theta)
    t_and_state1 = t_and_state

    N = int(T / h)
    adj_vf = AdjointVectorField(vf)
    args = t_and_state1, adj_y1, adj_theta
    args = fori_loop(0, N, grad_step, args)
    t_and_state0, adj_y0, adj_theta = args

    return adj_theta, adj_y0


# =================================================================================
# Custom vector-Jacobian product (vjp) rules for backpropagating the backward solve
# There are three functions:
# - _solve_backward: helper function
# - _solve_backward_fwd: backward pass
# - _solve_backward_bwd: continuous adjoint method
# =================================================================================


@eqx.filter_custom_vjp
def _solve_backward_adjoint(vjp_arg, h, T, self):
    """
    Helper backward solve function to allow continous adjoint vjp rules.
    """

    vf, y1 = vjp_arg
    return _solve_backward(vf, y1, h, T, self)


@_solve_backward_adjoint.def_fwd
def _solve_backward_adjoint_fwd(perturbed, vjp_arg, h, T, self):
    """
    Backward solve for vjp rule.

    **Arguments:**
    - perturbed: True/False PyTree used by Equinox to determine which elements require gradients (see docs)

    **Returns:**
    - y0: state at t=0
    - t_and_state: final solver state for vjp residual
    """

    vf, y1 = vjp_arg
    y0 = _solve_backward(vf, y1, h, T, self)
    t_and_state = (jnp.asarray(0.0)[None], y0)
    return y0, t_and_state


@_solve_backward_adjoint.def_bwd
def _solve_backward_adjoint_bwd(t_and_state, grad_obj, perturbed, vjp_arg, h, T, self):
    """
    Backpropagatation through backward solve.

    Implements the vjp rules for the continuous adjoint method.

    **Arguments:**
    - t_and_state: residuals from backward solve
    - grad_obj: adjoint state for output of backward solve (y0)
    - perturbed: True/False PyTree used by Equinox to determine which elements require gradients (see docs)

    **Returns:**
    - adj_theta: gradients w.r.t. parameters of vf
    - adj_y1: gradients w.r.t state y
    """

    def grad_step(i, args):
        t_and_state0, adj_y0, adj_theta = args
        t0, y0 = t_and_state0
        step = self.solver.step(adj_vf, h, t0, (y0, adj_y0, adj_theta))
        y1 = y0 + step[0]
        adj_y1 = adj_y0 + step[1]
        adj_theta = eqx.apply_updates(adj_theta, step[2])
        t1 = t0 + h
        return (t1, y1), adj_y1, adj_theta

    vf, y1 = vjp_arg
    adj_y0 = grad_obj
    adj_theta = eqx.filter(vf, eqx.is_inexact_array)
    adj_theta = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), adj_theta)
    t_and_state0 = t_and_state

    N = int(T / h)
    adj_vf = AdjointVectorField(vf)
    args = t_and_state0, adj_y0, adj_theta
    args = fori_loop(0, N, grad_step, args)
    t_and_state1, adj_y1, adj_theta = args

    return adj_theta, adj_y1
