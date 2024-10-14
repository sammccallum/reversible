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
IntegralState = Float[Array, " 1"]
ConcatState = Float[Array, " d+1"]


class CNFSolver(eqx.Module):
    """
    Continuous Normalising Flow (CNF) solver.

    Additionally implements the memory-efficient backpropagation algorithm for reversible solvers.
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

    def solve_backward(
        self,
        vf: AbstractVectorField,
        y1: State,
        h: float,
        T: float,
    ) -> Tuple[State, IntegralState]:
        """
        Solve the CNF backwards in time - over [T, 0].

        **Arguments:**
        - vf: Neural ODE vector field
        - y1: initial state
        - h: step size
        - T: terminal time

        **Returns:**
        - y0: state at t=0
        - I0: calculated integral term
        """
        return _solve_backward(vf, y1, h, T, self)


class IntegralVectorField(AbstractVectorField):
    """
    Wraps the Neural ODE vector field to create the concatenated vector field (Neural ODE vector field + integrand).
    """

    vf: AbstractVectorField
    data_size: int

    def __init__(self, vf: AbstractVectorField, data_size: int):
        """
        **Arguments:**
        - vf: Neural ODE vector field
        - data_size: dimension of the data
        """
        self.vf = vf
        self.data_size = data_size

    def __call__(self, t: Float[Array, " 1"], y: ConcatState) -> ConcatState:
        """
        **Arguments:**
        - t: time
        - y: ODE + Integral state
        """
        y = y[: self.data_size]

        fn = lambda y: self.vf(t, y)
        f, vjp_fn = jax.vjp(fn, y)
        eye = jnp.eye(self.data_size)
        (dfdy,) = jax.vmap(vjp_fn)(eye)
        trJ = jnp.trace(dfdy)[None]
        return jnp.concatenate([f, -trJ])


# =================================================================================
# Custom vector-Jacobian product (vjp) rules for backpropagating the backward solve
# There are three functions:
# - _solve_backward: helper function
# - _solve_backward_fwd: backward pass
# - _solve_backward_bwd: custom backpropagation
# =================================================================================


@eqx.filter_custom_vjp
def _solve_backward(vjp_arg, y1, h, T, self):
    """
    Helper backward solve function to allow custom vjp rules.
    """

    def backward_step(i, t_and_state):
        t1, y1, z1, I1 = t_and_state
        t0 = t1 - h
        z0 = z1 + self.solver.step(vf, -h, t1, y1)
        y0 = (
            (1 / self.l) * y1
            + (1 - (1 / self.l)) * z0
            - (1 / self.l) * self.solver.step(vf, h, t0, z0)
        )
        I0 = (
            I1
            + self.solver.step(integral_vf, -h, t1, jnp.concatenate([y1, I1]))[
                data_size
            ]
        )
        return (t0, y0, z0, I0)

    vf = vjp_arg
    data_size = y1.shape[0]
    integral_vf = IntegralVectorField(vf, data_size)

    N = int(T / h)
    t1 = jnp.asarray(T)[None]
    I1 = jnp.asarray(0.0)[None]
    t_and_state = (t1, y1, y1, I1)
    t_and_state = fori_loop(0, N, backward_step, t_and_state)
    _, y0, z0, I0 = t_and_state

    return y0, I0


@_solve_backward.def_fwd
def _solve_backward_fwd(perturbed, vjp_arg, y1, h, T, self):
    """
    Backward solve for vjp rule.

    **Arguments:**
    - perturbed: True/False PyTree used by Equinox to determine which elements require gradients (see docs)

    **Returns:**
    - y0: state at t=0
    - I0: calculated integral
    - t_and_state: final solver state for vjp residual
    """

    def backward_step(i, t_and_state):
        t1, y1, z1, I1 = t_and_state
        t0 = t1 - h
        z0 = z1 + self.solver.step(vf, -h, t1, y1)
        y0 = (
            (1 / self.l) * y1
            + (1 - (1 / self.l)) * z0
            - (1 / self.l) * self.solver.step(vf, h, t0, z0)
        )
        I0 = (
            I1
            + self.solver.step(integral_vf, -h, t1, jnp.concatenate([y1, I1]))[
                data_size
            ]
        )
        return (t0, y0, z0, I0)

    vf = vjp_arg
    data_size = y1.shape[0]
    integral_vf = IntegralVectorField(vf, data_size)

    N = int(T / h)
    t1 = jnp.asarray(T)[None]
    I1 = jnp.asarray(0.0)[None]
    t_and_state = (t1, y1, y1, I1)
    t_and_state = fori_loop(0, N, backward_step, t_and_state)
    _, y0, z0, I0 = t_and_state

    return (y0, I0), t_and_state


@_solve_backward.def_bwd
def _solve_backward_bwd(t_and_state, grad_obj, perturbed, vjp_arg, y1, h, T, self):
    """
    Backpropagatation through backward solve.

    Implements the vjp rules for backpropagating the backward solve of the reversible CNF method.

    **Arguments:**
    - t_and_state: residuals from backward solve
    - grad_obj: adjoint state for output of backward solve (y0, I0)
    - perturbed: True/False PyTree used by Equinox to determine which elements require gradients (see docs)

    **Returns:**
    - adj_theta: gradients w.r.t. parameters of vf
    """

    def forward_step(i, t_and_state):
        t0, y0, z0, I0 = t_and_state
        t1 = t0 + h
        y1 = self.l * y0 + (1 - self.l) * z0 + self.solver.step(vf, h, t0, z0)
        z1 = z0 - self.solver.step(vf, -h, t1, y1)
        I1 = (
            I0
            - self.solver.step(integral_vf, -h, t1, jnp.concatenate([y1, I0]))[
                data_size
            ]
        )
        return t1, y1, z1, I1

    def grad_step(i, args):
        t_and_state0, adj_y0, adj_z0, adj_I0, adj_theta = args
        t0, y0, z0, I0 = t_and_state0

        t_and_state1 = forward_step(i, t_and_state0)
        t1, y1, z1, I1 = t_and_state1

        _, grad_step_z0_fun = eqx.filter_vjp(self.solver.step, vf, h, t0, z0)
        _, grad_step_y1_fun = eqx.filter_vjp(self.solver.step, vf, -h, t1, y1)
        _, grad_integral_step_y1_fun = eqx.filter_vjp(
            self.solver.step, integral_vf, -h, t1, jnp.concatenate([y1, I1])
        )

        grad_step_z0 = grad_step_z0_fun(adj_y0)
        adj_z0 = adj_z0 + (1 - (1 / self.l)) * adj_y0 - (1 / self.l) * grad_step_z0[3]
        adj_z1 = adj_z0

        grad_step_y1 = grad_step_y1_fun(adj_z0)
        grad_integral_step_y1 = grad_integral_step_y1_fun(
            jnp.concatenate([jnp.zeros_like(y1), adj_I0])
        )
        adj_y1 = (
            grad_step_y1[3]
            + (1 / self.l) * adj_y0
            + grad_integral_step_y1[3][:data_size]
        )

        adj_I1 = adj_I0

        adj_theta = eqx.apply_updates(adj_theta, grad_step_y1[0])
        adj_theta = eqx.apply_updates(
            adj_theta,
            jax.tree_util.tree_map(lambda x: -(1 / self.l) * x, grad_step_z0[0]),
        )
        adj_theta = eqx.apply_updates(adj_theta, grad_integral_step_y1[0].vf)

        return t_and_state1, adj_y1, adj_z1, adj_I1, adj_theta

    vf = vjp_arg
    data_size = y1.shape[0]
    integral_vf = IntegralVectorField(vf, data_size)
    adj_y0 = grad_obj[0]
    adj_I0 = grad_obj[1]
    adj_z0 = jnp.zeros_like(t_and_state[2])
    adj_theta = eqx.filter(vf, eqx.is_inexact_array)
    adj_theta = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), adj_theta)
    t_and_state0 = t_and_state

    N = int(T / h)
    args = t_and_state0, adj_y0, adj_z0, adj_I0, adj_theta
    args = fori_loop(0, N, grad_step, args)
    t_and_state1, adj_y1, adj_z1, adj_I1, adj_theta = args

    return adj_theta
