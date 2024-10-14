from abc import abstractmethod

import equinox as eqx
import jax
from jaxtyping import Array, Float

from reversible_solvers.vector_field import AbstractVectorField

State = Float[Array, " d"]


class AbstractSolverStep(eqx.Module):
    """Abstract base class for all solver steps.

    Implements the step for a given solver, where y_{n+1} = y_n + step.

    This step is required for reversible solvers.
    """

    @abstractmethod
    def step(
        self,
        vf: AbstractVectorField,
        h: float,
        t: Float[Array, " 1"],
        y: State,
    ) -> State:
        """
        One solver step.

        **Arguments:**
        - vf: vector field
        - h: step size
        - t: time
        - y: state

        **Returns:**
        - step: a step of the solver, where y_{n+1} = y_n + step
        """

        pass


class Euler(AbstractSolverStep):
    """
    Euler's method. (Compatible with PyTree state)

    Calculates step, where y_{n+1} = y_n + step.
    """

    def step(
        self,
        vf: AbstractVectorField,
        h: float,
        t: Float[Array, "1"],
        y: State,
    ) -> State:
        return jax.tree_util.tree_map(lambda x: h * x, vf(t, y))


class Midpoint(AbstractSolverStep):
    """
    Midpoint method. (Compatible with PyTree state)

    Calculates step, where y_{n+1} = y_n + step.
    """

    def step(
        self,
        vf: AbstractVectorField,
        h: float,
        t: Float[Array, "1"],
        y: State,
    ) -> State:
        return jax.tree_util.tree_map(
            lambda x: h * x,
            vf(
                t + (h / 2),
                eqx.apply_updates(
                    y, jax.tree_util.tree_map(lambda x: (h / 2) * x, vf(t, y))
                ),
            ),
        )


class RK4(AbstractSolverStep):
    """
    Runge-Kutta 4. (Compatible with PyTree state)

    Calculates step, where y_{n+1} = y_n + step.
    """

    def step(
        self,
        vf: AbstractVectorField,
        h: float,
        t: Float[Array, "1"],
        y: State,
    ) -> State:
        k1 = vf(t, y)
        k2 = vf(
            t + (h / 2),
            eqx.apply_updates(y, jax.tree_util.tree_map(lambda x: (h / 2) * x, k1)),
        )
        k3 = vf(
            t + (h / 2),
            eqx.apply_updates(y, jax.tree_util.tree_map(lambda x: (h / 2) * x, k2)),
        )
        k4 = vf(
            t + h, eqx.apply_updates(y, jax.tree_util.tree_map(lambda x: h * x, k3))
        )
        return jax.tree_util.tree_map(
            lambda x: h * x,
            eqx.apply_updates(
                eqx.apply_updates(
                    eqx.apply_updates(
                        jax.tree_util.tree_map(lambda x: x / 6, k1),
                        jax.tree_util.tree_map(lambda x: x / 3, k2),
                    ),
                    jax.tree_util.tree_map(lambda x: x / 3, k3),
                ),
                jax.tree_util.tree_map(lambda x: x / 6, k4),
            ),
        )


class Dopri5(AbstractSolverStep):
    """
    Dormand-Prince 5/4 method. (Not compatible with PyTree state)

    Calculates step, where y_{n+1} = y_n + step.
    """

    def step(
        self,
        vf: AbstractVectorField,
        h: float,
        t: Float[Array, "1"],
        y: State,
    ) -> State:
        k1 = vf(t, y)
        k2 = vf(t + (h / 5), y + k1 * (h / 5))
        k3 = vf(t + (3 * h / 10), y + (k1 + 3 * k2) / 4 * (3 * h / 10))
        k4 = vf(t + (4 * h / 5), y + (11 * k1 - 42 * k2 + 40 * k3) / 9 * (4 * h / 5))
        k5 = vf(
            t + (8 * h / 9),
            y + (4843 * k1 - 19020 * k2 + 16112 * k3 - 477 * k4) / 1458 * (8 * h / 9),
        )
        k6 = vf(
            t + h,
            y
            + (477901 * k1 - 1806240 * k2 + 1495424 * k3 + 46746 * k4 - 45927 * k5)
            / 167904
            * h,
        )
        k7 = vf(
            t + h,
            y
            + (12985 * k1 + 64000 * k3 + 92750 * k4 - 45927 * k5 + 18656 * k6)
            / 142464
            * h,
        )
        return (
            h
            * (
                1921409 * k1
                + 9690880 * k3
                + 13122270 * k4
                - 5802111 * k5
                + 1902912 * k6
                + 534240 * k7
            )
            / 21369600
        )
