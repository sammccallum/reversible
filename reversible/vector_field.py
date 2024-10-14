from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Float

State = Float[Array, " d"]


class AbstractVectorField(eqx.Module):
    """Abstract base class for all vector fields.

    Implements vf = f(t, y).
    """

    @abstractmethod
    def __call__(
        self,
        t: Float[Array, " 1"],
        y: State,
    ) -> State:
        """
        Evaluate vector field.

        **Arguments:**
        - t: time
        - y: state

        **Returns:**
        - f(t, y): vector field at (t, y)
        """

        pass
