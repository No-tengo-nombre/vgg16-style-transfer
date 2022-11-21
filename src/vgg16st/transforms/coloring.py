import torch

from vgg16common import EPSILON, LOGGER
from vgg16st.exceptions import MethodException
from vgg16st.utils import center_tensor, cov_eigvals


class Coloring:
    def __init__(self, method="paper") -> None:
        self.method = method
        try:
            self.__function = globals()[f"__coloring_{method}"]
        except KeyError as e:
            raise MethodException(f"Coloring method {method} could not be found.") from e

    def __call__(self, *args, **kwargs):
        return self.__function(*args, **kwargs)


def __coloring_paper(content, parameter_content=None):
    LOGGER.info("Calculating coloring with paper method.")
    c, _ = center_tensor(content)

    if parameter_content is None:
        vals, vecs = cov_eigvals(content)
    else:
        vals, vecs = cov_eigvals(parameter_content)

    # Resize the matrix
    c_shape = c.shape
    c = c.reshape(c_shape[0], -1)

    # We remove negative values and zeros
    reduced_dimension = (vals > EPSILON).sum()
    LOGGER.info(f"Coloring reduced dimension {reduced_dimension}")
    vals = vals[:reduced_dimension]
    vecs = vecs[:, :reduced_dimension]
    vals_mat = torch.diag(vals.pow(0.5))

    # Apply the coloring transformation
    LOGGER.info(f"Coloring shapes {vecs.shape}, {vals_mat.shape}, {c.shape}")
    colored = vecs @ vals_mat @ vecs.T @ c
    return colored.reshape(*c_shape)
