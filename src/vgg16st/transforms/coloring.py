import torch

from vgg16common import EPSILON, LOGGER
from vgg16st.exceptions import MethodException
from vgg16st.functions import parameters_from_image


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
    if parameter_content is None:
        c, _, vals, vecs = parameters_from_image(content)
    else:
        c, _, vals, vecs = parameters_from_image(parameter_content)

    # Resize the matrix
    c_shape = c.shape
    c_mat = c.reshape(c_shape[0], -1)

    # We remove negative values and zeros
    reduced_dimension = (vals > EPSILON).sum()
    vals = vals[:reduced_dimension]
    vecs = vecs[:, :reduced_dimension]
    vals_mat = torch.diag(vals).pow(-0.5)

    # Apply the coloring transformation
    LOGGER.info(f"Coloring shapes {vecs.shape}, {vals_mat.shape}, {c_mat.shape}")
    colored = vecs @ vals_mat @ vecs.T @ c_mat
    return colored.reshape(*c_shape)
