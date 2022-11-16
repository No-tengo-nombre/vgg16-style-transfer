import torch

from vgg16st.exceptions import MethodException
from vgg16st.functions import parameters_from_image
from vgg16autoencoder import EPSILON


class Whitening:
    def __init__(self, method="paper") -> None:
        self.method = method
        try:
            self.__function = globals()[f"__whitening_{method}"]
        except KeyError as e:
            raise MethodException(f"Whitening method {method} could not be found.") from e

    def __call__(self, *args, **kwargs):
        return self.__function(*args, **kwargs)


def __whitening_paper(content, parameter_content=None):
    if parameter_content is None:
        c, _, vals, vecs = parameters_from_image(content)
    else:
        c, _, vals, vecs = parameters_from_image(parameter_content)

    # We remove negative values and zeros
    reduced_dimension = (vals > EPSILON).sum()
    vals = vals[:reduced_dimension]
    vecs = vecs[:, :reduced_dimension]

    # Apply the whitening transformation
    whitened = vecs @ torch.diag(vals).pow(-0.5) @ vecs.T @ c
    return whitened
