import torch

from vgg16st.exceptions import MethodException
from vgg16autoencoder import EPSILON


class Coloring:
    def __init__(self, method="paper") -> None:
        self.method = method
        try:
            self.__function = globals()[f"__coloring_{method}"]
        except KeyError as e:
            raise MethodException(f"Coloring method {method} could not be found.") from e

    def __call__(self, *args, **kwargs):
        return self.__function(*args, **kwargs)


def __coloring_paper(content):
    # Center the image
    channels, height, width = content.shape
    content_clone = content.clone()
    mean_val = torch.mean(content_clone, (1, 2))
    c = content_clone - mean_val.reshape(1, 1, -1).T

    # Calculate eigenvalues and eigenvectors of covariance matrix
    cov_mat = (c.reshape(channels, -1) @ c.reshape(channels, -1).T) / (height * width - 1)
    vals, vecs = torch.linalg.eig(cov_mat)
    vals = vals.real
    vecs = vecs.real

    # We remove negative values and zeros
    reduced_dimension = (vals > EPSILON).sum()
    vals = vals[:reduced_dimension]
    vecs = vecs[:, :reduced_dimension]

    # Apply the coloring transformation
    colored = vecs @ torch.diag(vals).pow(0.5) @ vecs.T @ c
    return colored, mean_val
