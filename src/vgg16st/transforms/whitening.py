import torch

from vgg16st.exceptions import MethodException


EPSILON = 1e-30


class Whitening:
    def __init__(self, method="paper") -> None:
        self.method = method
        try:
            self.__function = globals()[f"__whitening_{method}"]
        except KeyError as e:
            raise MethodException(f"Whitening method {method} could not be found.") from e

    def __call__(self, *args, **kwargs):
        return self.__function(*args, **kwargs)


def __whitening_paper(content):
    # Center the image
    channels, height, width = content.shape
    content_clone = content.clone()
    c = content_clone - torch.mean(content_clone, (1, 2)).reshape(1, 1, -1).T

    # Calculate eigenvalues and eigenvectors of covariance matrix
    cov_mat = (c.reshape(channels, -1) @ c.reshape(channels, -1).T) / (height * width - 1)
    vals, vecs = torch.linalg.eig(cov_mat)
    vals = vals.real
    vecs = vecs.real

    # We remove negative values and zeros
    reduced_dimension = (vals > EPSILON).sum()
    vals = vals[:reduced_dimension]
    vecs = vecs[:, :reduced_dimension]

    # Apply the whitening transformation
    whitened = vecs @ torch.diag(vals) @ vecs.T @ c
    return whitened
