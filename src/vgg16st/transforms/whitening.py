import torch

from vgg16st.exceptions import MethodException


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
    content_clone = content.clone()
    c = content_clone - torch.mean(content_clone)

    # Calculate eigenvalues and eigenvectors of covariance matrix
    cov_mat = (c @ c.T) / (c.shape[1] * c.shape[2] - 1)
    vals, vecs = torch.linalg.eig(cov_mat)
