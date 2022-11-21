import torch
from torchvision.transforms import Normalize


def center_tensor(image):
    # Calculate the mean for every channel
    mean_val = torch.mean(image, (1, 2))
    return image - mean_val.reshape(-1, 1, 1), mean_val


def cov_eigvals(image):
    channels, height, width = image.shape

    # Calculate eigenvalues and eigenvectors of covariance matrix
    cov_mat = (image.reshape(channels, -1) @ image.reshape(channels, -1).T) / (height * width - 1)
    vals, vecs = torch.linalg.eig(cov_mat)

    return vals.real, vecs.real


def minmax_normalize(image):
    min_vals = image.min(axis=1).values.min(axis=1).values
    max_vals = image.max(axis=1).values.max(axis=1).values
    normalization = Normalize(min_vals, max_vals - min_vals)
    return normalization(image)
