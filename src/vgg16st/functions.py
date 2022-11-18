import torch


def parameters_from_image(image):
    # Center the image
    channels, height, width = image.shape
    content_clone = image.clone()

    # Calculate the mean for every channel
    mean_val = torch.mean(content_clone, (1, 2))
    c = content_clone - mean_val.reshape(-1, 1, 1)

    # Calculate eigenvalues and eigenvectors of covariance matrix
    cov_mat = (c.reshape(channels, -1) @ c.reshape(channels, -1).T) / (height * width - 1)
    vals, vecs = torch.linalg.eig(cov_mat)
    vals = vals.real
    vecs = vecs.real

    return c, mean_val, vals, vecs
