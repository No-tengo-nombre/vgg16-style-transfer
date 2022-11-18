from vgg16autoencoder.logger import LOGGER
from vgg16st.transforms.whitening import Whitening
from vgg16st.transforms.coloring import Coloring
from vgg16st.functions import parameters_from_image


class WhiteningColoring:
    def __init__(self, alpha=1, method="paper", whitening_kwargs=None, coloring_kwargs=None) -> None:
        self.method = method
        self.alpha = alpha

        if whitening_kwargs is None:
            whitening_kwargs = {}

        if coloring_kwargs is None:
            coloring_kwargs = {}

        self.whitening = Whitening(method=method, **whitening_kwargs)
        self.coloring = Coloring(method=method, **coloring_kwargs)

    def __call__(self, content, style, alpha=None):
        # Determine the blending parameter
        LOGGER.info("Determining alpha.")
        if alpha is None:
            blending = self.alpha
        else:
            blending = alpha

        # Apply the wct
        _, style_mean, *_ = parameters_from_image(style)
        LOGGER.info("Applying whitening.")
        whitened_content = self.whitening(content)
        LOGGER.info("Applying coloring.")
        colored_content = self.coloring(whitened_content, style)

        # Readjust by the style means and blend
        LOGGER.info("Readjusting stylized image.")
        stylized_image = colored_content + style_mean.reshape(1, 1, -1).T

        LOGGER.info(f"Blending. Stylized shape: {stylized_image.shape}, content shape: {content.shape}.")
        blended_image = blending * stylized_image + (1 - blending) * content

        return blended_image
