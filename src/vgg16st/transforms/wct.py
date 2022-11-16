from vgg16st.transforms.whitening import Whitening
from vgg16st.transforms.coloring import Coloring


class WhiteningColoring:
    def __init__(self, method="paper", whitening_kwargs=None, coloring_kwargs=None) -> None:
        self.method = method

        if whitening_kwargs is None:
            whitening_kwargs = {}

        if coloring_kwargs is None:
            coloring_kwargs = {}

        self.whitening = Whitening(method=method, **whitening_kwargs)
        self.coloring = Coloring(method=method, **coloring_kwargs)

    def __call__(self, content, style):
        whitened_content, _ = self.whitening(content)
        colored_content, _ = self.coloring(whitened_content)

        whitened_style, _ = self.whitening(style)
        colored_style, _ = self.coloring(whitened_style)
