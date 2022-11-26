from setuptools import setup
from setuptools_rust import Binding, RustExtension


if __name__ == "__main__":
    setup(
        rust_extensions=[
            RustExtension("vgg16st.vgg16st", binding=Binding.PyO3),
        ]
    )