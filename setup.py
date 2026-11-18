from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    rust_extensions=[
        RustExtension(
            "smart_splitter._core",
            path="Cargo.toml",
            binding=Binding.PyO3,
            optional=False,
        )
    ]
)
