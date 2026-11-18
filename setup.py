import shutil
import warnings
from setuptools import setup
from setuptools_rust import Binding, RustExtension

rust_extensions = []
if shutil.which("cargo") is not None:
    rust_extensions.append(
        RustExtension(
            "smart_splitter._core",
            path="Cargo.toml",
            binding=Binding.PyO3,
            optional=True,
        )
    )
else:
    warnings.warn(
        "Rust/Cargo toolchain not found; installing smart-image-splitter with vectorized OpenCV/NumPy backend.",
        stacklevel=1,
    )

setup(rust_extensions=rust_extensions)
