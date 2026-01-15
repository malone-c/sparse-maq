import os
import sys
from pathlib import Path
from setuptools import setup, find_packages
from distutils.core import Extension
from Cython.Build import cythonize
import numpy as np
import pyarrow as pa

arrow_lib_dir = pa.get_library_dirs()[0]

if "darwin" in sys.platform:
    COMPILE_ARGS = ["-std=c++20", "-O2", "-pthread"]
    LINK_ARGS = ["-stdlib=libc++"]
elif "linux" in sys.platform:
    COMPILE_ARGS = ["-std=c++20", "-lstdc++", "-Wall", "-O2", "-pthread"]
    LINK_ARGS = ["-lstdc++", "-pthread"]
elif "win32" in sys.platform:
    COMPILE_ARGS = ["-Wall", "-O2"]
    LINK_ARGS = []
else:
    raise ImportError("Unknown OS.")

setup_dir = Path(os.path.abspath(os.path.dirname(__file__)))
core_dir = setup_dir / 'core' / 'src'

ext = Extension(
    "sparse_maq.ext",
    language="c++",
    sources=["sparse_maq" + os.path.sep + "mckpbindings.pyx"],
    extra_compile_args=COMPILE_ARGS,
    extra_link_args=LINK_ARGS,
    include_dirs=[str(core_dir), pa.get_include(), np.get_include()],
    library_dirs=[arrow_lib_dir],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    packages=find_packages(include=["sparse_maq"]),
    ext_modules=cythonize(ext, compiler_directives={"language_level": 3}),
)
