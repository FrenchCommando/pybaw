import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# baw_core cimports scipy.optimize.cython_optimize (Cython brentq), so scipy must be
# present at build time, not just at runtime.
extensions = [
    Extension(
        "pybaw.baw_core",
        ["pybaw/baw_core.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}))
