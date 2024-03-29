from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext

ext_modules = [
    Extension("u",
              ["u.pyx"],
              libraries=["m"],
              extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=['-fopenmp']
              )
]

setup(
    name="u",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules
)

# python setup.py build_ext
# --inplace option creates .so file in the current directory
