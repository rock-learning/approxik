import os
from distutils.core import setup, Extension
from distutils.sysconfig import get_config_vars
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np
import config


VERBOSE = 0


def path_list(paths):
    while ";;" in paths:
        paths = paths.replace(";;", ";")
    return paths.split(";")


def lib_names(libraries):
    names = []
    for lib in libraries:
        filename = lib.split(os.sep)[-1]
        libname = filename.replace("lib", "").replace(".so", "")
        names.append(libname)
    return names


def strict_prototypes_workaround():
    # Workaround to remove '-Wstrict-prototypes' from compiler invocation
    (opt,) = get_config_vars('OPT')
    os.environ['OPT'] = " ".join(
        flag for flag in opt.split() if flag != '-Wstrict-prototypes'
    )

if __name__ == "__main__":
    strict_prototypes_workaround()

    include_dirs = path_list(config.include_dirs)
    include_dirs.extend(["..", "../approxik", np.get_include()])
    include_dirs = list(set(include_dirs))
    # TODO .replace("lib/x86_64-linux-gnu", "lib")?
    library_dirs = path_list(config.library_dirs)
    library_dirs = list(set(library_dirs))
    extra_compile_args = config.extra_compile_args.strip().split(" ")
    extra_compile_args = list(set(extra_compile_args))

    if VERBOSE >= 1:
        print("=== Library directories:")
        print(library_dirs)
        print("=== Libraries:")
        print(libraries)
        print("=== Extra compile args:")
        print(extra_compile_args)
    libraries = lib_names(path_list(config.libraries))

    extension = Extension(
        name="approxik",
        sources=["approxik.pyx"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        define_macros=[("NDEBUG", '1')],
        language="c++",
        extra_compile_args=extra_compile_args +
            ["-Wno-cpp", "-Wno-unused-function"])

    setup(ext_modules=cythonize(extension))
