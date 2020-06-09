mtxSolverPetsc
==============

Small code to read a linear system in Matrix Market format (mtx), write as a binary format and solve the linear system using a PETSc solver.

Building
--------
The following programs and libraries are required:
- [CMake](https://cmake.org/) version 3.0 or later
- [Eigen](http://eigen.tuxfamily.org) version 3.3.4 or later (an even more recent version is needed for the GCC 7 series) [`EIGEN3_ROOT`]
- [Boost](http://www.boost.org/) [`BOOST_ROOT`]
- [PETSc](http://www.mcs.anl.gov/petsc/) 3.8 or above and MPI are required to build the PETSc interface. [`PETSC_DIR`, `PETSC_ARCH`] Note that for the release build of BLASTed, PETSc must be configured with `--with-memalign=64` on x86\_64 platforms.

Assuming that you are in the top-level BLASTed directory, to configure a release version with the PETSc interface, type

    mkdir build && cd build
	cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_BUILD_TYPE=Release ..

where `mpicc` and `mpicxx` are the C and C++ MPI compiler wrappers you want to use. This will build the library for use with PETSc with available block sizes 4 and 5 (the default block sizes). See the beginning of the top-level CMakeLists.txt file for all the options. To build,

    make -j4

and to run the tests,

	make test

A C++ compiler with C++ 14 support is required. The build is known to work with GCC 7.3, 8.1, 8.2, 9.1, Intel 2017, 2018, and Clang 6.0, 7.0 in a GNU/Linux environment. To build in other enviroments, tweaking the CMakeLists.txt file will be required.

To build the [Doxygen](http://www.stack.nl/~dimitri/doxygen/) documentation,

    cd path/to/BLASTed/doc
    doxygen blasted_doxygen.cfg

This will build HTML documentation in a subdirectory called html in the current directory.

Finally, from the build directory, one can issue

    make tags
   
to generate a tags file for [easier navigation of the source code in Vim](http://vim.wikia.com/wiki/Browsing_programs_with_tags). If you are using [Spacemacs](http://spacemacs.org), this is not needed; just use the shortcut `SPC p G`.

Usage
-----
- Convert matrix and RHS vector to PETSc's binary format using the executable `util_petsc_io`.
- Solve the linear system using the executable `petscsingle` by specifying a PETSc options file.

---

[![Built with Spacemacs](https://cdn.rawgit.com/syl20bnr/spacemacs/442d025779da2f62fc86c2082703697714db6514/assets/spacemacs-badge.svg)](http://spacemacs.org)
