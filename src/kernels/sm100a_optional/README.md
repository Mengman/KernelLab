# sm_100a — B200 optional

This directory is reserved and is not part of any current target; B200 availability must not
affect the main build or tests.

Starting with B01, add a separate CMake option, CUDA target, and test label. Do not put `100a` in
the main fat binary or mistake the current scalar CUDA smoke test for B200/TMEM support.
