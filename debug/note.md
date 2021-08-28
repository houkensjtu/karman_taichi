This folder contains previous versions of karman_taichi.

* 20210828
- Tested with 64x320 grid, cpu operation only; GPU operation will lead to error
- Used bicgstab for both momentum and p correction solving
- Calculation speed can be further improved by implementing multigrid
- Multiple linear solvers reside in the file, needs to be cleaned
