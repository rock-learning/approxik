# Evaluation

Here you can find the scripts that generate the results for the paper and
some additional plots.

## TODO

* add penalties for acceleration to enforce smootheness of trajectories
* write benchmark script for obstacle environment
* write benchmark script for ball-throwing

## Running the scripts

Make sure that you have pytransform (library for transformations) and BOLeRo
(our library for behavior learning) installed. Both are not published yet.
The following scripts are available:

* plot_viapoints.py - generates a plot that visualizes the viapoint environment
  (`viapoints.pdf`)
* benchmark_viapoints.py - generates benchmark results for the viapoint
  environment
* plot_obstacles.py - generates plots that visualize the obstacle environment
  (`obstacles1.pdf` and `obstacles2.pdf`)
* benchmark_obstacles.py - TODO
* benchmark_throwing.py - TODO

The following IPython notebooks can then be used to make a plot for the paper:

* evaluate_viapoint.ipynb - generate a plot for the viapoint environment
* evaluate_obstacle.ipynb - TODO
* evaluate_throwing.ipynb - TODO
