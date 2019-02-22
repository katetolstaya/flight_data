# Auto-Air Traffic Control

## Report
View the current version of the write-up [here](https://www.overleaf.com/read/fkygphjtxkwf).

## Dependencies

- Python 2.7
- SciPy 1.1.0
- configparser

## Train model
To train model, run:
`python2 train.py`

This will generate a file in the model folder with the filename corresponding to `grid_filename` in `params.cfg`.
A trained model is provided:  `./model/grid8.pkl`

## Test model
To test model, run:
`python2 test.py`

This will load the trained model listed for `grid_filename` in `params.cfg` and display a plot of a generated trajectory. 
If you close this window, the program will go on to another test case and show another graph. 

The generated figures should look like this:
![alt text](https://github.com/katetolstaya/flight_data/blob/master/traj.png "Expert and learned trajectory")

The expert trajectory is denoted in green, while the learner's sequence of motion primitives is denoted with red arrows, and a blue spline interpolates these points. 

## Change parameters
To modify experiment parameters, change `params.cfg` and retrain the model.

## To do for paper:
0) Arrows instead of circles for animated plot??
1) Record video of 4 planes landing and get screenshots
2) Plot of IRL cost while learning cost grid
3) Plot of IRL cost while learning spacing
4) Demo with slower or faster airplanes

## To do for code:
1) Move text parsing code into its own folder
2) Move plotting utils
3) Organize compressed dataset

## Future directions
1) Demo in a simulator
2) Enforce trajectory start time and end time - should this change the heuristic?
3) Take-offs 
4) Prune high cost trajectories (as if high cost regions are fake obstacles)
5) Refine splines by adding control points sampled using a Variational Autoencoder ( won't generalize to landing vs takeoff)
