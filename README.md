# Auto-Air Traffic Control

## Report
View the IROS 2019 submission [here](https://github.com/katetolstaya/flight_data/blob/master/iros2019.pdf) and the video on [Youtube](https://youtu.be/5HasgHNl-XY).


## Dependencies

- Python 2.7
- SciPy 1.1.0
- configparser

## Train routing model
To train model, run:
`python2 train_routing.py`

This will generate a file in the model folder with the filename corresponding to `grid_filename` in `params.cfg`.
A trained model is provided:  `./models/grid19.pkl`



## Test routing model
To test model, run:
`python2 test_single.py`

This will load the trained model listed for `grid_filename` in `params.cfg` and display a plot of a generated trajectory. 
If you close this window, the program will go on to another test case and show another graph. 

The expert trajectory is denoted in green, while the learner's sequence of motion primitives is denoted with red arrows, and a blue spline interpolates these points. 

## Train spacing model
To train model, run:
`python2 train_spacing.py`

## Test spacing model
To train model, run:
`python2 test_multiple.py`


## Change parameters
To modify experiment parameters, change `cfg/params.cfg` and retrain the model.

## Future directions
1) Demo in a simulator
2) Enforce trajectory start time and end time - should this change the heuristic?
3) Take-offs 
4) Prune high cost trajectories (as if high cost regions are fake obstacles)
5) Refine splines by adding control points sampled using a Variational Autoencoder ( won't generalize to landing vs takeoff)
