# Auto-Air Traffic Control

## Dependencies

- Python 2
- SciPy 1.0.0
- configparser

## To train model, run:
`python2 train.py`

This will generate a file in the model folder with the filename corresponding to `grid_filename` in `params.cfg`
For example, `./model/grid6.pkl`

## To test model, run:
`python2 test.py`

This will load the trained model listed for `grid_filename` in `params.cfg` and display plots of generated trajectories. 

## To modify experiment parameters:
Change `params.cfg` and retrain model.
