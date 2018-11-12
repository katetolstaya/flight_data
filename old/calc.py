import numpy as np

from parameters import Parameters

total_costs = [2.99704018e+02, 9.70430723e+05, 5.45264286e+04, 9.70976287e+06]
total_actual_costs = [8.21606242e+01, 1.24070225e+06, 4.41053982e+04, 1.24114338e+07]



params = Parameters()

print [x / actual_costs[3] for x in actual_costs]
print [x / my_costs[3] for x in my_costs]
