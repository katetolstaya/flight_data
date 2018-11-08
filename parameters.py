import numpy as np

class Parameters:
    def __init__(self):



        # data set params
        self.fname = 'flights20160113'
        self.fname = 'flights20160111'
        self.fname = 'flights20160112'
        self.airport = "KSEA"
        self.center_t = 32000.0 #25000.0 + 4000.0 #1452748000.0 + 9000.0  # + 300.0
        self.range_t = 100000 #500.0 #200.0 #500.0

        self.alt_lim =  2500.0
        self.dist_lim = 50000.0 # 100000.0

        self.min_time = 0
        self.start_t = 0
        self.end_t = 0

        self.time_delta = 1000

        self.time_limit = 2.0

        self.scale = 1000.0
        # planner params
        self.turning_radius = 3000.0 / self.scale  #3000.0 #50.0 #100.0
        self.rise_rate = 1000.0 / self.scale #50.0
        self.velocity = 20.0 / self.scale  #20.0

        ## switch rates for interpolation/planner
        self.turning_radius_interp = 100.0 / self.scale  #3000.0 #50.0 #100.0
        self.rise_rate_interp = 1000.0 / self.scale #50.0

        self.turning_radius_learn = 20000.0 / self.scale  #3000.0 #50.0 #100.0
        self.rise_rate_learn = 1000.0 / self.scale #50.0

        self.w1 = 1.0
        self.w2 = 1.0
        self.w3 = 1.0

        # self.w1 = 0.0
        # self.w2 = 1000.0
        # self.w3 = 0.0

        if self.airport == "KSEA":
            self.lat0 = 47.4435903
            self.lon0 = -122.2960726
            self.alt0 = 109.37  # meters
        else:
            raise NotImplementedError

    def switch_params(self, mode="learn"):
        if mode == "learn":
            self.turning_radius = self.turning_radius_learn / self.scale
            self.rise_rate = self.turning_radius_learn / self.scale
            self.time_limit = 5.0
        elif mode == "interp":
            self.turning_radius = self.turning_radius_interp / self.scale
            self.rise_rate = self.turning_radius_interp / self.scale
            self.time_limit = 0.01



