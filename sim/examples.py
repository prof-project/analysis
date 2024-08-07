import numpy as np
from utils import *
from sim import *

def scenario_arbitrage(param_type=random_type ,num_intervals=10, num_iters=10, debug=False, num_users=10, kickback_percentage=.9, min_param=0.01, max_param=4):
    """Run arbitrage simulation

    Args:
        param_type ('same' or 'rndm', optional): whether each user has a different random ('rndm') preference or the 'same' random preference. Defaults to 'same'.
        num_intervals (int, optional): number of params to try. Defaults to 10.
        num_iters (int, optional): number of iterations of each parameter. Defaults to 10.
        debug (bool, optional): print extra debug information. Defaults to False.
        num_users (int, optional): number of users of PROF/MEVShare. Defaults to 2.
        kickback_percentage (float, optional): percentage of profit returned to MEVShare users. Defaults to .9.
        min_param (float, optional): minimum parameter. Defaults to 0.01.
        max_param (float, optional): maximum parameter. Defaults to 5.

    Returns:
        None
    """
    np.random.seed(0)

    params = default_params[param_type](4)
    print("params", params)
    print("param_type", param_type)

    experiment = Experiment(param_type=param_type, num_iters=1000, users_range=[10,20], params=params, debug=debug)

    experiment.run_all(all_stats=True, save_state=True)

    print("params", params)
    experiment.print_stats(stat_names=[random_walk_stat])
    experiment.plot_stats_line(stat_names=[random_walk_stat], by="param", merge=False)

def scenario_paper():
    np.random.seed(0)

    experiment = Experiment(param_type=max_demand_type, params=[0.25, .5, .75, 1, 2, 4, 8, 16], users_range=list(range(20,101)), num_iters=1000, file_dir="paper_data")

    experiment.run_all(all_stats=True, save_state=True)
    experiment.print_stats(stat_names=[user_util_stat])
    experiment.plot_stats_line(stat_names=[user_util_stat], by="param", merge=False)    

if __name__ == "__main__":
    scenario_paper()