from dask.distributed import Client, LocalCluster, performance_report, SSHCluster
import os
from pls.workflows.execute_workflow import train, test, evaluate
import itertools



def run_train():
    for exp in exps:
        path = os.path.join(exp, "config.json")
        train(path)

def run_test():
    for exp in exps:
        path = os.path.join(exp, "config.json")
        test(path)

def main_cluster():
    """
    Running experiments parallelly using dask
    """
    client = Client("134.58.41.100:8786")
    ## some dask computation
    futures = client.map(train, exps)
    results = client.gather(futures)

def main_cluster_test():
    """
    Running experiments parallelly using dask
    """
    client = Client("134.58.41.100:8786")
    ## some dask computation
    futures = client.map(test, exps)
    results = client.gather(futures)

def run_evaluate():
    folder = exps[0]
    # model_at_step = 100000
    model_at_step = "end"
    mean_reward, n_deaths = evaluate(folder, model_at_step=model_at_step, n_test_episodes=100)
    print(mean_reward, n_deaths)

if __name__ == "__main__":
    hyper_parameters = {
        "exp_folders": [
            "experiments",
        ],
        "domains": [
            # "stars/small0",
            # "pacman/small3"
            "carracing/map0"
        ],
        "exps": [
            "PLPGperf",
            # "epsVSRLthres0.1",
            # "PLPG_LTnoisy",
            # "PLPG_STnoisy",
            # "PLPGnoisy",
            # "VSRLperf",
            # "VSRLthres"
            ],
        "seeds":
            [
            "seed1"
            ]
    }
    cwd = os.path.join(os.path.dirname(__file__))
    lengths = list(map(len, list(hyper_parameters.values())))
    lists_of_indices = list(map(lambda l: list(range(l)), lengths))
    combinations = list(itertools.product(*lists_of_indices))

    exps = []
    for combination in combinations:
        hyper = dict.fromkeys(hyper_parameters.keys())
        hyper["exp_folders"] = hyper_parameters["exp_folders"][combination[0]]
        hyper["domains"] = hyper_parameters["domains"][combination[1]]
        hyper["exps"] = hyper_parameters["exps"][combination[2]]
        hyper["seeds"] = hyper_parameters["seeds"][combination[3]]
        folder = os.path.join(cwd,
                              hyper["exp_folders"],
                              hyper["domains"],
                              hyper["exps"],
                              hyper["seeds"],
                              )
        exps.append(folder)
    # run_train()
    run_test()