import random
import numpy as np
import datetime
from MTMFEA.run import run_mtmfea

if __name__ == "__main__":
    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)

    start = datetime.datetime.now()
    run_mtmfea(
        pop_size = 10,
        subpop_num = 3,
        structure_shape = (5, 5),
        experiment_name = "Thrower-v0-MTMFEA",
        max_evaluations = 200,
        train_iters = 300,
        num_cores = 10,
        env_name = 'Thrower-v0',
    )
    end = datetime.datetime.now()

    print(start)
    print(end)
