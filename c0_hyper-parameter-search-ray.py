from functools import partial
import os
import torch
import torch.nn as nn

from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler


### fine-tune RNN on biosignals for AROUSAL & VALENCE. 

# optimise the GRU’s hidden representations’ size 
# the number of stacked GRU layers 
# the learning rate.
# consider both unidirectional and bidirectional GRUs.
# Window size and step length of the segmentation
# rnn_dropout
    
def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    data_dir = os.path.abspath("./data")
    load_data(data_dir)
    config = {
        "model_dim": tune.choice([32*2**i for i in range(5)]),
        "rnn_n_layers": tune.choice([2, 4, 8]),
        "rnn_bi": tune.choice([True, False]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "win_len": tune.choice([50*i for i in range(1, 5)]),
        "hop_len": tune.choice([25*i for i in range(1, 5)]),
        "rnn_dropout": tune.choice([0.1*i for i in range(10)]),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train_cifar, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = best_checkpoint.to_dict()

    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=100, max_num_epochs=100, gpus_per_trial=1)