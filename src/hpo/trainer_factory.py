from src.trainers.RevINMixer import Scenario1Trainer, Scenario2Trainer

def make_trainer(scenario, params, config, seed=42, trial=None):
    TrainerClass = {
        1: Scenario1Trainer,
        2: Scenario2Trainer
    }[scenario]

    return TrainerClass(
        seq_length=params["seq_length"],
        ff_dim=params["ff_dim"],
        dropout=params["dropout"],
        pred_len=config["pred_len"],
        n_block=params["n_block"],
        batch_size=params["batch_size"],
        lr=params["lr"],
        epochs=config["epochs"],
        patience=config["patience"],
        holding_cost=config["holding_cost"],
        lead_time=config["lead_time"],
        ordering_cost=config["ordering_cost"],
        seed=int(seed),
        trial=trial
    )