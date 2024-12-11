from sweep_base import Uncertainty_Sweep
import wandb

def main():
    print(wandb.__file__)

    wandb.init()

    cfg = wandb.config

    unc_model = Uncertainty_Sweep(cfg)
    unc_model.run(10000)
    unc_model.eval()

if __name__ == "__main__":
    main()