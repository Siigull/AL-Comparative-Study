from sweep_base import Random_Sweep
import wandb

def main():
    wandb.init()

    cfg = wandb.config
    unc_model = Random_Sweep(cfg)
    unc_model.run()
    unc_model.eval()

if __name__ == "__main__":
    main()