from sweep_base import Random_Sweep
import wandb
from wandb_reporter import wandb_report

def main():
    wandb.init()

    cfg = wandb.config
    unc_model = Random_Sweep(cfg)
    unc_model.run()
    f1, eval_arr, y_test, predict_arr, classes = unc_model.eval()

    wandb.log({"eval/f1": f1})

    wandb_report(eval_arr, y_test, predict_arr, classes)

if __name__ == "__main__":
    main()