from sweep_base import LAL_Sweep
import wandb
from wandb_reporter import wandb_report

def main():
    wandb.init()

    cfg = wandb.config
    lal_model = LAL_Sweep(cfg)
    lal_model.run()
    f1, eval_arr, y_test, predict_arr, classes = lal_model.eval()

    wandb.log({"eval/f1": f1})

    wandb_report(eval_arr, y_test, predict_arr, classes)

if __name__ == "__main__":
    main()