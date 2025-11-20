import splitfolders
import hydra
from omegaconf import DictConfig
import os


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def split_data(cfg: DictConfig):
    input_folder = cfg.paths.raw_data_dir
    output_folder = cfg.paths.processed_data_dir
    seed = cfg.seed

    if os.path.exists(output_folder):
        print(f"Output directory {output_folder} already exists, skipping data splitting.")
        return

    print(f"Splitting data from {input_folder} to {output_folder}, seed: {seed}")

    splitfolders.ratio(
        input_folder,
        output=output_folder,
        seed=seed,
        ratio=(0.8, 0.1, 0.1),
        group_prefix=None,
        move=False
    )
    print("Data splitting completed.")


if __name__ == "__main__":
    split_data()