import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.loggers import WandbLogger # 如果需要 WandB

from src.data.datamodule import ResearchDataModule
from src.models.lightning_module import ResearchSystem


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    dm = ResearchDataModule(
        data_dir=cfg.paths.processed_data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers
    )

    model = ResearchSystem(
        lr=cfg.training.lr,
        num_classes=cfg.model.num_classes
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.paths.checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}"
    )

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_cb, early_stop_cb],
        # logger=WandbLogger(...) if cfg.logger.enabled else None
    )

    trainer.fit(model, datamodule=dm)

    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()