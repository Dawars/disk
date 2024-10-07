import argparse
import os
import signal
from pathlib import Path

from skimage.color.rgb_colors import khaki

# do this before importing numpy! (doing it right up here in case numpy is dependency of e.g. json)
os.environ["MKL_NUM_THREADS"] = "1"  # noqa: E402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # noqa: E402

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.callbacks import BatchSizeFinder

import json

from disk.data.datamodule import DataModuleTraining
from disk.model.disk_module import DiskModule
from opt import get_opts
from disk.utils.training_utils import create_exp_name

def train_model(args):
    pl.seed_everything(42)
    exp_name = create_exp_name(args.exp_name, args)
    print('Start training of ' + exp_name)
    model = DiskModule(args)

    jobId = os.getenv("SLURM_JOB_ID")
    taskId = os.getenv('SLURM_ARRAY_JOB_ID')
    job_id = int(taskId) if taskId else int(jobId)
    ckpt_dir: Path = args.save_dir / exp_name / str(job_id)
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='{epoch}-best_vcre',
        save_top_k=1,
        verbose=True,
        monitor='val_vcre/auc_vcre',
        mode='max'
    )
    lr_monitoring_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

    config_dict = vars(args)
    for key, value in config_dict.items():
        if isinstance(value, Path):  # Check if the value is a Path object
            config_dict[key] = str(value)  # Convert Path object to string

    (ckpt_dir / "config.json").write_text(json.dumps(config_dict, indent=4, ensure_ascii=False))

    logger = WandbLogger(project="disk", name=exp_name, id=str(job_id), resume="allow", config=config_dict)

    trainer = pl.Trainer(devices=args.num_gpus,
                         # log_every_n_steps=cfg.TRAINING.LOG_INTERVAL,
                         # val_check_interval=cfg.TRAINING.VAL_INTERVAL,
                         # limit_val_batches=cfg.TRAINING.VAL_BATCHES,
                         max_epochs=args.num_epochs,
                         logger=logger,
                         callbacks=[checkpoint_callback,
                                    # BatchSizeFinder()
                                    lr_monitoring_callback],
                         num_sanity_val_steps=0,
                         # gradient_clip_val=0.,
                         plugins=SLURMEnvironment(requeue_signal=signal.SIGHUP, auto_requeue=False),
                         enable_progress_bar=True,
                         )

    datamodule_end = DataModuleTraining(args, args.batch_size)

    if args.resume:
        ckpt_path = args.resume
    else:
        ckpt_path = None

    trainer.fit(model, datamodule_end, ckpt_path=ckpt_path)


if __name__ == '__main__':
    args = get_opts()
    train_model(args)