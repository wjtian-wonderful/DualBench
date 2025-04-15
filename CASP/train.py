from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data.text_image_dm import TextImageDataModule
from models.wrapper import CASPWrapper, CASP
import sys
import os
import torch
from tools.logger import TextLogger
sys.path.append('models/beats')
torch.set_float32_matmul_precision('high')

def main():

    # 创建 TensorBoardLogger 实例
    tensorboard_logger = TensorBoardLogger(save_dir="logs", name="CASP-5s-finetune-use_v2ctrain")
    # 创建自定义 TextLogger 实例
    log_dir = tensorboard_logger.log_dir

    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    text_logger = TextLogger(save_dir=log_dir, name="CASP_txt")

    # 配置 ModelCheckpoint 回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,     # 设置保存检查点的目录
        filename='{epoch}-{step}',  # 检查点文件的命名格式
        save_top_k=-1,              # 保留所有检查点
        every_n_train_steps=1000,    # 每500个训练步骤保存一次
        save_weights_only=False      # 只保存模型权重
    )
    # 初始化超参数
    hparams = {
        "precision": 32,
        "max_epochs": 32,
        "strategy": "ddp_find_unused_parameters_true",
        "d_model": 768,
        "learning_rate": 2e-4,
        "batch_size": 64,
        "num_workers": 2,
        "shuffle": True,
        "log_step": 5,
        "checkpoint_path": '/root/logs/CASP/version_31/checkpoints/epoch=25-step=100000.ckpt'

        # "checkpoint_path": None
    }
    checkpoint_path = hparams["checkpoint_path"]


    dm = TextImageDataModule(
                            batch_size=hparams["batch_size"],
                            num_workers=hparams["num_workers"],
                            shuffle = hparams["shuffle"]
    )
    
    train_dataloader = dm.train_dataloader()
    train_dataloader_len = len(train_dataloader)
    

    trainer = Trainer(
        precision=hparams["precision"],
        max_epochs=hparams["max_epochs"],
        strategy=hparams["strategy"],
        callbacks=[checkpoint_callback],  # 添加回调到 Trainer 配置中
        logger=[tensorboard_logger],  # 添加多个日志记录器到 Trainer 配置中
        )
        
    model = CASPWrapper(d_model=hparams["d_model"], 
                        max_train_steps=train_dataloader_len * trainer.max_epochs,
                        learning_rate=hparams["learning_rate"],
                        log_step=hparams["log_step"],
                        )

    # 记录超参数
    text_logger.log_hyperparams(hparams)
    trainer.fit(model, dm.train_dataloader(), ckpt_path=checkpoint_path)

if __name__ == '__main__':
    main()
