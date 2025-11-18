import numpy as np
import os
import time
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from config import Config
from data.dataset import MedicalDataset
from model.model import LungCancerClassifier
from model.trainer import Trainer
import random
import torch.nn as nn
import multiprocessing
import pandas as pd


def set_seed(config):
    """设置随机种子"""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_metrics(metrics, stage, epoch, save_dir):
    """将指标追加到同一个CSV文件"""
    file_path = os.path.join(save_dir, 'training_metrics.csv')

    # 构建结果字典（添加epoch和stage信息）
    results = {
        'epoch': epoch,
        'stage': stage,
        'loss': metrics.get('loss', np.nan),  # 添加loss记录
        'accuracy': metrics.get('acc', np.nan),
        'precision': metrics.get('precision', np.nan),
        'recall': metrics.get('recall', np.nan),
        'f1': metrics.get('f1', np.nan),
        'auc': metrics.get('auc', np.nan),
        'npv': metrics.get('npv', np.nan),
        'specificity': metrics.get('specificity', np.nan)
    }

    # 创建DataFrame并保存
    df = pd.DataFrame([results])

    # 如果文件不存在则写入表头，否则追加
    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)


def main():
    # 初始化配置
    config = Config()
    set_seed(config)

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 加载预处理后的 npy 文件
    data = np.load('data/' + config.data_path, allow_pickle=True)

    # 获取标签
    labels = np.array([sample['label'] for sample in data])
    subs = [sample['subject_id'] for sample in data]
    print(1111, len(subs))
    # structure = np.array([sample['structured'] for sample in data])
    print(np.sum(labels == 1), np.sum(labels == 0))
    # 数据分割
    train_idx, val_idx = train_test_split(
        np.arange(len(data)),
        test_size=0.1,
        stratify=labels,
        random_state=config.seed
    )
    train_data = data[train_idx]
    val_data = data[val_idx]

    # 添加过滤统计
    def count_nonzero_image(data):
        return sum(1 for sample in data if not np.all(sample['image_vec'] == 0))

    print(f"原始训练样本: {len(train_data)}，含图像样本: {count_nonzero_image(train_data)}")
    print(f"原始验证样本: {len(val_data)}，含图像样本: {count_nonzero_image(val_data)}")
    pos_r = float(np.sum(labels == 0) / np.sum(labels == 1))
    print(pos_r)
    # 创建数据集
    train_dataset = MedicalDataset(train_data)
    val_dataset = MedicalDataset(val_data)
    print(len(train_dataset), len(val_dataset))
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        drop_last=True
    )

    # 初始化模型
    model = LungCancerClassifier(config)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(config.device)

    # 初始化训练器
    trainer = Trainer(model, config, pos_r)

    best_val_auc = 0.0

    # 训练循环
    for epoch in range(config.num_epochs):
        start_time = time.time()

        # 训练阶段
        train_loss, train_metrics, train_results = trainer.train_epoch(train_loader)
        train_metrics['loss'] = train_loss
        save_metrics(train_metrics, 'train', epoch + 1, config.output_dir)

        # 验证阶段（每 10 个 epoch）
        if (epoch + 1) % 10 == 0:
            # trainer.clamp_paras()
            val_loss, val_metrics, val_results = trainer.evaluate(val_loader)
            val_metrics['loss'] = val_loss
            save_metrics(val_metrics, 'val', epoch + 1, config.output_dir)

            # 更新最佳模型
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                torch.save(model.state_dict(), os.path.join(config.output_dir, 'best_model.pth'))

                # 保存预测结果
            train_results.to_csv(config.output_dir + f"/train_predictions_epoch_{epoch}.csv", index=False)
            val_results.to_csv(config.output_dir + f"/val_predictions_epoch_{epoch}.csv", index=False)

        # 打印日志
        total_time = time.time() - start_time
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print(f"Throughput: {len(train_loader) / total_time:.2f} batches/s")
        print(f"[Train] Loss: {train_loss:.4f} | AUC: {train_metrics['auc']:.3f}, Accuracy: {train_metrics['acc']:.3f}")
        if (epoch + 1) % 10 == 0:
            print(f"[Val] AUC: {val_metrics['auc']:.3f}, Accuracy: {val_metrics['acc']:.3f}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
