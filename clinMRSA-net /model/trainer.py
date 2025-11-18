# model/trainer.py
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score, confusion_matrix
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import pandas as pd


class Trainer:
    def __init__(self, model, config, pos_r=1):
        self.model = model
        self.config = config
        pos_weight = torch.tensor([1.0, 2], device=config.device)
        self.criterion = nn.CrossEntropyLoss(weight=pos_weight)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    @staticmethod
    def _calculate_ci(metric_func, y_true, y_pred, y_probs=None, n_bootstraps=1000):
        """计算指标的95%置信区间"""
        stats = []
        rng = np.random.RandomState(42)

        for _ in range(n_bootstraps):
            indices = rng.choice(len(y_true), len(y_true), replace=True)

            if y_probs is not None:  # AUC需要概率
                score = metric_func(
                    y_true[indices],
                    y_probs[indices][:, 1] if y_probs.shape[1] == 2 else y_probs[indices]
                )
            else:
                score = metric_func(
                    y_true[indices],
                    y_pred[indices]
                )
            stats.append(score)

        return np.percentile(stats, [2.5, 97.5])

    def _calculate_metrics(self, y_true, y_pred, y_probs):
        """统一计算所有指标及其置信区间"""
        metrics = {}
        # print(111,y_probs[:,1])
        # 基础指标
        metrics['acc'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred)
        metrics['auc'] = roc_auc_score(y_true, y_probs[:, 1] if y_probs.shape[1] == 2 else y_probs)

        # 置信区间计算
        # metrics['precision_ci'] = self._calculate_ci(precision_score, y_true, y_pred)
        # metrics['recall_ci'] = self._calculate_ci(recall_score, y_true, y_pred)
        # metrics['f1_ci'] = self._calculate_ci(f1_score, y_true, y_pred)
        # metrics['auc_ci'] = self._calculate_ci(roc_auc_score, y_true, None, y_probs)

        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # 计算NPV和特异度
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics['npv'] = npv
        metrics['specificity'] = specificity
        return metrics

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        all_subject_ids = []
        for batch in tqdm(train_loader, desc="Training"):
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                if torch.isnan(batch['text']).any() or torch.isnan(batch['structured']).any():
                    print("WARNING: Input data contains NaN!")
                outputs = self.model(
                    batch['text'].to(self.config.device, non_blocking=True),
                    batch['structured'].to(self.config.device, non_blocking=True),
                    batch['image_vec'].to(self.config.device, non_blocking=True),
                )

                loss = self.criterion(outputs, batch['label'].to(self.config.device))

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1).cpu().detach().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(batch['label'].cpu().numpy())
            all_subject_ids.extend(batch['subject_id'])

        results_df = pd.DataFrame({
            'subject_id': all_subject_ids,
            'true_label': all_labels,
            'pred_label': all_preds,
            'prob_0': [p[0] for p in all_probs],
            'prob_1': [p[1] for p in all_probs]
        })
        y_pred = np.argmax(all_probs, axis=1)
        metrics = self._calculate_metrics(all_labels, y_pred, np.array(all_probs))
        return total_loss / len(train_loader), metrics, results_df

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        all_subject_ids = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                outputs = self.model(
                    batch['text'].to(self.config.device),
                    batch['structured'].to(self.config.device),
                    batch['image_vec'].to(self.config.device, non_blocking=True)
                )

                loss = self.criterion(outputs, batch['label'].to(self.config.device))

                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1).cpu().detach().numpy()
                preds = np.argmax(probs, axis=1)
                labels = batch['label'].cpu().numpy()

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_subject_ids.extend(batch['subject_id'])
            results_df = pd.DataFrame({
                'subject_id': all_subject_ids,
                'true_label': all_labels,
                'pred_label': all_preds,
                'prob_0': [p[0] for p in all_probs],
                'prob_1': [p[1] for p in all_probs]
            })

            y_pred = np.argmax(all_probs, axis=1)
            metrics = self._calculate_metrics(all_labels, y_pred, np.array(all_probs))

            avg_loss = total_loss / len(val_loader)

        return avg_loss, metrics, results_df

    def clamp_paras(self):
        # 在训练循环中添加
        for param in self.model.parameters():
            param.register_hook(lambda grad: torch.clamp(grad, -0.5, 0.5))  # 梯度裁剪
