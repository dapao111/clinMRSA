# model/model.py
import torch.nn as nn
import torch
import math
import numpy as np


class LightweightImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),  # Added batch norm
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),  # Added batch norm
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),  # Added batch norm
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, config.hidden_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.block(x)


class FeatureInteractionLayer(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.input_features = input_features
        self.num_pairs = input_features * (input_features - 1) // 2

        # 交互特征处理网络
        self.interaction_net = nn.Sequential(
            nn.Linear(self.num_pairs, 16),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(16, input_features)
        )

    def forward(self, x):
        """
            输入: [batch, seq_len, features=6]
            输出: [batch, seq_len, features*2]
            """
        batch, seq_len, _ = x.shape

        # 生成特征对索引
        i, j = torch.tril_indices(self.input_features, self.input_features, offset=-1)

        # 计算所有特征对乘积 [batch, seq_len, num_pairs]
        interactions = x[:, :, i] * x[:, :, j]

        # 处理交互特征 [batch, seq_len, input_features]
        transformed = self.interaction_net(interactions)

        # 拼接原始特征 [batch, seq_len, 6+6=12]
        return torch.cat([x, transformed], dim=-1)


import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageCNN(nn.Module):
    def __init__(self, config, input_shape=(3, 224, 224)):
        super(ImageCNN, self).__init__()
        self.input_shape = input_shape
        self.hidden_dim = config.hidden_dim

        # 卷积层设计
        self.conv_layers = nn.Sequential(
            # 输入: (3, 224, 224)
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (64, 112, 112)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (128, 56, 56)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (256, 28, 28)

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: (512, 14, 14)
        )

        # 自适应池化到固定尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # 输出: (512, 7, 7)

        # 展平后映射到隐藏维度
        self.flatten_dim = 512 * 7 * 7
        self.fc = nn.Linear(self.flatten_dim, self.hidden_dim)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x 形状: (batch_size, 3, 224, 224)
        features = self.conv_layers(x)  # 输出: (batch_size, 512, 14, 14)
        features = self.adaptive_pool(features)  # 输出: (batch_size, 512, 7, 7)
        features = features.view(features.size(0), -1)  # 展平: (batch_size, 512*7*7)
        output = self.fc(features)  # 映射到隐藏维度: (batch_size, hidden_dim)
        return output


class LungCancerClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 文本编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(config.text_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(0.5)
        )
        # 结构化特征编码器
        self.struct_encoder = nn.Sequential(
            FeatureInteractionLayer(config.struct_dim),
            nn.Linear(config.struct_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(0.3)
        )
        self.struct_icd_encoder = nn.Sequential(
            # FeatureInteractionLayer(config.struct_dim),
            nn.Linear(13, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(0.3)
        )
        # 时间差编码器：处理时间特征并增强
        self.time_encoder = nn.Sequential(
            nn.Linear(5, config.hidden_dim),  # 输入为时间差的两个特征
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(0.1)
        )
        # self.img_encoder = nn.Sequential(
        #     nn.Linear(2048, 768),  # 输入为时间差的两个特征
        #     nn.GELU(),
        #     nn.LayerNorm(768),
        #     nn.Dropout(0.1),
        #     nn.Linear(768, config.hidden_dim),  # 输入为时间差的两个特征
        #     nn.GELU(),
        #     nn.LayerNorm(config.hidden_dim),
        #     nn.Dropout(0.1)
        # )
        # self.img_encoder = ImageCNN(config)
        self.img_encoder = LightweightImageEncoder(config)
        self.img_encoder_ark = nn.Sequential(
            nn.Linear(1376, 768),  # 输入为时间差的两个特征
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Dropout(0.7),
            nn.Linear(768, config.hidden_dim),  # 输入为时间差的两个特征
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(0.7)
        )
        # Transformer时序编码
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=512,
                nhead=1,
                dim_feedforward=512),
            num_layers=1)

        # 分类器
        # self.classifier = nn.Sequential(
        #     nn.Linear(config.hidden_dim + config.hidden_dim , 64),
        #     nn.GELU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(64, 2)
        # )
        self.pool = nn.Linear(512, 1)
        self.latent_feature_extractor = nn.Sequential(
            nn.Conv1d(config.max_seq_length, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            # 1D池化
            nn.MaxPool1d(2, stride=2),
            # 自适应池化统一输出尺寸
            nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            # 1D池化
            nn.MaxPool1d(2, stride=2),
            nn.AdaptiveAvgPool1d(1),
            # 展平
            nn.Flatten(),
            # 全连接层
            nn.Linear(32, config.hidden_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, 64),
            # nn.Dropout(0.5),
            # nn.BatchNorm1d(64),
            nn.GELU(),
            # ResidualBlock(config.hidden_dim),
            nn.Linear(64, 32),
            ResidualBlock(32),  # 可以添加更多残差块
            nn.Linear(32, 2)
        )

    def forward(self, text, structured, img=None):
        # 文本特征
        text_feat = self.text_encoder(text)
        # 结构化特征
        # struct_feat = self.struct_encoder(structured)
        icd = structured[:, :, :13]
        struct_icd_feat = self.struct_icd_encoder(icd)
        # img_features = self.img_encoder_ark(img).squeeze(1)
        img_features = self.img_encoder(img)

        # 提取时间差特征（最后两维）
        time_features = structured[:, :, -5:]  # [batch, seq_len, 5]
        time_encoded = self.time_encoder(time_features)

        #结构数据特征
        # structured_data = torch.cat([struct_icd_feat, time_encoded], dim=-1)
        structured_data = struct_icd_feat + time_encoded

        combined = torch.cat([structured_data, text_feat], dim=-1)
        # combined = text_feat + structured_data  # 时间信息融合

        combined = combined.permute(1, 0, 2)
        temporal_feat = self.transformer(combined)
        # print(temporal_feat[-1],img_features)
        # 分类
        to_classified_features = torch.cat([temporal_feat[-1],img_features], dim=-1)
        logit = self.classifier(to_classified_features)
        # print(logit.size())

        # logit = self.classifier(struct_icd_feat.reshape(struct_icd_feat.shape[0],-1))
        return logit
