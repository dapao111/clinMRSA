# # data/dataset.py
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import re
# import math
#
#
# class MedicalDataset(Dataset):
#     def __init__(self, df, embedder, config):
#         self.df = df
#         self.embedder = embedder
#         self.max_records = config.max_seq_length
#         self.config = config
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         patient_data = row['patient_records']  # 获取患者数据字典
#         label = row['label']
#         return self._process_patient(patient_data, label)
#
#     def _process_icd(self, record):
#         """动态解析ICD代码"""
#         icd_version = 0 if record.get('icd_version') == 9 else 1
#         raw_code = str(record.get('icd_code', '')).strip().upper()
#
#         if not raw_code or raw_code == 'NAN':
#             return None  # 标记为无效记录
#
#         # 清理所有分隔符（. 或空格）
#         cleaned_code = re.sub(r'[\. ]', '', raw_code)
#
#         if icd_version == 1:  # ICD-10
#             main_code = cleaned_code[:3] if len(cleaned_code) >= 3 else cleaned_code.ljust(3, 'X')
#             sub_code_part = cleaned_code[3:] if len(cleaned_code) > 3 else ''
#             # 子分类保留前两位，不足补0（根据需求调整）
#             sub_code = sub_code_part.ljust(2, '0')[:2] if sub_code_part else '00'
#         else:  # ICD-9
#             main_code = cleaned_code[:3] if len(cleaned_code) >= 3 else cleaned_code.ljust(3, '0')
#             sub_code_part = cleaned_code[3:] if len(cleaned_code) > 3 else ''
#             # 子分类补零至两位（如'3'→'03'）
#             sub_code = sub_code_part.zfill(2)[:2] if sub_code_part else '00'
#
#         encoding = [
#             icd_version,
#             hash(main_code) % 100,
#             hash(sub_code) % 50
#         ]
#         return encoding
#
#     def _process_patient(self, patient_data, label):
#         text_embeddings = []
#         structured_features = []
#
#         for record in patient_data['records']:
#             # === 过滤无效记录 ===
#             icd_encoding = self._process_icd(record) or [0, 0, 0]
#             icd_encoding = [float(x) if not np.isnan(x) else 0.0 for x in icd_encoding]
#
#             # === 文本特征 ===
#             text_content = record.get('text_content', '')
#             icd_long_title = record.get('long_title', '')
#             if isinstance(text_content, float) and math.isnan(text_content):
#                 text_content = ''
#             if isinstance(icd_long_title, float) and math.isnan(icd_long_title):
#                 icd_long_title = ''
#             combined_text = f"{icd_long_title}: {text_content}" if icd_long_title else text_content
#
#             if combined_text.strip():
#                 print(len(combined_text))
#                 text_emb = self.embedder.embed_text(combined_text)[0]
#             else:
#                 text_emb = np.zeros(self.config.text_dim)  # 空文本填充
#             text_embeddings.append(text_emb)
#
#             # === 结构化特征 ===
#             struct_feat = icd_encoding.copy()
#             value = record.get('seq_num', 0)
#             # clean_value = 0 if pd.isna(value) or value == "" else float(value)
#
#             struct_feat.append(0 if math.isnan(value) else value)
#             # === 时间特征 ===
#             time_diff = (record['record_time'] - patient_data['first_diagnosis_time']).days
#             bins = [30, 180, 365, 730]  # 按天数划分
#             time_bin = np.digitize(abs(time_diff), bins)
#             time_onehot = np.eye(len(bins) + 1)[time_bin]  # 独热编码
#             # struct_feat.extend([
#             #     time_diff / 365.0,  # 年标准化
#             #     np.log(abs(time_diff) + 1e-6)  # 对数变换
#             # ])
#             struct_feat.extend(time_onehot.astype(float).tolist())
#             structured_features.append(struct_feat)
#
#         # 填充处理
#         text_embeddings = self._pad_sequences(text_embeddings, self.config.text_dim)
#         structured_features = self._pad_sequences(structured_features, self.config.struct_dim)
#
#         return {
#             'text': torch.FloatTensor(text_embeddings),
#             'structured': torch.FloatTensor(structured_features),
#             'label': torch.tensor(label, dtype=torch.long)
#         }
#
#     def _pad_sequences(self, sequences, feat_dim):
#         if len(sequences) < self.max_records:
#             pad = np.zeros((self.max_records - len(sequences), feat_dim))
#             sequences = np.concatenate([np.array(sequences), pad])
#         else:
#             sequences = np.array(sequences[:self.max_records])
#         return sequences

import numpy as np
import torch
from torch.utils.data import Dataset


class MedicalDataset(Dataset):
    def __init__(self, data):
        """
        初始化数据集，加载预处理后的数据。
        :param data_path: 预处理数据文件的路径
        """
        self.data =data
        self.valid_indices = [
            i for i, sample in enumerate(data)
            if not np.all(sample['image_vec'] == 0)
        ]
    def __len__(self):
        """
        返回数据集长度
        """
        return len(self.valid_indices)  # 返回有效样本数量

        # return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        :param idx: 索引值
        """
        actual_idx = self.valid_indices[idx]
        sample = self.data[actual_idx]
        # sample = self.data[idx]
        return {
            'subject_id': sample['subject_id'],
            'text': torch.FloatTensor(sample['text']),
            'structured': torch.FloatTensor(sample['structured']),
            'image_vec': torch.tensor(sample['image_vec'], dtype=torch.float32),
            'label': torch.tensor(sample['label'], dtype=torch.long)

        }

