import numpy as np
import pandas as pd
import os
import re
import sys
import ast

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from model.utils import TextEmbedder, ImageEmbedder
from config_mul import Config


def extract_study_id_from_path(jpg_path):
    """从图像路径中提取study_id"""
    # 路径格式：.../files/p10/p10000032/s50414267/xxxx.jpg
    parts = jpg_path.split('/')
    if len(parts) >= 4 and parts[-2].startswith('s'):
        try:
            return int(parts[-2][1:])  # 移除's'前缀并转为整数
        except ValueError:
            pass
    return None


def get_report_text_from_jpg_path(jpg_path):
    """从图像路径获取对应的报告文本"""
    # 提取study_id
    study_id = extract_study_id_from_path(jpg_path)
    if study_id is None:
        return ""

    # 构建报告文件路径：将.jpg替换为.txt，并保留目录结构
    report_path = jpg_path.replace('.jpg', '.txt')

    # 尝试读取报告文件
    try:
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        print(f"读取报告文件失败: {report_path}, 错误: {e}")

    return ""


def segment_average_pooling(arr, target_length):
    # arr: (seq_len, dim)
    seq_len = len(arr)
    if seq_len == target_length:
        return np.array(arr)
    indices = np.linspace(0, seq_len, target_length + 1, dtype=int)
    pooled = []
    for i in range(target_length):
        start, end = indices[i], indices[i + 1]
        pooled.append(np.mean(arr[start:end], axis=0))
    return np.stack(pooled)


def icd_encoding_to_binary(icd_encoding):
    # 定义每个元素的位长度
    bits = [2, 7, 4]

    binary_str = ""
    for i, value in enumerate(icd_encoding):
        # 将整数转为二进制字符串并填充指定位数
        bin_val = bin(value)[2:].zfill(bits[i])
        binary_str += bin_val

    # 将字符串转为 0/1 整数向量
    return [int(bit) for bit in binary_str]


def preprocess_data(df, embedder, config, save_path):
    processed_data = []
    idx = 0
    print("总条数:", len(df))
    for _, row in df.iterrows():
        idx = idx + 1
        print("正在处理第" + str(idx) + "条数据")

        patient_data = row['patient_records']
        label = row['label']
        text_embeddings = []
        structured_features = []
        report_text = ""
        for record in patient_data['records']:
            # === ICD 编码处理 ===
            icd_version = 0 if record.get('icd_version') == 9 else 1
            raw_code = str(record.get('icd_code', '')).strip().upper()

            if not raw_code or raw_code == 'NAN':
                icd_encoding = [0, 0, 0]  # 无效记录
            else:
                cleaned_code = re.sub(r'[\. ]', '', raw_code)
                if icd_version == 1:  # ICD-10
                    main_code = cleaned_code[:3] if len(cleaned_code) >= 3 else cleaned_code.ljust(3, 'X')
                    sub_code_part = cleaned_code[3:] if len(cleaned_code) > 3 else ''
                    sub_code = sub_code_part.ljust(2, '0')[:2] if sub_code_part else '00'
                else:  # ICD-9
                    main_code = cleaned_code[:3] if len(cleaned_code) >= 3 else cleaned_code.ljust(3, '0')
                    sub_code_part = cleaned_code[3:] if len(cleaned_code) > 3 else ''
                    sub_code = sub_code_part.zfill(2)[:2] if sub_code_part else '00'

                icd_encoding = [
                    icd_version,
                    hash(main_code) % 100,
                    hash(sub_code) % 10
                ]
            icd_encoding = icd_encoding_to_binary(icd_encoding)

            # === 文本嵌入处理 ===
            text_content = record.get('note_snippet', '')
            icd_long_title = record.get('long_title', '')

            # 处理NaN
            text_content = '' if pd.isna(text_content) else str(text_content)
            icd_long_title = '' if pd.isna(icd_long_title) else str(icd_long_title)

            # === 添加过滤逻辑 ===
            # 如果诊断描述包含"staphylococcus aureus"但不包含"history"，则置为空
            if icd_long_title:
                lower_title = icd_long_title.lower()
                if "staphylococcus aureus" in lower_title and "history" not in lower_title:
                    icd_long_title = ''

            # 组合基础文本
            if len(icd_long_title) and len(text_content):
                combined_text = f"{icd_long_title}: {text_content}"
            elif len(icd_long_title):
                combined_text = icd_long_title
            else:
                combined_text = text_content

            # === 添加MIMIC-CXR报告文本 ===
            # if config.reports_flag:
            #     report_text = ""
            #     rep_paths = getattr(row, 'cxr_reports', [])
            #     if isinstance(rep_paths, float) or rep_paths is None or len(rep_paths)==0:
            #         rep_paths = patient_data['records'][0].get('cxr_reports', [])
            #     # print(rep_paths)
            #     if rep_paths:
            #         # 如果jpg_paths是字符串，转换为列表
            #         if isinstance(rep_paths, str):
            #             try:
            #                 rep_paths = ast.literal_eval(rep_paths)
            #             except:
            #                 rep_paths = rep_paths.strip("[]").split(",")
            #                 rep_paths = [p.strip().strip("'\"") for p in rep_paths]
            #
            #         # 尝试从每个路径获取报告，直到成功
            #         for jpg_path in rep_paths:
            #             report_text = get_report_text_from_jpg_path(jpg_path)
            #             if report_text:
            #                 break
            #
            #     # 合并所有文本
            #     if report_text:
            #         print(report_text)
            #         if combined_text:
            #             combined_text = f"{combined_text}\n\nCXR Report:\n{report_text}"
            #         else:
            #             combined_text = f"CXR Report:\n{report_text}"

            if config.reports_flag and report_text == "":
                report_text = getattr(row, 'cxr_reports', "")
                if len(report_text) == 0:
                    report_text = patient_data['records'][0].get('cxr_reports', "")

                if report_text:
                    print("合并cxr报告信息")
                    if combined_text:
                        combined_text = f"{combined_text}\n\nCXR Report:\n{report_text}"
                    else:
                        combined_text = f"CXR Report:\n{report_text}"

            # === 文本嵌入 ===
            if combined_text.strip():
                text_emb = embedder.embed_text(combined_text)
            else:
                text_emb = np.zeros(config.text_dim)  # 空文本填充

            text_embeddings.append(text_emb)

            # === 结构化特征处理 ===
            struct_feat = icd_encoding.copy()
            value = record.get('seq_num', 0)
            struct_feat.append(0 if np.isnan(value) else value)

            # 确保日期是datetime对象
            event_time = pd.to_datetime(record['event_time'])
            diagnosis_time = pd.to_datetime(patient_data['first_diagnosis_time'])

            # 计算时间差
            time_diff = (event_time - diagnosis_time).days
            bins = [30, 180, 365, 730]
            time_bin = np.digitize(abs(time_diff), bins)
            time_onehot = np.eye(len(bins) + 1)[time_bin]
            struct_feat.extend(time_onehot.astype(float).tolist())

            structured_features.append(struct_feat)

        # === 填充处理 ===
        if len(text_embeddings) < config.max_seq_length:
            pad_emb = np.zeros((config.max_seq_length - len(text_embeddings), config.text_dim))
            text_embeddings = np.concatenate([text_embeddings, pad_emb])
        else:
            text_embeddings = np.array(text_embeddings[:config.max_seq_length])

        if len(structured_features) < config.max_seq_length:
            pad_struct = np.zeros((config.max_seq_length - len(structured_features), config.struct_dim))
            structured_features = np.concatenate([structured_features, pad_struct])
        else:
            structured_features = np.array(structured_features[:config.max_seq_length])

        # # 获取图像路径列表
        jpg_paths = getattr(row, 'jpg_paths', [])
        if isinstance(jpg_paths, float) or jpg_paths is None or len(jpg_paths) == 0:
            jpg_paths = patient_data['records'][0].get('jpg_paths', [])
        if jpg_paths and isinstance(jpg_paths, str):
            try:
                jpg_paths = ast.literal_eval(jpg_paths)
            except:
                jpg_paths = jpg_paths.strip("[]").split(",")
                jpg_paths = [p.strip().strip("'\"") for p in jpg_paths]

        # 生成图像向量
        image_embedder = ImageEmbedder(config)
        image_vector = image_embedder.embed_images(list(jpg_paths))

        # 保存处理结果
        processed_data.append({
            'subject_id': row['subject_id'],
            'text': text_embeddings,
            'structured': structured_features,
            'image_vec': image_vector.astype(np.float32),  # 图像向量
            'label': label
        })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, processed_data)
    pro_df = pd.DataFrame(processed_data)
    pro_df.to_csv("all_ad_records_processed_data_1w_14days_info.csv", index=False)  # 保存到文件


if __name__ == "__main__":
    config = Config()
    embedder = TextEmbedder(config.bert_model)

    # 加载数据
    control = pd.read_pickle(config.control_path)
    case = pd.read_pickle(config.case_path)
    print(len(control), len(case))
    # control = control.sample(n=10000, random_state=42)
    control['label'], case['label'] = 0, 1
    full_df = pd.concat([control, case])
    print(len(control), len(case))

    # 预处理并保存
    preprocess_data(full_df, embedder, config, Config.data_path)
