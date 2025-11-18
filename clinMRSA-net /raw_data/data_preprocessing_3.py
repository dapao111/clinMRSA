import pandas as pd
from datetime import timedelta, datetime
import os
import re
import ast
import numpy as np

meta_df = pd.read_csv(
    "/data/dupllin/mimic_data_iv_3.1/physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz")


def parse_dicom_datetime(date_str, time_str):
    try:
        date_part = datetime.strptime(str(date_str), "%Y%m%d")
        time_str = str(time_str).split('.')[0]  # 移除毫秒部分
        if len(time_str) == 6:
            time_part = datetime.strptime(time_str, "%H%M%S").time()
        elif len(time_str) > 0:
            time_part = datetime.strptime(time_str.zfill(6), "%H%M%S").time()
        else:
            time_part = datetime.min.time()
        return datetime.combine(date_part.date(), time_part)
    except:
        return None


meta_df['study_datetime'] = meta_df.apply(
    lambda x: parse_dicom_datetime(x['StudyDate'], x['StudyTime']),
    axis=1
)

initial_count = len(meta_df)
meta_df = meta_df.dropna(subset=['study_datetime'])
print(f"Remove invalid time records: {initial_count - len(meta_df)}/{initial_count}")


def build_jpg_path(row, base_path):
    p_folder = f"p{str(row['subject_id'])[:2]}"
    p_id = f"p{row['subject_id']}"
    s_id = f"s{row['study_id']}"
    return os.path.join(
        base_path, 'files', p_folder, p_id, s_id, f"{row['dicom_id']}.jpg"
    )


base_jpg_path = '/data/dupllin/mimic_data_iv_3.1/physionet.org/files/mimic-cxr-jpg/2.1.0'
meta_df['jpg_path'] = meta_df.apply(
    lambda x: build_jpg_path(x, base_jpg_path),
    axis=1
)


def validate_path(path):
    return os.path.exists(path) if isinstance(path, str) else False


meta_df['path_exists'] = meta_df['jpg_path'].apply(validate_path)
existing_paths = meta_df['path_exists'].sum()
print(f"Valid image path: {existing_paths}/{len(meta_df)} ({existing_paths / len(meta_df):.2%})")

# Screen valid image records
valid_meta_df = meta_df[meta_df['path_exists']].copy()


# Get the CXR report content
def get_cxr_report(jpg_path):
    # Replace the dataset part in the path
    report_base_path = jpg_path.replace("mimic-cxr-jpg", "mimic-cxr")

    # Get the path of the "study id" directory
    study_dir = os.path.dirname(report_base_path)  # /.../s12345678

    # Get the name of the study folder
    study_folder_name = os.path.basename(study_dir)

    # Build the path of the report file
    parent_dir = os.path.dirname(study_dir)  # /.../p10012345
    report_path = os.path.join(parent_dir, f"{study_folder_name}.txt")

    try:
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        else:
            study_report_path = os.path.join(study_dir, "report.txt")
            if os.path.exists(study_report_path):
                with open(study_report_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            print(f"The report document does not exist: {report_path} 或 {study_report_path}")
    except Exception as e:
        print(f"Failed to read the report file: {report_path}, 错误: {e}")

    return ""


# Filter the images within the time window and obtain the report
def filter_images(subject_id, base_time, max_time_diff, time_window=7):
    #
    patient_images = valid_meta_df[valid_meta_df['subject_id'] == subject_id]

    patient_images['days_diff'] = (base_time - patient_images['study_datetime']).dt.total_seconds() / (24 * 3600)

    if isinstance(max_time_diff, pd.Timedelta):
        max_time_diff_days = max_time_diff.total_seconds() / (24 * 3600)
    else:
        max_time_diff_days = max_time_diff

    # print(subject_id,patient_images['days_diff'])
    # if subject_id == 10076263:
    #     print(base_time, patient_images['study_datetime'],111)
    #     print(patient_images['days_diff'], max_time_diff_days)
    valid_images = patient_images[
        (patient_images['days_diff'] >= 0)
        # & (patient_images['days_diff'] <= time_window)
        # & (patient_images['days_diff'] <= max_time_diff_days + 0.01)
        ]

    jpg_paths = []
    cxr_reports = []
    seen_reports = set()  # 避免重复报告

    for _, row in valid_images.iterrows():
        jpg_path = row['jpg_path']
        report = get_cxr_report(jpg_path)

        if report and report not in seen_reports:
            jpg_paths.append(jpg_path)
            cxr_reports.append(report)
            seen_reports.add(report)

    return jpg_paths, cxr_reports


#
def process_mrsa_data():
    # 加载数据
    df = pd.read_csv('./MRSA829.csv', parse_dates=['anchor_time', 'event_time'])
    print(len(df))

    # 预处理
    def preprocess_data(df):
        orig_patients = df['subject_id'].nunique()
        print("初始患者人数：", orig_patients)
        first_diagnosis = df.groupby('subject_id')['anchor_time'].first()

        processed_df = df.merge(
            first_diagnosis,
            on='subject_id',
            suffixes=('_record', '_first')
        )

        processed_df['time_diff'] = processed_df['anchor_time_first'] - processed_df['event_time']
        mask = (processed_df['time_diff'] > timedelta(0))# & (processed_df['time_diff'] <= pd.Timedelta(days=7))

        return processed_df[mask].sort_values(['subject_id', 'event_time'])

    filtered_df = preprocess_data(df)
    print(filtered_df['time_diff'].max())

    meta_table = (
        filtered_df
        .groupby('subject_id')
        .apply(lambda x: {
            'first_diagnosis_time': x['anchor_time_first'].iloc[0],
            'record_count': len(x),
            'records': x[[
                'event_time', 'record_type', 'icd_version', 'seq_num',
                'icd_code', 'diagnosis_description', 'note_snippet'
            ]].to_dict('records')
        })
        .reset_index(name='patient_records')
    )

    max_time_diffs = filtered_df.groupby('subject_id')['time_diff'].max()
    meta_table = meta_table.merge(
        max_time_diffs.rename('max_time_diff'),
        on='subject_id'
    )

    result = meta_table.apply(
        lambda x: filter_images(
            x['subject_id'],
            x['patient_records']['first_diagnosis_time'],
            x['max_time_diff']
        ),
        axis=1
    )

    meta_table['jpg_paths'] = result.apply(lambda x: x[0])
    meta_table['cxr_reports'] = result.apply(lambda x: x[1])

    patients_with_reports = meta_table[meta_table['cxr_reports'].apply(len) > 0]
    print(f"含有效报告的患者: {len(patients_with_reports)}/{len(meta_table)}")

    meta_table.to_pickle('MRSA_meta_table_wjpg_reports_filtered_alldays_allp_val_wr.pkl')

    patient_jpg_map = dict(zip(meta_table['subject_id'], meta_table['jpg_paths']))
    patient_report_map = dict(zip(meta_table['subject_id'], meta_table['cxr_reports']))

    filtered_df['jpg_paths'] = filtered_df['subject_id'].map(patient_jpg_map)
    filtered_df['cxr_reports'] = filtered_df['subject_id'].map(patient_report_map)
    filtered_df.to_csv('filtered_MRSA_records_wjpg_reports_filtered_alldays_allp_val_wr.csv', index=False)

    return meta_table


def process_control_data():
    # 加载数据
    df = pd.read_csv('./mrsa_analysis_20250829_150347.csv', parse_dates=['anchor_time', 'event_time'])
    unique_subjects = df['subject_id'].unique()
    print(len(df))

    # np.random.seed(42)
    # sampled_subjects = np.random.choice(unique_subjects, size=10000, replace=False)
    # print(sampled_subjects)
    # df = df[df['subject_id'].isin(sampled_subjects)]

    # 预处理
    def preprocess_data(df):
        first_diagnosis = df.groupby('subject_id')['anchor_time'].first().reset_index()

        processed_df = df.merge(
            first_diagnosis,
            on='subject_id',
            how='left',
            suffixes=('_record', '_first')
        )

        processed_df['time_diff'] = processed_df['anchor_time_first'] - processed_df['event_time']
        # mask = (processed_df['time_diff'] > pd.Timedelta(0))
        mask = (processed_df['time_diff'] > timedelta(0)) #& (processed_df['time_diff'] <= pd.Timedelta(days=7))

        return processed_df[mask].sort_values(['subject_id', 'event_time'])

    filtered_df_control = preprocess_data(df)
    print(filtered_df_control['time_diff'].max())
    meta_table = (
        filtered_df_control
        .groupby('subject_id')
        .apply(lambda x: {
            'first_diagnosis_time': x['anchor_time_first'].iloc[0],
            'record_count': len(x),
            'records': x[[
                'event_time', 'record_type', 'icd_version', 'seq_num',
                'icd_code', 'diagnosis_description', 'note_snippet'
            ]].to_dict('records')
        })
        .reset_index(name='patient_records')
    )

    max_time_diffs = filtered_df_control.groupby('subject_id')['time_diff'].max()
    meta_table = meta_table.merge(
        max_time_diffs.rename('max_time_diff'),
        on='subject_id'
    )
    result = meta_table.apply(
        lambda x: filter_images(
            x['subject_id'],
            x['patient_records']['first_diagnosis_time'],
            x['max_time_diff']
        ),
        axis=1
    )

    meta_table['jpg_paths'] = result.apply(lambda x: x[0])
    meta_table['cxr_reports'] = result.apply(lambda x: x[1])

    patients_with_reports = meta_table[meta_table['cxr_reports'].apply(len) > 0]
    print(f"Patients with valid reports: {len(patients_with_reports)}/{len(meta_table)}")

    meta_table.to_pickle('control_meta_table_wjpg_reports_filtered_alldays_1w_val_27.pkl')

    patient_jpg_map = dict(zip(meta_table['subject_id'], meta_table['jpg_paths']))
    patient_report_map = dict(zip(meta_table['subject_id'], meta_table['cxr_reports']))

    filtered_df_control['jpg_paths'] = filtered_df_control['subject_id'].map(patient_jpg_map)
    filtered_df_control['cxr_reports'] = filtered_df_control['subject_id'].map(patient_report_map)
    filtered_df_control.to_csv('filtered_control_records_wjpg_reports_filtered_alldays_1w_val_27.csv', index=False)

    return meta_table


if __name__ == "__main__":
    print("Start processing the MRSA group data...")
    mrsa_meta = process_mrsa_data()

    print("\nStart processing the data of the control group...")
    control_meta = process_control_data()

