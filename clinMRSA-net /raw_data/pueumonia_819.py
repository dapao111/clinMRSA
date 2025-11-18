import pandas as pd
import os
import re

# 定义肺炎诊断的ICD代码列表
# pneumonia_icd9_codes = [
#     '480.0', '480.1', '480.2', '480.3', '480.8', '480.9', '481', '482.0', '482.1', '482.2',
#     '482.3', '482.30', '482.31', '482.32', '482.39', '482.4', '482.40', '482.41', '482.42',
#     '482.49', '482.8', '482.81', '482.82', '482.83', '482.84', '482.89', '482.9', '483',
#     '483.0', '483.1', '483.8', '484', '485', '486', '487.0', '507.0', '507.1', '507.8', '508.0',
#     '511.0', '511.1', '511.8', '511.9', '513.0', '513.1'
# ]
#
# pneumonia_icd10_codes = [
#     'J12.0', 'J12.1', 'J12.2', 'J12.3', 'J12.8', 'J12.9', 'J13', 'J14', 'J15.0', 'J15.1',
#     'J15.2', 'J15.20', 'J15.21', 'J15.22', 'J15.3', 'J15.4', 'J15.5', 'J15.6', 'J15.7',
#     'J15.8', 'J15.9', 'J16.0', 'J16.8', 'J17', 'J17.0', 'J17.1', 'J17.2', 'J17.3', 'J17.8',
#     'J17.9', 'J18.0', 'J18.1', 'J18.2', 'J18.8', 'J18.9', 'J69.0', 'J69.1', 'J69.8', 'J70.2',
#     'J70.3', 'J70.4', 'J80', 'J81.0', 'J81.1', 'J85.1', 'J85.2', 'J86.9', 'J95.4', 'J95.82'
# ]
#
# # 定义肺炎诊断代码前缀（用于通配符匹配）
# pneumonia_icd9_prefixes = ['480', '481', '482','483', '484', '485', '486', '487', '507', '508', '511', '513']
# pneumonia_icd10_prefixes = ['J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J69', 'J70', 'J80', 'J81', 'J85', 'J86', 'J95']

#ICD核对

pneumonia_icd9_codes = ['00322', '0064', '01160', '01161', '01162', '01163', '01164', '01165', '01166',
                        '0382', '0521', '0551', '0730', '11505', '11515', '11595', '1304', '4800', '4801',
                        '4802', '4803', '4808', '4809', '481', '4820', '4821', '4822', '48230', '48231',
                        '48232', '48239', '48240', '48241', '48242', '48249', '48281', '48282', '48283',
                        '48284', '48289', '4829', '4830', '4831', '4838', '4841', '4843', '4845', '4846',
                        '4847', '4848', '485', '486', '4870', '48801', '48811', '48881', '4957', '4958',
                        '4959', '5060', '5070', '5071', '5078', '5100', '5109', '51630', '51632', '51633',
                        '51635', '51636', '51637', '5171', '7700', '99731', '99732', 'V0382', 'V066', 'V1261']

pneumonia_icd10_codes = ['A0103', 'A0222', 'A065', 'A3700', 'A3701', 'A3710', 'A3711', 'A3780', 'A3781',
                         'A3790', 'A3791', 'A403', 'A5004', 'A5484', 'B012', 'B052', 'B0681', 'B250', 'B7781',
                         'B953', 'B960', 'B961', 'J09X1', 'J100', 'J1000', 'J1001', 'J1008', 'J110', 'J1100',
                         'J1108', 'J12', 'J120', 'J121', 'J122', 'J123', 'J128', 'J1281', 'J1282', 'J1289',
                         'J129', 'J13', 'J14', 'J15', 'J150', 'J151', 'J152', 'J1520', 'J1521', 'J15211',
                         'J15212', 'J1529', 'J153', 'J154', 'J155', 'J156', 'J157', 'J158', 'J159', 'J16',
                         'J160', 'J168', 'J17', 'J18', 'J180', 'J181', 'J182', 'J188', 'J189', 'J200', 'J67',
                         'J678', 'J679', 'J680', 'J69', 'J690', 'J691', 'J698', 'J8281', 'J8282', 'J8411',
                         'J84111', 'J84113', 'J84114', 'J84116', 'J84117', 'J842', 'J851', 'J852', 'J954',
                         'J95851', 'O2901', 'O29011', 'O29012', 'O29013', 'O29019', 'O740', 'O8901', 'P23',
                         'P230', 'P231', 'P232', 'P233', 'P234', 'P235', 'P236', 'P238', 'P239', 'Z8701']

septicemia_sepsis_icd9_codes = ['0031', '0223', '0380', '03810', '03811', '03812', '03819', '0382', '0383',
                                '03840', '03841', '03842', '03843', '03844', '03849', '0388', '0389', '0545',
                                '77181', '67020', '67022', '67024', '99591', '99592']

septicemia_sepsis_icd10_codes = ['A021', 'A227','A267', 'A327', 'A40', 'A400', 'A401', 'A403', 'A408', 'A409',
                                 'A41','A410', 'A4101', 'A4102', 'A411', 'A412', 'A413', 'A414', 'A415', 'A4150',
                                 'A4151', 'A4152', 'A4153', 'A4159', 'A418', 'A4181', 'A4189', 'A419', 'A427',
                                 'A5486', 'B377', 'O0337', 'O0387', 'O0487', 'O0737', 'O0882', 'O85', 'O8604',
                                 'P36', 'P360', 'P361', 'P3610', 'P3619', 'P362', 'P363', 'P3630', 'P3639',
                                 'P364', 'P365', 'P368', 'P369', 'R652', 'R6520', 'R6521', 'T8144', 'T8144XA',
                                 'T8144XD', 'T8144XS']


def is_pneumonia_diagnosis(row):
    """
    Check whether the given line contains a diagnosis of pneumonia
    """
    if row['record_type'] != 'Diagnosis':
        return False

    icd_code = str(row['icd_code']).strip()

    # Check the ICD-9 code
    if row['icd_version'] == 9:
        if icd_code in pneumonia_icd9_codes:
            return True
        # for prefix in pneumonia_icd9_prefixes:
        #     if icd_code.startswith(prefix):
        #         return True
        elif icd_code in septicemia_sepsis_icd9_codes:
            return True
        return False

    # Check the ICD-10 code
    elif row['icd_version'] == 10:
        if icd_code in pneumonia_icd10_codes:
            return True
        # for prefix in pneumonia_icd10_prefixes:
        #     if icd_code.startswith(prefix):
        #         return True
        elif icd_code in septicemia_sepsis_icd10_codes:
            return True
        return False

    return False


def get_image_count(img_paths):
    """Calculate the number of valid images"""
    if isinstance(img_paths, list):
        return len(img_paths)
    elif isinstance(img_paths, str) and img_paths.startswith('['):
        try:
            return len(eval(img_paths))
        except:
            return 0
    return 0


# Process MRSA data
# mrsa_df = pd.read_csv('filtered_MRSA_records_wjpg_filtered.csv')
mrsa_df = pd.read_csv('filtered_MRSA_records_wjpg_reports_filtered_14days_allp_val.csv')
print(f"The number of MRSA patients before treatment: {mrsa_df['subject_id'].nunique()}")

# Screening for the diagnosis of pneumonia
pneumonia_mask = mrsa_df.apply(is_pneumonia_diagnosis, axis=1)
mrsa_pneumonia_df = mrsa_df[pneumonia_mask]

# Obtain the ID of the pneumonia patient
pneumonia_patients = mrsa_pneumonia_df['subject_id'].unique()
print(f"The number of pneumonia patients: {len(pneumonia_patients)}")

# Screen all records of patients with pneumonia diagnoses
mrsa_pneumonia_full = mrsa_df[mrsa_df['subject_id'].isin(pneumonia_patients)].copy()

# Add image counting
mrsa_pneumonia_full['image_count'] = mrsa_pneumonia_full['jpg_paths'].apply(get_image_count)

# Screen patients with images
patients_with_images = mrsa_pneumonia_full[mrsa_pneumonia_full['image_count'] > 0]['subject_id'].unique()
mrsa_pneumonia_with_images = mrsa_pneumonia_full[mrsa_pneumonia_full['subject_id'].isin(patients_with_images)]

print(f"The number of MRSA patients after treatment: {mrsa_pneumonia_full['subject_id'].nunique()}")
print(f"The number of pneumonia patients with images: {len(patients_with_images)}")

# Save the results
mrsa_pneumonia_with_images.to_csv('filtered_MRSA_pneumonia&septicemia_records_wjpg_reports_allp.csv', index=False)

# Processing control group data
# control_df = pd.read_csv('filtered_control_records_wjpg_filtered.csv')
control_df = pd.read_csv('filtered_control_records_wjpg_reports_filtered_14days_1w_val_27.csv')
print(f"The number of patients in the control group before treatment: {control_df['subject_id'].nunique()}")

# Screening for the diagnosis of pneumonia
pneumonia_mask_control = control_df.apply(is_pneumonia_diagnosis, axis=1)
control_pneumonia_df = control_df[pneumonia_mask_control]

# Obtain the ID of the pneumonia patient
pneumonia_patients_control = control_pneumonia_df['subject_id'].unique()
print(f"The number of pneumonia patients in the control group: {len(pneumonia_patients_control)}")

# Screen all records of patients with pneumonia diagnoses
control_pneumonia_full = control_df[control_df['subject_id'].isin(pneumonia_patients_control)].copy()

# Add image counting
control_pneumonia_full['image_count'] = control_pneumonia_full['jpg_paths'].apply(get_image_count)
# Screen patients with images
patients_with_images_control = control_pneumonia_full[control_pneumonia_full['image_count'] > 0]['subject_id'].unique()
control_pneumonia_with_images = control_pneumonia_full[
    control_pneumonia_full['subject_id'].isin(patients_with_images_control)]
# control_cxr_df = control_df.copy()
# control_cxr_df['image_count'] = control_cxr_df['jpg_paths'].apply(get_image_count)
#
# patients_with_images_control = control_cxr_df[control_cxr_df['image_count'] > 0]['subject_id'].unique()
# control_pneumonia_with_images = control_cxr_df[
#     control_cxr_df['subject_id'].isin(patients_with_images_control)]
print(f"The number of patients in the control group after treatment: {control_pneumonia_full['subject_id'].nunique()}")
print(f"The number of patients in the control group with images: {len(patients_with_images_control)}")

# Saving the results
control_pneumonia_with_images.to_csv('filtered_control_pneumonia&septicemia_records_wjpg_reports_allp.csv', index=False)


def create_structured_metadata(df, group_col, time_col):
    """
    创建结构化元数据（只包含有影像的患者）
    """
    # 先筛选有影像的患者
    df = df[df['image_count'] > 0].copy()

    meta_table = (
        df
        .groupby(group_col)
        .apply(lambda x: {
            'first_diagnosis_time': x[time_col].iloc[0],
            'record_count': len(x),
            'image_count': x['image_count'].iloc[0],  # 添加影像计数
            'records': x[[
                'event_time',
                'record_type',
                'icd_version',
                'seq_num',
                'icd_code',
                'diagnosis_description',
                'note_snippet',
                'jpg_paths',
                'cxr_reports'
            ]].to_dict('records')
        })
        .reset_index(name='patient_records')
    )
    return meta_table


mrsa_meta_pneumonia = create_structured_metadata(
    mrsa_pneumonia_with_images,
    'subject_id',
    'anchor_time_first'
)
mrsa_meta_pneumonia.to_pickle('MRSA_pneumonia&septicemia_meta_table_wjpg_reports.pkl')

control_meta_pneumonia = create_structured_metadata(
    control_pneumonia_with_images,
    'subject_id',
    'anchor_time_first'
)
control_meta_pneumonia.to_pickle('control_pneumonia&septicemia_meta_table_wjpg_reports.pkl')


