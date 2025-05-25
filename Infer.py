import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from catboost import CatBoostClassifier
import re
import sys
import warnings

# Bỏ qua tất cả warning (bao gồm cả SettingWithCopyWarning)
warnings.filterwarnings("ignore")

def normalize_col(col):
    # Chuẩn hóa: chữ thường, thay mọi ký tự không phải chữ/số thành _
    return re.sub(r'\W+', '_', col.lower())

def infer_from_row(input_csv, row_idx):
    # Đọc toàn bộ file input
    df_input = pd.read_csv(input_csv)
    # Loại bỏ cột 'ID' và 'target' nếu có (không phân biệt hoa thường)
    drop_cols = [col for col in df_input.columns if col.lower() in ['id', 'target']]
    df_input = df_input.drop(columns=drop_cols, errors='ignore')
    # Lấy hàng thứ row_idx (bắt đầu từ 0)
    input_row = df_input.iloc[row_idx].to_dict()

    # --- Step 1: Fill missing value ---
    data_file = '/home/ubuntu/1/BTL_Mining/data_80_imputed_no_id.csv'
    df = pd.read_csv(data_file)
    for col in df.columns:
        if col not in input_row:
            input_row[col] = None
    input_df = pd.DataFrame([input_row])[df.columns]
    concat_df = pd.concat([df, input_df], ignore_index=True)
    imputer = KNNImputer(n_neighbors=5)
    imputed = imputer.fit_transform(concat_df)
    imputed_input = pd.DataFrame([imputed[-1]], columns=df.columns)

    # --- Step 2: Assign cluster ---
    cluster0_path = '/home/ubuntu/1/BTL_Mining/cluster_0.csv'
    cluster1_path = '/home/ubuntu/1/BTL_Mining/cluster_1.csv'
    cluster_0 = pd.read_csv(cluster0_path)
    cluster_1 = pd.read_csv(cluster1_path)
    drop_cols = [col for col in ['target', 'cluster'] if col in cluster_0.columns]
    cluster_0 = cluster_0.drop(columns=drop_cols, errors='ignore')
    cluster_1 = cluster_1.drop(columns=drop_cols, errors='ignore')
    features = [col for col in imputed_input.columns if col in cluster_0.columns]
    centroid_0 = cluster_0[features].mean()
    centroid_1 = cluster_1[features].mean()
    input_values = imputed_input[features].iloc[0]
    dist_0 = np.linalg.norm(input_values - centroid_0)
    dist_1 = np.linalg.norm(input_values - centroid_1)
    assigned_cluster = 0 if dist_0 < dist_1 else 1

    # --- Step 3: Extract important features ---
    if assigned_cluster == 0:
        features_file = '/home/ubuntu/1/BTL_Mining/top_features_cluster_0.csv'
        model_path = '/home/ubuntu/1/BTL_Mining/catboost_model_cluster_0.cbm'
    else:
        features_file = '/home/ubuntu/1/BTL_Mining/top_features_cluster_1.csv'
        model_path = '/home/ubuntu/1/BTL_Mining/catboost_model_cluster_1.cbm'
    top_features = pd.read_csv(features_file)['feature'].tolist()
    columns_to_save = top_features
    extracted = imputed_input[columns_to_save].copy()
    extracted['cluster'] = assigned_cluster

    # --- Step 4: Predict True/False ---
    model = CatBoostClassifier()
    model.load_model(model_path)
    model_features = model.feature_names_
    input_cols_norm = {normalize_col(col): col for col in extracted.columns}
    model_features_norm = {normalize_col(feat): feat for feat in model_features}
    rename_dict = {}
    for norm_feat, feat in model_features_norm.items():
        if norm_feat in input_cols_norm:
            rename_dict[input_cols_norm[norm_feat]] = feat
    X = extracted.rename(columns=rename_dict)
    if 'cluster' in X.columns:
        X = X.drop(columns=['cluster'])
    for feat in model_features:
        if feat not in X.columns:
            X[feat] = 0
    X = X[model_features]
    prediction = model.predict(X)[0]
    print(bool(prediction))
    return bool(prediction)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Cách dùng: python Infer.py <đường_dẫn_file_csv> <số_thứ_tự_dòng_bắt_đầu_từ_0>")
        print("Ví dụ: python Infer.py /home/ubuntu/1/BTL_Mining/data_10_1.csv 1")
        sys.exit(1)
    input_csv = sys.argv[1]
    row_idx = int(sys.argv[2])
    infer_from_row(input_csv, row_idx)