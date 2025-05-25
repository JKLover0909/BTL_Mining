import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import re
import warnings

warnings.filterwarnings("ignore")

def normalize_col(col):
    return re.sub(r'\W+', '_', col.lower())

def infer_from_row_dict(input_row):
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
    # Lấy xác suất dự đoán class 1
    proba = model.predict_proba(X)[0][1]
    return proba

if __name__ == "__main__":
    input_csv = '/home/ubuntu/1/BTL_Mining/data_10_1.csv'
    df = pd.read_csv(input_csv)
    drop_cols = [col for col in df.columns if col.lower() == 'id']
    df = df.drop(columns=drop_cols, errors='ignore')
    y_true = []
    y_score = []
    for idx, row in df.iterrows():
        input_row = row.drop('target').to_dict()
        target = row['target']
        proba = infer_from_row_dict(input_row)
        y_true.append(target)
        y_score.append(proba)
        print(f"Row {idx}: target={target}, predict_proba={proba:.4f}")
    # Tính AUROC
    auroc = roc_auc_score(y_true, y_score)
    # Chuyển xác suất thành nhãn (ngưỡng 0.5)
    y_pred = [1 if s >= 0.5 else 0 for s in y_score]
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print(f"\nAUROC trên toàn bộ tập: {auroc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")