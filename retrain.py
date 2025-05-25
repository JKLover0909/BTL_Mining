import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, classification_report
import re
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

def normalize_col(col):
    return re.sub(r'\W+', '_', col.lower())

def train_catboost_and_report_save(data, cluster_name, model_path, logf=None):
    X = data.drop(columns=['target'])
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = CatBoostClassifier(verbose=0, random_state=42)
    model.fit(X_train, y_train)
    model.save_model(model_path)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_pred_proba)
    msg = (
        f"\n{cluster_name}:\n"
        f"AUROC: {auroc:.4f}\n"
        f"Đã lưu model vào: {model_path}\n"
    )
    print(msg)
    if logf is not None:
        logf.write(msg + "\n")

def retrain_from_csv(input_csv, log_file):
    df_input = pd.read_csv(input_csv)
    if 'target' in df_input.columns:
        df_input['target'] = df_input['target'].apply(lambda x: True if x == 1.0 else False)
    drop_cols = [col for col in df_input.columns if col.lower() == 'id']
    df_input = df_input.drop(columns=drop_cols, errors='ignore')

    data_file = 'data_80_imputed_no_id.csv'
    df = pd.read_csv(data_file)
    for col in df.columns:
        if col not in df_input.columns:
            df_input[col] = None
    df_input = df_input[df.columns]
    concat_df = pd.concat([df, df_input], ignore_index=True)
    imputer = KNNImputer(n_neighbors=5)
    imputed = imputer.fit_transform(concat_df)
    imputed_input = pd.DataFrame(imputed[-len(df_input):], columns=df.columns)

    cluster0_path = 'cluster_0.csv'
    cluster1_path = 'cluster_1.csv'
    cluster_0 = pd.read_csv(cluster0_path)
    cluster_1 = pd.read_csv(cluster1_path)
    drop_cols = [col for col in ['target', 'cluster'] if col in cluster_0.columns]
    cluster_0 = cluster_0.drop(columns=drop_cols, errors='ignore')
    cluster_1 = cluster_1.drop(columns=drop_cols, errors='ignore')
    features = [col for col in imputed_input.columns if col in cluster_0.columns]
    centroid_0 = cluster_0[features].mean()
    centroid_1 = cluster_1[features].mean()
    dists_0 = np.linalg.norm(imputed_input[features] - centroid_0, axis=1)
    dists_1 = np.linalg.norm(imputed_input[features] - centroid_1, axis=1)
    assigned_clusters = np.where(dists_0 < dists_1, 0, 1)

    data_by_cluster = {0: [], 1: []}
    for idx, assigned_cluster in enumerate(assigned_clusters):
        if assigned_cluster == 0:
            features_file = 'top_features_cluster_0.csv'
        else:
            features_file = 'top_features_cluster_1.csv'
        top_features = pd.read_csv(features_file)['feature'].tolist()
        row = imputed_input.iloc[idx][top_features].copy()
        data_by_cluster[assigned_cluster].append(row.values.tolist() + [df_input.iloc[idx]['target']])

    # Tạo DataFrame cho từng cluster và train lại theo chuẩn step 4
    with open(log_file, "a") as logf:
        for cluster_id in [0, 1]:
            if cluster_id == 0:
                features_file = 'top_features_cluster_0.csv'
                model_path = 'catboost_model_cluster_0.cbm'
                cluster_name = "CatBoost filtered_cluster_0_balanced"
            else:
                features_file = 'top_features_cluster_1.csv'
                model_path = 'catboost_model_cluster_1.cbm'
                cluster_name = "CatBoost filtered_cluster_1_balanced"
            top_features = pd.read_csv(features_file)['feature'].tolist()
            rows = data_by_cluster[cluster_id]
            if not rows:
                logf.write(f"\n[{datetime.now()}] Cluster {cluster_id}: No valid sample found, skip retrain.\n")
                continue
            df_cluster = pd.DataFrame(rows, columns=top_features + ['target'])
            # Loại bỏ các dòng không có nhãn
            df_cluster = df_cluster[df_cluster['target'].notnull()]
            if len(df_cluster) == 0:
                logf.write(f"\n[{datetime.now()}] Cluster {cluster_id}: No valid label found, skip retrain.\n")
                continue
            train_catboost_and_report_save(df_cluster, cluster_name, model_path, logf=logf)
            logf.write("-" * 40 + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Cách dùng: python retrain.py <đường_dẫn_file_csv> <log_file.txt>")
        sys.exit(1)
    input_csv = sys.argv[1]
    log_file = sys.argv[2]
    retrain_from_csv(input_csv, log_file)