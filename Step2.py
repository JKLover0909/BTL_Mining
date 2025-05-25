import pandas as pd
import numpy as np

def assign_cluster_by_centroid(imputed_input_row, cluster0_path, cluster1_path):
    # Đọc dữ liệu các cluster
    cluster_0 = pd.read_csv(cluster0_path)
    cluster_1 = pd.read_csv(cluster1_path)

    # Loại bỏ cột 'target' nếu có
    drop_cols = [col for col in ['target', 'cluster'] if col in cluster_0.columns]
    cluster_0 = cluster_0.drop(columns=drop_cols, errors='ignore')
    cluster_1 = cluster_1.drop(columns=drop_cols, errors='ignore')

    # Lấy danh sách feature chung
    features = [col for col in imputed_input_row.columns if col in cluster_0.columns]

    # Tính centroid cho mỗi cluster
    centroid_0 = cluster_0[features].mean()
    centroid_1 = cluster_1[features].mean()

    # Lấy giá trị input
    input_values = imputed_input_row[features].iloc[0]

    # Tính khoảng cách Euclidean tới từng centroid
    dist_0 = np.linalg.norm(input_values - centroid_0)
    dist_1 = np.linalg.norm(input_values - centroid_1)

    assigned_cluster = 0 if dist_0 < dist_1 else 1
    print(f"Khoảng cách đến centroid cluster_0: {dist_0:.4f}")
    print(f"Khoảng cách đến centroid cluster_1: {dist_1:.4f}")
    print(f"Input được gán vào cluster: {assigned_cluster}")

    # Thêm cột cluster vào DataFrame và lưu lại
    imputed_input_row_with_cluster = imputed_input_row.copy()
    imputed_input_row_with_cluster['cluster'] = assigned_cluster
    imputed_input_row_with_cluster.to_csv('/home/ubuntu/1/BTL_Mining/example_imputed.csv', index=False)
    print("Đã lưu example_imputed.csv với cột cluster.")

    return assigned_cluster

if __name__ == "__main__":
    # Đọc input đã impute từ Step1
    imputed_input = pd.read_csv('/home/ubuntu/1/BTL_Mining/example_imputed.csv')

    cluster0_path = '/home/ubuntu/1/BTL_Mining/cluster_0.csv'
    cluster1_path = '/home/ubuntu/1/BTL_Mining/cluster_1.csv'

    assign_cluster_by_centroid(imputed_input, cluster0_path, cluster1_path)