import pandas as pd

# Đọc input đã fill missing value và đã có cột cluster
input_filled = pd.read_csv('/home/ubuntu/1/BTL_Mining/example_imputed.csv')

# Đọc giá trị cluster từ cột 'cluster'
assigned_cluster = int(input_filled['cluster'].iloc[0])

# Đọc danh sách feature quan trọng tương ứng
if assigned_cluster == 0:
    features_file = '/home/ubuntu/1/BTL_Mining/top_features_cluster_0.csv'
else:
    features_file = '/home/ubuntu/1/BTL_Mining/top_features_cluster_1.csv'

top_features = pd.read_csv(features_file)['feature'].tolist()

# Lọc các feature quan trọng và giữ lại cột cluster
columns_to_save = top_features + ['cluster']
extracted = input_filled[columns_to_save]

# Lưu ra file mới
extracted.to_csv('/home/ubuntu/1/BTL_Mining/example_extracted.csv', index=False)
print("Đã lưu các feature quan trọng và cột cluster vào example_extracted.csv")