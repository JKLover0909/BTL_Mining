import pandas as pd
from catboost import CatBoostClassifier
import re

def normalize_col(col):
    # Chuẩn hóa: chữ thường, thay mọi ký tự không phải chữ/số thành _
    return re.sub(r'\W+', '_', col.lower())

# Đọc dữ liệu đã extract và cluster
input_extracted = pd.read_csv('/home/ubuntu/1/BTL_Mining/example_extracted.csv')

# Đọc giá trị cluster
assigned_cluster = int(input_extracted['cluster'].iloc[0])

# Chọn model tương ứng với cluster
if assigned_cluster == 0:
    model_path = '/home/ubuntu/1/BTL_Mining/catboost_model_cluster_0.cbm'
else:
    model_path = '/home/ubuntu/1/BTL_Mining/catboost_model_cluster_1.cbm'

# Load model CatBoost
model = CatBoostClassifier()
model.load_model(model_path)

# Lấy tên feature đúng từ model
model_features = model.feature_names_

# Chuẩn hóa tên cột input và tên feature model để ánh xạ
input_cols_norm = {normalize_col(col): col for col in input_extracted.columns}
model_features_norm = {normalize_col(feat): feat for feat in model_features}

# Tạo dict đổi tên cột input về đúng tên feature model
rename_dict = {}
for norm_feat, feat in model_features_norm.items():
    if norm_feat in input_cols_norm:
        rename_dict[input_cols_norm[norm_feat]] = feat

X = input_extracted.rename(columns=rename_dict)

# Loại bỏ cột 'cluster' nếu có
if 'cluster' in X.columns:
    X = X.drop(columns=['cluster'])

# Bổ sung các feature còn thiếu với giá trị 0
for feat in model_features:
    if feat not in X.columns:
        X[feat] = 0

# Đảm bảo đúng thứ tự cột
X = X[model_features]

# Dự đoán giá trị True/False
prediction = model.predict(X)[0]
print(f"Kết quả phân loại cho dữ liệu đầu vào: {bool(prediction)}")