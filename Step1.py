import pandas as pd
from sklearn.impute import KNNImputer

# Đọc dữ liệu gốc (không có cột ID)
data_file = '/home/ubuntu/1/BTL_Mining/data_80_imputed_no_id.csv'
df = pd.read_csv(data_file)

# Đọc file input chứa 1 hàng cần fill missing value
input_file = '/home/ubuntu/1/BTL_Mining/example.csv'
input_row = pd.read_csv(input_file).iloc[0].to_dict()

# Đảm bảo input_row có đủ tất cả các cột, nếu thiếu thì thêm giá trị None
for col in df.columns:
    if col not in input_row:
        input_row[col] = None

# Chuyển input_row thành DataFrame 1 hàng, đúng thứ tự cột
input_df = pd.DataFrame([input_row])[df.columns]

# Ghép input vào cuối dữ liệu gốc để cùng impute
concat_df = pd.concat([df, input_df], ignore_index=True)

# KNNImputer với n_neighbors=5
imputer = KNNImputer(n_neighbors=5)
imputed = imputer.fit_transform(concat_df)

# Lấy lại hàng cuối cùng (input đã được impute)
imputed_input = pd.DataFrame([imputed[-1]], columns=df.columns)

# Lưu ra file mới
imputed_input.to_csv('/home/ubuntu/1/BTL_Mining/example_imputed.csv', index=False)
print("Đã lưu input sau khi KNNImputer xử lý missing value vào example_imputed.csv")