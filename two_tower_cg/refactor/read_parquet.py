import pandas as pd

file_path = r'd:\Study\UNIVERSITY\THIRD YEAR\Business Analytics\final assignment\hm_recsys_core\two_tower_cg\refactor\output\inference\inference_results.parquet'
df = pd.read_parquet(file_path)
print(df.head())
