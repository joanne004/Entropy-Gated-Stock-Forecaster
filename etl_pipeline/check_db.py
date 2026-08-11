import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://stockuser:stockpass@localhost:5432/stockdb")

print("price_features:  ", pd.read_sql("SELECT COUNT(*) FROM price_features", engine).iloc[0,0], "rows")
print("sentiment_scores:", pd.read_sql("SELECT COUNT(*) FROM sentiment_scores", engine).iloc[0,0], "rows")
print("fused_features:  ", pd.read_sql("SELECT COUNT(*) FROM fused_features", engine).iloc[0,0], "rows")
