"""
빠른 디버깅을 위해 작은 데이터셋 생성
100개 사이트, t=1~20 (train), t=21~22 (test)
"""
import pandas as pd
import numpy as np

# 2a_7 데이터 로드
train_df = pd.read_csv('data/2a/2a_7_train.csv')
test_df = pd.read_csv('data/2a/2a_7_test.csv')

print(f"Original train: {len(train_df)} rows, time range: {train_df['t'].min()}-{train_df['t'].max()}")
print(f"Original test: {len(test_df)} rows, time range: {test_df['t'].min()}-{test_df['t'].max()}")

# 전체 데이터 합치기
all_df = pd.concat([train_df, test_df], ignore_index=True)
print(f"Combined: {len(all_df)} rows")

# 사이트 수 줄이기 (100개만)
unique_coords = all_df[['x', 'y']].drop_duplicates()
print(f"Total unique sites: {len(unique_coords)}")

sampled_coords = unique_coords.sample(n=min(100, len(unique_coords)), random_state=42)
print(f"Sampled sites: {len(sampled_coords)}")

# 샘플링된 좌표만 필터링
all_small = all_df.merge(sampled_coords, on=['x', 'y'])

# 시간 분할: t=1~20 (train), t=21~22 (test)
train_small = all_small[all_small['t'] <= 20].copy()
test_small = all_small[(all_small['t'] >= 21) & (all_small['t'] <= 22)].copy()

print(f"\nSmall train: {len(train_small)} rows, time range: {train_small['t'].min()}-{train_small['t'].max()}")
print(f"Small test: {len(test_small)} rows, time range: {test_small['t'].min()}-{test_small['t'].max()}")
print(f"Sites in train: {train_small[['x', 'y']].drop_duplicates().shape[0]}")
print(f"Sites in test: {test_small[['x', 'y']].drop_duplicates().shape[0]}")

# 저장
train_small.to_csv('data/2a/2a_7_train_small.csv', index=False)
test_small.to_csv('data/2a/2a_7_test_small.csv', index=False)

print("\n✓ Small datasets saved!")
print("  - data/2a/2a_7_train_small.csv")
print("  - data/2a/2a_7_test_small.csv")
