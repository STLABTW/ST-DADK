# Grid Search 실험 시스템 사용 가이드

## 개요

모든 하이퍼파라미터 조합에 대해 자동으로 실험을 수행하고, 결과를 구조화된 CSV 파일로 저장하는 시스템입니다.

## 구조 변경 사항

### 이전 방식
- 32개의 개별 config 파일 생성 (config_1.yaml ~ config_32.yaml)
- 각 config를 순차적으로 실행
- 결과가 개별 폴더에 분산 저장
- 수동으로 결과 수집 및 정리 필요

### 새로운 방식
- **단일 base config 파일** (config_st_interp.yaml)
- **파라미터 그리드 정의** (run_grid_search.py에서)
- **자동 조합 생성 및 실행**
- **통합 CSV 파일 출력**:
  - `grid_search_summary.csv`: 각 config의 요약 통계
  - `grid_search_detail.csv`: 각 iteration의 raw 값
  - `grid_search_configs.csv`: 전체 config 정보

## 파일 구조

```
ST-DADK/
├── configs/
│   └── config_st_interp.yaml          # Base configuration
├── scripts/
│   ├── train_st_interp.py             # 기존 학습 스크립트 (수정됨)
│   └── run_grid_search.py             # 새로운 Grid Search 러너
└── results/
    └── YYYYMMDD_grid_search/
        ├── grid_search_summary.csv     # 요약 통계
        ├── grid_search_detail.csv      # 상세 결과
        ├── grid_search_configs.csv     # Config 정보
        └── config001_uni_lrn_site_10_cor/
            ├── config.yaml
            ├── experiments/
            │   ├── 1/
            │   ├── 2/
            │   └── ...
            └── summary/
                ├── summary_statistics.json
                └── all_experiments.csv
```

## 사용 방법

### 1. 파라미터 그리드 정의

`scripts/run_grid_search.py`에서 실험하고 싶은 파라미터 조합을 정의:

```python
param_grid = {
    'spatial_init_method': ['uniform', 'gmm'],
    'spatial_learnable': [True, False],
    'obs_method': ['site-wise', 'random'],
    'obs_ratio': [0.1, 0.3],
    'obs_spatial_pattern': ['corner', 'uniform'],
}
```

### 2. 실험 실행

**순차 실행:**
```bash
python scripts/run_grid_search.py --config configs/config_st_interp.yaml
```

**병렬 실행 (권장):**
```bash
python scripts/run_grid_search.py \
    --config configs/config_st_interp.yaml \
    --parallel \
    --n_jobs 10
```

**출력 디렉토리 지정:**
```bash
python scripts/run_grid_search.py \
    --config configs/config_st_interp.yaml \
    --output_dir results/my_experiment \
    --parallel \
    --n_jobs 10
```

### 3. 결과 분석

#### Summary CSV (grid_search_summary.csv)

각 config의 평균 성능:

```python
import pandas as pd

df = pd.read_csv('results/20251203_grid_search/grid_search_summary.csv')

# Best performers
df_sorted = df.sort_values('test_rmse_mean')
print(df_sorted.head(10))

# Factor analysis
for method in df['spatial_init_method'].unique():
    subset = df[df['spatial_init_method'] == method]
    print(f"{method}: {subset['test_rmse_mean'].mean():.4f}")
```

#### Detail CSV (grid_search_detail.csv)

각 iteration의 raw 값:

```python
df_detail = pd.read_csv('results/20251203_grid_search/grid_search_detail.csv')

# Specific config analysis
config1 = df_detail[df_detail['config_id'] == 1]
print(config1[['experiment_id', 'test_rmse', 'test_mae']])

# Statistical tests
from scipy import stats
group1 = df_detail[df_detail['config_id'] == 1]['test_rmse']
group2 = df_detail[df_detail['config_id'] == 2]['test_rmse']
t_stat, p_value = stats.ttest_ind(group1, group2)
```

## 출력 파일 상세

### 1. grid_search_summary.csv

각 config당 1행, 요약 통계 포함:

| 컬럼 | 설명 |
|------|------|
| config_id | Config 번호 (1-32) |
| tag | Config 식별자 |
| spatial_init_method | 'uniform' or 'gmm' |
| spatial_learnable | True or False |
| obs_method | 'site-wise' or 'random' |
| obs_ratio | 0.1 or 0.3 |
| obs_spatial_pattern | 'corner' or 'uniform' |
| n_experiments | 반복 실험 수 |
| test_rmse_mean | Test RMSE 평균 |
| test_rmse_std | Test RMSE 표준편차 |
| test_rmse_min | Test RMSE 최소값 |
| test_rmse_max | Test RMSE 최대값 |
| test_rmse_median | Test RMSE 중앙값 |
| ... | (다른 metric들도 동일) |

### 2. grid_search_detail.csv

각 config의 각 iteration마다 1행:

| 컬럼 | 설명 |
|------|------|
| config_id | Config 번호 |
| tag | Config 식별자 |
| experiment_id | Iteration 번호 (1-10) |
| spatial_init_method | Config 설정 |
| ... | (다른 config 설정들) |
| test_rmse | 해당 iteration의 Test RMSE |
| test_mae | 해당 iteration의 Test MAE |
| test_mse | 해당 iteration의 Test MSE |
| valid_rmse | 해당 iteration의 Valid RMSE |
| ... | (다른 metric들) |
| total_time_seconds | 학습 시간 (초) |

### 3. grid_search_configs.csv

전체 config 정보 저장 (JSON 형태):

| 컬럼 | 설명 |
|------|------|
| config_id | Config 번호 |
| tag | Config 식별자 |
| config_json | 전체 config (JSON 문자열) |

## 시각화 예시

### 1. Heatmap 비교

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('results/20251203_grid_search/grid_search_summary.csv')

# Pivot for heatmap
pivot = df.pivot_table(
    values='test_rmse_mean',
    index=['spatial_init_method', 'spatial_learnable'],
    columns=['obs_method', 'obs_ratio', 'obs_spatial_pattern']
)

plt.figure(figsize=(15, 6))
sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r')
plt.title('Test RMSE across all configurations')
plt.tight_layout()
plt.savefig('heatmap.png')
```

### 2. Factor 효과 비교

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/20251203_grid_search/grid_search_summary.csv')

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

factors = ['spatial_init_method', 'spatial_learnable', 
           'obs_method', 'obs_ratio', 'obs_spatial_pattern']

for ax, factor in zip(axes.flatten(), factors):
    data = df.groupby(factor)['test_rmse_mean'].agg(['mean', 'std'])
    data.plot(kind='bar', y='mean', yerr='std', ax=ax)
    ax.set_title(f'Effect of {factor}')
    ax.set_ylabel('Test RMSE')

plt.tight_layout()
plt.savefig('factor_effects.png')
```

### 3. Interaction Plot

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/20251203_grid_search/grid_search_summary.csv')

# uni_fix vs gmm_lrn by obs_spatial_pattern
df_unifix = df[(df['spatial_init_method']=='uniform') & (df['spatial_learnable']==False)]
df_gmmlrn = df[(df['spatial_init_method']=='gmm') & (df['spatial_learnable']==True)]

patterns = df['obs_spatial_pattern'].unique()
unifix_means = [df_unifix[df_unifix['obs_spatial_pattern']==p]['test_rmse_mean'].mean() 
                for p in patterns]
gmmlrn_means = [df_gmmlrn[df_gmmlrn['obs_spatial_pattern']==p]['test_rmse_mean'].mean() 
                for p in patterns]

plt.figure(figsize=(10, 6))
plt.plot(patterns, unifix_means, 'o-', label='uni_fix', linewidth=2)
plt.plot(patterns, gmmlrn_means, 's-', label='gmm_lrn', linewidth=2)
plt.xlabel('Observation Pattern')
plt.ylabel('Test RMSE')
plt.title('Interaction: Method × Observation Pattern')
plt.legend()
plt.grid(True)
plt.savefig('interaction_plot.png')
```

## 장점

1. **재현성**: 모든 실험이 동일한 base config에서 시작
2. **확장성**: 파라미터 추가가 쉬움 (param_grid만 수정)
3. **분석 용이**: 구조화된 CSV로 다양한 분석 가능
4. **병렬화**: 모든 config를 동시에 실행 가능
5. **추적성**: 모든 config와 결과가 자동으로 저장

## 추가 파라미터 실험하기

다른 파라미터를 추가하고 싶다면 `run_grid_search.py`에서:

```python
param_grid = {
    # 기존 파라미터
    'spatial_init_method': ['uniform', 'gmm'],
    'spatial_learnable': [True, False],
    
    # 새로운 파라미터 추가
    'lr': [1e-3, 5e-3, 1e-2],
    'hidden_dims': [[128, 128], [256, 256, 128]],
    'dropout': [0.0, 0.1, 0.2],
}
```

## 문제 해결

### 메모리 부족
```bash
# n_jobs 줄이기
python scripts/run_grid_search.py --parallel --n_jobs 5
```

### 특정 config만 재실행
```python
# Python에서 직접 실행
from scripts.train_st_interp import run_multiple_experiments
import yaml

with open('configs/config_st_interp.yaml') as f:
    config = yaml.safe_load(f)

config['spatial_init_method'] = 'gmm'
config['obs_ratio'] = 0.3
# ... 다른 설정

summary = run_multiple_experiments(config, 'results/rerun', 'cpu', parallel=True)
```
