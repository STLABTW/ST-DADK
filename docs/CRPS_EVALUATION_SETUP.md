# CRPS 評估設定：論文 vs 我們的實作

## 論文 / 競賽的 CRPS 是在什麼設定下報的？

- **KAUST Competition 說明**（搜尋結果）：  
  *"Test sets used a **10% observation split** for validation."*  
  即：**同一批資料裡，10% 當「觀測」、其餘當「測試」**，不是用「不同時間」切 train/test。

- **本 repo 的 Table 4.4**（`run_table_4_4.py`）：
  - 使用 **`data_file = 'data/2b/2b_8.csv'`**（單一完整檔），**沒有**用 `2b_8_train.csv` / `2b_8_test.csv`。
  - **`obs_ratio = 0.1`**：每個時間點（或整體）約 10% 當觀測、90% 當 test。
  - 也就是：**同一時間軸上，10% 觀測 / 90% 當 test 的「空間插值」設定**。

因此：**論文裡的 CRPS（約 0.08–0.17、Table 4.4）是在「10% 觀測 / 90% test」這套設定下報的**，不是「前 90 步 train、後 10 步 test」的時序預測。

---

## 兩套設定的對照

| 項目 | 論文 / Table 4.4（我們對齊的） | Provider train/test（2b_8_train + 2b_8_test） |
|------|--------------------------------|-----------------------------------------------|
| 資料 | 單檔 `2b_8.csv`（全時段） | `2b_8_train.csv` + `2b_8_test.csv` |
| Train | 10% 觀測（同一批時間） | t=1..90 全部 |
| Test | 90% 未觀測（同一批時間） | t=91..100 全部 |
| 任務 | 空間插值（同時間補點） | 時序預測（預測未來 10 步） |
| 我們實作 | `use_provider_split=False`，`obs_ratio=0.1` | `use_provider_split=True` |
| 我們 CRPS | 最佳約 **~0.13**（接近論文） | **~0.72**（較難，數字不能直接比） |

---

## 結論

- **要對齊論文的 CRPS（~0.08）**：用 **10% obs / 90% test** 那套（單檔 `2b_8.csv` + `obs_ratio=0.1`，不要開 `use_provider_split`）。
- **Provider 的 train/test 檔**：是「前 90 步 vs 後 10 步」的時序切分，和論文報 CRPS 的設定不同；在這套下 CRPS 較高是預期現象。
