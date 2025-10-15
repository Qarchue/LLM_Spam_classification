

## 簡介

這是一個 LLM 垃圾信檢測模型的實作專案



---






## 展示

[result_1]: https://raw.githubusercontent.com/Qarchue/LLM_Spam_classification/blob/main/images/result_1.png
![實作結果][result_1] 


---


## 專案資料夾結構

```
LLM_Spam_classification/
    ├── gpt2/             = GPT-2 模型檔案（774M 目錄）
    ├── images/           = 展示圖片與結果
    ├── *.csv             = 訓練/驗證/測試資料（train.csv / validation.csv / test.csv）
    ├── review_classifier.pth = 訓練好的模型權重
    ├── sms_spam_collection/ = 原始資料集（SMSSpamCollection.tsv）
    ├── main.py           = 主程式，用於訓練模型
    ├── app.py           = 處理 Chainlit 前端
    ├── start.py          = 專案啟動腳本
    ├── requirements.txt  = pip 依賴
    ├── accuracy-plot.pdf = 訓練結果（accuracy）
    └── loss-plot.pdf     = 訓練結果（loss）
```

---



## 貢獻



程式撰寫: Qarchue

