import requests
import zipfile
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.ch05 import download_and_load_gpt2, load_weights_into_gpt


def download_and_unzip_spam_data(
    url,
    zip_path,
    extracted_path,
    data_file_path,
):
    """ 下載並解壓縮 SMS Spam Collection 資料集，並將其轉換為 tsv 格式。"""
    if data_file_path.exists():
        print(f"{data_file_path} 資料已存在，跳過下載步驟。")
        return

    # 下載資料
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    
    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    # 解壓縮資料
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # 把資料變成 tsv 格式
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"資料已下載並儲存於 {data_file_path}")


def create_balanced_dataset(df: pandas.DataFrame) -> pandas.DataFrame:
    """創建一個平衡的資料集，使 "ham" 和 "spam" 類別的數量相等。"""
    
    # 取得 'spam' 的資料集
    spam_subset = df[df["Label"] == "spam"]
    
    # 取得 'ham' 的資料集
    ham_subset = df[df["Label"] == "ham"]
    
    # 取得 'spam' 的數量
    num_spam = spam_subset.shape[0]
    
    # 隨機抽樣 'ham' 範例，使其數量與 'spam' 相等
    ham_subset = ham_subset.sample(num_spam, random_state=123)

    # 合併 'ham' 和 'spam' 為新的資料集
    balanced_df = pandas.concat([ham_subset, spam_subset])

    return balanced_df


def random_split(
    df: pandas.DataFrame,
    train_frac: float,
    validation_frac: float,
) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    """
    隨機將 DataFrame 分割為訓練集、驗證集和測試集。
    參數:
        df: 要分割的 DataFrame。
        train_frac: 訓練集所佔的比例 (0 < train_frac < 1)。
        validation_frac: 驗證集所佔的比例 (0 < validation_frac < 1)，
                        且 train_frac + validation_frac < 1。
    回傳: 三個 DataFrame，分別為訓練集、驗證集和測試集。
    """
    
    # 隨機排列整個 DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # 計算分割索引 也就是依照資料量與比例來決定分割點
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # 分割資料集
    train_df = df[:train_end]
    """訓練集"""
    validation_df = df[train_end:validation_end]
    """驗證集"""
    test_df = df[validation_end:]
    """測試集"""

    return train_df, validation_df, test_df


class SpamDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        tokenizer: tiktoken.Encoding,
        max_length: int = None,
        pad_token_id: int = 50256,
    ):
        """
        資料集初始化
        參數:
            csv_file: 包含資料的 CSV 檔案路徑。
            tokenizer: 用於編碼文字的分詞器。
            max_length: 序列的最大長度。如果為 None，則使用資料中最長的序列長度。
            pad_token_id: 用於填充的標記 ID。
        """
        # 讀取 CSV 檔案
        self.data = pandas.read_csv(csv_file)
        
        # 編碼文字資料
        self.encoded_texts: list[list[int]] = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # 如果序列長度超過 max_length，則截斷序列
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # 將不夠長的序列填充到最長序列
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
        
    def _longest_encoded_length(self):
        """取得資料集中最長的編碼序列長度。"""
        return len(max(self.encoded_texts, key=len))
    

    def __getitem__(self, index):
        encoded: list[int] = self.encoded_texts[index]
        """編碼成數字串列後的一段文字"""
        label = self.data.iloc[index]["Label"]
        """標籤 代表這段文字是垃圾訊息還是正常訊息"""
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)



def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """計算各資料集的分類準確度。"""
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device):
    """計算單一批次的交叉熵損失。"""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """計算分類損失。"""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """評估模型在訓練集和驗證集上的損失。"""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    device, 
    num_epochs,
    eval_freq, 
    eval_iter,
):
    """微調 GPT 模型以進行分類任務。"""
    # 初始化列表
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # 主訓練循環
    for epoch in range(num_epochs):
        model.train()  # 將模型設定為訓練模式

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 重置上一次批次迭代的損失梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # 計算損失梯度
            optimizer.step()  # 使用損失梯度更新模型權重
            examples_seen += input_batch.shape[0]  # 新功能: 追蹤範例而不是標記
            global_step += 1

            # 可選評估步驟
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 計算每個時期後的準確率
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 繪製訓練和驗證損失與時期的關係圖
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="微調 GPT 模型以進行分類"
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help=("This flag runs the model in test mode for internal testing purposes. "
              "Otherwise, it runs the model as it is used in the chapter (recommended).")
    )
    args = parser.parse_args()

    ########################################
    # 下載並準備資料集
    ########################################

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    try:
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    except (requests.exceptions.RequestException, TimeoutError) as e:
        print(f"Primary URL failed: {e}. Trying backup URL...")
        url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

    df = pandas.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)

    ########################################
    # 建立資料載入器
    ########################################
    
    # 使用 GPT-2 的分詞器
    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = SpamDataset(
        csv_file="train.csv",
        max_length=None,
        tokenizer=tokenizer,
    )

    val_dataset = SpamDataset(
        csv_file="validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer,
    )

    test_dataset = SpamDataset(
        csv_file="test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer,
    )

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    ########################################
    # 載入預訓練模型
    ########################################

    # 用於測試的小型 GPT 模型
    if args.test_mode:
        BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": 120,
            "drop_rate": 0.0,
            "qkv_bias": False,
            "emb_dim": 12,
            "n_layers": 1,
            "n_heads": 2
        }
        model = GPTModel(BASE_CONFIG)
        model.eval()
        device = "cpu"

    # Code as it is used in the main chapter
    else:
        CHOOSE_MODEL = "gpt2-large (774M)"
        INPUT_PROMPT = "Every effort moves"

        BASE_CONFIG = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 1024,  # Context length
            "drop_rate": 0.0,        # Dropout rate
            "qkv_bias": True         # Query-key-value bias
        }

        model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

        assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
            f"Dataset length {train_dataset.max_length} exceeds model's context "
            f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
            f"`max_length={BASE_CONFIG['context_length']}`"
        )

        model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

        model = GPTModel(BASE_CONFIG)
        load_weights_into_gpt(model, params)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ########################################
    # 修改預訓練模型
    ########################################

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(123)

    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
    model.to(device)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    ########################################
    # 微調修改後的模型
    ########################################

    start_time = time.time()
    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    
    print(f"已完成訓練，耗時 {execution_time_minutes:.2f} 分鐘。")

    ########################################
    # 繪製結果
    ########################################

    # 損失圖
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

    # 準確度圖
    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

    torch.save(model.state_dict(), "review_classifier.pth")