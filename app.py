from pathlib import Path
import sys

import tiktoken
import torch
import chainlit

from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.ch06 import classify_review


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():

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

    tokenizer = tiktoken.get_encoding("gpt2")

    model_path = Path(".") / "review_classifier.pth"
    if not model_path.exists():
        print(
            f"Could not find the {model_path} file."
        )
        sys.exit()

    # 建立模型
    model = GPTModel(BASE_CONFIG)

    # 將模型改為分類器：在最後加一層線性輸出頭
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

    # 載入權重、搬到 device 並設為 eval 模式
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return tokenizer, model


# 取得供 Chainlit 使用的 tokenizer 與 model
tokenizer, model = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    Chainlit 的主處理函式：接收使用者輸入並回傳分類結果。
    """
    user_input = message.content

    # 使用 classify_review 將輸入分類（0/1），max_length 可視需要調整
    label = classify_review(user_input, model, tokenizer, device, max_length=120)

    # 把結果回傳到 Chainlit 前端介面
    await chainlit.Message(content=f"{label}").send()
