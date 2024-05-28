from llava.train.train_cot import train

if __name__ == "__main__":
    train_cot(attn_implementation="flash_attention_2")
