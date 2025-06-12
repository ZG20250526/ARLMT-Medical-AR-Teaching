from llava.train.train import train
# from llava.train.train20240827 import train
if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
