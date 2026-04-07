import gdown

url = "https://huggingface.co/yakhyo/face-parsing/resolve/main/79999_iter.pth"
gdown.download(url, "79999_iter.pth", quiet=False)
