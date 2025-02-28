import kagglehub

import kagglehub

# Download latest version
path = kagglehub.dataset_download("badasstechie/celebahq-resized-256x256")

print("Path to dataset files:", path)
print("Try this command: mv {path}/celeba_hq_256/ ./data/")