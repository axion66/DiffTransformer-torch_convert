# DiffTransformer-torch_convert

Turn their differential transformer[https://arxiv.org/pdf/2410.05258] into pytorch-dependent style for better understand

### Example

Here’s a simple example demonstrating how to use the `DiffAttn` layer:

```python
from layers import DiffAttn
import torch

# Example tensor with shape (batch_size, seq_len, embed_dim)
# In this case, we have a single batch of 2048 tokens, each represented by a 3072-dimensional embedding
tensor = torch.ones((1, 2048, 3072))

# Create an instance of the DiffAttn layer
model = DiffAttn(embed_dim=3072, layer_index=1)

output = model(tensor)

# Print the output shape
print(output.shape)
```

## Citation

@article{Ye2024DifferentialTransformer,
title = {Differential Transformer},
author = {Tianzhu Ye and Li Dong and Yuqing Xia and Yutao Sun and Yi Zhu and Gao Huang and Furu Wei},
year = {2024},
url = {https://arxiv.org/pdf/2410.05258},
}
