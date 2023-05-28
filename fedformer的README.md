# DeepTD-LSP

DeepTD-LSP (Deep Temporal Decomposition Long-term Series Prediction) model is constructed based on an encoder-decoder structure and combined with an attention mechanism. It uses the frequency disassembly module for series disassembly, and splits the series into trend information, seasonal information and noise. The model is tested on three real-world datasets, and the experimental results show that the DeepTD-LSP model outperforms the other five new long-term time series prediction models.

## Frequency Disassembly
<!-- |![Figure1](https://user-images.githubusercontent.com/44238026/171341166-5df0e915-d876-481b-9fbe-afdb2dc47507.png)| -->
|![Figure1](./images/模型图.png)|
|:--:| 
| *Figure 1. Overall structure of FEDformer* |

<!-- |![image](https://user-images.githubusercontent.com/44238026/171343471-7dd079f3-8e0e-442b-acc1-d406d4a3d86a.png) | ![image](https://user-images.githubusercontent.com/44238026/171343510-a203a1a1-db78-4084-8c36-62aa0c6c7ffe.png) -->
|![image](./results/多变量.png) | ![image](./results/单变量.png)
|:--:|:--:|
| *Figure 2. Frequency Enhanced Block (FEB)* | *Figure 3. Frequency Enhanced Attention (FEA)* |


## Main Results
![image](https://user-images.githubusercontent.com/44238026/171345192-e7440898-4019-4051-86e0-681d1a28d630.png)


## Get Started

1. Install Python >=3.6,PyTorch 1.9.0.
2. Download data. You can obtain all the six benchmarks from [[Autoformer](https://github.com/thuml/Autoformer)] or [[Informer](https://github.com/zhouhaoyi/Informer2020)].
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the multivariate and univariate experiment results by running the following shell code separately:

```bash

```

