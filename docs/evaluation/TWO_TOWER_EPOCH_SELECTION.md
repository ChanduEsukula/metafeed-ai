# MetaFeed AI — Two-Tower Epoch Selection

## Experiment

- Dataset: KuaiRec 2.0
- Training sample: 50,000 positive interactions
- Training hardware: NVIDIA Tesla T4
- Model: PyTorch two-tower retrieval
- Embedding dimension: 32
- Batch size: 64
- Seed: 42
- Training epochs: 10
- Selection metric: validation NDCG@10

## Best checkpoint

Epoch 7 was selected using validation NDCG@10.

| Metric | Popularity | Two-Tower Epoch 7 |
|---|---:|---:|
| Recall@10 | 0.00119555 | 0.00134688 |
| NDCG@10 | 0.00819479 | 0.00877137 |
| HitRate@10 | 0.08922249 | 0.08044186 |

Evaluated users: 7,061.

NDCG@10 improved approximately 7.04% relative to popularity.
Recall@10 improved approximately 12.66%.
HitRate@10 remained lower than popularity.

These are validation results, not final test results.
Statistical significance has not been established.

## Reproducibility

Best checkpoint SHA-256:
6b04ce960ebf5248737ec2809549ff5088b69597eac5107e20633623275e59d9

Sample SHA-256:
60000f6cd4af64d17c8dca4c5c907115ed6232367a8a22922bd4bde624496dd7

The local experiment archive is stored in artifacts/colab/
and intentionally excluded from Git.

## Next steps

- Run a bounded 500K-positive-interaction training experiment.
- Select checkpoints using validation data.
- Evaluate the final selected configuration on the held-out test split.
- Develop and benchmark the deployed recommendation API.
- Investigate TorchRec after establishing the PyTorch baseline.
