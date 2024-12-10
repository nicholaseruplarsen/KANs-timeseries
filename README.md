# DLinear

Change `num_workers` from `10` to `0` for the repo to work out of the box.
```python
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
```

Download the most commonly used time series datasets for benchmarking from https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy or https://github.com/thorhojhus/ssl_fts/tree/main/data.