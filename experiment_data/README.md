# Downloading experiment data

Use `download_exp_data.conf.example` as a template for the SSH hosts, then run:

```sh
cp download_exp_data.conf.example download_exp_data.conf
./download_exp_data q1i_direct_gpu
```

The script syncs source data into `data/<experiment>/source`, the remote
`sink/bid-only` data into `data/<experiment>/sink`, and, when `gpu_host` is a hostname,
`/tmp/imputation-perf.csv` into `perf_data/<experiment>`.

Optional `gpu_perf` and `rdma_perf` hosts download `/tmp/perf_gpu.csv` and
`/tmp/rdma_perf_pre.csv` and `/tmp/rdma_perf_post.csv` into the same
performance-data directory. They can also be enabled for one invocation with
`-g HOST` and `-r HOST`. Pass `-external`
to use these external servers and skip the normal `gpu_host` performance file.

Use `-c FILE` for another config file or `-n` to preview the rsync operations.
