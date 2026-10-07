# FlinkE2C Artifact

Clone with submodules:

```sh
git clone --recursive https://github.com/maxhuettner/flinke2c-edbt.git
```

## Contents

| Path | Description |
|---|---|
| `flinke2c/` | FlinkE2C: modified Apache Flink (branch `flinke2c`) |
| `flinke2c_runtime/` | FlinkE2C external runtime (Rust, Java/Rust UDF execution) |
| `flinke2c_experiment_scripts/` | Environment setup (Terraform, Ansible, Docker) and experiment runs for Flink and NES; contains queries Q1–Q8 |
| `streaming_exp_management/` | Experiment scripts for QI and QF (filter update, crash, slowdown, parallel runs) |
| `streaming_tcp_harness/` | TCP source and sink used to feed and measure all systems |
| `CEPless/` | CEPless baseline |
| `experiment_data/` | Raw experiment data, notebooks and plotting scripts |

## Queries

Q1–Q8:
- Flink: `flinke2c_experiment_scripts/exp_management/queries/flink/`
- NES: `flinke2c_experiment_scripts/exp_management/queries/nes/`

QI and QF:
- Flink: `streaming_exp_management/flink/queries/`
- NES: `streaming_exp_management/nes/queries/`
