# OPD (Open On-Policy Distillation)

这个仓库是在 `slime` 训练框架基础上裁剪出来的一套 OPD 复现实验代码，当前主入口是 [train.py](/idfsdata/yexuyan/OPD/train.py)，示例运行脚本是 [examples/run_opd_qwen3_14b_to_4b_science_bg.sh](/idfsdata/yexuyan/OPD/examples/run_opd_qwen3_14b_to_4b_science_bg.sh)。

## 本地 `slime` 环境补齐

这份仓库并不只依赖 `requirements.txt` 里的纯 Python 包，还依赖两个已经打过本地补丁的 editable 源码树：

- `SGLang`: `/idfsdata/yexuyan/slime_deps/sglang` at `24c91001cf99ba642be791e099d358f4dfe955f5`
- `Megatron-LM`: `/idfsdata/yexuyan/slime_deps/Megatron-LM` at `3714d81d418c9f1bca4594fc35f9e8289f652862`

如果这台机器上已经有一个可用的 `slime` Conda 环境，最稳妥的做法是直接克隆它，再把 editable 安装重新指回当前仓库和本地依赖源码：

```bash
bash scripts/bootstrap_slime_opd_env.sh
```

默认脚本会：

- 优先从 `/idfsdata/yexuyan/conda_envs/slime` 克隆到 `/idfsdata/yexuyan/conda_envs/slime-opd`
- 重新安装本地 editable 包：当前仓库、`sglang`、`Megatron-LM`
- 跑一次环境预检，确认模型、数据和关键依赖都能被当前 Python 解释器看到

如果目标环境已经存在，脚本会直接复用它并修正 editable 安装。

## 环境预检

只检查环境，不真正启动 teacher server 或训练：

```bash
bash examples/run_opd_qwen3_14b_to_4b_science_bg.sh --preflight
```

这个预检会确认：

- 当前 Conda 环境里能导入 `torch`、`ray`、`sglang`、`megatron.core`、`flash_attn`、`ring_flash_attn`
- `slime` 是否指向当前仓库
- `sglang` / `Megatron-LM` 是否指向本机的本地源码目录
- 示例默认使用的 teacher / student 模型和数据文件是否存在

## 正式启动示例

```bash
bash examples/run_opd_qwen3_14b_to_4b_science_bg.sh
```

脚本默认使用下面这些本机路径：

- Conda 环境：`/idfsdata/yexuyan/conda_envs/slime-opd`
- Teacher 模型：`/idfsdata/yexuyan/OPD/models/Qwen3-14B`
- Student 模型：`/idfsdata/yexuyan/OPD/models/Qwen3-4B`
- 数据：`/idfsdata/yexuyan/OPD/data/science_with_answer.jsonl`

如果你想覆盖默认值，可以在启动前设置环境变量，例如：

```bash
CONDA_ENV_PATH=/idfsdata/yexuyan/conda_envs/slime \
TEACHER_MODEL=/some/other/teacher \
STUDENT_MODEL=/some/other/student \
bash examples/run_opd_qwen3_14b_to_4b_science_bg.sh --preflight
```
