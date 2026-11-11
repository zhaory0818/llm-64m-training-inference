# 项目过程与产物索引

以下均以仓库根目录为起点。权重、数据、原始日志和虚拟环境保存在本机，不随源码交付；表中路径是本地资产位置，不承诺克隆后自动存在。

| 阶段 | 做了什么 | 主要产物（相对路径） |
|---|---|---|
| 环境/数据 | 独立环境、依赖适配、正式JSONL校验 | `.venv/`；`dataset/`；`docs/learning/SETUP_REPORT.md` |
| 预训练 | 从随机初始化训练2轮 | `logs/cloud_deploy/20261015_bitahub/completed/minimind/out/pretrain_768.pth`；同目录上级`checkpoints/`、原始日志 |
| SFT | 从自己的预训练初始化，完成2轮 | `out/full_sft_768.pth`；`logs/cloud_deploy/20261016_bitahub_sft/completed/minimind/` |
| 医疗LoRA | 冻结基座训练适配器，完成10轮 | `out/lora_medical_768.pth`；`out/merge_medical_768.pth`；`checkpoints/lora_medical_768_resume.pth` |
| DPO | 从通用SFT独立初始化，完成1轮 | `out/dpo_768.pth`；`checkpoints/dpo_768_resume.pth`；`logs/dpo_local/` |
| 蒸馏 | 作者MoE教师→自己的Dense学生，保存至5,400步后主动停止 | `logs/cloud_deploy/20261019_bitahub_distillation/completed/minimind-distillation/out/full_dist_768.pth`；其`checkpoints/` |
| GRPO | 3次实际更新 | `logs/rlaif_local/experience_grpo_256_20261028_192357/`：`out/`、`checkpoints/`、`result.json` |
| CISPO | 3次实际更新 | `logs/rlaif_local/experience_cispo_256_20261028_194226/` |
| PPO | 3批rollout，Actor/Critic各6次更新 | `logs/ppo_local/experience_256_20261028_200319/` |
| Agent RL | 最新一次独立体验3次更新 | `logs/agent_local/experience_256_20261029_125452/`；之前各次不是累计续训 |
| 导出 | 自己SFT转换为可加载目录并核验 | `docs/learning/generated/serving_20261030/sft_export/` |
| API/工具/WebUI | 普通/流式请求、执行工具、网页聊天 | `logs/serving_local/`；`docs/learning/SERVING_*_REPORT.md` |
| 自训模型诊断 | 36题双解码、完整输出、审阅与短文损失 | `logs/diagnosis_local/ability_20261031_110859/` |
| 作者Dense对照 | 固定发布权重，同条件诊断，独立实现和长度补测 | `logs/diagnosis_local/author_sft_20261031_132827/` |

## 必须区分的文件

- `out/full_sft_768.pth`：自己训练的通用SFT，SHA256 `2334de2f14733fffc4ef4f71fb1d318d5940a1a295bcdfb0549e49d17538946f`。
- `out/full_sft_768_moe.pth`：下载的作者MoE教师，**不是自己训练的MoE**。
- 作者Dense SFT：`docs/learning/generated/basic_diagnosis_20261031/author_reference_edba70ec/weights/full_sft_768.pth`，SHA256 `5b8c49f6c9d965092e651cafeeaeb8705558632b3fd4ac8ab319cbc5a3cbc4a0`，不是Zero文件。
- `lora_medical_768.pth`是适配器，必须配合对应基座；`merge_medical_768.pth`是合并完整模型，不能再重复叠加同一适配器。
- `*_resume.pth`含优化器、进度等续训状态；普通推理不需要加载这些大文件。

## 日志和曲线

本地完整报告入口为`docs/learning/FIRST_PART_SUMMARY.md`。各阶段图表原件在`docs/learning/generated/pretrain_20261015/`、`sft_20261016/`、`lora_20261018/`、`dpo_20261019/`；可分享副本集中在本目录`figures/`。

`logs/`保留终端、配置、CSV、SwanLab本地记录和代表性权重。清理后，技术试跑/失败运行和被后一次代表性体验替代的检查点已删除，但轻量证据保留；旧报告是历史记录，其中部分临时文件路径不再存在。删除项的原路径、大小、哈希和理由均见`docs/learning/generated/latest_closeout.json`所指目录的清理清单与执行回执。

## 数据和外部资产

训练数据位于`dataset/*.jsonl`，包括两份mini数据、医疗LoRA、DPO、RLAIF和Agent相关数据，按已有来源与许可证说明使用。奖励模型原目录在项目的同级`internlm2-1_8b-reward`，属于共享外部依赖，本次没有删除或搬动。

机器可读的主要资产信息见本目录`evidence/asset_manifest.json`。发布源码不会附带这些大文件，也不把下载的作者资产算作自己的训练产物。
