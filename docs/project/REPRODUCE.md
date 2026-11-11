# 运行与复现说明

## 1. 环境与资产准备

已实际使用：Windows Python3.12.7、torch2.6.0+cu124、transformers4.57.6、datasets3.6.0、tokenizers0.22.2、numpy1.26.4、pyarrow20.0.0。云端为Python3.11/PyTorch2.6.0的4090镜像，环境细节并非与Windows完全相同。

本机现有环境入口是`.venv/Scripts/python.exe`。不要为再次演示重装系统CUDA或修改驱动。干净机器需按适用系统建立独立环境、安装匹配的PyTorch构建，再参照`environment/`记录配置；冻结清单是已装环境记录，不能保证任意操作系统直接安装全部包都成功。本次没有重新搭建并验证一台全新机器。

源码不携带模型和数据。自己的通用SFT应复制到`out/full_sft_768.pth`，并核对[资产索引](ARTIFACTS.md)中的SHA。作者公开模型可从[官方发布仓库](https://huggingface.co/jingyaogong/minimind-3-pytorch)另行获取，但其效果不等于本项目自己的模型，必须标注来源。

## 2. 本机演示命令

在仓库根目录运行；以下均为推理，不会更新模型。

```powershell
# 仅严格加载检查
.\.venv\Scripts\python.exe -X utf8 tools/project_demo.py --check-only --device cpu

# 通用SFT，固定贪心；采样体验可另加 --sample
.\.venv\Scripts\python.exe -X utf8 tools/project_demo.py --prompt "请介绍一下自己。" --max-new-tokens 128

# 医疗适配器：与原通用SFT一起加载
.\.venv\Scripts\python.exe -X utf8 tools/project_demo.py --adapter out/lora_medical_768.pth --prompt "介绍一下健康生活习惯。" --max-new-tokens 128

# DPO完整权重
.\.venv\Scripts\python.exe -X utf8 tools/project_demo.py --checkpoint out/dpo_768.pth --prompt "你好。" --max-new-tokens 128

# 预训练续写：使用与聊天不同的输入方式
.\.venv\Scripts\python.exe -X utf8 tools/project_demo.py --checkpoint logs/cloud_deploy/20261015_bitahub/completed/minimind/out/pretrain_768.pth --mode pretrain --prompt "人工智能是" --max-new-tokens 128

# 本地API：复用原项目路由，127.0.0.1监听；Ctrl+C停止
.\.venv\Scripts\python.exe -X utf8 tools/project_demo.py --serve --port 8998
```

Linux环境把解释器改成`.venv/bin/python`。新入口会按脚本位置确定仓库根目录，先加载datasets以适配Windows Arrow DLL顺序，原生模型严格加载；支持的范围是本项目Dense768/8层，不是所有模型格式的通用加载器。

另一终端可测试API：

```powershell
$requestBody = @{model='minimind'; messages=@(@{role='user';content='你好。'}); max_tokens=128; stream=$false} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8998/v1/chat/completions -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($requestBody))
```

HTTP成功只表示接口工作，不代表回答正确。API返回tool_calls不等于程序已执行工具；历史工具闭环由客户端执行和回填。

## 3. 已有WebUI与导出体验

在本机原工作区可运行原先已验证的：

```powershell
.\.venv\Scripts\python.exe -X utf8 docs/learning/setup_helpers/run_webui_local.py
```

它依赖本机保留的导出目录及验证记录，不是干净源码包开箱即有的资产。WebUI直接加载导出模型，不经过API；历史工具体验和最大轮数说明见原服务报告。轻量发布副本保留上游`scripts/web_demo.py`，不打包带本机路径的生成网页代码；公开运行时需按上游说明放入兼容模型并核对生成长度设置。

## 4. 历史正式训练配置

原生训练代码在`trainer/`，数据在`dataset/`。下列是实际关键参数记录，不是建议立即重新训练：

| 阶段 | 数据/轮数 | 关键设置 |
|---|---|---|
| 预训练 | pretrain_t2t_mini，2轮 | batch32，accumulation8，seq768，BF16，lr5e-4，workers8，seed42 |
| SFT | sft_t2t_mini，2轮 | batch16，accumulation1，seq768，BF16，lr1e-5，workers8，seed42；从自己的预训练开始 |
| 医疗LoRA | 25,276条，10轮 | rank16，393,216个可训练参数；实际其余配置以保留记录为准 |
| DPO | 17,166对，1轮 | 实际配置和偏好对处理以阶段记录为准，不能把预训练/SFT参数机械套用 |

云端预训练原命令为在`trainer/`下执行`../.venv/bin/python -u train_pretrain.py --max_seq_len 768 --use_wandb`；SFT为`../.venv/bin/python -u train_full_sft.py --use_wandb`。这些依赖当时固定代码与默认值，复跑时应把配置显式核对，而非依赖未来版本默认不变。`--use_wandb`在本项目记录中实际连接SwanLab适配。

重新训练前另建输出目录，避免覆盖`out/`已有成果。短程RL为观察机制做过batch、候选数、生成长度、调度长度及奖励模型CPU放置调整，不能称全部原默认正式长训练。

## 5. 清理后的行为

Arrow数据缓存已清理，下次加载完整JSONL会重新生成缓存，首次耗时与磁盘写入增加；数据本身仍在。试跑权重和重复上传压缩包不再保留，不能直接用旧的临时路径恢复试跑；正式和代表性体验的普通权重及resume保留。

## 6. 无需模型的诊断评分复核

交付副本包含冻结题集、原始回答及原严格评分器，可以离线重算，不需要模型或训练数据：

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/score_diagnosis.py
.\.venv\Scripts\python.exe -X utf8 tools/score_diagnosis.py --answers docs/project/evidence/basic_diagnosis/author_sft_answers.json
```

这复核的是固定原始输出的严格评分，不是重新生成回答，也不自动重评0/1/2开放审阅。完整训练、资源和监督审计的历史辅助程序保留在本机学习档案，不全部混入轻量发布副本。
# 效率开关与复跑（2026-10-31新增）

CLI及HTTP/SSE共用显式生成设置，未加开关时保持原行为：

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/project_demo.py --cached-sdpa --last-logits --prompt "请介绍一下自己。" --max-new-tokens 128
.\.venv\Scripts\python.exe -X utf8 tools/project_demo.py --serve --cached-sdpa --last-logits --port 8998
```

原SFT入口可追加`--dynamic_padding 1 --pad_to_multiple 8`，默认0。它只是可选数据整理，完整重训仍需另定输出、资源和检查点策略；本次性能实验没有覆盖正式权重。

完整基准需要自己的SFT、医疗LoRA及合并权重、官方SFT JSONL和CUDA。以下会新建隔离实验，不是日常体验必需步骤；新抽样会重新记录哈希和行号，精确复核本次结果请使用保留的冻结数据与CSV：

```powershell
.\.venv\Scripts\python.exe -X utf8 experiments/resume_evidence/prepare.py --data-path dataset/sft_t2t_mini.jsonl
.\.venv\Scripts\python.exe -X utf8 experiments/resume_evidence/benchmark_inference.py
.\.venv\Scripts\python.exe -X utf8 experiments/resume_evidence/benchmark_training.py
.\.venv\Scripts\python.exe -X utf8 experiments/resume_evidence/benchmark_cached_sdpa.py
.\.venv\Scripts\python.exe -X utf8 experiments/resume_evidence/summarize.py
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p "test_*.py" -v
```

先准备再顺序执行，避免GPU任务同时争用。基准使用本项目固定SHA的个人权重；换权重应建立新协议并更新哈希，不能静默沿用旧成绩。数据冻结文件、模型和完整JSONL不随轻量源码发布。

