# CaMRec

CaMRec 是 MultiModalRecommendation 项目中的多模态推荐模型，结合协同过滤、图结构和跨模态注意力以整合视觉与文本特征。本文档介绍环境配置、数据准备、模型运行以及超参数搜索的完整流程。

## 目录结构速览
- `src/`：训练脚本、模型定义（`models/camrec.py`）、配置文件与工具函数。
- `data/`：默认放置已对齐的多模态特征与交互数据，子目录按数据集划分（如 `baby/`、`sports/`）。
- `preprocessing/`：从原始 Amazon/MicroLens 数据构建上述特征的脚本与 Notebook。
- `evaluation/`、`reports/`：评估脚本与论文报告（可选）。

## 环境配置
推荐使用 Conda 管理依赖（提供了 GPU 版本的 PyTorch 与图学习库），也可使用 `pip` 快速安装核心依赖。

### Conda（推荐）
```bash
conda env create -f src/environment.yml
conda activate Graph
```
- `environment.yml` 默认生成名为 `Graph` 的 Python 3.10 环境，包含 PyTorch 2.4、PyG、DGL 等依赖。
- 如果需要自定义 CUDA 版本，可修改文件中的 `pytorch`, `pytorch-cuda` 等条目后再创建环境。

### Pip/Virtualenv（轻量）
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
- `requirements.txt` 需 GPU/图学习支持，请额外安装与本机匹配的 PyTorch、torch-geometric、dgl 等包。
- 运行脚本前，请确保当前 Shell 的 `PYTHONPATH` 包含 `src/`，例如：`export PYTHONPATH=$(pwd)/src:$PYTHONPATH`。

## 数据准备
CaMRec 依赖已经提取好的视觉与文本特征。项目默认提供 Amazon 子集（Baby/Clothing/Sports）


### 获取现成数据集
1. 下载官方整理的数据：
   - Amazon Baby/Clothing/Sports：<https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing>
   - MicroLens 视频推荐：<https://drive.google.com/drive/folders/14UyTAh_YyDV8vzXteBJiy9jv8TBDK43w?usp=drive_link>
2. 解压后放入 `data/` 目录，形成如下结构（以 `baby` 数据集为例）：
   ```text
   CaMRec/
   ├── data/
   │   └── baby/
   │       ├── baby.inter           # 交互文件，列包含 userID/itemID/x_label 等
   │       ├── image_feat.npy       # 预提取图像特征
   │       ├── text_feat.npy        # 预提取文本特征
   │       ├── user_graph_dict.npy  # （可选）用户侧图结构
   │       ├── item_graph_dict.npy  # （可选）物品侧图结构
   │       └── ...                  # 首次训练时会生成 image_adj_*.pt、text_adj_*.pt 等缓存
   ```
3. 配置文件 `src/configs/dataset/<dataset>.yaml` 会描述文件名、字段名等信息，如需自定义文件名请同步修改。
4. 默认交互文件中的 `x_label` 字段用于划分训练/验证/测试集（0/1/2）。若希望过滤冷启动用户，可在 `overall.yaml` 中控制 `filter_out_cod_start_users`。

### 使用自定义数据
1. 根据 `preprocessing/README.md` 提供的流程依次执行 Notebook/脚本，完成数据清洗、划分、特征抽取与 ID 重映射。
2. 将生成的 `*.inter`、`*feat.npy` 与图结构文件放入新的 `data/<your_dataset>/` 目录，并新增对应的 `src/configs/dataset/<your_dataset>.yaml`。
3. 运行 CaMRec 时使用 `-d <your_dataset>` 指定新数据集名称。

## 运行 CaMRec 模型
1. 切换到源码目录并确保环境已激活：
   ```bash
   cd src
   export PYTHONPATH=$(pwd):$PYTHONPATH
   ```
2. 启动训练（示例：在 GPU 0 上训练 `baby` 数据集）：
   ```bash
   python main.py -m CAMREC -d baby -g 0
   ```
   - `-m/--model`：模型名称，CaMRec 对应 `CAMREC`（需与 `configs/model/` 文件名一致）。
   - `-d/--dataset`：数据集标识，需在 `configs/dataset/` 中存在对应 YAML。
   - `-g/--gpu`：CUDA 设备编号；如需在 CPU 上运行，可在 `src/configs/overall.yaml` 中将 `use_gpu` 设为 `False`。
3. 训练日志会写入 `src/log/`，模型检查点默认保存在 `src/saved/`（可通过 `checkpoint_dir` 配置修改）。
4. 训练完成后，日志会给出在验证集表现最优的超参数组合及其对应的测试指标。

## 超参数搜索
CaMRec 的超参数由 `src/configs/model/CAMREC.yaml` 管理，并通过 `utils/quick_start.py` 实现网格搜索。

1. **编辑待搜索参数**：在 YAML 中将超参数写成列表形式，例如：
   ```yaml
   learning_rate: [0.0005, 0.001]
   diff_weight: [0.25, 0.5]
   timesteps: [10, 20]
   ```
2. **声明搜索空间**：确保这些键包含在文末的 `hyper_parameters` 列表中，例如：
   ```yaml
   hyper_parameters: ["diff_weight", "ssl_weight", "w1", "w2", "walk_length", "timesteps", "learning_rate", "choosing_tmp"]
   ```
   只会对列表中出现的参数执行笛卡尔积组合；将某个值改成单元素列表即可固定该超参数。
3. **启动训练**：执行与普通训练相同的命令。`quick_start` 会遍历所有组合并记录最佳验证/测试结果。
4. **查看结果**：训练日志中包含每组组合的指标统计，以及最终的最佳配置；可将日志中的最佳参数回填到 YAML 中以固定该配置。

## 常见排查
- **依赖冲突**：若 Conda 与 pip 版本差异较大，建议完全使用 `src/environment.yml`。升级 PyTorch 时请同步升级 `torch-geometric` 与 `torch-scatter` 等包。
- **数据路径错误**：确保从 `src/` 目录运行脚本；如需从仓库根目录执行，可显式传入 `--data_path`（或在 `overall.yaml` 中修改）。
- **缓存图不匹配**：更改 `image_knn_k`/`text_knn_k` 后，删除旧的 `image_adj_*`、`text_adj_*` 文件以触发重新构建。

欢迎在 Issues 中反馈使用过程中的问题或改进建议。

## 一键 Demo 体验
项目提供 `demo/camrec_demo.py`，整合了数据巡检、参数配置与训练流程，搭配 Rich 彩色终端输出，便于快速展示 CAMREC 的训练全貌。

```bash
python demo/camrec_demo.py --dataset baby --gpu 0 --epochs 5
```

- 若使用 CPU，可设置 `--gpu -1`，脚本会自动切换到非 GPU 模式。
- 默认仅训练少量轮次并不保存权重，可通过 `--save-model` 开启模型持久化。
- 运行结束后，脚本会提示最新日志路径，便于查看完整的指标与训练细节。
