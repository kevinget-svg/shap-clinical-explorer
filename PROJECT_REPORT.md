# SHAP Clinical Explorer -- 项目总结报告

---

## 1. 项目概述

**项目名称：** 临床试验探索性分析（SHAP-based Clinical Trial Exploratory Analysis）

**核心目标：** 利用机器学习算法 -- 特别是 SHAP (SHapley Additive exPlanations) 值 -- 对临床试验数据进行探索性分析。通过 Beeswarm、Summary Bar、Dependence Plot、Waterfall、Force Plot 等可视化手段，揭示各个特征如何影响临床结局。项目的侧重点是**可解释性**：在临床/监管语境下解释模型的决策过程，而非单纯追求预测性能的最大化。

### 1.1 技术栈

| 层级 | 依赖 |
|------|------|
| 语言 | Python 3.12 |
| ML / SHAP | `shap >= 0.46`、`survshap >= 0.4`、`scikit-learn >= 1.5`、`xgboost >= 2.1` |
| 生存分析 | `scikit-survival >= 0.22` |
| 统计建模 | `statsmodels >= 0.14` |
| 数据 I/O | `pyreadstat`（SAS `.sas7bdat` / R `.RData`）、`openpyxl`（`.xlsx`）、`pandas` |
| 数值计算 | `numpy >= 2.0` |
| 可视化 | `matplotlib >= 3.9`（支持 300 DPI 出版级输出） |
| Web 应用 | `streamlit >= 1.40` |
| 代码质量 | `ruff`、`mypy`（strict 模式） |
| 测试 | `pytest >= 8.3`、`pytest-cov >= 5.0` |

### 1.2 四种终点类型

| 终点类型 | 适用模型 | 关键输出 |
|----------|----------|----------|
| **Continuous** (连续型) | Linear Regression、GLM (Gaussian)、Random Forest、XGBoost | Beeswarm、Dependence、Waterfall |
| **Binary** (二分类) | Logistic Regression、GLM (Binomial)、XGBoost Classifier | + ROC Curve、AUC |
| **Survival** (生存) | Cox Proportional Hazards、Random Survival Forest | + SurvSHAP(t) 时间依赖分析 |
| **Count** (计数/Poisson) | Poisson GLM、Negative Binomial、RF、XGBoost | + Observed vs Predicted 校准图 |

---

## 2. 技术架构

### 2.1 目录结构（`code/` 重命名后）

```
SHAP/
├── core/                   # 核心逻辑（原 code/，因 stdlib 冲突重命名）
│   ├── __init__.py
│   ├── data_loader.py      # 多格式临床数据读取（csv/xlsx/sas7bdat/RData）
│   ├── preprocessing.py    # ADaM 标准清洗、缺失值处理、特征工程
│   ├── modeling.py         # 模型训练、自动算法选择、超参数调优
│   ├── shap_analysis.py    # SHAP/SurvSHAP 值计算、特征重要性排序
│   ├── visualization.py    # 出版级可视化（Beeswarm/Bar/Dependence/Waterfall/RCT）
│   ├── synthetic_data.py   # 合成数据生成器（4 种终点 × 2 种试验设计）
│   ├── pipeline.py         # 命令行 pipeline 入口
│   └── streamlit_app.py    # Streamlit 交互式 Dashboard
├── shared/                 # 共享配置与常量
│   ├── __init__.py
│   ├── config.py           # 路径、随机种子、颜色方案、matplotlib 全局设置
│   └── constants.py        # 临床领域常量（ADaM 列名、终点到模型映射）
├── data/                   # 原始数据（只读，不纳入版本控制）
├── output/                 # 模型结果、SHAP 值 CSV、图表 PNG/SVG
├── tests/                  # 单元测试（66 个 case，5 个文件）
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_modeling.py
│   ├── test_shap_analysis.py
│   └── test_visualization.py
├── reference/              # 关键文献（SHAP 方法学、SurvSHAP、AKI 临床范例）
├── .streamlit/
│   └── config.toml         # Streamlit Cloud 主题与服务器配置
├── streamlit_app.py        # Streamlit Cloud 部署根入口（包装 core.streamlit_app）
├── runtime.txt             # Python 版本声明（3.12）
├── requirements.txt        # 精简依赖（Streamlit Cloud 部署用）
├── pyproject.toml          # 项目元数据、lint/mypy/pytest 配置
└── CLAUDE.md               # Claude Code 辅助开发指南
```

### 2.2 流水线架构

```
原始数据 (.sas7bdat / .RData / .xlsx / .csv)
    │
    ▼
[1] data_loader       → 读取原始数据，根据后缀选择 reader；支持 pyreadstat 元数据保留
    │
    ▼
[2] preprocessing     → 数字特征标准化、分类特征 One-Hot 编码、train/test 划分 (80/20)
    │
    ▼
[3] modeling          → 模型自动选择 + 小样本 (N<200) 自动超参数调优 (GridSearchCV)
    │
    ▼
[4] shap_analysis     → 模型特定 Explainer 选择 (Tree/Linear/Kernel/SurvSHAP)
    │
    ▼
[5] visualization     → Beeswarm / Bar / Dependence / Waterfall / RCT Comparison
    │
    ▼
output/ (*.csv, *.png, *.svg)
```

**关键设计决策：**
- 随机种子统一为 `42`（`shared.config.SEED`）
- 训练/测试原则上 80/20 划分，生存终点保持删失比例一致
- 小样本 (N<200) 自动触发 5 折分层交叉验证 + GridSearchCV
- 每张图同时输出 PNG (300 DPI) 和 SVG 两种格式

### 2.3 Streamlit App 架构

根入口文件 `streamlit_app.py` 是一个轻量级 wrapper，其核心逻辑仅 4 行：

```python
# streamlit_app.py -- Streamlit Cloud Community Cloud 入口
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
import core.streamlit_app  # side-effect import 执行所有 Streamlit UI 代码
```

真正的 app 逻辑在 `core/streamlit_app.py`，由以下组件构成：

**Sidebar（侧边栏）：**
- 数据源选择：Demo (Synthetic) / Upload File
- Demo 模式：终点类型、试验设计、模型选择下拉框、样本量滑块
- Upload 模式：文件上传组件（支持 csv/xlsx/sas7bdat/RData）
- "Run Analysis" 按钮 → 状态持久化到 `st.session_state.analysis_triggered`

**5 个 Tab 页面：**

| Tab | 名称 | 内容 |
|-----|------|------|
| Tab 1 | Overview | SHAP Beeswarm + Summary Bar + 终点特定面板（ROC/SurvSHAP/Count 校准图） |
| Tab 2 | Feature Dependence | 可交互选择的 Dependence Plot（目标特征 + 交互特征着色） |
| Tab 3 | Individual Explain | Force Plot（基于 shap.save_html 完整 JS 嵌入）+ Waterfall + SurvSHAP(t) 个体分解 |
| Tab 4 | RCT Comparison | 治疗组/对照组 SHAP 重要性对比 + Count 终点校准面板 |
| Tab 5 | Data Export | CSV 下载 (SHAP 值 / 特征重要性) + 全部 SVG 图打包 zip 下载 + 数据预览 |

**终点特定面板：** 根据当前终点类型动态显示不同视图：
- Binary: ROC 曲线（AUC 标注）
- Survival: SurvSHAP(t) 时间依赖面板 + 个体分解
- Count: Observed vs Predicted 校准图 + 面板

---

## 3. 核心功能

### 3.1 四种终点建模

`ModelTrainer` 类支持以下模型路由：

| 终点 | `model_type=auto` 时的默认模型 | 可用模型 |
|------|-------------------------------|----------|
| Continuous | XGBoost Regressor | `rf`、`xgb`、`glm` |
| Binary | XGBoost Classifier | `rf`、`xgb`、`glm` |
| Survival | Random Survival Forest | `rsf`、`cox` |
| Count | XGBoost Regressor (objective=`count:poisson`) | `rf`、`xgb`、`glm` |

XGBoost 作为"auto"默认选择，原因是在所有四种终点上性能鲁棒且 TreeSHAP 计算高效。当样本量 <SMALL_SAMPLE_THRESHOLD (200)，自动触发 GridSearchCV 超参数搜索。

### 3.2 SHAP 分析

`SHAPAnalyzer` 类实现三层解释架构：

| 层次 | 聚合方式 | 可视化 |
|------|----------|--------|
| **Global** | `mean(|SHAP|)` 跨所有测试样本 | Summary Bar Plot |
| **Cohort** | 按特征值分组后 SHAP 分布 | Beeswarm Plot、Dependence Plot |
| **Local** | 单样本的 φ_i 向量 | Force Plot、Waterfall Plot |

**算法选择决策：**

```
模型是树模型 (rf/xgb)？
    Yes → shap.TreeExplainer (interventional perturbation, O(TLD²))
    No ↓
模型是线性模型 (glm)？
    Yes → shap.LinearExplainer (解析解: β_i × (x_i - μ_i))
    No ↓
模型是生存模型？
    Yes → survshap.SurvSHAP (时间依赖采样估计)
    No ↓
KernelSHAP (模型无关, 较慢)
```

### 3.3 SurvSHAP(t) 生存分析

生存终点使用 `survshap` 包实现时间依赖的 SHAP 分析：

1. `SurvivalModelExplainer` 包装模型 + 训练背景数据
2. `ModelSurvSHAP` 以 `calculation_method="sampling"` 拟合（`treeshap` 在 scikit-survival 集成模型中存在可加性问题）
3. 输出 3D 数组 `(n_samples, n_features, n_times)`，通过时间轴聚合得到标准 2D SHAP 值
4. 额外提供 `survshap_values_` (3D) 和 `survshap_times_` (时间网格) 用于 SurvSHAP(t) 专属可视化

### 3.4 RCT 对比

当试验设计为 `RCT_TWO_ARM` 且数据包含 `ARM` 列时：

- `plot_rct_comparison` 函数将测试集按 ARM 分组（Treatment vs Control）
- 对每个组分别计算 `mean(|SHAP|)`，生成分组柱状图
- 用于揭示治疗组和对照组中特征重要性排名的差异
- Count 终点额外提供 `plot_count_panel` 校准面板

### 3.5 数据导出

- **SHAP 值 CSV：** 每个测试样本 × 每个特征的 SHAP 值矩阵
- **特征重要性 CSV：** 按 `mean(|SHAP|)` 降序排列
- **全部图形打包 (SVG zip)：** 一键生成所有可视化图表的 SVG 矢量文件，打包下载

---

## 4. 关键问题与解决方案

### 4.1 Python stdlib `code` 模块冲突（影响最大）

**问题现象：**
`from code.xxx import ...` 总是失败，提示找不到模块或导入的是 stdlib 的 `code` 模块。所有 `code.*` import 全部断掉。

**根因：**
Python 3.12+ 的标准库中增加了一个 `code` 模块（提供交互式解释器支持）。当项目目录名为 `code/` 时，sys.path 中的当前工作目录优先级高于 site-packages，理论上应该能导入。但由于 Python 的 `importlib` 对 stdlib 模块有缓存机制，`import code` 被解释器解析为 stdlib 的 `code` 模块，导致 `from code.streamlit_app import ...` 失败。

**解决方案：**
将 `code/` 重命名为 `core/`，所有 import 路径从 `code.xxx` 更新为 `core.xxx`：
- `core/__init__.py`
- `core/streamlit_app.py` 内部 import
- `core/pipeline.py` 帮助文档
- `pyproject.toml` 中的 `[project.scripts]` 和 `[tool.setuptools.packages]`
- `streamlit_app.py` 根入口
- `CLAUDE.md` 中所有 `code/` 引用

**教训：** 绝不将 Python 包命名为与 stdlib 模块相同的名称。Python 版本升级可能引入新的 stdlib 模块，导致旧代码突然失效。

### 4.2 Binary Endpoint SHAP 3D 数组处理

**问题现象：**
二元分类模型（XGBoost、LogisticRegression）在 shap 0.46+ 版本中，`explainer.shap_values(X_test)` 返回的是 `(n_samples, n_features, n_classes)` 的 3D NumPy 数组，而旧版本返回的是 `[array_class0, array_class1]` 的 list。代码原本只处理了 list 类型，导致 3D 数组进入后续的 `ndim == 2` 分支，引发形状不匹配错误。

**根因：**
shap 0.46 版本更新了 TreeExplainer 的内部实现，统一将多类别/二分类的 SHAP 值以 3D 数组返回，而不是旧版的 list-of-arrays。

**解决方案：**
在 `core/shap_analysis.py` 的 `compute()` 方法中增加 `ndim == 3` 的分支：

```python
if isinstance(self.shap_values_, list):
    # Multi-class: take positive class
    self.shap_values_ = self.shap_values_[1]
elif self.shap_values_.ndim == 3:
    # Binary classification: (n_samples, n_features, n_classes) → positive class
    self.shap_values_ = self.shap_values_[:, :, 1]
```

**注意：** shap 更新后，Binary 分类的 `shap.TreeExplainer` 在未指定 `check_additivity=False` 时可能会抛出 additivity check 失败警告。解决方法是在调用 `explainer.shap_values()` 时传入 `check_additivity=False`（因为解释的是 log-odds，而非概率，additivity 在概率空间不成立）。

### 4.3 SurvSHAP 测试集行数不匹配

**问题现象：**
Beeswarm / Summary Bar 图生成时崩溃，报错 `shap_values` 的行数与 `X_test`（用作颜色数组）的行数不一致。

**根因：**
`compute_survival()` 方法为控制运行时间，通过 `max_samples` 参数将测试集抽样到最多 N 行（默认 50）。但 `run_analysis()` 返回结果时，使用原始的 `Xte` 作为 `X_test`，其行数为完整的测试集大小（如 60），而 `shap_values` 的行数为抽样后的 `min(X_test.shape[0], max_samples)`（如 50），两者不一致。

**解决方案：**
在 `core/streamlit_app.py` 的 `run_analysis()` 函数中，SurvSHAP 计算完成后同步对齐 `Xte` 和 `yte`：

```python
if endpoint == EndpointType.SURVIVAL:
    shap_vals = analyzer.compute_survival(
        model, X_train_df, X_test_df, ytr, feature_names,
        model_type=actual_type, max_samples=50,
    )
    # Align X_test / y_test with subsampled SHAP values
    n_sv = shap_vals.shape[0]
    Xte = Xte[:n_sv]
    yte = yte[:n_sv]
```

### 4.4 Streamlit 按钮状态丢失

**问题现象：**
点击"Run Analysis"按钮后分析成功执行，显示结果。当用户在 Tab 2（Feature Dependence）切换下拉框选择不同特征时，页面立即跳回首页（Welcome 页面），之前的所有分析结果丢失。

**根因：**
`st.button()` 只在被点击的**当次** rerun 中返回 `True`。用户切换 `st.selectbox` 下拉框也会触发一次 rerun，此时 `st.button()` 返回 `False`，导致 `if run_clicked:` 条件不满足，后续的分析结果展示代码被跳过，回到 Welcome 状态。

**解决方案：**
使用 `st.session_state` 持久化按钮点击状态：

```python
run_clicked = st.sidebar.button("▶ Run Analysis", type="primary")

if run_clicked:
    st.session_state.analysis_triggered = True

if not st.session_state.get("analysis_triggered", False):
    # 显示 Welcome 页面
    st.stop()

# 后续分析结果展示代码
```

`st.session_state` 的值在 Streamlit 的 rerun 之间保持，因此下拉框等交互不会丢失状态。如需重新运行分析，在按钮回调中清空并重新触发即可。

### 4.5 SHAP Force Plot JavaScript 不加载

**问题现象：**
在 Streamlit Tab 3（Individual Explain）中，Force Plot 显示为空白或仅显示静态 SVG 片段，D3.js 交互功能不工作。

**根因：**
`shap.plots.force().html()` 方法返回的是**仅包含 plot div 片段**的 HTML，不包含 `<script>` 标签（D3.js、lodash、shap 自身的 JS）。shap 官方文档对应的 `shap.save_html()` 才会输出包含完整 JS bundle 的独立 HTML 文件。

**解决方案：**
使用 `shap.save_html()` 写入临时文件，再读取完整内容嵌入 Streamlit：

```python
force_plot = shap.plots.force(
    expected_value, shap_vals[sample_idx],
    feature_names=feature_names, matplotlib=False,
)
tmp_html = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
try:
    shap.save_html(tmp_html.name, force_plot)
    with open(tmp_html.name) as f:
        force_html = f.read()
finally:
    Path(tmp_html.name).unlink(missing_ok=True)
st.components.v1.html(force_html, height=200, scrolling=True)
```

**注意：** 临时文件必须在 `finally` 块中清理（`unlink`），以防止磁盘泄漏。

### 4.6 Streamlit Cloud 部署 -- Python 3.14 + shap 不兼容

**问题现象：**
Streamlit Cloud 部署后，app 启动失败。日志显示 `numba` 安装失败，`shap` 依赖的 `llvmlite` 找不到兼容版本。

**根因：**
Streamlit Community Cloud 在 2025 年 12 月后将默认 Python 版本升级为 **3.14.4**。而 `shap >= 0.46` 依赖 `numba`，`numba` 依赖 `llvmlite`。这两个包的稳定版截至 2025 年底只支持到 Python 3.13，暂不支持 Python 3.14。

**解决方案：**
在 Streamlit Cloud 的 Manage App → Settings → Advanced 中，将 Python version 手动选择为 **3.12**。

**注意：** 项目仓库中包含 `runtime.txt` 文件（内容为 `3.12`），但 Streamlit Cloud **不读取** `runtime.txt` 来决定 Python 版本。该文件仅用于 Heroku 等传统 PaaS 平台。Streamlit Cloud 的 Python 版本必须在 Web UI 中手动设置。

---

## 5. 测试策略

### 5.1 测试概况

| 指标 | 数值 |
|------|------|
| 总测试数 | 66 个 |
| 测试模块 | 5 个 |
| 覆盖率框架 | pytest + pytest-cov |
| 配置 | `pyproject.toml` 中 `[tool.pytest.ini_options]` |

### 5.2 测试分布

| 测试文件 | 测试函数数 | 覆盖范围 |
|----------|-----------|----------|
| `test_data_loader.py` | 7 | 多格式数据加载（csv/xlsx）、列校验、缺失检测 |
| `test_preprocessing.py` | 11 | Preprocessor fit/transform、train_test_split、One-Hot 编码、ARM 保留 |
| `test_modeling.py` | 24 | 4 种端点 × auto + 指定模型类型、小样本 GridSearchCV、指标评估 |
| `test_shap_analysis.py` | 9 | Tree/Linear/Kernel explainer、Binary 3D 数组处理、特征重要性排序、CSV 保存 |
| `test_visualization.py` | 15 | Beeswarm/Bar/Dependence/Waterfall/RCT/ROC 图生成、PNG+SVG 双格式输出 |

### 5.3 pytest 配置

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "-v --tb=short --strict-markers -p no:debugging"
```

### 5.4 `-p no:debugging` 说明

在 `code/` 目录时期，Python 3.12+ 的 pytest 会在收集测试时触发 `import code`，加载 stdlib 的 `code` 模块（交互式调试器），如果项目也有 `code/` 包则产生冲突导致收集失败。`-p no:debugging` 禁用 pytest 调试器插件解决了此问题。重命名为 `core/` 后，该选项不再必需，但保留也无害。

### 5.5 运行测试

```bash
# 完整测试套件
pytest tests/ -v

# 带覆盖率报告
pytest tests/ -v --cov=core --cov-report=term-missing

# 单个模块
pytest tests/test_modeling.py -v
```

---

## 6. Streamlit Cloud 部署总结

### 6.1 部署文件清单

| 文件 | 作用 |
|------|------|
| `streamlit_app.py` | Streamlit Cloud 入口点（社区云要求根目录有 streamlit_app.py） |
| `requirements.txt` | 精简版依赖列表（不含 survshap/scikit-survival/pyreadstat/openpyxl） |
| `runtime.txt` | Python 版本声明（但 Cloud 不读取，需手动在 UI 设置） |
| `.streamlit/config.toml` | 主题色、字体、文件上传大小上限 (100MB)、匿名统计关闭 |

### 6.2 requirements.txt 内容说明

`requirements.txt` 是 Cloud 部署的精简版，包含 Demo 模式所需的全部依赖：

```
numpy>=2.0
pandas>=2.2
shap>=0.46
scikit-learn>=1.5
xgboost>=2.1
statsmodels>=0.14
matplotlib>=3.9
streamlit>=1.40
```

全量版依赖（含 `survshap`、`scikit-survival`、`pyreadstat`、`openpyxl`）定义在 `pyproject.toml` 的 `[project.dependencies]` 中。如需启用 Survival 端点或文件上传功能，需在 Cloud 的 requirements.txt 中取消对应行的注释，同时在 UI 中将 Python 版本设为 3.12（因为 `scikit-survival` 在 Cloud 默认的 3.14 环境下安装会失败）。

### 6.3 部署流程

1. 将代码推到 GitHub 仓库
2. 在 [share.streamlit.io](https://share.streamlit.io) 连接 GitHub，选择仓库和分支
3. Streamlit Cloud 自动检测 `streamlit_app.py` 作为入口点
4. **关键步骤：** Manage App → Settings → Advanced → Python version → 选择 **3.12**
5. 触发部署，Cloud 自动 `pip install -r requirements.txt` 并启动 app

### 6.4 错误诊断

第一版部署时添加了以下诊断代码（`core/streamlit_app.py` 第 32-67 行）：

```python
try:
    import shap
    from shared.config import ...
    from core.synthetic_data import ...
    from core.preprocessing import ...
    from core.modeling import ...
    from core.shap_analysis import ...
    from core.visualization import ...
except Exception:
    import traceback
    st.error(f"Import failed:\n\n```\n{traceback.format_exc()}\n```")
    st.stop()
```

这确保当依赖安装失败或版本不兼容时，真实的错误 traceback 会直接显示在 Streamlit UI 中，而不是被 Cloud 平台吞掉显示为通用的 "Something went wrong"。

---

## 7. 经验教训

### 7.1 包命名避开 stdlib（最高优先级）

将项目目录命名为 `code/` 是最具破坏力的早期决策。Python 3.12+ 的 stdlib 中包含 `code` 模块，导致所有 `from code.xxx import ...` 失败。此后所有 Python 项目都应遵循以下规则：

- **永远不要**将包命名为 `code`、`test`、`json`、`http`、`email`、`io`、`os` 等 stdlib 已有模块名
- 使用更具描述性的名称（如 `core`、`lib`、`src`、`app`）
- 如果必须使用，在 `pyproject.toml` 中确保包的 `find` 配置正确

### 7.2 Streamlit 状态管理

- `st.button()` 的语义是"瞬时的" -- 仅在点击那一刻返回 `True`。任何后续的 widget 交互都会使按钮状态归零。
- 需要跨 rerun 持久化的状态必须放入 `st.session_state`。
- 使用 `st.cache_data` 装饰器可以跨 rerun 缓存昂贵计算结果（本项目 pipeline 函数缓存 1 小时），避免不必要的重计算。

### 7.3 Streamlit Cloud 的 Python 版本管理

- **`runtime.txt` 被 Streamlit Cloud 忽略。** Python 版本必须在 Manage App → Settings → Advanced 中手动选择。
- Streamlit Cloud 的默认 Python 版本会随时间升级（当前为 3.14.4），导致许多科学计算库（numba/llvmlite、scikit-survival）无法安装。
- 对于依赖 `shap` 或 `numba` 的项目，务必将 Python 固定为 3.12。
- 建议在 `requirements.txt` 中添加显式的 `python_version` 注释和说明，提醒部署者手动设置。

### 7.4 SHAP 在不同模型类型下的返回值变化

- **shap 0.46+ 的 API 变更：** Binary 分类不再返回 `[array_c0, array_c1]` 的 list，改为 `(n, m, 2)` 的 3D 数组。
- **TreeExplainer additivity：** 解释 log-odds 而非概率时，additivity check 应设置为 `False`。
- **SurvSHAP treeshap 可加性问题：** scikit-survival 集成模型与 `treeshap` 计算模式存在已知的可加性验证失败问题，统一使用 `sampling` 模式更安全。

### 7.5 SurvSHAP 性能与抽样

- SurvSHAP(t) 的计算时间复杂度约为 `O(k × N_bg × N_explain × N_times)`，在默认参数下可能耗时数十分钟。
- **必须对背景数据和解释样本进行抽样**（本项目默认背景 80、解释样本 50），否则在 Web app 中会导致请求超时。
- 抽样后务必同步对齐 `X_test`/`y_test` 的行数，避免后续可视化函数形状不匹配。
- 建议在 UI 中提示用户 SurvSHAP 结果为近似估计，可通过增加 `max_samples` 提高精度（但影响响应时间）。

### 7.6 Streamlit `import` 缓存导致 Widget 交互失效

**问题：** 根入口 `streamlit_app.py` 使用 `import core.streamlit_app` 加载真正的 App 代码。第一次运行时 Python 执行模块代码并缓存到 `sys.modules`，Streamlit UI 正常渲染。但当用户在 Tab 2 改变下拉框时，Streamlit 重跑根入口脚本，`import` 返回的是 `sys.modules` 中已缓存的模块——**模块级 UI 代码（`st.sidebar`、`st.tabs` 等）不再执行**，页面内容消失，用户看到白屏或跳回首页。

**根因：** Streamlit 在一轮 rerun 中不会重新 import 已缓存的 Python 模块。而 App 的 UI 渲染逻辑全部放在模块顶层，这些代码只在首次 import 时执行一次。

**修复：** 将根入口的 `import core.streamlit_app` 替换为 `exec()` 内联执行源码：

```python
# 旧方案（有 bug）：
import core.streamlit_app

# 新方案（正确）：
_app_path = _project_root / "core" / "streamlit_app.py"
_source = _app_path.read_text()
_code = compile(_source, str(_app_path), "exec")
exec(_code, {"__name__": "__main__", "__file__": str(_app_path)})
```

**关键细节：** `exec()` 的 globals 字典中必须显式传入 `__file__`，因为 `core/streamlit_app.py` 内部使用 `Path(__file__).resolve().parent.parent` 定位项目根目录。`exec` 不会像 `import` 那样自动设置 `__file__`。

**避免方法：**
- 如果使用根 wrapper + import 模式部署 Streamlit App，务必在 wrapper 中使用 `exec()` 而非 `import`。
- 或者直接取消 wrapper，将 App 代码内联到 Streamlit Cloud 期望的入口文件中。
- 即使 App 第一版能正常渲染，也要测试 widget 交互（切换下拉框、滑动滑块、点击按钮）确认 rerun 后 UI 不丢失。

### 7.7 可选依赖的导入策略

**问题：** 为精简 `requirements.txt` 将 `scikit-survival` 和 `survshap` 注释掉，但 `core/modeling.py` 在模块顶层 `import sksurv`。选择非 Survival endpoint 时 App 正常，但选择 Survival endpoint 时触发 `ModuleNotFoundError`。

**根因：** 两个因素叠加导致：
1. 部署依赖列表中移除了 Survival 端点需要的包。
2. 模块顶层无条件 `import sksurv`，即使当前请求不需要 Survival 功能也会触发导入错误。

**修复（两步）：**
1. **将模块顶层 import 改为函数内按需 import**——`RandomSurvivalForest`、`CoxPHSurvivalAnalysis`、`concordance_index_censored` 分别移到 `_train_cox()`、`_train_rsf()`、`evaluate()` 内部，只有真正执行 Survival 训练/评估时才触发导入。
2. **恢复 `requirements.txt` 中的依赖**——部署环境必须包含所有端点可能用到的包。

**避免方法：**
- **顶层只导入"必定存在"的依赖**（numpy、pandas、sklearn、xgboost）。可选功能/端点的依赖放在函数内部按需导入。
- **`requirements.txt` 覆盖所有端点**。如果某个依赖体积大或编译慢，在文档中标注它属于哪个端点，但不要直接从部署依赖中删除。
- 可以保留 `pyproject.toml` 的 `optional-dependencies` 分组用于本地开发，但 Streamlit Cloud 只读 `requirements.txt`，两者要分开维护。

### 7.8 其他

- **pyproject.toml 统一配置：** 将 ruff、mypy、pytest、setuptools 的配置全部集中在 `pyproject.toml` 中，避免根目录散落 `.flake8`、`setup.cfg`、`mypy.ini` 等文件。
- **富依赖管理：** `requirements.txt` 用于最小化部署依赖，`pyproject.toml` 用于开发全量依赖 + optional-dependencies。两者职责分离，避免混淆。
- **Data 目录不纳入版本控制：** `.gitignore` 中排除了所有数据文件和输出文件，仅保留 `.gitkeep` 占位。外部协作时需要额外提供数据或使用 `synthetic_data.py` 生成模拟数据。
- **文献驱动开发：** `reference/` 目录存放了 4 篇核心参考文献（SHAP 统一框架、Shapley 算法综述、SurvSHAP(t)、AKI 临床验证），所有算法选择和可视化设计都可在文献中找到理论依据。

---

*报告生成日期：2026-04-30*
