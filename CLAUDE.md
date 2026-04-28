# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 1. Project Vision & Goals

**Project Name:** 临床试验探索性分析 (SHAP-based Clinical Trial Exploratory Analysis)

**Core Objective:** Leverage machine learning algorithms—specifically SHAP (SHapley Additive exPlanations) values—to perform exploratory analysis on clinical trial data. Use visualizations (Beeswarm plots, summary bar charts, dependence plots, waterfall plots) to reveal how individual features influence clinical outcomes.

The emphasis is on **interpretability**: the project exists to explain model decision-making processes in a clinical/regulatory context, not merely to maximize predictive performance.

---

## 2. Tech Stack

| Layer       | Dependency                                                                 |
|-------------|----------------------------------------------------------------------------|
| Language    | Python 3.12                                                                |
| ML / SHAP   | `shap`, `survshap`, `scikit-learn`, `xgboost`                             |
| Statistics  | `statsmodels` (GLM, Logistic, Cox-PH)                                     |
| Data I/O    | `pyreadstat` (SAS `.sas7bdat` / R `.RData`), `openpyxl` (`.xlsx`), `pandas` |
| Numerics    | `numpy`                                                                    |
| Plotting    | `matplotlib` (high-resolution, publication-ready figures)                 |
| Testing     | `pytest`, `pytest-cov`                                                     |
| Linting     | `ruff`, `mypy`                                                             |

---

## 3. Project Architecture & Directory Structure

```
SHAP/
├── data/           # Raw datasets (.sas7bdat, .RData, .xlsx). READ-ONLY.
├── code/           # Core logic scripts: data cleaning, modeling, visualization
│   ├── __init__.py
│   ├── data_loader.py       # Read & validate clinical data files
│   ├── preprocessing.py     # ADaM-standard cleaning & feature engineering
│   ├── modeling.py          # ML + traditional statistical model wrappers
│   ├── shap_analysis.py     # SHAP value computation & interpretation
│   └── visualization.py     # Publication-ready plotting functions
├── shared/         # Shared configuration, paths, constants, preprocessing params
│   ├── __init__.py
│   ├── config.py            # Paths, environment settings, seed values
│   └── constants.py         # Clinical domain constants (endpoint types, trial designs)
├── output/         # Generated model results, SHAP value CSVs, final figures
├── reference/      # Key literature: methodology references for SHAP & clinical prediction
├── tests/          # Unit tests mirroring code/ structure
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_modeling.py
│   ├── test_shap_analysis.py
│   └── test_visualization.py
├── CLAUDE.md
└── requirements.txt
```

---

## 4. Core Domain Logic

### 4.1 Data Standards

All data processing must follow **ADaM (Analysis Data Model)** conventions:
- **ADSL** (Subject-Level): demographics, population flags, stratification factors
- **ADTTE** (Time-to-Event): survival endpoints with censor indicators
- **ADLB** (Lab): laboratory results with visit/time structure

### 4.2 Endpoint Types

| Type          | Model(s) to Apply                                      |
|---------------|--------------------------------------------------------|
| Continuous    | Linear Regression, GLM (Gaussian), Random Forest       |
| Binary        | Logistic Regression, GLM (Binomial), XGBoost Classifier|
| Survival      | Cox Proportional Hazards, Random Survival Forest       |
| Count/Poisson | Poisson GLM, Negative Binomial                         |

### 4.3 Trial Design Types

- **Single-arm:** descriptive SHAP analysis; no treatment comparison.
- **RCT / 2-arm:** SHAP computed on treatment flag + covariates; interaction effects emphasized.
- **Parallel Multi-cohort:** SHAP by cohort; aggregated summary with multi-panel visualizations.

### 4.4 Train/Test Split

- Fixed ratio: **80% train / 20% test** (`random_state=42` as project-wide seed).
- For small sample size (N < 200): enforce **stratified k-fold cross-validation (k=5)** with automated hyperparameter tuning (GridSearchCV / RandomizedSearchCV) and bootstrap-based robustness checks.
- For survival endpoints: split must preserve the censoring proportion in both partitions.

### 4.5 SHAP Workflow

1. Train model on training set.
2. Compute SHAP values on test set using model-appropriate explainer:
   - Tree-based models → `shap.TreeExplainer`
   - Linear models → `shap.LinearExplainer`
   - Survival models → `survshap.SurvSHAP`
3. Aggregate SHAP values into global feature-importance ranking.
4. Generate visualizations: Beeswarm, Summary Bar, Dependence, Waterfall (per subject).

---

## 5. Coding Standards & Principles

### 5.1 Testing (Non-Negotiable)

Every new function or module **must** have a corresponding unit test. Run tests with:

```bash
# All tests
pytest tests/ -v

# Single test file
pytest tests/test_modeling.py -v

# With coverage report
pytest tests/ -v --cov=code --cov-report=term-missing
```

### 5.2 Visualization Standards (Publication-Ready)

所有图表必须符合医学期刊发表标准。以下规范部分参考了 `reference/` 中两篇核心文献的例图样式。

#### 5.2.1 全局渲染参数

```python
# matplotlib 全局设置 (在 visualization.py 模块入口调用)
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    # 字体
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "SimSun", "DejaVu Sans"],
    "font.size": 9,
    # 刻度
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    # 轴线
    "axes.linewidth": 0.5,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # 图例
    "legend.fontsize": 8,
    "legend.frameon": False,
    "legend.loc": "best",
    # 网格 (默认关闭，按需开启)
    "axes.grid": False,
})
```

中文字体回退链: `Arial` → `SimSun`（宋体）→ `DejaVu Sans`。Windows 下 SimSun 自动可用；macOS/Linux 需确保已安装或由 matplotlib 自动回退。

#### 5.2.2 配色方案

严格使用 `shared/config.py` 中定义的 `CLINICAL_COLORS`：

| 角色 | 色号 | 用途 |
|------|------|------|
| `primary` | `#1f77b4` | 主模型 / 治疗组 |
| `secondary` | `#ff7f0e` | 对照模型 / 对照组 |
| `positive` | `#2ca02c` | 正向 SHAP 贡献 / 保护因素 |
| `negative` | `#d62728` | 负向 SHAP 贡献 / 危险因素 |
| `neutral` | `#7f7f7f` | 中性特征 / 参考线 |
| `treatment` | `#9467bd` | 治疗标识 / 亚组 |
| `control` | `#17becf` | 对照标识 |

Beeswarm 图中 SHAP 值颜色映射: `negative` (低特征值) → `neutral` (中位) → `primary` (高特征值)，使用连续渐变。

#### 5.2.3 图表类型详细规范

##### SHAP Beeswarm Plot (Summary Plot)
参考: AKI 论文 Fig.3

| 属性 | 要求 |
|------|------|
| 画布尺寸 | 8×6 inches (单列) / 12×6 inches (双列) |
| Y 轴 | 特征名，按全局重要性降序排列 |
| X 轴 | SHAP value，标签 `SHAP value` |
| 点大小 | `s=8`（样本 <1000 时）；`s=3`（样本 ≥1000 时），alpha=0.6 |
| 颜色条 | 水平放置于图上方或下方，标签为特征原始值 |
| 叠加方式 | 垂直抖动叠加以展示密度分布 |
| 参考线 | `x=0` 处灰色虚线 (`color=neutral`, `linestyle='--'`, `linewidth=0.8`) |
| 排序规则 | `shap_values.abs.mean(0).argsort()` |

##### SHAP Summary Bar Plot (汇总柱状图)
参考: AKI 论文 Fig.3B

| 属性 | 要求 |
|------|------|
| 画布尺寸 | 8×6 inches |
| 方向 | 水平柱状图 (`plt.barh`) |
| Y 轴 | 特征名，按重要性降序排列 |
| X 轴 | `mean(|SHAP value|)`，标签 `Mean(|SHAP value|)` |
| 颜色 | 统一 `primary` 色 (`#1f77b4`) |
| 柱间距 | `height=0.6`，间距 0.2 |
| 数值标签 | 柱端右侧标注数值，`fontsize=7`，保留 3 位小数 |

##### SHAP Dependence Plot (依赖图)
参考: AKI 论文 Fig.4

| 属性 | 要求 |
|------|------|
| 画布尺寸 | 每个子图 4×3 inches，N 个特征排列为 2~4 列网格 |
| X 轴 | 特征原始值，标签为特征名 |
| Y 轴 | SHAP value，标签 `SHAP value for <feature>` |
| 颜色映射 | 用交互特征的值着色，颜色条置于子图右侧 |
| 散点 | `s=6`, `alpha=0.5`, `edgecolors='none'` |
| 趋势线 | 可选 LOWESS 平滑线 (`color='black'`, `linewidth=1.0`) |
| 水平参考 | `y=0` 处 `neutral` 色虚线 |

##### Force Plot (个体解释图)
参考: AKI 论文 Fig.5

| 属性 | 要求 |
|------|------|
| 方向 | 水平 single-sample force plot |
| 特征标注 | 特征名 + 贡献值，字体大小 7 |
| 颜色 | 红色系 (`negative`) = 推向高风险，蓝色系 (`primary`) = 推向低风险 |
| 输出 | 每张图标注模型输出值 (predicted probability / risk score) 和 base value |
| 画布尺寸 | 12×3 inches |

##### ROC Curve (ROC 曲线)
参考: AKI 论文 Fig.2

| 属性 | 要求 |
|------|------|
| 画布尺寸 | 6×6 inches (正方形) |
| X 轴 | `1 - Specificity`，范围 [0, 1] |
| Y 轴 | `Sensitivity`，范围 [0, 1] |
| 对角线 | `neutral` 色虚线 (`linewidth=0.8`) 表示随机分类器 |
| 曲线 | `primary` 色实线 (`linewidth=1.2`) |
| AUC 标注 | 图例中标注 `AUC = 0.xxx`（保留 3 位小数），图例置于右下角 |
| 刻度 | 间隔 0.2，`xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0]` |

##### SurvSHAP(t) 时间依赖图
参考: SurvSHAP 论文 Fig.3, Fig.11, Fig.12

| 属性 | 要求 |
|------|------|
| 画布尺寸 | 8×5 inches (单变量) / 10×6 inches (多变量) |
| X 轴 | 时间 `t`，标签 `Time` |
| Y 轴 | `SurvSHAP(t)` 值，标签 `SurvSHAP(t)` |
| 零线 | `y=0` 处 `neutral` 色虚线 |
| 正/负区域 | `positive` 色填充 >0 区域，`negative` 色填充 <0 区域 (alpha=0.15) |
| 变量曲线 | 每条曲线不同颜色 (按 CLINICAL_COLORS 顺序循环)，`linewidth=1.2` |
| 图例 | 变量名标注于曲线旁或独立图例，`fontsize=7` |

##### 局部 vs 全局重要性对比热力图
参考: SurvSHAP 论文 Fig.9, Fig.13

| 属性 | 要求 |
|------|------|
| 画布尺寸 | 10×8 inches |
| 布局 | 上下两个子图 (如 CPH / RSF)，共享 X 轴 |
| Y 轴 | 变量名 (按全局重要性排序) |
| X 轴 | 100 个观测样本 (或指定前 N 个) |
| 颜色映射 | `viridis` 或自定义: `negative`(低) → `neutral`(中) → `primary`(高) |
| 颜色条 | 右侧垂直放置，标签 `SHAP value` |
| 排序 | 按全局重要性降序 (最重要的变量在顶部) |

#### 5.2.4 输出规范

```python
def save_figure(fig, filename: str, formats: list[str] = ["png", "svg"]) -> None:
    """统一保存函数：同时输出 PNG (raster) + SVG (vector)。"""
    from shared.config import OUTPUT_DIR, FIGURE_DPI
    for fmt in formats:
        path = OUTPUT_DIR / f"{filename}.{fmt}"
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.05)
```

- 文件命名: `{figure_type}_{dataset_name}_{feature/cohort}.{fmt}`，如 `beeswarm_EXP1_full.png`
- 每张图必须同时输出 PNG (≥300 DPI) 和 SVG (矢量可编辑) 两种格式
- 图标题不嵌入图片文件，而是在论文撰写时通过排版软件添加

### 5.3 Code Quality

- **PEP8** compliance via `ruff`.
- **Type hints** required on all public functions and methods (`mypy` strict mode).
- **Data lineage:** every preprocessing step must log input shape, output shape, and the transformation applied. Use `shared/config.py` `LOG_CONFIG` for structured logging.

```bash
# Lint
ruff check code/ tests/

# Type check
mypy code/ --strict
```

### 5.4 Commit Messages

Use Chinese for commit messages (项目惯例). Format: `<type>: <description>`, e.g.:
- `feat: 添加 Cox 模型 SHAP 分析模块`
- `fix: 修正生存数据训练集/测试集划分逻辑`

---

## 6. Standard Analysis Pipeline

```
Raw Data (.sas7bdat / .RData / .xlsx)
    │
    ▼
[1] data_loader.py  ─── 读取原始数据，校验列名与格式
    │
    ▼
[2] preprocessing.py ─── ADaM 标准清洗、缺失值处理、特征工程
    │
    ▼
[3] modeling.py     ─── 80/20 划分 → 模型训练 → 超参数调优(小样本)
    │
    ▼
[4] shap_analysis.py ─── SHAP 值计算 → 特征重要性排序 → 交互效应检测
    │
    ▼
[5] visualization.py ─── Beeswarm / Summary Bar / Dependence / Waterfall
    │
    ▼
output/ (*.csv, *.png, *.svg)
```

### Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (once implemented)
python -m code.pipeline --input data/your_trial.sas7bdat --output output/

# Run tests
pytest tests/ -v

# Lint + type check
ruff check code/ tests/ && mypy code/ --strict
```

---

## 7. Reference Literature

The `reference/` directory stores key academic papers that inform the methodological approach of this project. All algorithms and statistical methods implemented in `code/` should align with the methodologies described in these references.

| Paper | Relevance |
|-------|-----------|
| `1-s2.0-S0950705122013302-main.pdf` | Core SHAP methodology and interpretable ML framework (Knowledge-Based Systems) |
| `Development and validation of a real-time prediction model for acute kidney injury in hospitalized patients.pdf` | Reference implementation: clinical prediction model development & validation workflow |

When implementing modeling or SHAP analysis logic, consult these papers for methodological justification of design choices (e.g., feature selection criteria, model evaluation metrics, SHAP visualization interpretation).

---

## 8. Key Design Decisions

- **Random seed:** `42` is the project-wide global seed. Import from `shared.config.SEED`.
- **SHAP vs. permutation importance:** SHAP is preferred because it provides both global and local (per-subject) explanations, which is critical for clinical interpretation.
- **`survshap`** is used for survival endpoints instead of standard `shap` because it correctly handles the time-dependent nature of survival predictions.
- **`pyreadstat`** is chosen over `sas7bdat`/`pyreadr` because it handles both SAS and R formats with metadata preservation (variable labels, value formats).
