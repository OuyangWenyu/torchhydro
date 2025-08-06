# 开发者指南

本项目使用现代 Python 工具链，基于 uv 进行依赖管理和构建。

## 🚀 快速开始

### 环境准备
```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆项目
git clone https://github.com/OuyangWenyu/torchhydro.git
cd torchhydro

# 安装依赖（自动创建虚拟环境）
uv sync --extra dev
```

### 常用命令
```bash
# 运行测试
uv run pytest

# 代码格式化
uv run black .
uv run isort .

# 代码检查
uv run flake8
uv run mypy torchhydro

# 构建包
uv build
```

## 📦 版本管理

使用 bump2version 进行版本管理：

```bash
# 升级版本
uv run bump2version patch   # 0.0.9 -> 0.0.10
uv run bump2version minor   # 0.0.9 -> 0.1.0  
uv run bump2version major   # 0.0.9 -> 1.0.0
```

版本号会自动更新：
- `pyproject.toml`
- `torchhydro/__init__.py`

## 🔄 CI/CD 工作流

项目配置了三个 GitHub Actions：

### `build.yml` - 持续集成
- **触发**：Push/PR 到 main/dev 分支
- **测试**：Python 3.10, 3.11, 3.12 on Ubuntu
- **功能**：安装依赖 → 下载测试数据 → 运行测试

### `pypi.yml` - 包发布
- **触发**：创建 Release 或手动触发
- **功能**：构建包 → 发布到 PyPI

### `docs.yml` - 文档部署
- **触发**：Push 到 main 分支
- **功能**：构建文档 → 部署到 GitHub Pages

## 🛠️ 开发流程

### 1. 开发新功能
```bash
# 创建功能分支
git checkout -b feature/new-feature

# 开发并测试
uv run pytest tests/

# 提交代码
git add .
git commit -m "feat: add new feature"
git push origin feature/new-feature
```

### 2. 发布新版本
```bash
# 升级版本
uv run bump2version patch

# 推送版本标签（自动触发 CI）
git push origin main --tags

# 在 GitHub 创建 Release（自动触发 PyPI 发布）
```

## 📋 项目结构

```
torchhydro/
├── .github/workflows/    # CI/CD 配置
├── torchhydro/          # 主要代码
├── tests/               # 测试代码
├── experiments/         # 实验脚本
├── pyproject.toml       # 项目配置和依赖
├── uv.lock             # 锁定文件
└── DEVELOPERS.md       # 本文件
```

## 🔧 开发工具配置

所有工具配置都在 `pyproject.toml` 中：

- **pytest**: 测试配置
- **black**: 代码格式化
- **isort**: 导入排序
- **flake8**: 代码检查  
- **mypy**: 类型检查
- **bumpversion**: 版本管理

## 📚 有用的链接

- [uv 文档](https://docs.astral.sh/uv/)
- [PyTorch 官网](https://pytorch.org/)
- [项目仓库](https://github.com/OuyangWenyu/torchhydro)