# 版本管理指南

你的项目现在支持多种版本管理方式。以下是不同工具的配置和使用方法：

## 🔧 当前配置（bumpversion）

### 已更新的配置
你的 `setup.cfg` 已经更新，现在 bumpversion 会同步更新三个位置的版本号：

```cfg
[bumpversion]
current_version = 0.0.9
commit = True
tag = True

[bumpversion:file:setup.py]
search = version='{current_version}'
replace = version='{new_version}'

[bumpversion:file:torchhydro/__init__.py]
search = __version__ = '{current_version}'
replace = __version__ = '{new_version}'

[bumpversion:file:pyproject.toml]
search = version = "{current_version}"
replace = version = "{new_version}"
```

### 使用方法
```bash
# 升级补丁版本 (0.0.9 -> 0.0.10)
bumpversion patch

# 升级次版本 (0.0.9 -> 0.1.0)
bumpversion minor

# 升级主版本 (0.0.9 -> 1.0.0)
bumpversion major
```

## 🚀 现代化替代方案

### 方案 1：使用 setuptools-scm（推荐）

这是最现代的方法，版本完全基于 Git tags 自动管理。

#### 更新 pyproject.toml
```toml
[build-system]
requires = ["setuptools>=61.0", "setuptools-scm[toml]>=6.2", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "torchhydro"
dynamic = ["version"]  # 版本由 setuptools-scm 管理
# ... 其他配置保持不变，移除 version = "0.0.9" 行

[tool.setuptools_scm]
write_to = "torchhydro/_version.py"
```

#### 更新 torchhydro/__init__.py
```python
# 替换现有的版本行
try:
    from torchhydro._version import __version__
except ImportError:
    __version__ = "unknown"
```

#### 使用方法
```bash
# 创建标签就会自动更新版本
git tag v0.1.0
git push origin v0.1.0

# 或者使用 GitHub Release 功能
# 版本号会自动从 git tag 中提取
```

### 方案 2：使用 semantic-release

全自动的版本管理，基于 commit message 自动决定版本号。

#### 安装和配置
```bash
npm install -g semantic-release
npm install -g @semantic-release/changelog
npm install -g @semantic-release/git
```

#### 创建 .releaserc.yml
```yaml
branches:
  - main
plugins:
  - "@semantic-release/commit-analyzer"
  - "@semantic-release/release-notes-generator"
  - "@semantic-release/changelog"
  - "@semantic-release/github"
  - "@semantic-release/git"
```

#### 使用方法
```bash
# 提交消息格式影响版本号
git commit -m "feat: add new feature"      # 次版本号 +1
git commit -m "fix: fix bug"               # 补丁版本号 +1
git commit -m "feat!: breaking change"     # 主版本号 +1

# 运行自动发布
semantic-release
```

### 方案 3：使用 uv 内置版本管理

uv 也有内置的版本管理功能：

```bash
# 使用 uv 管理版本
uv version patch    # 0.0.9 -> 0.0.10
uv version minor    # 0.0.9 -> 0.1.0
uv version major    # 0.0.9 -> 1.0.0
```

## 📋 推荐策略

### 对于个人项目（推荐 setuptools-scm）
```bash
# 1. 更新 pyproject.toml 使用动态版本
# 2. 通过 git tag 管理版本
git tag v0.1.0
git push origin v0.1.0
```

### 对于团队项目（推荐 semantic-release）
```bash
# 1. 配置 semantic-release
# 2. 规范化 commit message
# 3. 自动化发布流程
```

### 对于向后兼容（保持现有方式）
```bash
# 继续使用 bumpversion
bumpversion patch
```

## 🔄 迁移步骤

### 如果选择 setuptools-scm：

1. **更新 pyproject.toml**：
   - 添加 `setuptools-scm` 依赖
   - 将 `version = "0.0.9"` 改为 `dynamic = ["version"]`

2. **更新 __init__.py**：
   - 改为从 `_version.py` 导入版本

3. **创建初始标签**：
   ```bash
   git tag v0.0.9
   git push origin v0.0.9
   ```

### 如果选择 semantic-release：

1. **安装 semantic-release**
2. **配置 .releaserc.yml**
3. **规范化 commit message**
4. **集成到 GitHub Actions**

### 如果保持现有方式：

✅ 不需要任何额外操作，现有配置已经更新完成！

## 🎯 立即行动

### 测试当前配置
```bash
# 测试 bumpversion 是否正常工作
bumpversion --dry-run patch

# 查看会改变的文件
bumpversion --dry-run --verbose patch
```

### 实际升级版本
```bash
# 升级到 0.0.10
bumpversion patch

# 这会：
# 1. 更新 setup.py 中的版本
# 2. 更新 torchhydro/__init__.py 中的版本
# 3. 更新 pyproject.toml 中的版本
# 4. 创建 git commit
# 5. 创建 git tag
```

## 💡 最佳实践

1. **保持同步**：确保所有位置的版本号一致
2. **自动化**：使用 CI/CD 自动化版本发布
3. **语义化**：遵循语义化版本规范
4. **测试**：发布前在测试环境验证

## 🔍 验证版本一致性

```bash
# 检查各个位置的版本号
python -c "import torchhydro; print(torchhydro.__version__)"
python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])"
python setup.py --version
```

现在你的版本管理系统已经完全兼容新的 pyproject.toml 结构！🎉 