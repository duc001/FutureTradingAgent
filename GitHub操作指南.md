# PyCharm 项目提交到 GitHub 操作指南

## 前置条件

1. ✅ 已在 PyCharm 中配置好 GitHub Token
2. ✅ 已在 GitHub 上创建空仓库
3. ✅ 已配置 Git 用户信息（个人/公司区分）

---

## 完整操作流程

### 步骤一：初始化 Git 仓库

**方法1 - 使用 PyCharm 菜单：**
```
顶部菜单 → VCS → 启用版本控制集成... → 选择 Git → 确定
```

**方法2 - 使用终端：**
```bash
cd /Users/anjuke/PycharmProjects/你的项目名
git init
```

---

### 步骤二：创建 .gitignore 文件（推荐）

在项目根目录创建 `.gitignore` 文件，排除不需要提交的文件：

```gitignore
# Python
__pycache__/
*.py[cod]
*.so
.Python

# 虚拟环境
.venv/
venv/
ENV/

# IDE
.idea/
.vscode/

# 数据文件
*.xlsx
*.csv
*.json

# 日志
*.log

# 操作系统
.DS_Store
```

---

### 步骤三：添加文件到暂存区

**方法1 - 使用快捷键：**
```
选中文件或文件夹 → Ctrl + Alt + A (Mac: Cmd + Alt + A)
```

**方法2 - 右键菜单：**
```
右键点击文件 → Git → 添加
```

**验证：** 文件名变为绿色表示已添加成功

---

### 步骤四：首次提交（Commit）

1. 点击左下角 `提交` 标签（或按 `Cmd + K`）
2. 在提交信息框输入：`Initial commit`
3. 确认勾选要提交的文件
4. 点击 `提交` 按钮

---

### 步骤五：关联远程 GitHub 仓库

1. **获取远程仓库地址：**
   - 打开 GitHub 仓库页面
   - 点击绿色 `Code` 按钮
   - 复制 HTTPS 地址，如：`https://github.com/用户名/仓库名.git`

2. **在 PyCharm 中配置：**
   ```
   顶部菜单 → VCS → Git → 远程...
   → 点击左上角 + 号
   → 名称：origin（默认）
   → URL：粘贴 GitHub 仓库地址
   → 确定
   ```

---

### 步骤六：推送到 GitHub（Push）

**方法1 - 使用快捷键：**
```
Cmd + Shift + K
```

**方法2 - 使用菜单：**
```
顶部菜单 → VCS → Git → 推送...
```

**操作步骤：**
1. 确认分支是 `main` 或 `master`
2. 确认远程仓库是 `origin`
3. 勾选 "将当前分支设置为跟踪远程分支"（首次推送）
4. 点击 `推送` 按钮

---

### 步骤七：验证推送成功

1. 打开浏览器访问：`https://github.com/你的用户名/仓库名`
2. 确认文件已上传成功

---

## 日常开发流程

### 修改代码后的提交流程

```
1. 修改代码
   ↓
2. 添加到暂存区（Ctrl + Alt + A）
   ↓
3. 提交（Cmd + K）→ 填写提交信息
   ↓
4. 推送（Cmd + Shift + K）→ 推送到 GitHub
```

---

## 常用 Git 操作速查

| 操作 | PyCharm 菜单 | 快捷键（Mac） | 说明 |
|------|-------------|--------------|------|
| 添加文件 | Git → 添加 | Cmd + Alt + A | 添加到暂存区 |
| 提交 | Git → 提交 | Cmd + K | 本地提交 |
| 推送 | Git → 推送 | Cmd + Shift + K | 推送到远程 |
| 拉取 | Git → 拉取 | Cmd + T | 从远程拉取 |
| 查看历史 | Git → 显示历史记录 | - | 查看提交记录 |
| 撤销更改 | Git → 回滚 | - | 撤销未提交的修改 |

---

## 多账号配置说明

### 场景：公司有 GitLab，个人用 GitHub

**全局配置（默认个人账号）：**
```bash
git config --global user.name "GitHub用户名"
git config --global user.email "GitHub邮箱"
```

**公司项目单独配置：**
```bash
cd /path/to/公司项目
git config user.name "公司用户名"
git config user.email "公司邮箱"
```

**验证当前配置：**
```bash
git config user.name
git config user.email
```

---

## 常见问题解决

### 问题1：推送时要求输入用户名密码
**原因：** Token 未正确配置  
**解决：**
```
偏好设置 → 版本控制 → GitHub
→ 检查账号状态是否正常
→ 重新添加 Token
```

### 问题2：推送失败，提示分支不存在
**原因：** 首次推送未设置上游分支  
**解决：** 推送时勾选 "将当前分支设置为跟踪远程分支"

### 问题3：想修改远程仓库地址
**解决：**
```
VCS → Git → 远程...
→ 选中 origin → 点击编辑图标
→ 修改 URL → 确定
```

### 问题4：提交了不该提交的文件（如 .idea/）
**解决：**
```bash
# 从 Git 中删除但保留本地文件
git rm -r --cached .idea/

# 确保 .gitignore 中已添加该目录
# 然后重新提交
git commit -m "Remove .idea from tracking"
git push
```

---

## 项目目录规范

```
/Users/anjuke/PycharmProjects/          # Python 项目总目录
├── FutureTradingAgent/                 # 期货交易项目
│   ├── .git/                           # Git 仓库
│   ├── .gitignore                      # Git 忽略配置
│   ├── .venv/                          # 虚拟环境（不提交）
│   ├── .idea/                          # PyCharm 配置（不提交）
│   ├── main.py                         # 主程序
│   └── README.md                       # 项目说明
├── 其他项目1/
└── 其他项目2/
```

---

## 最佳实践建议

1. **频繁提交：** 每完成一个小功能就提交
2. **清晰的提交信息：** 说明做了什么改动
3. **及时推送：** 避免本地积压太多提交
4. **定期拉取：** 多人协作时先拉取再推送
5. **善用分支：** 新功能在独立分支开发

---

## 参考资源

- Git 官方文档：https://git-scm.com/doc
- GitHub 帮助：https://docs.github.com/cn
- PyCharm Git 集成：https://www.jetbrains.com/help/pycharm/using-git-integration.html

---

**最后更新：** 2026-05-06  
**适用工具：** PyCharm 汉化版、Git、GitHub
