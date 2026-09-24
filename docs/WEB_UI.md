# Web UI 使用说明（含 WSL / macOS 部署）

Web 页面由 FastAPI **同源静态托管**（`web/` 目录挂在 `/web`，`/` 自动重定向），
没有独立前端服务、没有构建步骤、没有 CORS 配置——服务起在哪，页面就在哪：

```
http://localhost:9000/          → 重定向到 /web/
http://localhost:9000/web/      → 页面本体
```

## 功能一览（三个 Tab）

| Tab | 对接接口 | 能力 |
|---|---|---|
| 🔍 文件搜索 | `/v1/search-agent/*` | 首查 → 候选卡片（分值条/证据/线索 chips）→ 追问收敛或直接确认 → 打开/分享/对比/备注/归档；重建索引 |
| 💬 端云聊天 | `/v1/chat` | 消息流 + 每条回复的路由元数据（source/reason/used_model/置信度/是否升级云端）；`force_cloud` 开关 |
| 🧾 报销材料 | `/v1/expense/*` | 收进来（抽取结果回显+缺件提示）/ 找回来（关键词+金额日期过滤）/ 拿出去（manifest 一键复制）；watch 目录重建索引 |

顶栏健康灯每 30s 轮询 `/health`。页面纯静态三件套（`index.html` / `style.css` /
`app.js`），改完刷新即生效，无需重启服务。

高级用法：`/web/?api=http://other-host:9000` 可把页面指向另一个后端
（跨源需要该后端自行允许 CORS；同源部署用不到）。

## WSL 下部署后端（Windows 浏览器访问）

### 基本形态：WSL2 + localhost 转发（零配置）

WSL2 默认把 WSL 内监听端口转发到 Windows 的 `localhost`。后端在 WSL 里正常起
（`run.sh` 默认绑 `0.0.0.0:9000`），Windows 浏览器直接开
**`http://localhost:9000/`** 即可，无需任何网络配置。

```powershell
# Windows 侧验证转发是否生效
curl.exe http://localhost:9000/health
```

若不生效，检查 `%UserProfile%\.wslconfig` 没有 `localhostForwarding=false`，
然后 `wsl --shutdown` 重进。

### systemd 托管（推荐，WSL 原生支持）

WSL（0.67.6+）支持 systemd，`deploy.sh` / `service.sh` 全套可用：

```bash
# WSL 内，一次性开启 systemd
sudo tee -a /etc/wsl.conf <<'EOF'
[boot]
systemd=true
EOF
```

```powershell
wsl --shutdown   # Windows 侧重启 WSL 生效
```

之后照常 `bash scripts/deploy.sh`（注意 WSL 发行版默认无 GPU，CPU 推理路径；
`--skip-models` 配合 Windows 侧已下好的权重目录也可以）。
唯一差别：WSL 实例随 Windows 关机而停，如需 Windows 开机即起后端，
在任务计划程序里加一条登录任务 `wsl -d <发行版> -u <用户> sudo systemctl start edge-cloud-agent`。

### 让局域网设备（手机/其他电脑）访问

WSL2 默认 NAT，外部设备到不了 `localhost:9000`，二选一：

**方案 A：镜像网络（Win11 22H2+，推荐）** —— `%UserProfile%\.wslconfig`：

```ini
[wsl2]
networkingMode=mirrored
```

`wsl --shutdown` 重进后，WSL 与 Windows 共享 IP，局域网设备直接访问
`http://<Windows的IP>:9000/`（Windows 防火墙需放行 9000 入站）。

**方案 B：端口代理（老版本 Windows）** —— Windows 管理员 PowerShell：

```powershell
netsh interface portproxy add v4tov4 listenport=9000 listenaddress=0.0.0.0 `
  connectport=9000 connectaddress=(wsl hostname -I).Trim()
New-NetFirewallRule -DisplayName "edge-cloud-agent" -Direction Inbound -LocalPort 9000 -Protocol TCP -Action Allow
```

WSL IP 每次重启会变，方案 B 需要重新执行（可做成登录脚本）。

### WSL 文件索引层：把 Windows 侧文件纳入索引

后端在 WSL 里，但要搜的文件在 Windows 侧（桌面/下载/图片/微信目录）。
索引层通过 `/mnt/c/...` 挂载路径直接覆盖它们，配置要点：

**1. 多根源目录**——`FILE_MEMORY_SOURCE_DIR` 支持逗号分隔多个根（路径可含空格）：

```bash
# 交互式：探测 /mnt/c/Users/* 下常见目录（含 WeChat Files/xwechat_files/QQ 等），多选写入 .env
bash scripts/wsl_sources.sh
bash scripts/wsl_sources.sh --print          # 只预览候选不写入
bash scripts/wsl_sources.sh /path/a /path/b  # 任意平台直接指定
```

**2. 排除目录剪枝**——跨 9P 扫描成本高，`FILE_MEMORY_SCAN_EXCLUDE_DIRS`
（默认含 `.git/node_modules/.venv/__pycache__` 等）会在遍历中整棵剪掉命中目录；
想跳过 AppData 大子树可自行追加。**不要把根目录直接设成 `/mnt/c/Users/<你>`**，
应选具体子目录，否则首扫会遍历整个用户目录。

**3. 性能预期**——首扫按文件量以分钟计（9P 单文件 stat/读 1MB hash 都比原生盘慢数倍）；
之后 hash/size 判重前置，未变更文件不读内容，120s watch 周期的增量成本低。
索引本体（`data/*.jsonl` + FAISS）在 WSL 文件系统内，不受 9P 影响。

**4. 打开动作的路径映射**——`file:///mnt/c/...` 对 Windows 浏览器没有意义。
WSL 环境下 open 动作会自动附带 `windows_path`（`C:\Users\...`），
Web UI 在动作结果区展示并支持一键复制，粘到资源管理器地址栏即可打开。
检测可用 `FILE_MEMORY_WSL_PATH_MAP=true/false` 强制覆盖。

**5. 幽灵清理按根保护**——某个根暂不可达（如 `/mnt/c` 未就绪）时，
属于它的索引记录不会被误删；只有可达根下确实消失的文件才被清理。

### 性能提示：权重放 WSL 文件系统内

模型权重（`models/`，~11 GiB）务必放在 WSL 自己的文件系统（如 `~/端云结合/models/`），
**不要放 `/mnt/c/...`**——跨文件系统 IO 会显著拖慢模型加载与 PLE 流式读取。
仓库整体都建议 clone 在 WSL 内，Windows 侧用 IDE 的 WSL remote 连接开发。
（注意与上一节区分：**源文件**本来就在 Windows 侧、必须经 /mnt/c 扫；
这里说的只是**权重与仓库自身**不要放 /mnt/c。）

## macOS 本机运行（开发机形态）

后端直接跑在 macOS 上（`make run` 前台，或 `service.sh start` PID 托管），
浏览器开 `http://localhost:9000/`。文件索引层的 macOS 要点：

**1. 源目录配置**——用平台助手探测常见目录（含微信/QQ/企业微信/钉钉沙盒与
iCloud Drive），多选写入 `.env`：

```bash
bash scripts/macos_sources.sh           # 交互选择（附 TCC 可读性检测）
bash scripts/macos_sources.sh --print   # 只预览候选
```

**2. TCC 隐私权限（macOS 特有，最常踩的坑）**——未授权进程读
`~/Desktop` / `~/Documents` / `~/Downloads` 会报 Operation not permitted，
扫描静默得到空结果。到「系统设置 → 隐私与安全性 → **完全磁盘访问权限**」
把**运行后端的那个 App**（Terminal / iTerm / PyCharm…）加进去，然后重启终端。
助手脚本会对每个候选目录做可读性检测并标注 ⚠️。

**3. iCloud Drive**——未下载到本地的文件是 `.icloud` 占位符，不在后缀白名单内，
自然跳过、不会触发下载；已下载文件正常入库。iCloud 同步目录的频繁变更可能让
watch 周期反复重算 hash，文件量大时可调大 `FILE_MEMORY_SCAN_INTERVAL_SECONDS`。

**4. 不要把根设成 `$HOME`**——`~/Library` 子树极大会拖垮扫描，选具体子目录；
外置卷（`/Volumes/...`）可以直接作为根，Spotlight/废纸篓等卷元数据目录已在
默认排除清单里。

**5. 打开动作**——`file://` URI 浏览器不允许从 http 页面跳转，Web UI 的
「复制路径」按钮会给纯 POSIX 路径，粘到 Finder「前往文件夹」（⌘⇧G）直达。

## 安全提醒

服务与页面均**无鉴权**。`localhost` 自用没有暴露面；一旦按上面方案对局域网开放，
同网段设备即可读写你的文件索引与报销数据，请自行评估（或只开镜像/代理给可信网络）。
