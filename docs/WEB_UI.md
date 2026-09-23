# Web UI 使用说明（含 WSL 部署）

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

### 性能提示：权重放 WSL 文件系统内

模型权重（`models/`，~11 GiB）务必放在 WSL 自己的文件系统（如 `~/端云结合/models/`），
**不要放 `/mnt/c/...`**——跨文件系统 IO 会显著拖慢模型加载与 PLE 流式读取。
仓库整体都建议 clone 在 WSL 内，Windows 侧用 IDE 的 WSL remote 连接开发。

## 安全提醒

服务与页面均**无鉴权**。`localhost` 自用没有暴露面；一旦按上面方案对局域网开放，
同网段设备即可读写你的文件索引与报销数据，请自行评估（或只开镜像/代理给可信网络）。
