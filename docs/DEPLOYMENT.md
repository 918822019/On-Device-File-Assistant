# 部署指南（Linux 服务器 + systemd）

> 一键部署：`bash scripts/deploy.sh`
> 服务管理：`bash scripts/service.sh {start|stop|restart|status|logs|health}`

本文覆盖生产部署全流程。本机开发仍用 `make run`（前台运行），两套互不干扰。

---

## 一、前置要求

| 项 | 要求 | 说明 |
|---|---|---|
| 系统 | Linux（systemd） | 无 systemd 环境（容器/开发机）自动降级为 PID 文件托管 |
| Python | 3.10 – 3.13 | `deploy.sh` 依次探测 python3.12/3.11/3.13/3.10/python3，也可 `--python` 指定 |
| 内存 | ≥ 12 GiB（建议 16 GiB） | E2B bfloat16 权重常驻 ~9.5 GiB + 运行时开销；不足会告警 |
| 磁盘 | ≥ 15 GiB 余量 | LLM ~9.6 GiB + embedding ~1.1 GiB + pip/缓存余量 |
| GPU | 不需要 | 默认 CPU 推理；deploy 会自动预装 CPU 版 torch 避免拉 CUDA 轮子 |
| 权限 | 普通用户 + sudo | 仅 systemd unit 安装一步需要提权，其余全部用户态 |

代码就位（二选一）：

```bash
git clone <repo-url> 端云结合 && cd 端云结合
# 或从开发机同步（含已下载权重，适合内网/离线服务器）：
rsync -avz --exclude .venv --exclude .git /path/to/端云结合/ user@server:/opt/edge-cloud-agent/
```

> **注意**：`.venv` 不要跨机器拷贝（平台相关），到服务器上由 deploy.sh 重建。

## 二、一键部署

```bash
cd /opt/edge-cloud-agent        # 仓库根目录
bash scripts/deploy.sh          # 交互式；自动化场景加 --yes
```

脚本按六步执行，每步幂等、可单独跳过，重复运行安全：

```
[1/6] 前置检查     内存/磁盘/Python/curl
[2/6] venv 与依赖  无 GPU 时预装 CPU 版 torch==2.6.0
[3/6] .env 配置    缺失时从 .env.example 生成；关键项体检
[4/6] 模型权重     检查 config.json + *.safetensors，缺失则确认后下载并回填 .env
[5/6] systemd      渲染 deploy/edge-cloud-agent.service → 安装 → enable → 启动
[6/6] 健康检查     轮询 /health，默认最长 900s（启动需同步加载 ~9.5GiB 模型）
```

### 参数一览

| 参数 | 作用 |
|---|---|
| `--yes` / `-y` | 非交互：权重下载不再确认（**非 tty 且未加此参数时会拒绝下载**，防误拉 10GB） |
| `--host H` / `--port P` | 监听地址/端口（默认 `0.0.0.0:9000`，写入 systemd unit） |
| `--skip-deps` | 跳过 venv/pip（需 `.venv` 已存在） |
| `--skip-models` | 跳过权重检查/下载（离线部署：权重已 rsync 就位时用） |
| `--skip-systemd` | 不装 unit（容器/无 root；配合 `service.sh` 的 PID 托管模式） |
| `--no-start` | 装好但不启动 |
| `--gpu` | 服务器有 NVIDIA GPU 时跳过 CPU torch 预装，走默认（CUDA）轮子 |
| `--python PATH` | 指定基础解释器 |
| `--uninstall` | 卸载 systemd 服务（不动代码/权重/data/） |

环境变量：`HEALTH_TIMEOUT`（健康检查秒数，默认 900）、`MIN_DISK_GIB`、`MIN_MEM_GIB`、
`HF_ENDPOINT`（HF 镜像，透传给下载）。

### 端口配置的正确姿势

`PORT` **不在 `.env` 里生效**（`run.sh` 在加载 .env 前就从进程环境取 `PORT`，.env 不覆盖
已设置的键）。改端口用下面任一方式：

```bash
bash scripts/deploy.sh --port 8080        # 重新渲染 unit 并重启（推荐）
# 或手工改 /etc/systemd/system/edge-cloud-agent.service 的 Environment=PORT=
# 后 systemctl daemon-reload && systemctl restart edge-cloud-agent
```

`service.sh` 会自动从 unit 文件读取端口做健康检查，无需重复指定。

## 三、服务管理

```bash
bash scripts/service.sh start     # 启动并等待健康检查通过
bash scripts/service.sh stop      # 停止（PID 模式先 SIGTERM 等 30s 优雅退出）
bash scripts/service.sh restart
bash scripts/service.sh status    # systemd 状态 + /health
bash scripts/service.sh logs -f   # 跟踪日志（systemd → journalctl；PID 模式 → logs/*.log）
bash scripts/service.sh logs -n 500
bash scripts/service.sh health    # 仅健康检查
```

等价 Make 目标：`make start / stop / restart / status / logs / deploy / uninstall`。

两种托管模式自动选择：装了 unit 走 systemd（开机自启、崩溃 10s 后自动拉起、
10 分钟内最多重启 5 次防刷爆）；否则 PID 文件托管（`.server.pid` + `logs/edge-cloud-agent.log`，
已 gitignore），适合容器与开发机。

## 三点五、WSL 部署特记

后端跑在 WSL2 时全套脚本同样可用（WSL 开启 systemd 后 `deploy.sh` 直接装 unit；
Windows 浏览器经 localhost 转发访问 `http://localhost:9000/` 的 Web UI）。
镜像网络/端口代理、权重不要放 `/mnt/c` 等注意事项见 **[WEB_UI.md](WEB_UI.md)**。

## 四、网络与客户端接入

- 服务默认监听 `0.0.0.0:9000`。放行端口：
  `sudo firewall-cmd --add-port=9000/tcp --permanent && sudo firewall-cmd --reload`
  （ufw：`sudo ufw allow 9000/tcp`；云主机另需安全组放行）
- Android 客户端把 base URL 指到 `http://<服务器IP>:9000` 即可；
  接口清单与示例见 [API.md](API.md)。
- 服务**无鉴权**，请勿直接暴露公网；跨网访问建议套一层内网穿透/反代 + BUC 等鉴权。

## 五、升级与回滚

```bash
cd /opt/edge-cloud-agent
git pull                                    # 或 rsync 新代码
bash scripts/deploy.sh --skip-models        # 只补依赖 + 重启（权重不动）
```

依赖没变化时可再省一步：`bash scripts/deploy.sh --skip-deps --skip-models`。
回滚 = `git checkout <旧提交>` 后重跑上面命令。数据（`data/*.jsonl`、FAISS 索引、
备注/归档状态）与代码解耦，升级不影响；跨版本字段增减由 `from_dict` 容错处理。

## 六、常见问题

| 症状 | 原因与处置 |
|---|---|
| 健康检查超时（900s） | CPU 加载 9.5GiB 权重可能就是要几分钟；先 `service.sh logs` 看是否仍在加载。真卡死常见于内存不足被 OOM kill（`dmesg | grep -i oom`）→ 加内存/swap，或 `--skip-models` 只跑 embedding 检索降级模式 |
| HF 下载失败/龟速 | 国内网络：`HF_ENDPOINT=https://hf-mirror.com bash scripts/deploy.sh`；或 `.env` 里 `EDGE_MODEL_SOURCE=modelscope` 后重跑（embedding 默认已走 modelscope） |
| 离线/内网服务器 | 在有网机器下好 `models/`，rsync 整个目录到服务器，`deploy.sh --skip-models` |
| 端口被占用 | `ss -ltnp | grep 9000` 找到占用进程；或换端口（见「端口配置的正确姿势」） |
| systemd 反复重启 | 10 分钟 5 次后 systemd 自动停止尝试（unit 已配 StartLimit）。`journalctl -u edge-cloud-agent -n 100` 定位；权重缺失不会崩（业务自动降级），崩溃多为 venv/依赖损坏 → 重跑 `deploy.sh` |
| `pip install` 拉了巨大 CUDA 轮子 | 服务器有 `nvidia-smi` 时 deploy 默认走 GPU 轮子；纯 CPU 机器不会。手工装依赖时用 `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu` 先行 |
| 想换监听用户/路径 | unit 是模板渲染产物：改 `deploy/edge-cloud-agent.service` 后重跑 `deploy.sh --skip-deps --skip-models` |
| 彻底卸载 | `bash scripts/deploy.sh --uninstall`（删 unit）；再按需删仓库目录（含 venv/权重/data） |

## 七、部署产物清单

```
scripts/deploy.sh                  一键部署入口（本文所有流程）
scripts/service.sh                 start/stop/restart/status/logs/health
scripts/run.sh                     进程启动（systemd ExecStart 复用它，勿删）
deploy/edge-cloud-agent.service    systemd unit 模板（__ROOT_DIR__ 等占位符）
/etc/systemd/system/edge-cloud-agent.service   渲染后的安装产物
logs/edge-cloud-agent.log          PID 托管模式的日志（systemd 模式走 journal）
.server.pid                        PID 托管模式的进程号文件
```
