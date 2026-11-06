# 本地自备 PostgreSQL / Redis 启动指南

`docker-compose.local.yml` 用于**复用宿主机已有的 PostgreSQL 和 Redis**，只启动 WeKnora 自身的三个服务。原始的 `docker-compose.yml` 不做任何修改，两者可以并存。

---

## 1. 启动了什么

| 服务 | 镜像 | 端口 | 说明 |
| --- | --- | --- | --- |
| `frontend` | `wechatopenai/weknora-ui` | `${FRONTEND_PORT:-80}:80` | NGINX + 静态资源，反代到 app |
| `app` | `wechatopenai/weknora-app` | `${APP_PORT:-8080}:8080` | Go 主服务，含 Asynq worker |
| `docreader` | `wechatopenai/weknora-docreader` | 50051（仅内网） | Python gRPC 文档解析 |

与官方 `docker-compose.yml` 的差异：

- **不启动 `postgres` / `redis`**，app 直连宿主机实例；
- 项目名 `weknora-local`、网络 `weknora-local`、卷 `weknora-local_*` 全部独立，不会和官方栈互相覆盖；
- 只保留核心三件套，SearXNG、MinIO、Qdrant、Neo4j 等 profile 服务都不在编排内。

---

## 2. 前置条件

### 2.1 Docker

Docker Desktop 4.x 及以上（WSL2 后端）。Windows 下所有命令在**仓库根目录**执行：

```cmd
cd /d D:\Projects\WeKnora
```

### 2.2 PostgreSQL

必须是**能安装扩展的实例**（用 `postgres` 超级用户连接即可）。四个扩展缺一不可：

| 扩展 | 用途 | 缺失后果 |
| --- | --- | --- |
| `uuid-ossp` | 初始迁移建表 | 000000 迁移失败 |
| `vector` | 向量检索 | 000002 迁移失败；`halfvec` 需 ≥0.7 |
| `pg_trgm` | 模糊匹配 | 000002 迁移失败 |
| `pg_search` | BM25 关键词检索 | 000002 迁移失败，**app 起不来** |

自检：

```cmd
psql -h 127.0.0.1 -p 5433 -U postgres -d WeKnora -c "SELECT name, default_version FROM pg_available_extensions WHERE name IN ('uuid-ossp','vector','pg_trgm','pg_search');"
```

推荐直接用 ParadeDB 官方镜像作为外部数据库（`pg_search` + `pgvector` 开箱可用）：

```cmd
docker run -d --name pg -p 5433:5432 ^
  -e POSTGRES_USER=postgres ^
  -e POSTGRES_PASSWORD=<你的密码> ^
  -e POSTGRES_DB=WeKnora ^
  -v pg-data:/var/lib/postgresql/data ^
  paradedb/paradedb:v0.25.11-pg17
```

> `CREATE DATABASE "WeKnora"` 必须带双引号，否则会被折叠成小写 `weknora`，而 app 用 `DB_NAME=WeKnora` 精确匹配，会报数据库不存在。

### 2.3 Redis

任意 6.x / 7.x 实例，默认连 `6379`，需要密码时在 `.env` 里配 `REDIS_PASSWORD`。

### 2.4 .env

```cmd
copy .env.example .env
```

至少确认以下项：

| 变量 | 说明 |
| --- | --- |
| `JWT_SECRET` | 登录 Token 签名，`openssl rand -hex 32` |
| `SYSTEM_AES_KEY` | 敏感字段加密，**32 字节**，`openssl rand -hex 16` |
| `SYSTEM_SIGNING_KEY` | 嵌入会话签名，`openssl rand -hex 32`；不填会回退到 AES Key |
| `DB_USER` / `DB_PASSWORD` / `DB_NAME` | 宿主机数据库账号 |
| `DB_PORT` | 宿主机数据库端口（本地编排会读取它） |
| `DB_HOST` | **写什么都不影响容器**，编排统一覆盖成 `host.docker.internal` |
| `REDIS_PASSWORD` / `REDIS_DB` | Redis 认证 |
| `WEKNORA_VERSION` | 镜像 tag，默认 `latest` |
| `APP_PORT` / `FRONTEND_PORT` | 对外端口 |

> `SYSTEM_AES_KEY` 一旦用于加密就**不能更换**，否则已保存的模型 API Key、向量库凭证等全部无法解密。

---

## 3. 配置优先级

编排里的 `environment` 会覆盖 `.env` 中的同名变量。真正生效的连接参数只有这几个：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `LOCAL_DB_HOST` | `host.docker.internal` | 容器访问宿主机的地址 |
| `LOCAL_DB_PORT` | `.env` 的 `DB_PORT`，再退回 `5432` | 宿主机 PostgreSQL 端口 |
| `LOCAL_REDIS_ADDR` | `host.docker.internal:6379` | 宿主机 Redis 地址 |
| `APP_PORT` | `8080` | app 对外端口 |
| `FRONTEND_PORT` | `80` | 前端对外端口 |
| `WEKNORA_VERSION` | `latest` | 三个镜像的 tag |

**不要**把 `LOCAL_DB_HOST` 改成 `127.0.0.1`：容器里的 `127.0.0.1` 指容器自己。临时覆盖端口用环境变量（Windows 写法）：

```cmd
:: cmd
set LOCAL_DB_PORT=5434 && docker compose -f docker-compose.local.yml up -d

:: PowerShell
$env:LOCAL_DB_PORT=5434; docker compose -f docker-compose.local.yml up -d
```

最省事的做法仍是直接改 `.env` 里的 `DB_PORT`。

查看最终解析结果（不启动容器）：

```cmd
docker compose -f docker-compose.local.yml config
```

---

## 4. 构建

镜像有两种来源：**拉官方镜像**（默认）或**从源码构建**。

### 4.1 拉取官方镜像

```cmd
docker compose -f docker-compose.local.yml pull
```

体积参考：app ≈1.7 GB、docreader ≈4.1 GB、ui ≈86 MB，首次拉取耗时取决于网速。

### 4.2 从源码构建

三个镜像 tag 与编排中的引用一致，构建完成后直接 `up -d` 就会用本地镜像：

```cmd
:: 一键构建（需要 make + Git Bash）
make docker-build-all
```

没有 `make` 时逐条执行：

```cmd
docker build -f docker/Dockerfile.app -t wechatopenai/weknora-app:latest .
docker build -f docker/Dockerfile.docreader -t wechatopenai/weknora-docreader:latest .
docker build -f frontend/Dockerfile -t wechatopenai/weknora-ui:latest frontend/
```

注意事项：

- app 默认链接 Rust 编写的 anydoc，构建需要 Rust 工具链且耗时长。不需要时加 `--build-arg WITH_ANYDOC=0`。
- **不要把本地构建和 `pull` 混用**：同名 tag 会被远程镜像覆盖。想同时保留两者，用独立 tag 构建并同步修改编排里的 `image`，或在 `.env` 里设 `WEKNORA_VERSION=dev` 后构建 `weknora-app:dev`。
- 纯前端改动可以只重建 ui，改 Go 代码只需重建 app。

---

## 5. 启动与停止

### 5.1 启动

```cmd
docker compose -f docker-compose.local.yml up -d
```

启动顺序由健康检查控制：`docreader healthy` → `app` 启动 → `app healthy` → `frontend` 启动。首次启动会执行数据库迁移，`app` 的健康检查有 60s `start_period`，属正常等待。

### 5.2 查看状态

```cmd
docker compose -f docker-compose.local.yml ps
```

期望三个服务都是 `Up`，其中 app 与 docreader 带 `(healthy)`。

### 5.3 重启

三种操作的语义不同，按场景选：

| 命令 | 行为 | 适用场景 |
| --- | --- | --- |
| `restart app` | 只重启进程，**不重读 `.env`**、不改动容器配置 | 进程卡死、内存异常 |
| `up -d app` | 对比配置，有变化则**重建容器**并重读 `.env` | **改完 `.env` 后必须用这个** |
| `down` + `up -d` | 全部删除重建（卷保留） | 编排文件结构改动、疑难问题 |

```cmd
docker compose -f docker-compose.local.yml restart app
docker compose -f docker-compose.local.yml up -d app
docker compose -f docker-compose.local.yml down
docker compose -f docker-compose.local.yml up -d
```

强制重建（配置没变但想重来）：

```cmd
docker compose -f docker-compose.local.yml up -d --force-recreate app
```

### 5.4 停止

```cmd
:: 停止并删除容器/网络，保留数据卷
docker compose -f docker-compose.local.yml down

:: 只停不删
docker compose -f docker-compose.local.yml stop

:: 停止并删除数据卷（会丢上传文件和临时图片，不可恢复）
docker compose -f docker-compose.local.yml down -v
```

---

## 6. 查看日志

```cmd
:: 实时跟随
docker compose -f docker-compose.local.yml logs -f app

:: 最近 200 行后退出
docker compose -f docker-compose.local.yml logs --tail=200 app

:: 多个服务一起看
docker compose -f docker-compose.local.yml logs -f app frontend

:: 只看最近 10 分钟
docker compose -f docker-compose.local.yml logs --since 10m app

:: 显示时间戳
docker compose -f docker-compose.local.yml logs -f -t app
```

### 6.1 启动阶段看什么

| 关键词 | 含义 |
| --- | --- |
| `[startup-env]` | 启动时实际生效的环境变量 |
| `Running database migrations...` | 开始迁移 |
| `Database migrated from version X to Y` | 迁移成功 |
| `dirty` | 迁移失败且标记为脏，需人工处理 |
| `Database connection failed` | 连不上 PostgreSQL，先查 host/port/密码 |
| `pg_search` / `extension` | 扩展缺失 |
| `SSRF` | 外联地址被安全校验拦截 |
| `panic` / `FATAL` | 进程级错误 |

### 6.2 常用过滤

Git Bash / WSL：

```bash
docker compose -f docker-compose.local.yml logs --tail=2000 app | grep -Ei "error|fatal|panic|migrat"
```

cmd / PowerShell 用 `findstr`：

```cmd
docker compose -f docker-compose.local.yml logs --tail=2000 app | findstr /I "error fatal panic migrat"
```

### 6.3 进容器排查

```cmd
docker compose -f docker-compose.local.yml exec app sh
docker compose -f docker-compose.local.yml exec app curl -s http://localhost:8080/health
```

---

## 7. 启动后验证

```cmd
:: 后端健康
curl http://127.0.0.1:8080/health
:: 期望 {"status":"ok"}

:: 前端
curl -o NUL -w "%{http_code}" http://127.0.0.1/
:: 期望 200

:: 迁移状态与扩展
docker exec pg psql -U postgres -d WeKnora -c "SELECT version, dirty FROM schema_migrations;"
docker exec pg psql -U postgres -d WeKnora -c "SELECT extname, extversion FROM pg_extension WHERE extname IN ('pg_search','vector','pg_trgm','uuid-ossp');"
```

`dirty = f` 且扩展齐全即可正常使用。浏览器打开 `http://127.0.0.1`（`FRONTEND_PORT` 非 80 时带端口），注册账号后在「设置 → 模型」配置对话模型和向量模型，来源选 **API**，不需要本机 Ollama。

---

## 8. 常见问题

| 现象 | 排查方向 |
| --- | --- |
| app 反复重启，日志 `Database connection failed` | 宿主机 PG 是否监听 `0.0.0.0`；`pg_hba.conf` 是否允许来自 Docker 网段；端口是否与 `.env` 的 `DB_PORT` 一致 |
| 迁移失败提到 `pg_search` | 宿主机 PG 未安装该扩展；换 ParadeDB 镜像或安装后重启 PG |
| 页面能打开但功能报错 | `schema_migrations` 是否 `dirty`；迁移失败后 app 仍可能启动 |
| 改了 `.env` 没生效 | 用了 `restart`；应使用 `up -d` 重建容器 |
| 客户端连 `localhost:5433` 超时 | Windows 下 `localhost` 会先走 `::1`，Docker 只转发 IPv4；改用 `127.0.0.1` |
| 模型 Base URL 报 SSRF 校验失败 | 目标域名解析到受限网段（常见于代理 fake-IP 的 `198.18.0.0/15`）；把域名或网段加入 `SSRF_WHITELIST_EXTRA` 后重建 app |
| 80 / 8080 端口被占用 | 改 `.env` 的 `FRONTEND_PORT` / `APP_PORT` 后 `up -d` |
| 文档解析一直"处理中" | 看 `docreader` 日志；扫描件还需要配置视觉模型 |
| 前端 502 | app 未 healthy；`depends_on` 会拦住正常启动，手动 `up -d frontend` 可能绕过健康检查 |

---

## 9. 数据与备份

| 数据 | 位置 | 说明 |
| --- | --- | --- |
| 上传文件 | 卷 `weknora-local_data-files` → `/data/files` | `STORAGE_TYPE=local` 时使用 |
| docreader 临时图 | 卷 `weknora-local_docreader-tmp` | 与 app 只读共享，可随时清空 |
| 业务数据 + 向量 | **宿主机 PostgreSQL**（不在编排内） | 备份用 `pg_dump` |
| 队列 | 宿主机 Redis | 丢失只会丢未完成任务 |

```cmd
docker run --rm -v weknora-local_data-files:/data -v %cd%:/backup alpine tar czf /backup/files.tar.gz -C /data .
docker exec pg pg_dump -U postgres -d WeKnora -Fc -f /tmp/WeKnora.dump
docker cp pg:/tmp/WeKnora.dump ./WeKnora.dump
```

---

## 10. 升级

```cmd
:: 1. 备份数据库（见上一节）
:: 2. 拉新镜像
docker compose -f docker-compose.local.yml pull
:: 3. 重建
docker compose -f docker-compose.local.yml up -d
```

- 迁移在 app 启动时自动执行，`AUTO_MIGRATE=true`；
- 指定版本：在 `.env` 里改 `WEKNORA_VERSION=v0.9.0` 再 `pull` + `up -d`；
- 跨大版本升级前先读对应 release notes，部分版本要求先升级数据库扩展或更换签名算法；
- 回滚镜像不会回滚数据库结构，降级前必须从备份恢复。

---

## 11. 命令速查

```cmd
docker compose -f docker-compose.local.yml config          :: 校验并打印最终配置
docker compose -f docker-compose.local.yml pull            :: 拉镜像
docker compose -f docker-compose.local.yml up -d           :: 启动/重建
docker compose -f docker-compose.local.yml ps              :: 状态
docker compose -f docker-compose.local.yml logs -f app     :: 实时日志
docker compose -f docker-compose.local.yml restart app     :: 重启进程
docker compose -f docker-compose.local.yml up -d app       :: 重建单个服务（重读 .env）
docker compose -f docker-compose.local.yml exec app sh     :: 进容器
docker compose -f docker-compose.local.yml stop            :: 停止
docker compose -f docker-compose.local.yml down            :: 停止并删除容器
docker compose -f docker-compose.local.yml down -v         :: 连同数据卷一起删除（危险）
```
