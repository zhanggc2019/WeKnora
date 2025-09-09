#!/bin/bash

# WeKnora 自动初始化脚本
# 使用环境变量完成系统初始化，跳过Web界面

set -e

echo "=== WeKnora 自动初始化 ==="

# 检查必要的环境变量
if [ -z "$LLM_API_KEY" ]; then
    echo "错误: 请设置 LLM_API_KEY 环境变量"
    echo "示例: export LLM_API_KEY=your-openai-api-key"
    exit 1
fi

if [ -z "$LLM_BASE_URL" ]; then
    LLM_BASE_URL="https://api.openai.com/v1"
    echo "使用默认 LLM_BASE_URL: $LLM_BASE_URL"
fi

# 检查服务是否启动
if ! curl -f http://localhost:8080/api/v1/health > /dev/null 2>&1; then
    echo "错误: WeKnora 服务未启动，请先启动服务"
    exit 1
fi

# 检查是否已初始化
INIT_STATUS=$(curl -s http://localhost:8080/api/v1/initialization/status | jq -r '.data.initialized')
if [ "$INIT_STATUS" = "true" ]; then
    echo "系统已初始化，无需重复操作"
    exit 0
fi

# 创建初始化配置
cat > /tmp/init-config.json << EOF
{
  "llm": {
    "source": "remote",
    "modelName": "${LLM_MODEL_NAME:-gpt-3.5-turbo}",
    "baseUrl": "${LLM_BASE_URL}",
    "apiKey": "${LLM_API_KEY}"
  },
  "embedding": {
    "source": "remote",
    "modelName": "${EMBEDDING_MODEL_NAME:-text-embedding-ada-002}",
    "baseUrl": "${EMBEDDING_BASE_URL:-$LLM_BASE_URL}",
    "apiKey": "${EMBEDDING_API_KEY:-$LLM_API_KEY}",
    "dimension": ${EMBEDDING_DIMENSION:-1536}
  },
  "rerank": {
    "modelName": "${RERANK_MODEL_NAME:-}",
    "baseUrl": "${RERANK_BASE_URL:-}",
    "apiKey": "${RERANK_API_KEY:-}"
  },
  "multimodal": {
    "enabled": ${ENABLE_MULTIMODAL:-false},
    "storageType": "${STORAGE_TYPE:-minio}"
  },
  "documentSplitting": {
    "chunkSize": ${CHUNK_SIZE:-1000},
    "chunkOverlap": ${CHUNK_OVERLAP:-200},
    "separators": ["\\n\\n", "\\n", "。", "！", "？", ".", "!", "?"]
  }
}
EOF

echo "正在执行系统初始化..."

# 调用初始化API
RESPONSE=$(curl -s -X POST http://localhost:8080/api/v1/initialization/initialize \
  -H "Content-Type: application/json" \
  -d @/tmp/init-config.json)

if echo "$RESPONSE" | grep -q '"success":true'; then
    echo "✅ 系统初始化成功！"
    echo "现在可以访问 http://localhost 使用系统"
else
    echo "❌ 初始化失败:"
    echo "$RESPONSE"
    exit 1
fi

# 清理临时文件
rm -f /tmp/init-config.json

echo "=== 初始化完成 ==="