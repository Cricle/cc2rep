# cc2rep

轻量 Rust 代理：对外暴露 OpenAI Responses 风格接口，对内转发到上游 `chat/completions`。

## 特性

- **自动探测**：启动时自动探测上游 provider 能力（tool_choice、reasoning 等），无需手动配置
- **本地工具执行**：支持配置本地命令作为工具，代理自动执行并回传结果
- **并行工具调用**：多个工具调用可并行执行
- **Reasoning 支持**：自动提取和转发 reasoning_content
- **会话管理**：内存存储会话，定时清理过期条目
- **多 provider 支持**：已测试 DeepSeek、MiMo 等

## 快速开始

```bash
# 最小配置只需 URL + API Key + Model
cargo run -- --config ./config.example.json
```

Codex 或其他客户端将 `base_url` 指向这个服务，并使用 `proxy_api_key` 作为 Bearer token。

## 配置

配置文件使用 JSON，示例见 [config.example.json](./config.example.json)。

### 最小配置

```json
{
  "proxy_host": "127.0.0.1",
  "proxy_port": 8800,
  "proxy_api_key": "your-proxy-key",
  "upstream_base_url": "https://api.deepseek.com/v1",
  "upstream_chat_path": "/chat/completions",
  "upstream_model": "deepseek-v4-pro",
  "upstream_api_key": "your-api-key",
  "upstream_api_key_header_name": "Authorization",
  "upstream_api_key_prefix": "Bearer "
}
```

### 完整配置

```json
{
  "proxy_host": "127.0.0.1",
  "proxy_port": 8800,
  "proxy_api_key": "your-proxy-key",
  "upstream_base_url": "https://api.deepseek.com/v1",
  "upstream_chat_path": "/chat/completions",
  "upstream_model": "deepseek-v4-pro",
  "upstream_api_key": "your-api-key",
  "upstream_headers": {},
  "upstream_body": {},
  "upstream_api_key_header_name": "Authorization",
  "upstream_api_key_prefix": "Bearer ",
  "request_timeout_seconds": 120.0,
  "upstream_supports_image_input": false,
  "drop_input_reasoning": false,
  "drop_tools": false,
  "response_ttl_seconds": 3600,
  "max_auto_tool_rounds": 8,
  "local_tools": {
    "get_weather": {
      "command": "sh",
      "args": ["-c", "read JSON; echo $JSON | jq -r '.arguments.city'"],
      "stdin_json": true,
      "output_json": true,
      "timeout_seconds": 5.0
    }
  },
  "model_aliases": {
    "gpt-5-codex": "deepseek-v4-pro"
  }
}
```

### 配置项说明

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `proxy_host` | `127.0.0.1` | 监听地址 |
| `proxy_port` | `8800` | 监听端口 |
| `proxy_api_key` | 必填 | 代理 API Key |
| `upstream_base_url` | 必填 | 上游 API 地址 |
| `upstream_chat_path` | `/v1/chat/completions` | chat/completions 路径 |
| `upstream_model` | 必填 | 上游模型名称 |
| `upstream_api_key` | 必填 | 上游 API Key |
| `request_timeout_seconds` | `120.0` | 请求超时（秒） |
| `response_ttl_seconds` | `3600` | 会话过期时间（秒） |
| `max_auto_tool_rounds` | `8` | 最大自动工具轮次 |
| `upstream_supports_image_input` | `false` | 上游是否支持图片输入 |
| `drop_input_reasoning` | `false` | 丢弃输入中的 reasoning |
| `drop_tools` | `false` | 丢弃工具，强制纯文本 |

### 自动探测

启动时会自动探测以下能力，无需手动配置：

- `upstream_supports_named_tool_choice` — 命名 tool_choice 支持
- `upstream_supports_tool_choice_required` — `tool_choice: "required"` 支持
- `upstream_supports_reasoning_content` — reasoning_content 支持

探测失败时保守降级（默认 false），不影响启动。

### tool_choice 兼容性

代理会自动处理 tool_choice 不兼容的情况：

- 命名 tool_choice 不支持 → 回退到 `"required"`
- `"required"` 不支持 → 回退到 `"auto"`
- 上游返回 tool_choice 错误 → 自动去掉 tool_choice 重试

### 本地工具

配置 `local_tools` 后，代理会自动执行匹配的工具调用：

```json
{
  "local_tools": {
    "tool_name": {
      "command": "python3",
      "args": ["-c", "import json,sys; d=json.load(sys.stdin); print(json.dumps({'result': eval(d['arguments']['expr'])}))"],
      "stdin_json": true,
      "output_json": true,
      "timeout_seconds": 10.0
    }
  }
}
```

- `stdin_json: true` — 通过 stdin 传入 `{"name", "call_id", "arguments"}`
- `output_json: true` — 要求 stdout 输出合法 JSON
- 多个工具调用会并行执行

### 模型别名

```json
{
  "model_aliases": {
    "gpt-5-codex": "deepseek-v4-pro"
  }
}
```

客户端请求 `gpt-5-codex` 时，实际使用 `deepseek-v4-pro`。

## 支持的 API

- `POST /v1/responses` — 创建响应（流式/非流式）
- `GET /v1/responses/{id}` — 获取响应
- `GET /v1/responses/{id}/input_items` — 获取输入项
- `POST /v1/responses/{id}/cancel` — 取消响应
- `DELETE /v1/responses/{id}` — 删除响应
- `GET /healthz` — 健康检查

## 已测试 Provider

| Provider | 模型 | 基础对话 | 流式 | Reasoning | Tool Calls |
|----------|------|----------|------|-----------|------------|
| DeepSeek | deepseek-v4-pro | ✓ | ✓ | ✓ | ✓ |
| DeepSeek | deepseek-reasoner | ✓ | ✓ | ✓ | ✓ (自动重试) |
| MiMo | mimo-v2.5-pro | ✓ | ✓ | ✓ | ✓ |

## 构建

```bash
# 开发构建
cargo build

# 运行测试
cargo test

# 发布构建
cargo build --release
```

## 许可证

MIT
