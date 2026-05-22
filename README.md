# cc2rep

轻量 Rust 代理：对外暴露 `POST /v1/responses`，对内转发到上游 `chat/completions`。

## 运行

```bash
cargo run -- --config ./config.example.json
```

Codex 或其他客户端将 `base_url` 指向这个服务，并使用 `proxy_api_key` 作为 Bearer token。

## 配置

配置文件使用 JSON，示例见 [config.example.json](./config.example.json)。

当前版本支持：

- `POST /v1/responses`
- `GET /healthz`
- 非流式和 SSE 流式文本输出
- `strict_protocol` 和 `metadata.response_proxy.compatibility.ignored_fields`

当前版本不支持：

- `GET /v1/responses/{id}`
- `cancel/delete`
- 多模态
- tool execution
