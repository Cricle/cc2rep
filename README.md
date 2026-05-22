# cc2rep

轻量 Rust 代理：对外暴露 OpenAI Responses 风格接口，对内转发到上游 `chat/completions`。

发布说明：

- CI 产出的 Linux release 二进制使用 `musl` 静态链接，便于直接分发运行。

## 运行

```bash
cargo run -- --config ./config.example.json
```

Codex 或其他客户端将 `base_url` 指向这个服务，并使用 `proxy_api_key` 作为 Bearer token。

## 配置

配置文件使用 JSON，示例见 [config.example.json](./config.example.json)。

`upstream_body` 可用于给上游请求追加自定义 JSON 字段，例如一些 provider 专有参数。代理自身生成的字段如 `model`、`messages`、`stream`、`tools` 仍然优先覆盖同名键。

兼容开关：

- `drop_input_reasoning`: 丢弃客户端输入里的 `reasoning` 项，避免继续把思考链历史传给上游。
- `drop_tools`: 丢弃客户端声明的 `tools`、`tool_choice` 以及输入里的 `function_call` / `function_call_output`，强制退化为纯文本对话。

当前版本支持：

- `POST /v1/responses`
- `GET /v1/responses/{id}`
- `GET /v1/responses/{id}/input_items`
- `POST /v1/responses/{id}/cancel`
- `DELETE /v1/responses/{id}`
- `GET /healthz`
- 非流式和 SSE 流式输出
- `previous_response_id` 历史回放
- `function` tools、`function_call`、`function_call_output` 闭环
- 可选的代理内本地工具自动执行
- 多模态输入中的 `image_url` / `input_image` 转发
- `strict_protocol` 和 `metadata.response_proxy.compatibility.ignored_fields`

说明：

- 响应对象和输入项仅保存在内存中，不落库。
- 图片输入转发需要启用 `upstream_supports_image_input`。
- 未配置 `local_tools` 时，工具执行仍由客户端按 `function_call` / `function_call_output` 往返完成。
- 配置了 `local_tools` 后，非流式请求里命中的 `function` 工具会由代理自动执行，并自动把 `function_call_output` 回传给上游继续完成对话。
- 配置了 `local_tools` 后，流式请求也会自动执行命中的本地工具；客户端看到的是自动续跑后的最终回答流，而不是中间的内部工具轮次。

`local_tools` 约定：

- `command` / `args` 定义本地可执行命令。
- 默认会把 `{"name","call_id","arguments"}` 作为 JSON 写入工具进程的 `stdin`。
- 默认要求工具从 `stdout` 输出合法 JSON，代理会把这段 JSON 作为 `function_call_output.output` 继续发给上游。
- 如果你的工具只输出纯文本，可以把 `output_json` 设为 `false`。
