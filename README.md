# LLMWereWolf 大语言模型狼人杀

## 项目描述
本项目是一个简易的基于文本的狼人游戏实现的大型语言模型（LLM）游戏。


## 配置API密钥
项目使用`config.json`文件来存储API配置。请在项目根目录下创建`config.json`文件，并按照以下格式填入您的API信息：

```json
{
    "api_configs": [
        {
            "api_base": "https://example.com/api/v1",
            "model_name": "model-name",
            "api_key": "your-api-key-here"
        },
        // 更多配置...
    ]
}
```

**注意**：请勿将`config.json`文件上传到公共仓库，以保护您的API密钥安全。

## 使用示例
1. 启动游戏：
   ```bash
   python main.py
   ```

2. 可以在chat_logs里看到之前的记录。