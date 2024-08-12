## CUSTOMIZATION

The application can be customised using a configuration file. The config file must be a JSON file. The default configuration is as follows:

```json
{
    "llm": {
        "disable_chat_history": false,
        "model": "llama3.1:8b-instruct-q4_0"
    },
    "stt": {
        "device": "torch device identifier (`cuda` if available; otherwise `cpu`",
        "generation_args": {
            "batch_size": 8
        },
        "model": "openai/whisper-small.en"
    },
    "tts": {
        "device": "torch device identifier (`cuda` if available; otherwise `cpu`",
        "model": "tts_models/en/ljspeech/glow-tts"
    }
}
```

When you use a configuration file, it overrides the default configuration but does not overwrite it. So you can partially modify the configuration if you desire. For instance, if you do not wish to use speech recognition and only want to provide prompts through text, you can disable that by using a config file with the following configuration:

```json
{
  "stt": null
}
```

Similarly, you can disable the audio synthesiser, or both, to only use the virtual assistant in text mode.

If you only want to modify the device on which you want to load a particular type of model, without changing the other default attributes of the model, you could use:

```json
{
  "tts": {
    "device": "cpu"
  }
}
```

### Configuration Attributes

#### `llm` - Language Model Configuration

- `llm.device`: Torch device identifier (e.g., `cpu`, `cuda`, `mps`) on which the pipeline will be allocated.
- `llm.disable_chat_history`: Boolean indicating whether to disable or enable chat history. Enabling chat history will make interactions more dynamic, as the model will have access to previous contexts, but it will consume more processing power. Disabling it will result in less interactive conversations but will use fewer processing resources.
- `llm.model`: Name of the text-generation model tag on Ollama. Ensure this is a valid model tag that exists on your machine.
- `llm.system_prompt`: Give a system prompt to the model. If the underlying model does not support a system prompt, an error will be raised.

#### `stt` - Speech-to-Text Model Configuration

- `tts.device`: Torch device identifier (e.g., `cpu`, `cuda`, `mps`) on which the pipeline will be allocated.
- `stt.generation_args`: Object containing generation arguments accepted by Hugging Face's speech recognition pipeline.
- `stt.model`: Name of the speech recognition model on Hugging Face. Ensure this is a valid model ID that exists on Hugging Face.

#### `tts` - Text-to-Speech Model Configuration

- `tts.device`: Torch device identifier (e.g., `cpu`, `cuda`, `mps`) on which the pipeline will be allocated.
- `tts.generation_args`: Object containing generation arguments accepted by Coqui's TTS API.
- `tts.model`: Name of the text-to-speech model supported by the Coqui's TTS Toolkit. Ensure this is a valid model ID.
