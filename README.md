# june

## Local Voice Chatbot: Ollama + HF Transformers + Coqui TTS Toolkit

- [OVERVIEW](#overview)
- [INSTALLATION](#installation)
- [USAGE](#usage)
- [CUSTOMIZATION](#customization)
- [FAQ](#frequently-asked-questions)


## OVERVIEW

[**june**](https://github.com/mezbaul-h/june) is a local voice chatbot that combines the power of Ollama (for language model capabilities), Hugging Face Transformers (for speech recognition), and the Coqui TTS Toolkit (for text-to-speech synthesis). It provides a flexible, privacy-focused solution for voice-assisted interactions on your local machine, ensuring that no data is sent to external servers.

![demo-text-only-interaction](demo.gif)

### Interaction Modes

- **Text Input/Output:** Provide text inputs to the assistant and receive text responses.
- **Voice Input/Text Output:** Use your microphone to give voice inputs, and receive text responses from the assistant.
- **Text Input/Audio Output:** Provide text inputs and receive both text and synthesised audio responses from the assistant.
- **Voice Input/Audio Output (Default):** Use your microphone for voice inputs, and receive responses in both text and synthesised audio form.


## INSTALLATION

### Pre-requisites
- [**Ollama**](https://github.com/ollama/ollama)
- [**Python**](https://www.python.org/downloads/) 3.10 or greater (with _pip_)
- **Python** development package (e.g. `apt install python3-dev` for Debian) — **only for GNU/Linux**
- **PortAudio** development package (e.g. `apt install portaudio19-dev` for Debian) — **only for GNU/Linux**
- **PortAudio** (e.g. `brew install portaudio` using Homebrew) — **only for macOS**
- [**Microsoft Visual C++**](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 14.0 or greater — **only for Windows**

### From Source

#### Method 1: Direct Installation

To install **june** directly from the GitHub repository:

```shell
pip install git+https://github.com/mezbaul-h/june.git@master
```

#### Method 2: Clone and Install

Alternatively, you can clone the repository and install it locally:

```shell
git clone https://github.com/mezbaul-h/june.git
cd june
pip install .
```


## USAGE

Pull the language model (default is `llama3.1:8b-instruct-q4_0`) with Ollama first, if you haven't already:

```shell
ollama pull llama3.1:8b-instruct-q4_0
```

Next, run the program (with default configuration):

```shell
june-va
```

This will use [llama3.1:8b-instruct-q4_0](https://ollama.com/library/llama3.1:8b-instruct-q4_0) for LLM capabilities, [openai/whisper-small.en](https://huggingface.co/openai/whisper-small.en) for speech recognition, and `tts_models/en/ljspeech/glow-tts` for audio synthesis.

You can also customize behaviour of the program with a json configuration file:

```shell
june-va --config path/to/config.json
```

> [!NOTE]
> The configuration file is optional. To learn more about the structure of the config file, see the [Customization](#customization) section.


## CUSTOMIZATION

The application can be customised using a configuration file. The config file must be a JSON file. The default configuration is as follows:

```json
{
    "llm": {
        "disable_chat_history": false,
        "model": "llama3.1:8b-instruct-q4_0"
    },
    "stt": {
        "device": "torch device identifier (`cuda` if available; otherwise `cpu`)",
        "generation_args": {
            "batch_size": 8
        },
        "model": "openai/whisper-small.en"
    },
    "tts": {
        "device": "torch device identifier (`cuda` if available; otherwise `cpu`)",
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


## Frequently Asked Questions

### Q: How does the voice input work?

After seeing the `[system]> Listening for sound...` message, you can speak directly into the microphone. Unlike typical voice assistants, there's no wake command required. Simply start speaking, and the tool will automatically detect and process your voice input. Once you finish speaking, maintain silence for 3 seconds to allow the assistant to process your voice input.

### Q: Can I clone a voice?

Many of the models (e.g., `tts_models/multilingual/multi-dataset/xtts_v2`) supported by Coqui's TTS Toolkit support voice cloning. You can use your own speaker profile with a small audio clip (approximately 1 minute for most models). Once you have the clip, you can instruct the assistant to use it with a custom configuration like the following:

```json
{
  "tts": {
    "model": "tts_models/multilingual/multi-dataset/xtts_v2",
    "generation_args": {
      "language": "en",
      "speaker_wav": "/path/to/your/target/voice.wav"
    }
  }
}
```

### Q: Can I use a remote Ollama instance with june?

Yes, you can easily integrate a remotely hosted Ollama instance with june instead of using a local instance. Here's how to do it:
1. Set the `OLLAMA_HOST` environment variable to the appropriate URL of your remote Ollama instance.
2. Run the program as usual.

#### Example:

To use a remote Ollama instance, you would use a command like this:

```shell
OLLAMA_HOST=http://localhost:11434 june-va
```
