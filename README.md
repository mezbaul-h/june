## 📣 Bulletins

- Now supports

# june

## Local Voice Chatbot: Ollama + HF Transformers + Coqui TTS Toolkit

- [OVERVIEW](#overview)
- [INSTALLATION](#installation)
- [USAGE](#usage)
- [CUSTOMIZATION](#customization)
- [FAQ](#faq)


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

You can also use the REST API service instead of the command-line interface:

```shell
june-va --serve
```

This will start the server at 8000 port. To learn more about the REST API, see [REST-API.md](docs/REST-API.md).

You can also customize behaviour of the program with a json configuration file:

```shell
june-va --config path/to/config.json
```

> [!NOTE]
> The configuration file is optional. To learn more about the structure of the config file, see the [Customization](#customization) section.


## CUSTOMIZATION

See [Customization.md](docs/Customization.md).


## FAQ

See [FAQ.md](docs/FAQ.md).
