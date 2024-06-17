# june


## OVERVIEW

**june** is a command-line application that serves as a high-level wrapper around the Hugging Face Transformers library. It provides a user-friendly interface to interact with large language models (LLMs) for both text-based and audio-based communication. The app supports customization through a configuration file, allowing users to select specific models and adjust generation parameters.

### Features

- **Text-Based Interaction:** Communicate with the LLM by typing text inputs. The LLM generates text responses based on the input.
- **Audio-Based Interaction:** Speak to the application, which uses a speech-to-text model to transcribe the audio. The transcribed text is sent to the LLM, and the generated response is synthesized using a text-to-speech model and played through the speaker.
- **Customizable Configurations:** Use a configuration file to select models for speech-to-text, text generation, and text-to-speech, as well as to adjust various generation parameters of the LLM.


## SETUP

### Pre-requisites:
- [Ollama](https://github.com/ollama/ollama)
- Python `3.10+` (with _pip_)
- Ubuntu `22.04+`

**NOTE:** These are not strict pre-requisites. The project was built and tested on Python `3.10.14` and Ubuntu `24.04 LTS`, so it should run on any Debian-based OS with reasonably recent versions of Python 3.

You will also need the following native package installed on your machine:

```shell
apt install portaudio19-dev  # requirement for PyAudio
```

### Steps

1. Clone the project and go into the directory:
    ```shell
    git clone https://github.com/mezbaul-h/june.git
    cd june
    ```
2. Install project dependencies:
    ```shell
    pip install -r requirements.txt
    ```
    or if you **do not have a GPU**, install CPU-specific dependencies with this:
    ```shell
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
    ```
    Just a reminder, running the program on a CPU will be very slow, since LLM inference is inefficient on CPUs, especially for larger models.

You are now ready to use the program!


## USAGE

This is a CLI program. You can execute it by running the library module as a script:

```shell
python -m june --config path/to/config.json
```

**NOTE:** The configuration file is mandatory and the app cannot run without it. To learn more about the structure of the config file, see the [Configuration](#configuration) section.

## CONFIGURATION

The application can be customized using a config file. The config file should be a JSON file like the following:

```json
{
  "llm": {
    "device": "cuda",
    "disable_chat_history": true,
    "generation_args": {
        "max_new_tokens": 200,
        "num_beams": 1,
        "return_full_text": false
    },
    "model":"stabilityai/stablelm-2-zephyr-1_6b",
    "system_prompt": "You are a pirate chatbot who always responds in pirate speak!"
  },
  "stt": {
    "device": "cpu",
    "generation_args": {
      "batch_size": 8
    },
    "model": "openai/whisper-small.en"
  },
  "tts": {
    "device": "cpu",
    "generation_args": {
      "speaker_embeddings": 999
    },
    "model": "microsoft/speecht5_tts"
  }
}
```

### Configuration Keys

#### `llm` - Language Model Configuration

- `llm.device`: Torch device identifier (e.g., `cpu`, `cuda`, `mps`) on which the pipeline will be allocated. Defaults to `cuda` if available; otherwise, defaults to `cpu`.
- `llm.disable_chat_history`: Boolean indicating whether to disable or enable chat history. Enabling chat history will make interactions more dynamic, as the model will have access to previous contexts, but it will consume more processing power. Disabling it will result in less interactive conversations but will use fewer processing resources.
- `llm.generation_args`: Object containing generation arguments accepted by Hugging Face's text-generation pipeline.
- `llm.model`: Name of the text-generation model on Hugging Face. Ensure this is a valid model ID that exists on Hugging Face. This field is **required**.
- `llm.system_prompt`: Give a system prompt to the model. If the underlying model does not support a system prompt and raises an error, it will be suppressed. This field is **optional**.

#### `stt` - Speech-to-Text Model Configuration

- `tts.device`: Torch device identifier (e.g., `cpu`, `cuda`, `mps`) on which the pipeline will be allocated. Defaults to `cuda` if available; otherwise, defaults to `cpu`.
- `stt.generation_args`: Object containing generation arguments accepted by Hugging Face's speech recognition pipeline.
- `stt.model`: Name of the speech recognition model on Hugging Face. Ensure this is a valid model ID that exists on Hugging Face. This field is optional; if omitted, the app will operate in text-based mode without speech recognition functionality.

#### `tts` - Text-to-Speech Model Configuration

- `tts.device`: Torch device identifier (e.g., `cpu`, `cuda`, `mps`) on which the pipeline will be allocated. Defaults to `cuda` if available; otherwise, defaults to `cpu`.
- `tts.generation_args`: Object containing generation arguments accepted by Hugging Face's text-to-speech pipeline.
- `tts.model`: Name of the text-to-speech model on Hugging Face. Ensure this is a valid model ID that exists on Hugging Face. This field is optional; if omitted, the app will operate in text-based mode without speech recognition functionality.


## TESTED MODELS

### LLM

- Berkeley Starling
- Google Gemma
- Meta Llama 3
- Microsoft Phi-3
- Mistral AI Mistral/Mixtral
- Nous Hermes
- OpenChat 3.5
- Qwen2
- Stability AI Stable LM

### Speech-to-Text

- OpenAI Whisper

### Text-to-Speech

- Microsoft SpeechT5
