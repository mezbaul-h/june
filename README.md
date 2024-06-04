# june

june is a fun and interactive command-line application that leverages the latest advancements in Speech-to-Text (STT), Large Language Models (LLM), and Text-to-Speech (TTS) technologies. This playful assistant is designed to make your interactions more engaging and efficient.

### How It Works:

1. **Voice Prompt:** The program begins by displaying a prompt and listens for your input through the microphone.
2. **Speech-to-Text (STT):** our spoken words are transcribed into text using _OpenAI_'s _Whisper_.
3. **Large Language Model (LLM):** The transcribed text is processed by _Microsoft_'s _Phi-3 (Phi-3-Mini-128K-Instruct)_, which generates a thoughtful and relevant response.
4. **Text-to-Speech (TTS):** The response is then converted back into natural-sounding speech using _XTTSv2_'s text-to-speech API.
5. **Voice Output:** Finally, the response is played back to you through your speaker, creating a seamless conversational experience.


## SETUP

### Pre-requisites:
- Python `3.10+` _(with pip)_
- Ubuntu `22.04+`

_NOTE_: These are not strict pre-requisites. The project was built and tested on Python `3.10.14` and Ubuntu `24.04 LTS`, so it should run on any Debian-based OS with reasonably recent versions of Python 3.

You will also need the following native package installed on your machine:
```shell
apt install ffmpeg  # requirement for openai-whisper
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
    Just a reminder, running the program on a CPU will be very slow, since LLM inference is inefficient on CPUs, especially for a model of this size.

You are now ready to use the program!


## USAGE

This is a CLI program. You can execute it by running:

```shell
python -m june
```

Initially, you will be given a chance to provide a **system prompt**, which instructs the LLM model to behave in a certain way. An example prompt could be:

```shell
[system]> Respond like a detective solving a mystery with every answer.
```

The program will then download (only once; subsequent runs use previously downloaded files) and load the LLM, Text-to-Speech, and Speech-to-Text models, and start listening to your microphone. Say something, and it will reply both on the CLI and through your speaker. You can have a conversation on pretty much any topic you like.
