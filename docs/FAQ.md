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
