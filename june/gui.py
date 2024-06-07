import sys
import time

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .models import llm, stt, tts

model_types = {
    "llm": llm,
    "stt": stt,
    "tts": tts,
}
model_labels = {
    "llm": "LLM",
    "stt": "Speech to Text",
    "tts": "Text to Speech",
}


class LoadModelWorker(QThread):
    model_loaded = pyqtSignal(str, str, object)

    def __init__(self, model_type, model_name):
        super().__init__()
        self.model_type = model_type
        self.model_name = model_name

    def run(self):
        model = model_types[self.model_type].all_models[self.model_name]
        model.noop()

        # Emit the signal when the model is loaded
        self.model_loaded.emit(self.model_type, self.model_name, model)


class ModelWorker(QThread):
    model_state = pyqtSignal(str)

    def __init__(self, models):
        super().__init__()
        self.models = models
        self._should_stop = False

    def run(self):
        # system_initial_context = input("[system]> ")
        generation_args = {
            "max_new_tokens": 200,
            "num_beams": 1,
        }

        llm_model = self.models["llm"]["object"]
        context_id = "cli-chat"

        stt_model = self.models["stt"]["object"]
        tts_model = self.models["tts"]["object"]

        while not self._should_stop:
            self.model_state.emit("listening")
            audio_data = stt_model.record_audio()

            if audio_data is not None:
                self.model_state.emit("transcribing")
                transcription = stt_model.transcribe(audio_data)

                user_input = transcription.strip()

                if user_input:
                    print(f"[user]> {user_input}")

                    if user_input.lower() in ["exit", "halt", "stop", "quit"]:
                        break

                    reply = llm_model.generate(user_input, context_id=context_id, generation_args=generation_args)
                    print(f"[assistant]> {reply['content']}")
                    self.model_state.emit("speaking")
                    tts_model.speak(reply["content"])
            else:
                print("No sound detected.")

            time.sleep(1)  # Pause briefly before next listening

    def stop(self):
        self._should_stop = True


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("june the assistant")
        self.setGeometry(300, 300, 400, 250)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layouts
        self.main_layout = QVBoxLayout()
        central_widget.setLayout(self.main_layout)

        self.models = {
            k: {
                "dropdown": self.create_model_dropdown(f"{model_labels[k]} Model", "---", k),
                "object": None,
            }
            for k in model_types.keys()
        }
        self.load_model_worker = None
        self.model_worker = None

        # Mic icon label
        self.mic_icon = QLabel()
        self.set_mic_icon("green")
        self.main_layout.addWidget(self.mic_icon)

        # Loading label
        self.loading_label = QLabel("")
        self.main_layout.addWidget(self.loading_label)

    def models_loaded(self):
        for k in self.models.keys():
            if not self.models[k]["object"]:
                return False

        return True

    def create_model_dropdown(self, label, placeholder, model_type):
        label = QLabel(label)

        self.main_layout.addWidget(label)

        dropdown = QComboBox()
        dropdown.addItem(placeholder)
        dropdown.addItems(list(model_types[model_type].all_models.keys()))
        dropdown.currentIndexChanged.connect(lambda: self.load_model(dropdown, model_type))

        self.main_layout.addWidget(dropdown)

        return dropdown

    def set_mic_icon(self, color):
        colors = {
            "green": "mic_green.png",
            "red": "mic_red.png",
            "disabled": "mic_disabled.png",
            "bouncing_black": "mic_bouncing_black.gif",  # Assuming you have a gif for bouncing effect
        }
        pixmap = QPixmap(colors[color])
        self.mic_icon.setPixmap(pixmap.scaled(50, 50, Qt.KeepAspectRatio))

    def set_model_dropdowns_disabled(self, disabled):
        for k in self.models.keys():
            self.models[k]["dropdown"].setDisabled(disabled)

    def load_model(self, dropdown, model_type):
        if dropdown.currentIndex() == 0:
            # no model selected
            ...
        else:
            model_name = dropdown.currentText()
            self.loading_label.setText(f"Loading {model_type}:{model_name}...")
            self.set_mic_icon("disabled")
            self.set_model_dropdowns_disabled(True)

            if self.model_worker:
                self.model_worker.stop()
                self.model_worker.wait()
                self.model_worker = None

            # Create and start the worker thread
            self.load_model_worker = LoadModelWorker(model_type, model_name)
            self.load_model_worker.model_loaded.connect(self.on_model_loaded)
            self.load_model_worker.start()

    @pyqtSlot(str, str, object)
    def on_model_loaded(self, model_type, model_name, model_object):
        # Store the loaded model
        self.models[model_type]["object"] = model_object

        # Update the UI
        self.loading_label.setText("")
        self.set_mic_icon("green")
        self.set_model_dropdowns_disabled(False)
        print(f"{model_type} model {model_name} loaded successfully.")

        self.load_model_worker.wait()
        self.load_model_worker = None

        if self.models_loaded():
            # Create and start the worker thread
            self.model_worker = ModelWorker(self.models)
            self.model_worker.model_state.connect(self.on_model_state_changed)
            self.model_worker.start()

    @pyqtSlot(str)
    def on_model_state_changed(self, model_state):
        print("state changed", model_state)

    def start_listening(self):
        self.set_mic_icon("green")
        QTimer.singleShot(2000, self.start_processing)  # Simulate listening for 2 seconds

    def start_processing(self):
        self.set_mic_icon("red")
        QTimer.singleShot(2000, self.start_replying)  # Simulate processing for 2 seconds

    def start_replying(self):
        self.set_mic_icon("bouncing_black")
        QTimer.singleShot(2000, self.finish_replying)  # Simulate replying for 2 seconds

    def finish_replying(self):
        self.set_mic_icon("green")


def main():
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Unhandled exception: {e}")
        sys.exit(1)
