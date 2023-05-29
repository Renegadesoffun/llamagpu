import sys
import re
import logging
from PyQt5.QtWidgets import QApplication, QFileDialog, QVBoxLayout, QPushButton, QLabel, QSlider, QMainWindow, QTextEdit, QProgressBar, QAction, QStatusBar, QMessageBox, QWidget, QLineEdit
from PyQt5.QtCore import Qt, QProcess, QThread, pyqtSignal
import subprocess
import concurrent.futures
import os

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

# Clear the log file
with open('llama_gui.log', 'w') as f:
    pass

logging.basicConfig(filename='llama_gui.log', level=logging.INFO)


class LlamaThread(QThread):
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    loading_signal = pyqtSignal()

    def __init__(self, llama_file, gpu_layers, parent=None):
        super().__init__(parent)
        self.llama_file = llama_file
        self.gpu_layers = gpu_layers
        self.process = None
        self.ready = False

    def run(self):
        cmd = 'D:\\llama-master-ac7876a-bin-win-cublas-cu12.1.0-x64\\main.exe -m {} -r user: --interactive-first --gpu-layers {}'.format(self.llama_file, str(self.gpu_layers))
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.run_subprocess, cmd)
            return_value = future.result()

    def run_subprocess(self, cmd):
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_CONSOLE)
        
        while True:
            if self.process.poll() is not None:
                break

            if self.process.stdout is not None:
                output = self.process.stdout.readline().decode()
                if output:
                    if " If you want to submit another line, end your input with " in output:
                        self.ready = True
                    if self.ready:
                        self.output_signal.emit(output.strip())  # strip to remove newline

            if self.process.stderr is not None:
                error = self.process.stderr.readline().decode()
                if error:
                    if 'loading' in error.lower():
                        continue
                    elif 'fully loaded' in error.lower():
                        self.loading_signal.emit()
                    else:
                        error_message = f'Error: {error}'
                        logging.error(error_message)



    def write(self, message):
        if self.process is not None and self.process.poll() is None and self.process.stdin is not None:
            self.process.stdin.write((f"### Instruction: {message}\n\n### Response:\n").encode())
            self.process.stdin.flush()
            self.process.stdin.write(('\n').encode())
            self.process.stdin.flush()



    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()


class LlamaGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Llama Program')
        self.llama_file = ''
        self.gpu_layers = 0
        self.process = None
        
        self.llama_thread = None

        # Create controls
        self.browse_button = QPushButton('Browse...')
        self.browse_button.clicked.connect(self.browse_for_file)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(60)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self.slider_value_changed)

        self.slider_label = QLabel('GPU Layers: 0')

        self.run_button = QPushButton('Run LLaMA')
        self.run_button.clicked.connect(self.run_program)
        self.run_button.setEnabled(False)

        self.stop_button = QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_program)
        self.stop_button.setEnabled(False)

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText('Enter your message here...')

        self.ai_output = QTextEdit()
        self.ai_output.setReadOnly(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        # Create a QVBoxLayout object for the main layout
        layout = QVBoxLayout()

        # Create a QLabel object for the LLaMA file label
        self.llama_file_label = QLabel("LLaMA File:")

        # Create a QLineEdit object for the LLaMA file line edit
        self.llama_file_line_edit = QLineEdit()

        self.gpu_layers_label = QLabel("GPU Layers:")

        # Create a QSlider object for the GPU layers slider
        self.gpu_layers_slider = QSlider(Qt.Orientation.Horizontal)

        # Connect the valueChanged signal to the slider_value_changed slot
        self.gpu_layers_slider.valueChanged.connect(self.slider_value_changed)

        # Create a QLabel object for the GPU layers value label
        self.gpu_layers_value_label = QLabel("0")

        # Create a QTextEdit object for the output text edit
        self.output_text_edit = QTextEdit()

        # Add the LLaMA file label to the layout
        layout.addWidget(self.llama_file_label)

        # Add the LLaMA file line edit to the layout
        layout.addWidget(self.llama_file_line_edit)

        # Add the GPU layers label to the layout
        layout.addWidget(self.gpu_layers_label)

        # Add other widgets to the layout
        layout.addWidget(self.browse_button)
        layout.addWidget(self.gpu_layers_slider)
        layout.addWidget(self.gpu_layers_value_label)
        layout.addWidget(self.run_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.output_text_edit)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.user_input)
        layout.addWidget(self.ai_output)

        # Create a central widget
        central_widget = QWidget()

        # Set the layout for the central widget
        central_widget.setLayout(layout)

        # Set the central widget for the QMainWindow
        self.setCentralWidget(central_widget)

        # Create a status bar and set it as the status bar for the main window
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Add a label to the status bar
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)

        # Create menu bar
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')
        open_action = QAction('Open', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.browse_for_file)
        file_menu.addAction(open_action)
        save_action = QAction('Save', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close_application)
        file_menu.addAction(exit_action)

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Create a button to send user input
        self.send_button = QPushButton('Send')
        self.send_button.clicked.connect(self.send_user_input)

        # Add the send button and connect the enter key to the send_user_input method
        layout.addWidget(self.send_button)
        self.user_input.returnPressed.connect(self.send_user_input)
        if self.llama_thread:
            self.llama_thread.loading_signal.connect(self.handle_loading_complete)

    def handle_loading_complete(self):
        self.progress_bar.setValue(100)


    def display_error(self, error):
        QMessageBox.warning(self, 'Error', error)

    def browse_for_file(self):
        filename, _ = QFileDialog.getOpenFileName(None, 'Select llama file', '', '*.bin')
        if filename:
            self.llama_file = filename
            self.llama_file_line_edit.setText(filename)  # Update the QLineEdit with the selected filename
            self.run_button.setEnabled(True)

    def slider_value_changed(self, value):
        self.gpu_layers = value
        self.gpu_layers_value_label.setText(str(value))  # Update the QLabel with the current slider value

    def run_program(self):
        if not self.llama_file:
            QMessageBox.warning(self, 'Error', 'No LLaMA file selected')
            return

        if self.llama_thread and self.llama_thread.isRunning():
            QMessageBox.warning(self, 'Error', 'LLaMA program already running')
            return

        self.ai_output.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        self.llama_thread = LlamaThread(self.llama_file, self.gpu_layers)

        self.llama_thread.output_signal.connect(self.handle_output)

        self.llama_thread.finished_signal.connect(self.handle_finished)
        self.llama_thread.start()

        print('LLaMA program started')  # Print a message when the LLaMA program starts

        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_program(self):
        if self.llama_thread and self.llama_thread.isRunning():
            self.llama_thread.stop()
        self.stop_button.setEnabled(False)

        print('LLaMA program stopped')  # Print a message when the LLaMA program stops

    def send_user_input(self):
        # Get the user input and clear the input field
        user_input = self.user_input.text()
        self.user_input.clear()

        # Add the user input to the AI output with the 'user:' prefix
        self.ai_output.append(f'user: {user_input}')

        # Send the user input to the AI
        if self.llama_thread and self.llama_thread.isRunning():
            self.llama_thread.write(user_input)
            print('Message sent to LLaMA program')  # Print a message when a message is sent to the LLaMA program
        else:
            QMessageBox.warning(self, 'Error', 'LLaMA program not running')


    def handle_output(self, output):
        self.ai_output.append(f'ai: {output}')

    def handle_finished(self):
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.hide()

    def save_file(self):
        filename, _ = QFileDialog.getSaveFileName(None, 'Save output', '', '*.txt')
        if filename:
            with open(filename, 'w') as f:
                f.write(self.ai_output.toPlainText())

    def close_application(self):
        if self.llama_file:
            message_box = QMessageBox()
            message_box.setText(f'Do you want to save changes to {self.llama_file}?')
            message_box.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            message_box.setDefaultButton(QMessageBox.Save)
            result = message_box.exec_()
            if result == QMessageBox.Save:
                self.save_file()
            elif result == QMessageBox.Cancel:
                return
        QApplication.quit()

if __name__ == '__main__':
    app = QApplication([])
    window = LlamaGUI()
    window.show()
    sys.exit(app.exec_())
