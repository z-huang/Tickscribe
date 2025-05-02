# Tickscribe: Lightweight Real-time Audio Transcription and Summarization

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/z-huang/Tickscribe/commits/main)
[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-GPL-yellow.svg)](https://www.gnu.org/licenses/gpl-3.0)
Tickscribe is a nimble and efficient tool for real-time audio transcription, capable of providing immediate text output from your microphone. It goes a step further by offering optional Language Model (LLM) summarization, making it ideal for capturing meeting notes, lectures, and voice memos quickly and effectively.

## ✨ Key Features:

* **Instant Real-time Audio Transcription:** Witness spoken words transform into text as you speak, providing immediate feedback and a live transcript.
* **Intelligent LLM Summarization (Optional):** Leverage the power of Language Models to condense lengthy transcripts into concise and informative summaries, saving you time and effort.
* **Effortless Setup and Usage:** Designed with simplicity in mind, Tickscribe is easy to install and run, requiring minimal configuration to get started.
* **Local Storage:** Store your audio and transcriptions locally.

## 🎬 Demo Video:

[Link to your Demo Video Here (e.g., YouTube, Loom)]

*(Replace the bracketed text above with a link to your demonstration video. Seeing Tickscribe in action will significantly enhance understanding and engagement.)*

## 🛠️ Installation:

Follow these straightforward steps to get Tickscribe up and running on your system:

1.  **Prerequisites:** Ensure you have **Python 3.11** installed on your machine. You can create a conda enrivonment with `conda create -n tickscribe python=3.11`

2.  **Clone the Repository:** If you haven't already, clone the Tickscribe repository from GitHub:
    ```bash
    git clone https://github.com/z-huang/Tickscribe.git
    cd Tickscribe
    ```

3.  **Install Dependencies:** Navigate to the project directory in your terminal and install the required Python packages:

    Please read [RealtimeSTT's](https://github.com/KoljaB/RealtimeSTT?tab=readme-ov-file#installation) installation guide for `pip install RealtimeSTT`
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run Tickscribe:** Execute the main script to start the transcription process:
    ```bash
    python main.py
    ```

## 🚀 Roadmap: Exciting Features in Development!

We're continuously working to enhance Tickscribe with the following features:

* **Enhanced LLM Summarization Capabilities:** Exploring more advanced Language Models and summarization techniques to provide even more insightful and context-aware summaries.
* **Intuitive Graphical User Interface (GUI):** Developing a user-friendly interface for easier interaction, configuration, and visualization of the transcription and summarization processes.
* **Multi-Language Support:** Expanding language options beyond the initial implementation to cater to a global user base.

## 🔗 References and Inspiration:

Tickscribe draws inspiration and utilizes concepts from the following excellent projects:

* **MLX Hugging Face Integration:** Leveraging the power of MLX for efficient on-device model execution. ([https://huggingface.co/mlx-community](https://huggingface.co/mlx-community))
* **MLX Core Library:** Building upon the flexible and performant MLX framework. ([https://github.com/ml-explore/mlx](https://github.com/ml-explore/mlx))
* **RealtimeSTT:** Learning from and potentially integrating techniques for robust real-time speech-to-text. ([https://github.com/KoljaB/RealtimeSTT](https://github.com/KoljaB/RealtimeSTT))

## 🙏 Contributing:

We welcome contributions from the community! If you have ideas for improvements, bug fixes, or new features, please feel free to open an issue or submit a pull request.

## 📜 License:

This project is licensed under the [GPL-3.0 license](https://www.gnu.org/licenses/gpl-3.0.html). See the `LICENSE` file for more details.