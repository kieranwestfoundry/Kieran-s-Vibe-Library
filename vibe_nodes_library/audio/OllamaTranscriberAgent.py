try:
    import static_ffmpeg

    static_ffmpeg.add_paths()
except ImportError:
    pass  # Handled inside _process()

import os
import tempfile
from typing import Any

from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterMode,
    ParameterTypeBuiltin,
)
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_types.parameter_audio import ParameterAudio
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File
from griptape_nodes.traits.options import Options


class LocalWhisperTranscriber(ControlNode):
    """Runs ASR locally using HuggingFace Transformers, with advanced chunking."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            ParameterAudio(
                name="input_audio",
                tooltip="Input audio file to transcribe",
                allow_output=False,
            )
        )

        self.add_parameter(
            ParameterBool(
                name="use_hf_custom_mode",
                tooltip="Toggle to download and use a custom HuggingFace model repo.",
                default_value=False,
                ui_options={"display_name": "HuggingFace Custom Mode"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="model_preset",
                tooltip="Select a standard Whisper model.",
                default_value="openai/whisper-base",
                traits={
                    Options(
                        choices=[
                            "openai/whisper-tiny",
                            "openai/whisper-base",
                            "openai/whisper-small",
                            "openai/whisper-large-v3",
                            "distil-whisper/distil-large-v3-turbo",
                        ]
                    )
                },
                ui_options={"display_name": "Model Preset"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="hf_repo_id",
                tooltip="Enter a HuggingFace repo ID (e.g., openai/whisper-large-v3).",
                default_value="openai/whisper-base",
                ui_options={"display_name": "HuggingFace Repo ID", "hide": True},
            )
        )

        self.add_parameter(
            ParameterString(
                name="device",
                tooltip="Hardware to run on.",
                default_value="auto",
                traits={Options(choices=["auto", "cpu", "cuda", "mps"])},
                ui_options={"display_name": "Compute Device"},
            )
        )

        # --- ADVANCED WHISPER PARAMETERS ---
        self.add_parameter(
            ParameterString(
                name="task",
                tooltip="Whether to transcribe the audio or translate it to English.",
                default_value="transcribe",
                traits={Options(choices=["transcribe", "translate"])},
            )
        )

        self.add_parameter(
            ParameterString(
                name="language",
                tooltip="ISO language code (e.g., 'en', 'fr'). Leave blank to auto-detect.",
                default_value="",
                ui_options={"placeholder_text": "e.g. en (optional)"},
            )
        )

        self.add_parameter(
            Parameter(
                name="chunk_length_s",
                type=ParameterTypeBuiltin.INT.value,
                tooltip="Audio chunk length in seconds. 30 is the native Whisper standard.",
                default_value=30,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Chunk Length (Seconds)"},
            )
        )

        self.add_parameter(
            Parameter(
                name="batch_size",
                type=ParameterTypeBuiltin.INT.value,
                tooltip="Number of chunks to process at once. Increase for faster GPU/MPS processing.",
                default_value=8,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Batch Size"},
            )
        )

        self.add_parameter(
            ParameterBool(
                name="return_timestamps",
                tooltip="Required for long audio files. Predicts timestamps to stitch chunks together.",
                default_value=True,
                ui_options={"display_name": "Return Timestamps"},
            )
        )
        # -----------------------------------

        self.add_parameter(
            ParameterString(
                name="status_log",
                tooltip="Displays downloading and processing status.",
                allow_input=False,
                ui_options={"multiline": True, "display_name": "Status Log"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="transcription",
                tooltip="Resulting transcription text",
                allow_input=False,
                allow_property=False,
            )
        )

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "use_hf_custom_mode":
            if value:
                self.show_parameter_by_name("hf_repo_id")
                self.hide_parameter_by_name("model_preset")
            else:
                self.hide_parameter_by_name("hf_repo_id")
                self.show_parameter_by_name("model_preset")
        return super().after_value_set(parameter, value)

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        try:
            try:
                import torch
                from huggingface_hub import try_to_load_from_cache
                from transformers import pipeline
            except ImportError:
                raise ImportError(
                    "Missing dependencies. "
                    "Please run: pip install transformers torch torchaudio huggingface_hub static-ffmpeg"
                )

            self.parameter_output_values["transcription"] = None
            self.parameter_output_values["status_log"] = "Starting process...\n"

            # Retrieve inputs
            audio_artifact = self.get_parameter_value("input_audio")
            use_custom = self.get_parameter_value("use_hf_custom_mode")
            device_str = self.get_parameter_value("device") or "auto"

            # Retrieve advanced pipeline args
            task = self.get_parameter_value("task") or "transcribe"
            language = self.get_parameter_value("language")
            chunk_length_s = self.get_parameter_value("chunk_length_s") or 30
            batch_size = self.get_parameter_value("batch_size") or 8
            return_timestamps = self.get_parameter_value("return_timestamps")
            if return_timestamps is None:
                return_timestamps = True

            model_id = (
                self.get_parameter_value("hf_repo_id")
                if use_custom
                else self.get_parameter_value("model_preset")
            )

            if not audio_artifact:
                raise ValueError("No input audio provided.")

            if device_str == "auto":
                device = (
                    "cuda:0"
                    if torch.cuda.is_available()
                    else "mps"
                    if torch.backends.mps.is_available()
                    else "cpu"
                )
            else:
                device = device_str

            config_path = try_to_load_from_cache(model_id, "config.json")
            if config_path is None:
                log_msg = f"Downloading '{model_id}' from HuggingFace...\n"
                print(log_msg)
                self.parameter_output_values["status_log"] += log_msg
            else:
                self.parameter_output_values["status_log"] += (
                    f"Loading '{model_id}' to {device}...\n"
                )

            # Parse the audio path
            if isinstance(audio_artifact, dict) and "value" in audio_artifact:
                audio_path = audio_artifact["value"]
            elif hasattr(audio_artifact, "value"):
                audio_path = audio_artifact.value
            else:
                raise ValueError("Unsupported audio format structure.")

            audio_bytes = File(audio_path).read_bytes()

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                temp_path = temp_audio.name
                temp_audio.write(audio_bytes)

            try:
                # Load pipeline
                pipe = pipeline(
                    "automatic-speech-recognition", model=model_id, device=device
                )

                # Build generate kwargs (HuggingFace explicitly uses this for task/language)
                generate_kwargs = {"task": task}
                if language and language.strip() != "":
                    generate_kwargs["language"] = language.strip()

                self.parameter_output_values["status_log"] += (
                    "Transcribing (chunking enabled)...\n"
                )

                # The magic happens here! Passing chunk_length_s and return_timestamps solves the 30-second limit
                result = pipe(
                    temp_path,
                    chunk_length_s=chunk_length_s,
                    batch_size=batch_size,
                    return_timestamps=return_timestamps,
                    generate_kwargs=generate_kwargs,
                )

                self.parameter_output_values["transcription"] = result.get(
                    "text", ""
                ).strip()
                self.parameter_output_values["status_log"] += "Transcription complete!"

            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            self.parameter_output_values["transcription"] = f"Error: {str(e)}"
            self.parameter_output_values["status_log"] += f"\nFailed: {str(e)}"
            raise RuntimeError(f"{self.name} failed: {str(e)}") from e
