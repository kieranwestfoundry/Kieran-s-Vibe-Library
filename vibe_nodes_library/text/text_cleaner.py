# Standard library imports
import re
from pathlib import Path
from typing import Any

# Third-party imports
import docx  # type: ignore[import-untyped]
import fitz  # type: ignore[import-untyped]

# Local/Griptape imports
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode


class ProcessTextDocument(SuccessFailureNode):
    """Extracts and cleans text from documents while preserving paragraph structure."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Add status parameters to report success or failure to the user
        self._create_status_parameters(
            result_details_tooltip="Details about the text extraction operation result",
            result_details_placeholder="Details on the load attempt will be presented here.",
        )

        # Input file path parameter
        self.add_parameter(
            Parameter(
                name="file_path",
                input_types=["str"],
                type="str",
                tooltip="Input document file (.txt, .md, .docx, .pdf)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "clickable_file_browser": True,
                    "display_name": "Document File Path",
                },
            )
        )

        # Output cleaned text parameter
        self.add_parameter(
            Parameter(
                name="cleaned_text",
                output_type="str",
                tooltip="Extracted and cleaned text",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={
                    "display_name": "Cleaned Text",
                    "multiline": True,
                    "is_full_width": True,
                },
            )
        )

    def _clean_text(self, raw_text: str) -> str:
        """Removes artifacts while preserving paragraph structure."""
        # Remove null bytes or weird control characters often found in PDFs
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", raw_text)
        
        # Replace 3 or more consecutive newlines with exactly 2 newlines (preserves paragraphs)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        # Replace 2 or more horizontal spaces/tabs with a single space
        text = re.sub(r"[ \t]+", " ", text)
        
        return text.strip()

    def process(self) -> None:
        """Process the file and set output."""
        # Reset execution state at start
        self._clear_execution_status()

        # Clear output values to prevent stale data on errors
        self.parameter_output_values["cleaned_text"] = None

        file_path = self.get_parameter_value("file_path")
        
        if not file_path:
            self._set_status_results(
                was_successful=False, 
                result_details="FAILURE: No file path provided."
            )
            return

        path = Path(file_path)
        if not path.exists():
            self._set_status_results(
                was_successful=False, 
                result_details=f"FAILURE: File not found at {file_path}"
            )
            return

        ext = path.suffix.lower()
        raw_text = ""

        try:
            # Extract text based on file type
            if ext in [".txt", ".md"]:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
                    
            elif ext == ".docx":
                doc = docx.Document(path)
                raw_text = "\n".join([para.text for para in doc.paragraphs])
                
            elif ext == ".pdf":
                with fitz.open(path) as pdf:
                    raw_text = "\n".join([page.get_text() for page in pdf])
                    
            else:
                self._set_status_results(
                    was_successful=False, 
                    result_details=f"FAILURE: Unsupported file type '{ext}'."
                )
                return

            # Clean and set output
            cleaned_result = self._clean_text(raw_text)
            self.parameter_output_values["cleaned_text"] = cleaned_result
            
            # Report success
            self._set_status_results(
                was_successful=True, 
                result_details=f"SUCCESS: Extracted and cleaned {len(cleaned_result)} characters."
            )

        except Exception as e:
            # Handle failure cases gracefully
            error_details = f"Failed to process document: {e}"
            self._set_status_results(
                was_successful=False, 
                result_details=f"FAILURE: {error_details}"
            )
            self._handle_failure_exception(e)