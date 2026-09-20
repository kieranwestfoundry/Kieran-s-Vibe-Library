import base64
import logging
from contextlib import suppress
from io import BytesIO
from typing import Any

import requests
from PIL import Image

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes_library.utils.image_utils import dict_to_image_url_artifact

logger = logging.getLogger(__name__)


class ImageToBase64(SuccessFailureNode):
    """Converts an image artifact or URL to a Base64 Data URI."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Input image parameter using the recommended ParameterImage type
        self.add_parameter(
            ParameterImage(
                name="image",
                tooltip="Input image to convert to a Base64 Data URI",
                allow_output=False,
            )
        )

        # Output base64 string parameter
        self.add_parameter(
            Parameter(
                name="base64_uri",
                tooltip="The generated Base64 Data URI",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        # Create standard success/failure tracking parameters
        self._create_status_parameters(
            result_details_tooltip="Details about the conversion operation result",
            result_details_placeholder="Conversion status will appear here.",
        )

    def _log(self, message: str) -> None:
        """Safe logging with exception suppression."""
        with suppress(Exception):
            logger.info(message)

    def _get_artifact(self, value: Any) -> ImageArtifact | ImageUrlArtifact | None:
        """Safely extracts the artifact from the parameter value."""
        if not value:
            return None
        # Handle serialized dictionary inputs
        if isinstance(value, dict):
            return dict_to_image_url_artifact(value)
        # Handle direct artifact objects
        if isinstance(value, (ImageArtifact, ImageUrlArtifact)):
            return value
        return None

    def _convert_to_base64_uri(self, image_artifact: ImageArtifact | ImageUrlArtifact) -> str:
        """Convert image artifact to base64 data URI."""
        
        # Handle ImageUrlArtifacts (including localhost URLs from static storage)
        if isinstance(image_artifact, ImageUrlArtifact):
            url = image_artifact.value
            self._log(f"Downloading image from URL: {url[:100]}...")

            response = requests.get(url, timeout=30)
            response.raise_for_status()
            image_bytes = response.content

            # Detect MIME type from headers
            mime_type = response.headers.get("content-type", "image/jpeg")
            if not mime_type.startswith("image/"):
                mime_type = "image/jpeg"

            base64_data = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:{mime_type};base64,{base64_data}"

        # Handle ImageArtifacts directly
        if isinstance(image_artifact, ImageArtifact):
            # PREFERRED: Use built-in properties if available
            if hasattr(image_artifact, "base64") and hasattr(image_artifact, "mime_type"):
                base64_data = image_artifact.base64
                mime_type = image_artifact.mime_type

                # If it already contains the data URI prefix, return it as-is
                if base64_data.startswith("data:"):
                    self._log("Using ImageArtifact.base64 (already has data URI format)")
                    return base64_data

                self._log(f"Using ImageArtifact.base64 with mime_type: {mime_type}")
                return f"data:{mime_type};base64,{base64_data}"

            # FALLBACK: Manual byte extraction and MIME detection
            self._log("Falling back to manual base64 encoding")
            image_bytes = None
            
            if hasattr(image_artifact, "value") and hasattr(image_artifact.value, "read"):
                image_artifact.value.seek(0)
                image_bytes = image_artifact.value.read()
            elif hasattr(image_artifact, "data"):
                if isinstance(image_artifact.data, bytes):
                    image_bytes = image_artifact.data
                elif hasattr(image_artifact.data, "read"):
                    image_artifact.data.seek(0)
                    image_bytes = image_artifact.data.read()

            if not image_bytes:
                raise ValueError("Unsupported ImageArtifact format or empty data payload.")

            # Detect MIME type using PIL
            mime_type = "image/jpeg"
            try:
                img = Image.open(BytesIO(image_bytes))
                format_to_mime = {
                    "JPEG": "image/jpeg",
                    "PNG": "image/png",
                    "WEBP": "image/webp",
                }
                mime_type = format_to_mime.get(img.format, "image/jpeg")
            except Exception as e:
                self._log(f"Could not automatically detect MIME type, defaulting to JPEG: {e}")

            base64_data = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:{mime_type};base64,{base64_data}"

        raise ValueError("Unsupported artifact type provided.")

    def process(self) -> None:
        """Main processing entry point."""
        # Ensure idempotent execution state
        self._clear_execution_status()
        self.parameter_output_values["base64_uri"] = None

        try:
            raw_value = self.get_parameter_value("image")
            image_artifact = self._get_artifact(raw_value)

            if not image_artifact:
                raise ValueError("No input image was provided.")

            base64_uri = self._convert_to_base64_uri(image_artifact)

            # Assign outputs and mark success
            self.parameter_output_values["base64_uri"] = base64_uri
            self._set_status_results(
                was_successful=True,
                result_details="SUCCESS: Image successfully converted to Base64 Data URI.",
            )

        except Exception as e:
            error_details = f"Failed to convert image to Base64: {str(e)}"
            self._log(error_details)
            self._set_status_results(
                was_successful=False,
                result_details=f"FAILURE: {error_details}",
            )
            self._handle_failure_exception(e)