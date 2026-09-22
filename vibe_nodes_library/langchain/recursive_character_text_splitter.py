from typing import Any

from griptape.artifacts import ErrorArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.node_types import AsyncResult, DataNode


class RecursiveCharacterTextSplitterNode(DataNode):
    """Splits text into chunks using LangChain's RecursiveCharacterTextSplitter

    and returns the results strictly as a standard Python list of raw strings.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            Parameter(
                name="text_input",
                tooltip="The long text document to be split.",
                type=ParameterTypeBuiltin.STR.value,
                input_types=["str", "TextArtifact", "any"],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "display_name": "Text Input",
                    "multiline": True,
                },
                default_value="",
            )
        )

        self.add_parameter(
            Parameter(
                name="chunk_size",
                tooltip="Maximum character count per chunk.",
                type=ParameterTypeBuiltin.INT.value,
                input_types=["int", "any"],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Chunk Size"},
                default_value=1000,
            )
        )

        self.add_parameter(
            Parameter(
                name="chunk_overlap",
                tooltip="Number of characters to overlap between chunks.",
                type=ParameterTypeBuiltin.INT.value,
                input_types=["int", "any"],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Chunk Overlap"},
                default_value=200,
            )
        )

        self.add_parameter(
            Parameter(
                name="separators",
                tooltip="List of characters to split on, ordered by priority.",
                type="list",
                input_types=["list", "list[str]"],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Separators"},
                default_value=["\n\n", "\n", " ", ""],
            )
        )

        # Configured strictly as a standard python list type
        self.add_parameter(
            Parameter(
                name="output_chunks",
                tooltip="Strict standard Python list containing the split text strings.",
                type="list",
                output_type="list",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Output Chunks"},
            )
        )

    def process(self) -> AsyncResult | None:
        """Processes the input text and splits it into a strict native Python list of text chunks.

        The actual split runs on a worker thread (`yield`ed instead of called directly) -
        process() executes inline on the engine's event loop otherwise, and a large
        text_input with a small chunk_size/chunk_overlap gap can produce a huge number
        of chunks, blocking or OOM-ing the whole engine, not just this node.
        """
        # Use conditional import so the node fails gracefully if LangChain isn't installed
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as e:
            self.parameter_output_values["output_chunks"] = ErrorArtifact(
                f"langchain_text_splitters library is not installed: {str(e)}"
            )
            return

        text_input = self.get_parameter_value("text_input")
        chunk_size = self.get_parameter_value("chunk_size")
        chunk_overlap = self.get_parameter_value("chunk_overlap")
        separators = self.get_parameter_value("separators")

        # Everything below can fail on bad upstream data (empty property fields,
        # wrapped artifacts, wrong types) - keep it all inside one guard instead of
        # letting int() etc. raise uncaught.
        try:
            text_content = str(text_input.value) if hasattr(text_input, "value") else str(text_input)

            if not text_content:
                self.parameter_output_values["output_chunks"] = []
                return

            def _to_int(value: Any, default: int) -> int:
                if value is None or value == "":
                    return default
                if hasattr(value, "value"):
                    value = value.value
                return int(value)

            safe_chunk_size = _to_int(chunk_size, 1000)
            safe_chunk_overlap = _to_int(chunk_overlap, 200)
            safe_separators = (
                separators if isinstance(separators, list) and separators else ["\n\n", "\n", " ", ""]
            )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=safe_chunk_size,
                chunk_overlap=safe_chunk_overlap,
                separators=safe_separators,
            )

            def _do_split() -> list[str] | ErrorArtifact:
                # Runs in a worker thread. Caught locally so a mid-split failure
                # (e.g. a pathological separators list) comes back as data, not
                # an exception thrown across the thread boundary.
                try:
                    return splitter.split_text(text_content)
                except Exception as e:
                    return ErrorArtifact(f"Error splitting text: {e}")

            # Assign directly as a native standard Python list of strings (or an
            # ErrorArtifact if _do_split failed).
            self.parameter_output_values["output_chunks"] = yield _do_split

        except Exception as e:
            # Trap and expose execution errors safely to prevent workflow crashes
            self.parameter_output_values["output_chunks"] = ErrorArtifact(
                f"Error splitting text: {str(e)}"
            )
