import logging
from typing import Any

import chromadb
from griptape.artifacts import ErrorArtifact, ListArtifact, TextArtifact
from griptape.tools import BaseTool
from griptape.utils.decorators import activity
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from schema import Literal, Schema

logger = logging.getLogger(__name__)


class LocalChromaDBToolNode(DataNode):
    """
    A single Griptape Node that outputs a ChromaDB tool for Agents to use.
    The Griptape Tool logic is encapsulated entirely within this single class.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # Configuration parameters for the Node UI
        self.add_parameter(
            ParameterString(
                name="persist_directory",
                tooltip="Local filesystem path to the ChromaDB directory.",
                default_value="./chroma_db",
                allow_property=True,
                allow_input=True,
            )
        )

        self.add_parameter(
            ParameterString(
                name="collection_name",
                tooltip="Name of the ChromaDB collection to query.",
                default_value="default_collection",
                allow_property=True,
                allow_input=True,
            )
        )

        # Output parameter that connects to the Agent Node
        self.add_parameter(
            Parameter(
                name="tool",
                tooltip="The configured ChromaDB Tool. Connect this to the 'tools' input of an Agent node.",
                type="Tool",
                output_type="Tool",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        """Instantiate the tool dynamically and push it to the output port."""
        db_path = self.get_parameter_value("persist_directory") or ""
        col_name = self.get_parameter_value("collection_name") or ""

        # ---------------------------------------------------------
        # Define the Tool inline so this remains a strict "single node"
        # ---------------------------------------------------------
        class InlineChromaDBTool(BaseTool):
            def __init__(self, persist_dir: str, coll_name: str, **kwargs):
                super().__init__(**kwargs)
                self.persist_dir = persist_dir
                self.coll_name = coll_name

            @activity(
                config={
                    "description": "Can be used to retrieve relevant documents from a local vector database based on a search query.",
                    "schema": Schema(
                        {
                            Literal(
                                "query",
                                description="The search query used to find relevant information in the database.",
                            ): str,
                            Literal(
                                "n_results",
                                description="The number of relevant documents to return. Defaults to 5.",
                            ): int,
                        }
                    ),
                }
            )
            def retrieve(self, params: dict) -> ListArtifact | ErrorArtifact:
                query_text = params["values"].get("query")
                n_results = params["values"].get("n_results", 5)

                try:
                    # Connection phase
                    client = chromadb.PersistentClient(path=self.persist_dir)
                    collection = client.get_collection(name=self.coll_name)

                    # Query phase
                    results = collection.query(
                        query_texts=[query_text], n_results=n_results
                    )

                    documents = results.get("documents", [[]])[0]

                    if not documents:
                        return TextArtifact(
                            "No relevant documents found for the given query."
                        )

                    return ListArtifact([TextArtifact(doc) for doc in documents])

                except Exception as e:
                    logger.error(f"ChromaDB Tool Error: {e}")
                    # Agent-friendly failure state
                    return ErrorArtifact(
                        f"Failed to connect or retrieve from ChromaDB: {str(e)}"
                    )

        # ---------------------------------------------------------
        # Output the newly created tool instance to the Node canvas
        # ---------------------------------------------------------
        self.parameter_output_values["tool"] = InlineChromaDBTool(
            persist_dir=db_path, coll_name=col_name
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        """Ensure paths are provided before the workflow runs."""
        errors = []
        if not self.get_parameter_value("persist_directory"):
            errors.append(
                ValueError(f"The '{self.name}' node requires a 'persist_directory'.")
            )
        if not self.get_parameter_value("collection_name"):
            errors.append(
                ValueError(f"The '{self.name}' node requires a 'collection_name'.")
            )
        return errors if errors else None
