import logging
from attr import define, field
from schema import Schema, Literal

# Core Griptape Tool imports
from griptape.tools import BaseTool
from griptape.utils.decorators import activity
from griptape.drivers import GriptapeCloudVectorStoreDriver

# Griptape Nodes imports
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode

# Standardize logger
logger = logging.getLogger(__name__)
for noisy_logger in ["griptape", "urllib3", "engine", "websockets"]:
    log = logging.getLogger(noisy_logger)
    log.setLevel(logging.CRITICAL)
    log.propagate = False

# ---------------------------------------------------------
# 1. THE GRIPTAPE TOOL (Execution Logic)
# ---------------------------------------------------------
@define
class PipelineKnowledgeBaseTool(BaseTool):
    """The underlying Griptape Tool that the Agent will execute."""
    
    api_key: str = field(kw_only=True)
    knowledge_base_id: str = field(kw_only=True)

    @activity(
        config={
            "description": "Can be used to search the studio knowledge base for pipeline documentation, definitions, or technical workflows.",
            "schema": Schema({
                Literal(
                    "query", 
                    description="The specific question or search term to look up in the vector database."
                ): str
            })
        }
    )
    def search_knowledge_base(self, params: dict) -> str:
        query_string = params.get("values", {}).get("query")
        
        if not query_string:
            return "Error: No query provided."

        try:
            driver = GriptapeCloudVectorStoreDriver(
                api_key=self.api_key,
                knowledge_base_id=self.knowledge_base_id,
            )

            results = driver.query(query=query_string)

            if not results:
                return f"No matches found in the knowledge base for '{query_string}'."

            formatted_matches = [f"--- Found {len(results)} matches ---"]
            for i, r in enumerate(results, 1):
                content = r.to_artifact().value
                formatted_matches.append(f"Result {i}:\n{content}\n")
                
            return "\n".join(formatted_matches)

        except Exception as e:
            logger.error(f"Vector Store Query Failed: {e}")
            return f"Error communicating with the knowledge base: {str(e)}"


# ---------------------------------------------------------
# 2. THE VISUAL NODE (Canvas Wrapper)
# ---------------------------------------------------------
class PipelineKnowledgeBaseToolNode(DataNode):
    """The visual node that appears in the Griptape Nodes UI to configure and output the tool."""
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "Tools"
        self.description = "Creates a Vector Store Knowledge Base Tool for an Agent."

        # UI: API Key Input
        self.add_parameter(
            Parameter(
                name="api_key",
                input_types=["str"],
                type="str",
                default_value="",
                tooltip="Griptape Cloud API Key.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"password": True} # Masks the text in the UI
            )
        )

        # UI: Knowledge Base ID Input
        self.add_parameter(
            Parameter(
                name="knowledge_base_id",
                input_types=["str"],
                type="str",
                default_value="301f49f8-c009-4122-8c9b-c2fbd350d3b0",
                tooltip="The UUID of the Griptape Cloud Knowledge Base.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}
            )
        )

        # Output: The instantiated Tool
        self.add_parameter(
            Parameter(
                name="tool_out",
                output_type="Tool",
                type="Tool",
                tooltip="Connect this to the 'tools' port of an Agent.",
                allowed_modes={ParameterMode.OUTPUT},
                settable=False
            )
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = []
        if not self.get_parameter_value("api_key"):
            exceptions.append(ValueError(f"{self.name}: API Key is required."))
        if not self.get_parameter_value("knowledge_base_id"):
            exceptions.append(ValueError(f"{self.name}: Knowledge Base ID is required."))
        return exceptions if exceptions else None

    def process(self) -> None:
        """Instantiates the tool and outputs it to the graph."""
        api_key = self.get_parameter_value("api_key")
        kb_id = self.get_parameter_value("knowledge_base_id")

        # Create the tool instance
        tool_instance = PipelineKnowledgeBaseTool(
            api_key=api_key,
            knowledge_base_id=kb_id
        )

        # Output it downstream
        self.parameter_output_values["tool_out"] = tool_instance