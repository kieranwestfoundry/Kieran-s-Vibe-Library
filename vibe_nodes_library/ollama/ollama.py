import os
import json
import logging
from contextlib import suppress
from typing import Any

import requests

# Core Griptape Structures required for state passing
from griptape.structures import Agent
from griptape.memory.structure import ConversationMemory, Run
from griptape.artifacts import TextArtifact

# Import OpenAI driver for Local Emulation
from griptape.drivers import OpenAiChatPromptDriver
from griptape.tasks import PromptTask

# Import Griptape config to override global defaults
from griptape.configs import Defaults
from griptape.configs.drivers import DriversConfig

# Griptape Nodes Engine & UI Framework
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup, ParameterList
from griptape_nodes.exe_types.node_types import ControlNode, AsyncResult
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

logger = logging.getLogger(__name__)

class OllamaPromptNode(ControlNode):
    """
    Queries an Ollama server using the /api/chat endpoint to fully support 
    Griptape Conversation Memory and recursive Tool Execution.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "LLM"
        self.description = "Agent node supporting Ollama inference, memory, and Tool execution."

        # ---------------------------------------------------------
        # 1. Agent Port
        # ---------------------------------------------------------
        self.add_parameter(
            Parameter(
                name="agent",
                type="Agent",
                input_types=["Agent"],
                output_type="Agent",
                tooltip="Pass an existing agent to continue the chain, or leave empty to start a new one.",
                default_value=None,
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT}
            )
        )

        # ---------------------------------------------------------
        # 2. Tools Port
        # ---------------------------------------------------------
        self.add_parameter(
            ParameterList(
                name="tools",
                input_types=["Tool", "list[Tool]"],
                default_value=[],
                tooltip="Connect Griptape Tools for the LLM to use.",
                allowed_modes={ParameterMode.INPUT},
            )
        )

        # ---------------------------------------------------------
        # 3. Connection & Prompt
        # ---------------------------------------------------------
        self.add_parameter(
            Parameter(
                name="server_address",
                input_types=["str"],
                type="str",
                default_value="http://127.0.0.1:11434",
                tooltip="The IP/URL of the Ollama server.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}
            )
        )

        self.add_parameter(
            Parameter(
                name="model",
                input_types=["str"],
                type="str",
                default_value="llama3:latest",
                tooltip="The specific LLM model to query.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=[
                    "llama3:latest",
                    "llama3.1", 
                    "qwen3:14b", 
                    "deepseek-coder-v2:16b-lite-instruct-q4_K_M",
                    "phi3:3.8b",
                    "tinydolphin:latest",
                    "mistral"
                ])}
            )
        )
        
        self.add_parameter(
            Parameter(
                name="text_prompt",
                input_types=["str"],
                type="str",
                default_value="",
                tooltip="The input text to send to the LLM.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"multiline": True, "placeholder_text": "Enter your prompt here..."}
            )
        )

        # ---------------------------------------------------------
        # 4. Advanced Options Group
        # ---------------------------------------------------------
        self._options_group = ParameterGroup(
            name="advanced_model_options",
            ui_options={"display_name": "Advanced Options", "collapsed": True}
        )
        self.add_node_element(self._options_group)

        self.add_parameter(
            Parameter(
                name="thinking_mode",
                input_types=["bool"],
                type="bool",
                default_value=False,
                tooltip="Enable thinking mode (for models that support it, like Qwen3).",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                parent_element_name=self._options_group.name,
                ui_options={"display_name": "Thinking Mode"}
            )
        )

        self.add_parameter(
            Parameter(
                name="temperature",
                input_types=["float"],
                type="float",
                default_value=0.8,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                parent_element_name=self._options_group.name,
                traits={Slider(min_val=0.0, max_val=2.0)}
            )
        )

        self.add_parameter(
            Parameter(
                name="top_p",
                input_types=["float"],
                type="float",
                default_value=0.9,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                parent_element_name=self._options_group.name,
                traits={Slider(min_val=0.0, max_val=1.0)}
            )
        )

        self.add_parameter(
            Parameter(
                name="num_ctx",
                input_types=["int"],
                type="int",
                default_value=4096,
                tooltip="Context window size. Expand for DeepSeek/Qwen3 if VRAM permits.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                parent_element_name=self._options_group.name
            )
        )

        self.add_parameter(
            Parameter(
                name="num_predict",
                input_types=["int"],
                type="int",
                default_value=128,
                tooltip="Maximum tokens to generate. Increase for long code blocks.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                parent_element_name=self._options_group.name
            )
        )

        self.add_parameter(
            Parameter(
                name="repeat_penalty",
                input_types=["float"],
                type="float",
                default_value=1.1,
                tooltip="Penalty for repetition. Crucial for small models like TinyDolphin.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                parent_element_name=self._options_group.name,
                traits={Slider(min_val=1.0, max_val=2.0)}
            )
        )

        self.add_parameter(
            Parameter(
                name="seed",
                input_types=["int"],
                type="int",
                default_value=0,
                tooltip="Set to a specific integer (e.g. 42) for deterministic output during testing.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                parent_element_name=self._options_group.name
            )
        )

        self.add_parameter(
            Parameter(
                name="stop_sequences",
                input_types=["str"],
                type="str",
                default_value="",
                tooltip="Comma-separated strings that will halt generation (e.g. Observation:, \n\n).",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                parent_element_name=self._options_group.name
            )
        )
        
        # ---------------------------------------------------------
        # 5. Output
        # ---------------------------------------------------------
        self.add_parameter(
            Parameter(
                name="response_text",
                output_type="str",
                type="str",
                tooltip="The final generated text.",
                allowed_modes={ParameterMode.OUTPUT},
                settable=False
            )
        )

    # ---------------------------------------------------------
    # UTILITIES & VALIDATION
    # ---------------------------------------------------------
    def _log(self, message: str) -> None:
        with suppress(Exception):
            logger.info(f"{self.name}: {message}")

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["response_text"] = ""
        self.parameter_output_values["agent"] = None

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = []
        if not str(self.get_parameter_value("text_prompt") or "").strip():
            exceptions.append(ValueError(f"{self.name}: 'text_prompt' cannot be empty."))
        return exceptions if exceptions else None

    # ---------------------------------------------------------
    # TOOL HELPERS
    # ---------------------------------------------------------
    def _format_tools_for_ollama(self, tools: list) -> list:
        """Translates Griptape Tool schemas into Ollama's required JSON format."""
        ollama_tools = []
        for tool in tools:
            for activity in tool.activities():
                func_name = f"{tool.name}___{activity.name}"
                
                parameters = {}
                if hasattr(tool, "activity_schema"):
                    schema_obj = tool.activity_schema(activity)
                    parameters = schema_obj if isinstance(schema_obj, dict) else {}
                
                desc = getattr(activity, "description", "")
                if not desc and hasattr(tool, "activity_description"):
                    desc = tool.activity_description(activity)

                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "description": desc,
                        "parameters": parameters
                    }
                })
        return ollama_tools

    def _execute_griptape_tool(self, func_name: str, args: dict, tools: list) -> str:
        """Finds and executes the requested Python tool with LLM arguments."""
        try:
            tool_name, activity_name = func_name.split("___", 1)
            for tool in tools:
                if tool.name == tool_name:
                    method = getattr(tool, activity_name)
                    
                    if hasattr(tool, 'execute'):
                        res = tool.execute(method, {"values": args})
                        return str(res.value) if hasattr(res, 'value') else str(res)
                    else:
                        res = method({"values": args})
                        return str(res)
                        
            return f"Error: Tool {func_name} not found."
        except Exception as e:
            return f"Error executing tool {func_name}: {str(e)}"

    # ---------------------------------------------------------
    # EXECUTION
    # ---------------------------------------------------------
    def process(self) -> AsyncResult:
        """Yield to a background thread to prevent blocking the UI."""
        yield lambda: self._process()

    def _process(self) -> None:
        self._set_safe_defaults()

        server_addr = self.get_parameter_value("server_address") or "http://127.0.0.1:11434"
        model_name = self.get_parameter_value("model") or "llama3:latest"
        prompt_text = self.get_parameter_value("text_prompt")
        
        # 1. Gather Tools
        griptape_tools = self.get_parameter_list_value("tools") or []
        ollama_tools = self._format_tools_for_ollama(griptape_tools)

        # ------------------------------------------------------------------
        # PIPELINE FIX: OpenAI Emulation
        # We explicitly use an OpenAiChatPromptDriver pointed at the local M1.
        # This completely bypasses the 401 errors in built-in nodes because
        # Griptape knows how to deserialize OpenAI drivers natively.
        # ------------------------------------------------------------------
        local_base_url = f"{server_addr.rstrip('/')}/v1"
        
        # Force environment variables just in case a sub-module checks them
        os.environ["OPENAI_API_KEY"] = "ollama-dummy-key"
        os.environ["OPENAI_BASE_URL"] = local_base_url

        emulated_driver = OpenAiChatPromptDriver(
            model=model_name,
            base_url=local_base_url,
            api_key="ollama-dummy-key",
            use_native_tools=False, # <-- PIPELINE FIX: Bypasses Ollama's strict tool validation
            stream=True # mcp_task expects a streaming driver
        )
        
        with suppress(Exception):
            Defaults.drivers_config = DriversConfig(prompt_driver=emulated_driver)

        # 2. Agent Deserialization
        agent_state = self.get_parameter_value("agent")
        agent = Agent.from_dict(agent_state) if isinstance(agent_state, dict) else Agent()

        if agent.conversation_memory is None:
            agent.conversation_memory = ConversationMemory()

        if agent.tasks:
            agent.tasks[0].prompt_driver = emulated_driver
        else:
            agent.add_task(PromptTask(prompt_driver=emulated_driver))

        # 3. Build the Message Payload
        messages = []
        if agent.conversation_memory.runs:
            for run in agent.conversation_memory.runs:
                if run.input:
                    messages.append({"role": "user", "content": str(run.input.value)})
                if run.output:
                    messages.append({"role": "assistant", "content": str(run.output.value)})
        
        messages.append({"role": "user", "content": prompt_text})

        # 4. Options Setup
        raw_stops = self.get_parameter_value("stop_sequences")
        stop_list = [s.strip() for s in raw_stops.split(",")] if raw_stops else None

        inference_options = {
            "temperature": self.get_parameter_value("temperature"),
            "top_p": self.get_parameter_value("top_p"),
            "num_ctx": self.get_parameter_value("num_ctx"),
            "num_predict": self.get_parameter_value("num_predict"),
            "repeat_penalty": self.get_parameter_value("repeat_penalty"),
            "seed": self.get_parameter_value("seed"),
            "stop": stop_list,
            "thinking": self.get_parameter_value("thinking_mode"), 
        }
        # Strip out defaults/Nones so Ollama relies on model internals if unmodified
        inference_options = {k: v for k, v in inference_options.items() if v is not None and v != ""}

        endpoint = f"{server_addr.rstrip('/')}/api/chat"
        
        # 5. The Execution Loop
        max_iterations = 5
        iteration = 0
        final_response_text = ""

        try:
            while iteration < max_iterations:
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                    "options": inference_options
                }
                if ollama_tools:
                    payload["tools"] = ollama_tools

                self._log(f"Iteration {iteration+1}: Sending payload to Ollama...")
                response = requests.post(endpoint, json=payload, timeout=300)
                response.raise_for_status()
                
                response_data = response.json()
                message_obj = response_data.get("message", {})

                # Check if the LLM wants to use a tool
                if "tool_calls" in message_obj and message_obj["tool_calls"]:
                    messages.append(message_obj)
                    
                    for tool_call in message_obj["tool_calls"]:
                        func_name = tool_call.get("function", {}).get("name")
                        func_args = tool_call.get("function", {}).get("arguments", {})
                        
                        self._log(f"Executing Tool: {func_name} with args: {func_args}")
                        tool_result = self._execute_griptape_tool(func_name, func_args, griptape_tools)
                        self._log(f"Tool Result: {tool_result}")
                        
                        messages.append({
                            "role": "tool",
                            "content": tool_result
                        })
                    
                    iteration += 1
                else:
                    # The LLM generated standard text.
                    final_response_text = message_obj.get("content", "")
                    break
                    
            if iteration >= max_iterations:
                self._log("Warning: Reached maximum tool execution loop limit.")

            # 6. Finalize Agent Memory
            agent.conversation_memory.add_run(
                Run(
                    input=TextArtifact(value=prompt_text), 
                    output=TextArtifact(value=final_response_text)
                )
            )
            
            # 7. Output Setup
            self.parameter_output_values["response_text"] = final_response_text
            self.parameter_output_values["agent"] = agent.to_dict()
            
            self._log("Ollama inference complete.")
            
        except requests.exceptions.RequestException as e:
            self._set_safe_defaults()
            api_details = ""
            if e.response is not None:
                with suppress(Exception):
                    api_details = f" | Ollama Says: {e.response.json().get('error', e.response.text)}"
                    
            error_msg = f"API Error: {e}{api_details}"
            self._log(error_msg)
            raise RuntimeError(error_msg) from e
            
        except Exception as e:
            self._set_safe_defaults()
            self._log(f"Processing Error: {e}")
            raise RuntimeError(f"Processing Error: {e}") from e