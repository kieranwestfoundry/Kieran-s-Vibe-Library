import json
import logging
import os
from contextlib import suppress
from typing import Any, Dict

from griptape.rules import Rule, Ruleset

# Griptape Nodes Engine & UI Framework
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup
from griptape_nodes.exe_types.node_types import ControlNode, AsyncResult
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload

logger = logging.getLogger(__name__)

# Pipeline standard: store tool configs in a user or show-level directory
CONFIG_DIR = os.path.expanduser("~/.td_pipeline/configs")
ARCHETYPES_FILE = os.path.join(CONFIG_DIR, "llm_archetypes.json")

class ArchetypePresetNode(ControlNode):
    """
    Manages LLM Archetypes (Presets). Outputs inference options and Griptape Rulesets 
    to drive downstream Agent and Prompt nodes.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "LLM"
        self.description = "Saves and loads Model Presets (Archetypes) and System Rules."
        
        self._ensure_config_exists()

        # ---------------------------------------------------------
        # 1. Preset Management UI
        # ---------------------------------------------------------
        self._management_group = ParameterGroup(
            name="preset_management",
            ui_options={"display_name": "Preset Management", "collapsed": False}
        )
        self.add_node_element(self._management_group)

        self.add_parameter(
            Parameter(
                name="archetype_selector",
                input_types=["str"],
                type="str",
                default_value="Default TD",
                tooltip="Select an existing archetype.",
                allowed_modes={ParameterMode.PROPERTY},
                parent_element_name=self._management_group.name,
                traits={
                    Options(choices=self._get_archetype_names()),
                    Button(
                        full_width=False,
                        icon="download",
                        size="icon",
                        variant="secondary",
                        on_click=self._load_selected_archetype,
                        tooltip="Load selected archetype into parameters below."
                    ),
                }
            )
        )

        self.add_parameter(
            Parameter(
                name="new_archetype_name",
                input_types=["str"],
                type="str",
                default_value="",
                tooltip="Name for saving the current parameters as a new archetype.",
                allowed_modes={ParameterMode.PROPERTY},
                parent_element_name=self._management_group.name,
                #placeholder_text="e.g., VEX Code Optimizer...",
                traits={
                    Button(
                        full_width=False,
                        icon="save",
                        size="icon",
                        variant="default",
                        on_click=self._save_current_archetype,
                        tooltip="Save current parameters as a new archetype."
                    ),
                }
            )
        )

        # ---------------------------------------------------------
        # 2. Configurable Parameters (The "Preset" Data)
        # ---------------------------------------------------------
        self.add_parameter(
            Parameter(
                name="system_rules",
                input_types=["str"],
                type="str",
                default_value="You are a helpful Pipeline TD assistant.",
                tooltip="Base rules guiding the LLM's behavior.",
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={"multiline": True}
            )
        )

        self.add_parameter(
            Parameter(
                name="temperature",
                input_types=["float"],
                type="float",
                default_value=0.8,
                allowed_modes={ParameterMode.PROPERTY},
                traits={Slider(min_val=0.0, max_val=2.0)}
            )
        )

        self.add_parameter(
            Parameter(
                name="num_ctx",
                input_types=["int"],
                type="int",
                default_value=4096,
                allowed_modes={ParameterMode.PROPERTY}
            )
        )

        # ---------------------------------------------------------
        # 3. Outputs (Wired to downstream nodes)
        # ---------------------------------------------------------
        self.add_parameter(
            Parameter(
                name="out_ruleset",
                output_type="Ruleset",
                type="Ruleset",
                tooltip="Connect to an Agent node's 'rulesets' input.",
                allowed_modes={ParameterMode.OUTPUT},
                settable=False
            )
        )
        
        self.add_parameter(
            Parameter(
                name="out_temperature",
                output_type="float",
                type="float",
                tooltip="Connect to Ollama node's 'temperature' input.",
                allowed_modes={ParameterMode.OUTPUT},
                settable=False
            )
        )
        
        self.add_parameter(
            Parameter(
                name="out_num_ctx",
                output_type="int",
                type="int",
                tooltip="Connect to Ollama node's 'num_ctx' input.",
                allowed_modes={ParameterMode.OUTPUT},
                settable=False
            )
        )

    # ---------------------------------------------------------
    # Disk I/O & State Management
    # ---------------------------------------------------------
    def _ensure_config_exists(self):
        """Creates the pipeline config directory and base JSON if missing."""
        os.makedirs(CONFIG_DIR, exist_ok=True)
        if not os.path.exists(ARCHETYPES_FILE):
            base_config = {
                "Default TD": {
                    "system_rules": "You are a helpful Pipeline TD assistant. Write robust, PEP8 compliant Python.",
                    "temperature": 0.2,
                    "num_ctx": 4096
                },
                "Brainstormer": {
                    "system_rules": "You are a creative technical director exploring blue-sky workflow concepts.",
                    "temperature": 1.0,
                    "num_ctx": 8192
                }
            }
            with open(ARCHETYPES_FILE, 'w') as f:
                json.dump(base_config, f, indent=4)

    def _read_archetypes(self) -> Dict[str, Any]:
        with suppress(Exception):
            with open(ARCHETYPES_FILE, 'r') as f:
                return json.load(f)
        return {}

    def _get_archetype_names(self) -> list[str]:
        return list(self._read_archetypes().keys())

    # ---------------------------------------------------------
    # UI Callbacks
    # ---------------------------------------------------------
    def _load_selected_archetype(self, button: Button, payload: ButtonDetailsMessagePayload) -> None:
        """Reads the selected preset from disk and updates the node UI parameters."""
        selected = self.get_parameter_value("archetype_selector")
        archetypes = self._read_archetypes()
        
        if selected in archetypes:
            data = archetypes[selected]
            # Push values to the UI so the user sees the changes
            self.publish_update_to_parameter("system_rules", data.get("system_rules", ""))
            self.publish_update_to_parameter("temperature", data.get("temperature", 0.8))
            self.publish_update_to_parameter("num_ctx", data.get("num_ctx", 4096))
            logger.info(f"{self.name}: Loaded archetype '{selected}'")

    def _save_current_archetype(self, button: Button, payload: ButtonDetailsMessagePayload) -> None:
        """Saves current UI parameters to disk under a new name and updates the dropdown."""
        new_name = str(self.get_parameter_value("new_archetype_name") or "").strip()
        if not new_name:
            logger.warning(f"{self.name}: Cannot save. New archetype name is empty.")
            return

        archetypes = self._read_archetypes()
        archetypes[new_name] = {
            "system_rules": self.get_parameter_value("system_rules"),
            "temperature": self.get_parameter_value("temperature"),
            "num_ctx": self.get_parameter_value("num_ctx")
        }

        with open(ARCHETYPES_FILE, 'w') as f:
            json.dump(archetypes, f, indent=4)

        # Update the dropdown choices to include the newly saved item
        updated_names = list(archetypes.keys())
        self._update_option_choices("archetype_selector", updated_names, default_value=new_name)
        
        # Clear the input field for good UX
        self.publish_update_to_parameter("new_archetype_name", "")
        logger.info(f"{self.name}: Saved new archetype '{new_name}'")

    # ---------------------------------------------------------
    # Execution Graph (Passing data downstream)
    # ---------------------------------------------------------
    def process(self) -> AsyncResult:
        """Evaluates the node and passes parameters out the ports."""
        yield lambda: self._process()

    def _process(self) -> None:
        rules = self.get_parameter_value("system_rules")
        temp = self.get_parameter_value("temperature")
        ctx = self.get_parameter_value("num_ctx")

        # Convert the string prompt into a formal Griptape Ruleset
        ruleset_obj = Ruleset(name="Archetype Rules", rules=[Rule(rules)])

        # Assign outputs for downstream nodes
        self.parameter_output_values["out_ruleset"] = ruleset_obj
        self.parameter_output_values["out_temperature"] = temp
        self.parameter_output_values["out_num_ctx"] = ctx
        
        logger.info(f"{self.name}: Pushed archetype parameters downstream.")