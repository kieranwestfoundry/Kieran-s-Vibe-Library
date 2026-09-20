from __future__ import annotations

import copy
import logging
import re
from typing import Any

from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_json import ParameterJson

logger = logging.getLogger(__name__)


class AutoJsonTemplateFiller(DataNode):
    """Automatically populates a template JSON object using values from a source JSON object.

    Eliminates multi-step JsonExtractValue and JsonReplace chains by automatically matching keys,
    resolving string placeholders/paths, or recursively searching the source JSON for matching keys.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self.category = "json"
        self.description = (
            "Automatically populate a template JSON object using matching keys, "
            "paths, or placeholders from a source JSON."
        )

        self.add_parameter(
            ParameterJson(
                name="template_json",
                tooltip="Target JSON template structure (keys can be null, default strings, or path placeholders).",
                default_value={},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            ParameterJson(
                name="value_source",
                tooltip="Source JSON object containing values to map into the template.",
                default_value={},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            ParameterJson(
                name="json_output",
                tooltip="Resulting automatically populated JSON structure.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def _get_by_path(self, data: Any, path: str) -> tuple[bool, Any]:
        """Navigate nested dicts/lists using dot or bracket paths (e.g. 'payload.asset_type' or 'llm_payload[asset_type]')."""
        if not isinstance(data, (dict, list)) or not path:
            return False, None

        # Convert bracket access like [key] or ['key'] into dot notation .key
        clean_path = re.sub(r"\[(['\"]?)(.*?)\1\]", r".\2", path)
        tokens = [t for t in clean_path.split(".") if t]

        curr = data
        for token in tokens:
            if isinstance(curr, dict) and token in curr:
                curr = curr[token]
            elif isinstance(curr, list) and token.isdigit() and int(token) < len(curr):
                curr = curr[int(token)]
            else:
                return False, None
        return True, curr

    def _search_key_recursive(self, data: Any, target_key: str) -> tuple[bool, Any]:
        """Recursively search for a matching key name anywhere within a nested dictionary or list structure."""
        if isinstance(data, dict):
            if target_key in data:
                return True, data[target_key]
            for v in data.values():
                found, val = self._search_key_recursive(v, target_key)
                if found:
                    return True, val
        elif isinstance(data, list):
            for item in data:
                found, val = self._search_key_recursive(item, target_key)
                if found:
                    return True, val
        return False, None

    def _resolve_value(self, key: str, template_val: Any, source: Any) -> Any:
        """Resolve a single template field value using key matching, string paths, or recursive search."""
        # 1. Handle explicit string placeholders (e.g., "clean_text_chunk", "{{text_content}}", "llm_payload[asset_type]")
        if isinstance(template_val, str) and template_val.strip():
            clean_str = template_val.strip()
            match = re.fullmatch(r"\{\{\s*(.*?)\s*\}\}", clean_str)
            lookup_target = match.group(1) if match else clean_str

            # Try direct path/key lookup in source using placeholder string
            found, val = self._get_by_path(source, lookup_target)
            if found:
                return val

            # Try recursive key search in source using placeholder string
            found, val = self._search_key_recursive(source, lookup_target)
            if found:
                return val

        # 2. Try direct path lookup in source using the template field's own key name
        found, val = self._get_by_path(source, key)
        if found:
            return val

        # 3. Try recursive key search across all nested levels of source using the template field's key name
        found, val = self._search_key_recursive(source, key)
        if found:
            return val

        # 4. Return original template value if no match is found
        return template_val

    def _auto_fill(self, template: Any, source: Any) -> Any:
        """Recursively traverse template structure and populate values from source."""
        if isinstance(template, dict):
            filled_dict: dict[str, Any] = {}
            for key, val in template.items():
                if isinstance(val, (dict, list)):
                    filled_dict[key] = self._auto_fill(val, source)
                else:
                    filled_dict[key] = self._resolve_value(key, val, source)
            return filled_dict
        elif isinstance(template, list):
            return [self._auto_fill(item, source) for item in template]
        return template

    def process(self) -> None:
        """Process the template JSON and automatically fill values from the source JSON."""
        template_input = self.get_parameter_value("template_json") or {}
        value_source = self.get_parameter_value("value_source") or {}

        if not isinstance(template_input, (dict, list)):
            template_input = {}
        if not isinstance(value_source, (dict, list)):
            value_source = {}

        result = copy.deepcopy(template_input)

        try:
            populated_result = self._auto_fill(result, value_source)
            self.parameter_output_values["json_output"] = populated_result
        except Exception as e:
            logger.exception("AutoJsonTemplateFiller failed to process JSON template: %s", e)
            self.parameter_output_values["json_output"] = template_input
            raise RuntimeError(f"AutoJsonTemplateFiller node '{self.name}' failed: {e}") from e
