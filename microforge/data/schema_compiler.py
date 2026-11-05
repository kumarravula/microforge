# Copyright 2024 MicroForge Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Compile JSON-Schema tool definitions into structures used by grammar-
constrained decoding and the tool-classification router.

``SchemaCompiler`` produces a :class:`CompiledSchema` that maps every tool
name to an integer ID, records per-tool argument constraints, and pre-computes
token-level constraints so the decoder can mask out invalid continuations and
guarantee valid JSON output.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from microforge.tokenizer.builder import ToolTokenizer


# ─────────────────────────────────────────────────────────────────────────── #
# Data structures
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class ArgumentConstraint:
    """Constraints for a single tool argument.

    Attributes
    ----------
    name : str
        Argument name (key in the JSON object).
    dtype : str
        Expected JSON type (``"string"``, ``"integer"``, ``"number"``,
        ``"boolean"``, ``"array"``, ``"object"``).
    required : bool
        Whether the argument is required.
    enum_values : List[str]
        If non-empty, the argument must be one of these literal values.
    description : str
        Human-readable description (carried for debugging / prompts).
    """

    name: str
    dtype: str = "string"
    required: bool = False
    enum_values: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class ToolSchema:
    """Compiled representation of a single tool.

    Attributes
    ----------
    tool_id : int
        Dense integer identifier (used as the label for the router head).
    name : str
        Tool function name.
    description : str
        Short description of what the tool does.
    arguments : List[ArgumentConstraint]
        Per-argument constraints.
    required_args : List[str]
        Names of required arguments.
    """

    tool_id: int
    name: str
    description: str = ""
    arguments: List[ArgumentConstraint] = field(default_factory=list)
    required_args: List[str] = field(default_factory=list)


@dataclass
class TokenConstraint:
    """Allowed / disallowed token IDs at a given decoding state.

    Attributes
    ----------
    state : str
        A human-readable label for the decoding state (e.g.
        ``"after_arg_start"``, ``"inside_string_value"``).
    allowed_token_ids : FrozenSet[int]
        Token IDs that may legally follow. An empty set means "all allowed".
    disallowed_token_ids : FrozenSet[int]
        Token IDs that are explicitly forbidden.
    """

    state: str
    allowed_token_ids: FrozenSet[int] = field(default_factory=frozenset)
    disallowed_token_ids: FrozenSet[int] = field(default_factory=frozenset)


@dataclass
class CompiledSchema:
    """The full compiled output of :class:`SchemaCompiler`.

    Attributes
    ----------
    tool_names : List[str]
        Ordered list of tool names (index == tool_id).
    tool_name_to_id : Dict[str, int]
        Mapping from name → dense integer label.
    tool_schemas : Dict[str, ToolSchema]
        Per-tool compiled schema objects.
    argument_schemas : Dict[str, List[ArgumentConstraint]]
        Mapping from tool name → list of argument constraints.
    token_constraints : Dict[str, TokenConstraint]
        Named decoding-state constraints keyed by state label.
    num_tools : int
        Total number of tools (convenience field).
    json_structural_tokens : FrozenSet[int]
        Token IDs that correspond to JSON structural characters
        (``{``, ``}``, ``[``, ``]``, ``:`` , ``,`` , ``"``).
    """

    tool_names: List[str] = field(default_factory=list)
    tool_name_to_id: Dict[str, int] = field(default_factory=dict)
    tool_schemas: Dict[str, ToolSchema] = field(default_factory=dict)
    argument_schemas: Dict[str, List[ArgumentConstraint]] = field(default_factory=dict)
    token_constraints: Dict[str, TokenConstraint] = field(default_factory=dict)
    num_tools: int = 0
    json_structural_tokens: FrozenSet[int] = field(default_factory=frozenset)


# ─────────────────────────────────────────────────────────────────────────── #
# Compiler
# ─────────────────────────────────────────────────────────────────────────── #

class SchemaCompiler:
    """Compile raw JSON-Schema tool definitions for use in MicroForge.

    Parameters
    ----------
    tokenizer : ToolTokenizer
        A trained :class:`ToolTokenizer` (needed to resolve token IDs for
        grammar-constrained decoding masks).
    """

    # JSON structural characters whose token IDs we want to cache.
    _JSON_CHARS: Tuple[str, ...] = ("{", "}", "[", "]", ":", ",", '"')

    def __init__(self, tokenizer: ToolTokenizer) -> None:
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def compile_tools(self, tools: List[Dict[str, Any]]) -> CompiledSchema:
        """Compile a list of tool definitions into a :class:`CompiledSchema`.

        Parameters
        ----------
        tools : List[dict]
            Each dict should contain at minimum ``"name"`` and
            ``"parameters"``.  The ``"parameters"`` value is a JSON-Schema
            ``"object"`` with ``"properties"`` and optional ``"required"``.

        Returns
        -------
        CompiledSchema
        """
        schema = CompiledSchema()

        # 1. Build tool_name → id mapping
        for idx, tool_def in enumerate(tools):
            name: str = tool_def.get("name", f"tool_{idx}")
            schema.tool_names.append(name)
            schema.tool_name_to_id[name] = idx

            # 2. Parse argument constraints
            arg_constraints = self._parse_arguments(tool_def)
            required_args = self._get_required_args(tool_def)

            ts = ToolSchema(
                tool_id=idx,
                name=name,
                description=tool_def.get("description", ""),
                arguments=arg_constraints,
                required_args=required_args,
            )
            schema.tool_schemas[name] = ts
            schema.argument_schemas[name] = arg_constraints

        schema.num_tools = len(tools)

        # 3. Build token-level constraints for grammar-constrained decoding
        schema.json_structural_tokens = self._resolve_json_structural_tokens()
        schema.token_constraints = self._build_token_constraints(schema)

        return schema

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_arguments(tool_def: Dict[str, Any]) -> List[ArgumentConstraint]:
        """Extract :class:`ArgumentConstraint` objects from a tool definition."""
        params = tool_def.get("parameters", {})
        if not isinstance(params, dict):
            return []

        props = params.get("properties", {})
        required_set: Set[str] = set(params.get("required", []))

        constraints: List[ArgumentConstraint] = []
        for arg_name, arg_spec in props.items():
            if not isinstance(arg_spec, dict):
                continue
            constraints.append(
                ArgumentConstraint(
                    name=arg_name,
                    dtype=arg_spec.get("type", "string"),
                    required=(arg_name in required_set),
                    enum_values=[str(v) for v in arg_spec.get("enum", [])],
                    description=arg_spec.get("description", ""),
                )
            )
        return constraints

    @staticmethod
    def _get_required_args(tool_def: Dict[str, Any]) -> List[str]:
        """Return the list of required argument names for a tool."""
        params = tool_def.get("parameters", {})
        if isinstance(params, dict):
            return list(params.get("required", []))
        return []

    def _resolve_json_structural_tokens(self) -> FrozenSet[int]:
        """Map JSON punctuation characters to their token IDs."""
        ids: Set[int] = set()
        for ch in self._JSON_CHARS:
            token_ids = self.tokenizer.encode(ch, add_special_tokens=False)
            ids.update(token_ids)
        return frozenset(ids)

    def _build_token_constraints(
        self,
        schema: CompiledSchema,
    ) -> Dict[str, TokenConstraint]:
        """Pre-compute token constraints for key decoding states.

        The states modelled here correspond to positions inside the
        MicroForge wire format::

            [TOOL_START] <tool_name> [ARG_START] { ... } [ARG_END] [TOOL_END]

        Returns
        -------
        Dict[str, TokenConstraint]
            A mapping from state name to constraint.
        """
        constraints: Dict[str, TokenConstraint] = {}

        # --- State: right after [TOOL_START] → only tool-name tokens allowed
        tool_name_ids: Set[int] = set()
        for name in schema.tool_names:
            ids = self.tokenizer.encode(name, add_special_tokens=False)
            tool_name_ids.update(ids)
        # Also allow the "extract" pseudo-tool for extraction tasks
        extract_ids = self.tokenizer.encode("extract", add_special_tokens=False)
        tool_name_ids.update(extract_ids)

        constraints["after_tool_start"] = TokenConstraint(
            state="after_tool_start",
            allowed_token_ids=frozenset(tool_name_ids),
        )

        # --- State: right after <tool_name> → must be [ARG_START]
        arg_start_id = self.tokenizer.token_to_id("[ARG_START]")
        if arg_start_id is not None:
            constraints["after_tool_name"] = TokenConstraint(
                state="after_tool_name",
                allowed_token_ids=frozenset({arg_start_id}),
            )

        # --- State: right after [ARG_START] → must be "{" (JSON object start)
        open_brace_ids = set(self.tokenizer.encode("{", add_special_tokens=False))
        if open_brace_ids:
            constraints["after_arg_start"] = TokenConstraint(
                state="after_arg_start",
                allowed_token_ids=frozenset(open_brace_ids),
            )

        # --- State: inside JSON argument object → collect all valid key tokens
        all_key_ids: Set[int] = set()
        for tool_name, arg_list in schema.argument_schemas.items():
            for ac in arg_list:
                key_tokens = self.tokenizer.encode(
                    f'"{ac.name}"', add_special_tokens=False
                )
                all_key_ids.update(key_tokens)
        # Also allow JSON structural tokens and whitespace inside the object
        all_key_ids.update(schema.json_structural_tokens)
        constraints["inside_json_object"] = TokenConstraint(
            state="inside_json_object",
            allowed_token_ids=frozenset(),  # empty = all allowed (too many value tokens)
            disallowed_token_ids=frozenset(),
        )

        # --- State: after "}" → must be [ARG_END]
        arg_end_id = self.tokenizer.token_to_id("[ARG_END]")
        if arg_end_id is not None:
            constraints["after_json_close"] = TokenConstraint(
                state="after_json_close",
                allowed_token_ids=frozenset({arg_end_id}),
            )

        # --- State: after [ARG_END] → must be [TOOL_END]
        tool_end_id = self.tokenizer.token_to_id("[TOOL_END]")
        if tool_end_id is not None:
            constraints["after_arg_end"] = TokenConstraint(
                state="after_arg_end",
                allowed_token_ids=frozenset({tool_end_id}),
            )

        # --- State: after [TOOL_END] → [TOOL_START] (multi-tool) or [CONF]
        allowed_after_tool_end: Set[int] = set()
        tool_start_id = self.tokenizer.token_to_id("[TOOL_START]")
        conf_id = self.tokenizer.token_to_id("[CONF]")
        if tool_start_id is not None:
            allowed_after_tool_end.add(tool_start_id)
        if conf_id is not None:
            allowed_after_tool_end.add(conf_id)
        constraints["after_tool_end"] = TokenConstraint(
            state="after_tool_end",
            allowed_token_ids=frozenset(allowed_after_tool_end),
        )

        # --- Per-tool enum constraints (for grammar masking during value gen)
        for tool_name, arg_list in schema.argument_schemas.items():
            for ac in arg_list:
                if ac.enum_values:
                    enum_ids: Set[int] = set()
                    for val in ac.enum_values:
                        ids = self.tokenizer.encode(
                            f'"{val}"', add_special_tokens=False
                        )
                        enum_ids.update(ids)
                    state_key = f"enum_{tool_name}_{ac.name}"
                    constraints[state_key] = TokenConstraint(
                        state=state_key,
                        allowed_token_ids=frozenset(enum_ids),
                    )

        return constraints

    # ------------------------------------------------------------------ #
    # Utility: build multi-hot tool label
    # ------------------------------------------------------------------ #

    def build_tool_label(
        self,
        compiled: CompiledSchema,
        tool_names: List[str],
    ) -> List[float]:
        """Return a multi-hot float vector for the given tool names.

        Parameters
        ----------
        compiled : CompiledSchema
            The compiled schema (provides ``num_tools``).
        tool_names : List[str]
            Tools invoked in this example.

        Returns
        -------
        List[float]
            A ``num_tools``-length vector with 1.0 at active positions.
        """
        label = [0.0] * compiled.num_tools
        for name in tool_names:
            idx = compiled.tool_name_to_id.get(name)
            if idx is not None:
                label[idx] = 1.0
        return label

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    @staticmethod
    def compiled_to_dict(compiled: CompiledSchema) -> Dict[str, Any]:
        """Serialise a :class:`CompiledSchema` to a JSON-friendly dict."""
        return {
            "tool_names": compiled.tool_names,
            "tool_name_to_id": compiled.tool_name_to_id,
            "num_tools": compiled.num_tools,
            "argument_schemas": {
                name: [
                    {
                        "name": ac.name,
                        "dtype": ac.dtype,
                        "required": ac.required,
                        "enum_values": ac.enum_values,
                        "description": ac.description,
                    }
                    for ac in acs
                ]
                for name, acs in compiled.argument_schemas.items()
            },
        }

    @staticmethod
    def compiled_from_dict(data: Dict[str, Any]) -> CompiledSchema:
        """Deserialise a :class:`CompiledSchema` from a dict (no token
        constraints – those require a live tokenizer)."""
        cs = CompiledSchema(
            tool_names=data["tool_names"],
            tool_name_to_id=data["tool_name_to_id"],
            num_tools=data["num_tools"],
        )
        for tool_name, arg_dicts in data.get("argument_schemas", {}).items():
            constraints = [
                ArgumentConstraint(
                    name=ad["name"],
                    dtype=ad.get("dtype", "string"),
                    required=ad.get("required", False),
                    enum_values=ad.get("enum_values", []),
                    description=ad.get("description", ""),
                )
                for ad in arg_dicts
            ]
            cs.argument_schemas[tool_name] = constraints
            cs.tool_schemas[tool_name] = ToolSchema(
                tool_id=cs.tool_name_to_id.get(tool_name, -1),
                name=tool_name,
                arguments=constraints,
                required_args=[ac.name for ac in constraints if ac.required],
            )
        return cs
