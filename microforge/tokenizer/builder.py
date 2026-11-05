# Copyright 2024 MicroForge Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Domain-specific BPE tokenizer optimised for tool-calling and structured
extraction on ultra-lightweight edge models.

Wire format
-----------
Single tool call::

    [BOS] user query [SEP] [TOOL_START] tool_name [ARG_START] {"arg": "value"}
    [ARG_END] [TOOL_END] [CONF] 0.95 [EOS]

Multi-tool call::

    [BOS] user query [SEP]
    [TOOL_START] tool_a [ARG_START] {...} [ARG_END] [TOOL_END]
    [TOOL_START] tool_b [ARG_START] {...} [ARG_END] [TOOL_END]
    [CONF] 0.90 [EOS]

No-tool (empty) call::

    [BOS] user query [SEP] [EMPTY] [CONF] 0.00 [EOS]
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from tokenizers.normalizers import Lowercase, Sequence as NormSequence, NFD, StripAccents


# ---------------------------------------------------------------------------
# Special token definitions
# ---------------------------------------------------------------------------

SPECIAL_TOKENS: List[str] = [
    "[PAD]",        # 0
    "[BOS]",        # 1
    "[EOS]",        # 2
    "[TOOL_START]", # 3
    "[TOOL_END]",   # 4
    "[ARG_START]",  # 5
    "[ARG_END]",    # 6
    "[CONF]",       # 7
    "[SEP]",        # 8
    "[EMPTY]",      # 9
]

PAD_ID: int = 0
BOS_ID: int = 1
EOS_ID: int = 2
TOOL_START_ID: int = 3
TOOL_END_ID: int = 4
ARG_START_ID: int = 5
ARG_END_ID: int = 6
CONF_ID: int = 7
SEP_ID: int = 8
EMPTY_ID: int = 9


class ToolTokenizer:
    """BPE tokenizer tailored to MicroForge's tool-calling wire format.

    The tokenizer guarantees that every tool *name*, argument *key*, and common
    type keyword is encoded as a single token so the model can learn clean
    correspondences between queries and tool invocations.

    Parameters
    ----------
    vocab_size : int
        Maximum vocabulary size (default ``4096``).
    min_frequency : int
        Minimum frequency for BPE merges during training (default ``2``).
    """

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #

    def __init__(self, vocab_size: int = 4096, min_frequency: int = 2) -> None:
        self.vocab_size: int = vocab_size
        self.min_frequency: int = min_frequency

        # Will be set after training or loading
        self._tokenizer: Optional[Tokenizer] = None

        # Cached mappings populated after build
        self._special_token_ids: Dict[str, int] = {}
        self._tool_name_tokens: List[str] = []
        self._arg_name_tokens: List[str] = []

    # --------------------------------------------------------------------- #
    # Builder helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _make_base_tokenizer() -> Tokenizer:
        """Return an uninitialised BPE ``Tokenizer`` with sensible defaults."""
        tok = Tokenizer(models.BPE(unk_token="[PAD]"))
        # Normaliser: NFD → strip accents → lowercase
        tok.normalizer = NormSequence([NFD(), StripAccents(), Lowercase()])
        # Pre-tokeniser: split on whitespace + punctuation (keep delimiters)
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        return tok

    def _build_trainer(
        self,
        extra_single_tokens: Optional[List[str]] = None,
    ) -> trainers.BpeTrainer:
        """Return a ``BpeTrainer`` configured for tool-calling corpora."""
        special = list(SPECIAL_TOKENS)
        if extra_single_tokens:
            # De-duplicate while preserving order
            seen = set(special)
            for t in extra_single_tokens:
                if t not in seen:
                    special.append(t)
                    seen.add(t)
        return trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=special,
            show_progress=True,
        )

    @staticmethod
    def _extract_schema_atoms(schemas: List[Dict[str, Any]]) -> List[str]:
        """Pull out every tool name, arg name, type name, and enum value.

        These will be registered as *single* tokens so the model doesn't need
        to learn sub-word decompositions for critical identifiers.
        """
        atoms: List[str] = []
        seen: set = set()

        def _add(value: str) -> None:
            v = value.strip().lower()
            if v and v not in seen:
                atoms.append(v)
                seen.add(v)

        for schema in schemas:
            _add(schema.get("name", ""))
            params = schema.get("parameters", {})
            if isinstance(params, dict):
                props = params.get("properties", params)
                for arg_name, arg_spec in props.items():
                    _add(arg_name)
                    if isinstance(arg_spec, dict):
                        _add(arg_spec.get("type", ""))
                        for enum_val in arg_spec.get("enum", []):
                            _add(str(enum_val))
                        # Handle nested items
                        items = arg_spec.get("items", {})
                        if isinstance(items, dict):
                            _add(items.get("type", ""))
        return atoms

    # --------------------------------------------------------------------- #
    # Public training API
    # --------------------------------------------------------------------- #

    def train_from_corpus(self, texts: List[str]) -> "ToolTokenizer":
        """Train a BPE tokenizer from a raw text corpus.

        Parameters
        ----------
        texts : List[str]
            Raw training sentences / documents.

        Returns
        -------
        ToolTokenizer
            ``self``, for chaining.
        """
        self._tokenizer = self._make_base_tokenizer()
        trainer = self._build_trainer()
        self._tokenizer.train_from_iterator(texts, trainer=trainer)
        self._apply_post_processor()
        self._cache_special_ids()
        return self

    def build_from_tool_schemas(
        self,
        schemas: List[Dict[str, Any]],
        extra_corpus: Optional[List[str]] = None,
    ) -> "ToolTokenizer":
        """Build a tokenizer that treats every tool/arg name as a single token.

        The method extracts *atoms* (tool names, argument names, type keywords,
        enum values) from ``schemas`` and registers them as special single
        tokens.  An optional ``extra_corpus`` (e.g. user-query examples) can be
        provided so the BPE merges also cover natural-language fragments.

        Parameters
        ----------
        schemas : List[dict]
            Tool definitions in the MicroForge schema format.
        extra_corpus : List[str], optional
            Additional training sentences.

        Returns
        -------
        ToolTokenizer
            ``self``, for chaining.
        """
        atoms = self._extract_schema_atoms(schemas)
        self._tool_name_tokens = [
            s.get("name", "").strip().lower() for s in schemas if s.get("name")
        ]
        self._arg_name_tokens = []
        for s in schemas:
            params = s.get("parameters", {})
            props = params.get("properties", params) if isinstance(params, dict) else {}
            for k in props:
                key_lower = k.strip().lower()
                if key_lower not in self._arg_name_tokens:
                    self._arg_name_tokens.append(key_lower)

        self._tokenizer = self._make_base_tokenizer()
        trainer = self._build_trainer(extra_single_tokens=atoms)

        # Build a synthetic corpus from schemas so BPE has something to learn
        corpus: List[str] = []
        for schema in schemas:
            name = schema.get("name", "")
            desc = schema.get("description", "")
            corpus.append(f"{name} {desc}")
            params = schema.get("parameters", {})
            props = params.get("properties", params) if isinstance(params, dict) else {}
            for arg_name, arg_spec in props.items():
                arg_desc = arg_spec.get("description", "") if isinstance(arg_spec, dict) else ""
                corpus.append(f"{arg_name} {arg_desc}")
        if extra_corpus:
            corpus.extend(extra_corpus)

        # Also add JSON-structured examples so the tokenizer learns the format
        for schema in schemas:
            name = schema.get("name", "")
            params = schema.get("parameters", {})
            props = params.get("properties", params) if isinstance(params, dict) else {}
            sample_args: Dict[str, str] = {}
            for arg_name, arg_spec in props.items():
                if isinstance(arg_spec, dict):
                    enums = arg_spec.get("enum", [])
                    if enums:
                        sample_args[arg_name] = str(enums[0])
                    else:
                        sample_args[arg_name] = f"<{arg_spec.get('type', 'string')}>"
                else:
                    sample_args[arg_name] = "<value>"
            formatted = (
                f"[BOS] example query [SEP] [TOOL_START] {name} "
                f"[ARG_START] {json.dumps(sample_args)} [ARG_END] [TOOL_END] "
                f"[CONF] 0.95 [EOS]"
            )
            corpus.append(formatted)

        self._tokenizer.train_from_iterator(corpus, trainer=trainer)
        self._apply_post_processor()
        self._cache_special_ids()
        return self

    # --------------------------------------------------------------------- #
    # Post-processing & helpers
    # --------------------------------------------------------------------- #

    def _apply_post_processor(self) -> None:
        """Attach BOS/EOS wrapping as a ``TemplateProcessing`` post-processor."""
        assert self._tokenizer is not None
        bos = "[BOS]"
        eos = "[EOS]"
        bos_id = self._tokenizer.token_to_id(bos)
        eos_id = self._tokenizer.token_to_id(eos)
        if bos_id is not None and eos_id is not None:
            self._tokenizer.post_processor = processors.TemplateProcessing(
                single=f"{bos}:0 $A:0 {eos}:0",
                pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
                special_tokens=[(bos, bos_id), (eos, eos_id)],
            )

    def _cache_special_ids(self) -> None:
        """Cache special-token IDs for fast look-up."""
        assert self._tokenizer is not None
        for tok in SPECIAL_TOKENS:
            tid = self._tokenizer.token_to_id(tok)
            if tid is not None:
                self._special_token_ids[tok] = tid

    # --------------------------------------------------------------------- #
    # Encode / decode
    # --------------------------------------------------------------------- #

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """Encode ``text`` into a list of token IDs.

        Parameters
        ----------
        text : str
            Input text (may already include ``[BOS]`` / ``[EOS]`` markers).
        add_special_tokens : bool
            If ``True`` the post-processor appends ``[BOS]``/``[EOS]``.
            Set to ``False`` when the text already contains them.
        max_length : int, optional
            Truncate to this many tokens.

        Returns
        -------
        List[int]
            Token-ID sequence.
        """
        assert self._tokenizer is not None, "Tokenizer not trained / loaded."
        # Temporarily disable post-processor if caller doesn't want it
        if not add_special_tokens:
            saved_pp = self._tokenizer.post_processor
            self._tokenizer.post_processor = None  # type: ignore[assignment]
            enc = self._tokenizer.encode(text)
            self._tokenizer.post_processor = saved_pp
        else:
            enc = self._tokenizer.encode(text)
        ids = enc.ids
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: Union[List[int], Sequence[int]], skip_special: bool = False) -> str:
        """Decode token IDs back into a string.

        Parameters
        ----------
        ids : List[int]
            Token-ID sequence.
        skip_special : bool
            If ``True`` special tokens are omitted from the output.

        Returns
        -------
        str
            Decoded text.
        """
        assert self._tokenizer is not None, "Tokenizer not trained / loaded."
        return self._tokenizer.decode(list(ids), skip_special_tokens=skip_special)

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = False,
    ) -> List[List[int]]:
        """Encode a batch of texts.

        Parameters
        ----------
        texts : List[str]
            Input texts.
        add_special_tokens : bool
            Whether to add BOS/EOS via the post-processor.

        Returns
        -------
        List[List[int]]
            List of token-ID sequences.
        """
        assert self._tokenizer is not None, "Tokenizer not trained / loaded."
        if not add_special_tokens:
            saved_pp = self._tokenizer.post_processor
            self._tokenizer.post_processor = None  # type: ignore[assignment]
            encodings = self._tokenizer.encode_batch(texts)
            self._tokenizer.post_processor = saved_pp
        else:
            encodings = self._tokenizer.encode_batch(texts)
        return [e.ids for e in encodings]

    # --------------------------------------------------------------------- #
    # Token-ID helpers
    # --------------------------------------------------------------------- #

    def token_to_id(self, token: str) -> Optional[int]:
        """Return the integer ID for *token*, or ``None`` if unknown."""
        if self._tokenizer is None:
            return None
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Return the string token for *token_id*, or ``None``."""
        if self._tokenizer is None:
            return None
        return self._tokenizer.id_to_token(token_id)

    @property
    def pad_id(self) -> int:
        return self._special_token_ids.get("[PAD]", PAD_ID)

    @property
    def bos_id(self) -> int:
        return self._special_token_ids.get("[BOS]", BOS_ID)

    @property
    def eos_id(self) -> int:
        return self._special_token_ids.get("[EOS]", EOS_ID)

    @property
    def tool_start_id(self) -> int:
        return self._special_token_ids.get("[TOOL_START]", TOOL_START_ID)

    @property
    def tool_end_id(self) -> int:
        return self._special_token_ids.get("[TOOL_END]", TOOL_END_ID)

    @property
    def arg_start_id(self) -> int:
        return self._special_token_ids.get("[ARG_START]", ARG_START_ID)

    @property
    def arg_end_id(self) -> int:
        return self._special_token_ids.get("[ARG_END]", ARG_END_ID)

    @property
    def conf_id(self) -> int:
        return self._special_token_ids.get("[CONF]", CONF_ID)

    @property
    def sep_id(self) -> int:
        return self._special_token_ids.get("[SEP]", SEP_ID)

    @property
    def empty_id(self) -> int:
        return self._special_token_ids.get("[EMPTY]", EMPTY_ID)

    @property
    def special_token_ids(self) -> Dict[str, int]:
        """Return a copy of the special-token → ID mapping."""
        return dict(self._special_token_ids)

    @property
    def actual_vocab_size(self) -> int:
        """Return the real vocabulary size of the trained tokenizer."""
        assert self._tokenizer is not None, "Tokenizer not trained / loaded."
        return self._tokenizer.get_vocab_size()

    # --------------------------------------------------------------------- #
    # Format helpers
    # --------------------------------------------------------------------- #

    def format_tool_call(
        self,
        query: str,
        tool_calls: List[Dict[str, Any]],
        confidence: float = 1.0,
    ) -> str:
        """Produce the canonical wire-format string for a tool-calling example.

        Parameters
        ----------
        query : str
            The user query.
        tool_calls : List[Dict[str, Any]]
            Each dict has ``"name"`` and ``"arguments"`` keys.
        confidence : float
            Confidence score in ``[0, 1]``.

        Returns
        -------
        str
            Formatted string ready for :meth:`encode`.
        """
        parts: List[str] = [f"[BOS] {query} [SEP]"]
        if not tool_calls:
            parts.append("[EMPTY]")
        else:
            for call in tool_calls:
                name = call.get("name", "unknown")
                args = call.get("arguments", {})
                args_json = json.dumps(args, separators=(",", ":"))
                parts.append(
                    f"[TOOL_START] {name} [ARG_START] {args_json} [ARG_END] [TOOL_END]"
                )
        parts.append(f"[CONF] {confidence:.2f} [EOS]")
        return " ".join(parts)

    def format_extraction(
        self,
        text: str,
        extracted: Dict[str, Any],
    ) -> str:
        """Produce the canonical wire-format string for a structured extraction.

        Parameters
        ----------
        text : str
            Source text from which data is extracted.
        extracted : Dict[str, Any]
            The extracted key-value pairs.

        Returns
        -------
        str
            Formatted string ready for :meth:`encode`.
        """
        args_json = json.dumps(extracted, separators=(",", ":"))
        return (
            f"[BOS] {text} [SEP] [TOOL_START] extract "
            f"[ARG_START] {args_json} [ARG_END] [TOOL_END] [EOS]"
        )

    # --------------------------------------------------------------------- #
    # Persistence
    # --------------------------------------------------------------------- #

    def save(self, path: Union[str, Path]) -> None:
        """Save the tokenizer and metadata to *path*.

        Creates two files:
        - ``<path>/tokenizer.json`` – the HuggingFace ``tokenizers`` model.
        - ``<path>/meta.json`` – extra MicroForge metadata.
        """
        assert self._tokenizer is not None, "Nothing to save – train first."
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._tokenizer.save(str(path / "tokenizer.json"))

        meta = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "special_token_ids": self._special_token_ids,
            "tool_name_tokens": self._tool_name_tokens,
            "arg_name_tokens": self._arg_name_tokens,
        }
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ToolTokenizer":
        """Load a previously-saved tokenizer from *path*.

        Parameters
        ----------
        path : str | Path
            Directory that contains ``tokenizer.json`` and ``meta.json``.

        Returns
        -------
        ToolTokenizer
            Restored instance ready for :meth:`encode` / :meth:`decode`.
        """
        path = Path(path)
        tok_path = path / "tokenizer.json"
        meta_path = path / "meta.json"

        if not tok_path.exists():
            raise FileNotFoundError(f"No tokenizer.json in {path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(
            vocab_size=meta.get("vocab_size", 4096),
            min_frequency=meta.get("min_frequency", 2),
        )
        instance._tokenizer = Tokenizer.from_file(str(tok_path))
        instance._special_token_ids = {
            k: int(v) for k, v in meta.get("special_token_ids", {}).items()
        }
        instance._tool_name_tokens = meta.get("tool_name_tokens", [])
        instance._arg_name_tokens = meta.get("arg_name_tokens", [])
        return instance

    # --------------------------------------------------------------------- #
    # Dunder helpers
    # --------------------------------------------------------------------- #

    def __repr__(self) -> str:
        trained = self._tokenizer is not None
        return (
            f"ToolTokenizer(vocab_size={self.vocab_size}, "
            f"trained={trained})"
        )

    def __len__(self) -> int:
        if self._tokenizer is None:
            return 0
        return self._tokenizer.get_vocab_size()
