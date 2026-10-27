"""
Load, clean, and merge the prompt-safety corpora shared by the training and TFB notebooks.

Three Hugging Face datasets are combined (see ADR 0002, "Data"):

- ``nvidia/Nemotron-Safety-Guard-Dataset-v3`` ("v3"): human-labeled prompts in 12
  languages with fine-grained categories, plus a synthetic jailbreak slice. It supplies
  training data, the validation and test splits, and the held-out jailbreak test.
- ``ToxicityPrompts/PolyGuardMix``: about 1.9M prompts in 17 languages labeled by an
  LLM with MLCommons hazard codes. All of it goes to training.
- ``ToxicityPrompts/PolyGuardPrompts``: human-labeled translations of WildGuardTest in
  17 languages, used as a second, out-of-distribution test set.

Every source is converted to :class:`SafetyRecord`, cleaned on its own, and then
merged with cross-dataset rules that keep evaluation prompts out of training.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset, load_from_disk  # pyright: ignore[reportUnknownVariableType]
from huggingface_hub import snapshot_download  # pyright: ignore[reportUnknownVariableType]
from pydantic import BaseModel, ConfigDict
from transformers import PreTrainedTokenizerBase

NEMOTRON_V3_ID = "nvidia/Nemotron-Safety-Guard-Dataset-v3"
POLYGUARDMIX_ID = "ToxicityPrompts/PolyGuardMix"
POLYGUARDPROMPTS_ID = "ToxicityPrompts/PolyGuardPrompts"

NEMOTRON_LANGUAGES: tuple[str, ...] = (
    "ar", "de", "en", "es", "fr", "hi", "it", "ja", "ko", "nl", "th", "zh",
)  # fmt: skip
POLYGUARD_LANGUAGE_CODES: Mapping[str, str] = {
    "Arabic": "ar",
    "Chinese": "zh",
    "Czech": "cs",
    "Dutch": "nl",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Polish": "pl",
    "Portuguese": "pt",
    "Russian": "ru",
    "Spanish": "es",
    "Swedish": "sv",
    "Thai": "th",
}
ALL_LANGUAGES: tuple[str, ...] = tuple(sorted(POLYGUARD_LANGUAGE_CODES.values()))
SOURCE_LANGUAGE = "en"

NEMOTRON_SPLIT_FILES: Mapping[str, str] = {
    "train": "train.jsonl",
    "validation": "valid.jsonl",
    "test": "test.jsonl",
}
REDACTED_PROMPT = "REDACTED"
HUMAN_TAGS: tuple[str, ...] = ("generic", "adapted")
JAILBREAK_TAGS: tuple[str, ...] = ("jailbreaking",)

MLCOMMONS_TO_NEMOTRON: Mapping[str, str] = {
    "S4": "Sexual (minor)",
    "S6": "Unauthorized Advice",
    "S7": "PII/Privacy",
    "S8": "Copyright/Trademark/Plagiarism",
    "S10": "Hate/Identity Hate",
    "S11": "Suicide and Self Harm",
    "S14": "Malware",
}
MLCOMMONS_CODES: frozenset[str] = frozenset(f"S{index}" for index in range(1, 15))
MAPPED_CATEGORIES: tuple[str, ...] = tuple(sorted(set(MLCOMMONS_TO_NEMOTRON.values())))

# Bump whenever the cleaning or merge rules change, so stale caches are rebuilt.
CLEANING_VERSION = 1

TokenCounter = Callable[[Sequence[str]], list[int]]


class Source(StrEnum):
    """Dataset a record comes from."""

    NEMOTRON_V3 = "nemotron_v3"
    POLYGUARDMIX = "polyguardmix"
    POLYGUARDPROMPTS = "polyguardprompts"


class Label(StrEnum):
    """Prompt-level safety label."""

    SAFE = "safe"
    UNSAFE = "unsafe"


class CategoryCoverage(StrEnum):
    """
    How completely a record's ``violated_categories`` describe the prompt.

    ``ALL``: every v3 category is a known 0/1 target. ``MAPPED``: only the categories in
    :data:`MAPPED_CATEGORIES` are known. ``NONE``: no category target is known.
    """

    ALL = "all"
    MAPPED = "mapped"
    NONE = "none"


class SplitName(StrEnum):
    """Names of the merged splits."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    JAILBREAK_TEST = "jailbreak_test"
    POLYGUARDPROMPTS_TEST = "polyguardprompts_test"


# ---------------------------------------------------------------------------
# Raw row schemas: the contract between the files on disk and the cleaning code.
# ---------------------------------------------------------------------------


class NemotronRow(BaseModel):
    """One line of a v3 ``{language}/{split}.jsonl`` file (unused columns ignored)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    prompt: str
    prompt_label: Label
    violated_categories: str = ""
    tag: str


class PolyGuardMixMetadata(BaseModel):
    """The ``metadata`` struct of a PolyGuardMix row."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    language: str
    source: str


class PolyGuardMixRow(BaseModel):
    """One PolyGuardMix parquet row (unused columns ignored)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    prompt: str | None
    prompt_harm_label: Literal["yes", "no"] | None
    prompt_safety_categories: str | None
    metadata: PolyGuardMixMetadata


class PolyGuardPromptsRow(BaseModel):
    """One PolyGuardPrompts parquet row (unused columns ignored)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: int
    language: str
    prompt: str | None
    prompt_harm_label: Literal["harmful", "unharmful"] | None
    adversarial: bool


@dataclass(frozen=True, slots=True)
class NemotronRaw:
    """
    A v3 row together with the language and split of the file it was read from.

    :param split:       ``"train"``, ``"validation"`` or ``"test"``.
    :param language:    ISO code of the file's language directory.
    :param row:         The parsed row.
    """

    split: str
    language: str
    row: NemotronRow


@dataclass(frozen=True, slots=True)
class SafetyRecord:
    """
    A cleaned prompt in the schema shared by every source.

    :param id:                  Identifier shared by the translations of one prompt
                                (v3 and PolyGuardPrompts); unique per text for PolyGuardMix.
    :param source:              Dataset of origin.
    :param language:            ISO language code.
    :param tag:                 v3 tag, PolyGuardMix sub-source, or PolyGuardPrompts
                                ``"adversarial"`` / ``"vanilla"``.
    :param prompt:              Prompt text.
    :param prompt_label:        Safe or unsafe.
    :param violated_categories: v3 category names (first-seen order for v3, sorted for
                                PolyGuardMix); empty for safe prompts.
    :param caution_known:       Whether the v3 "Needs Caution" label is a known target.
    :param category_coverage:   Which category targets are known.
    :param num_tokens:          Token count including special tokens, before truncation.
    """

    id: str
    source: Source
    language: str
    tag: str
    prompt: str
    prompt_label: Label
    violated_categories: tuple[str, ...]
    caution_known: bool
    category_coverage: CategoryCoverage
    num_tokens: int


# ---------------------------------------------------------------------------
# Cleaning reports.
# ---------------------------------------------------------------------------


@dataclass
class NemotronCleaningReport:
    """
    Row counts removed by each v3 cleaning step, keyed by ``(split, language)``.

    :param raw:             Rows read in the requested languages.
    :param redacted:        Empty or ``REDACTED`` prompts.
    :param too_long:        Rows longer than the token limit.
    :param merged:          Repeated ``(language, prompt)`` rows folded into one record.
    :param conflicts:       Prompts whose repeats disagree on the label.
    :param leaked:          Prompts that occur in a later split, by text or English source.
    :param kept:            Records kept.
    """

    raw: Counter[tuple[str, str]] = field(default_factory=Counter[tuple[str, str]])
    redacted: Counter[tuple[str, str]] = field(default_factory=Counter[tuple[str, str]])
    merged: Counter[tuple[str, str]] = field(default_factory=Counter[tuple[str, str]])
    conflicts: Counter[tuple[str, str]] = field(default_factory=Counter[tuple[str, str]])
    too_long: Counter[tuple[str, str]] = field(default_factory=Counter[tuple[str, str]])
    leaked: Counter[tuple[str, str]] = field(default_factory=Counter[tuple[str, str]])
    kept: Counter[tuple[str, str]] = field(default_factory=Counter[tuple[str, str]])


@dataclass
class PolyGuardCleaningReport:
    """
    Row counts removed by each PolyGuardMix or PolyGuardPrompts cleaning step, per language.

    :param raw:             Rows read in the requested languages.
    :param missing:         Rows without a prompt or a label.
    :param merged:          Repeated ``(language, prompt)`` rows folded into one record.
    :param conflicts:       Prompts whose repeats disagree on the label.
    :param too_long:        Prompts longer than the token limit.
    :param kept:            Records kept.
    """

    raw: Counter[str] = field(default_factory=Counter[str])
    missing: Counter[str] = field(default_factory=Counter[str])
    merged: Counter[str] = field(default_factory=Counter[str])
    conflicts: Counter[str] = field(default_factory=Counter[str])
    too_long: Counter[str] = field(default_factory=Counter[str])
    kept: Counter[str] = field(default_factory=Counter[str])


@dataclass
class MergeReport:
    """
    Records removed by the cross-dataset rules.

    :param polyguardmix_in_v3:      PolyGuardMix prompts whose text matches a v3 prompt in
                                    the same language (the v3 human label wins).
    :param v3_eval_in_polyguardmix: v3 validation/test records, keyed by split, whose
                                    English source appears in PolyGuardMix.
    :param polyguardprompts_leaked: PolyGuardPrompts records whose id has a translation
                                    that matches a training prompt.
    :param split_sizes:             Final number of records per split.
    """

    polyguardmix_in_v3: int = 0
    v3_eval_in_polyguardmix: Counter[str] = field(default_factory=Counter[str])
    polyguardprompts_leaked: int = 0
    split_sizes: dict[str, int] = field(default_factory=dict[str, int])


@dataclass
class CleaningReport:
    """
    Everything removed while building the merged splits.

    :param nemotron:            v3 cleaning steps.
    :param polyguardmix:        PolyGuardMix cleaning steps.
    :param polyguardprompts:    PolyGuardPrompts cleaning steps.
    :param merge:               Cross-dataset rules.
    """

    nemotron: NemotronCleaningReport
    polyguardmix: PolyGuardCleaningReport
    polyguardprompts: PolyGuardCleaningReport
    merge: MergeReport


# ---------------------------------------------------------------------------
# Text helpers.
# ---------------------------------------------------------------------------

_WHITESPACE = re.compile(r"\s+")


def normalize_prompt(text: str) -> str:
    """
    Return the key used to detect the same prompt across datasets.

    Unicode NFKC folding, whitespace collapsing and case folding make copies that differ
    only in formatting compare equal.

    :param text:    Prompt text.
    :return:        Normalized text.
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip().casefold()


def parse_categories(raw: str | None) -> list[str]:
    """
    Split a ``violated_categories`` string into category names.

    :param raw: Comma-separated categories; ``None`` or empty means none.
    :return:    Stripped category names in their original order.
    """
    return [category.strip() for category in (raw or "").split(",") if category.strip()]


def make_token_counter(tokenizer: PreTrainedTokenizerBase, chunk_size: int = 4096) -> TokenCounter:
    """
    Wrap a tokenizer as a :data:`TokenCounter`.

    Texts are tokenized in chunks so that the token ids of a million prompts are never
    held at once.

    :param tokenizer:   Tokenizer of the encoder being trained.
    :param chunk_size:  Texts tokenized per call.
    :return:            A function mapping texts to token counts, special tokens included.
    """

    def count(texts: Sequence[str]) -> list[int]:
        counts: list[int] = []
        for start in range(0, len(texts), chunk_size):
            chunk = list(texts[start : start + chunk_size])
            input_ids = cast(
                list[list[int]], tokenizer(chunk, truncation=False, verbose=False)["input_ids"]
            )
            counts.extend(len(ids) for ids in input_ids)
        return counts

    return count


def count_unique_tokens(texts: Iterable[str], count_tokens: TokenCounter) -> dict[str, int]:
    """
    Count tokens once per distinct text.

    :param texts:           Texts, possibly repeated.
    :param count_tokens:    Maps a batch of texts to their token counts.
    :return:                Token count per distinct text.
    """
    unique = sorted(set(texts))
    return dict(zip(unique, count_tokens(unique), strict=True))


# ---------------------------------------------------------------------------
# Readers: the only functions that touch the files.
# ---------------------------------------------------------------------------


def read_nemotron_v3(root: Path, languages: Iterable[str]) -> list[NemotronRaw]:
    """
    Read the v3 JSONL files of ``languages`` under a dataset snapshot.

    :param root:        Snapshot directory containing one sub-directory per language.
    :param languages:   ISO codes to read.
    :return:            Every row with its split and language.
    """
    rows: list[NemotronRaw] = []
    for language in languages:
        for split, filename in NEMOTRON_SPLIT_FILES.items():
            with (root / language / filename).open(encoding="utf-8") as handle:
                rows.extend(
                    NemotronRaw(split, language, NemotronRow.model_validate_json(line))
                    for line in handle
                    if line.strip()
                )
    return rows


def parquet_rows(path: Path, columns: Sequence[str] | None) -> Iterator[dict[str, object]]:
    """
    Stream the rows of a parquet file one row group at a time.

    :param path:    Parquet file.
    :param columns: Columns to read; ``None`` reads all of them.
    :return:        Rows as dictionaries.
    """
    # pyarrow ships without type information, so its results are cast at this boundary.
    parquet = pq.ParquetFile(path)
    num_groups = cast(int, parquet.num_row_groups)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
    for group in range(num_groups):
        table = parquet.read_row_group(group, columns=columns)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
        yield from cast(list[dict[str, object]], table.to_pylist())  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]


def read_polyguardmix(
    paths: Iterable[Path], languages: Iterable[str]
) -> Iterator[tuple[str, PolyGuardMixRow]]:
    """
    Stream PolyGuardMix rows in ``languages`` from its parquet shards.

    Row groups are read one at a time with only the needed columns, so the 2.3 GB
    dataset never sits in memory at once.

    :param paths:       Parquet shard paths.
    :param languages:   ISO codes to keep.
    :return:            ``(language code, row)`` pairs.
    """
    wanted = set(languages)
    columns = ["prompt", "prompt_harm_label", "prompt_safety_categories", "metadata"]
    for path in paths:
        for item in parquet_rows(path, columns):
            row = PolyGuardMixRow.model_validate(item)
            code = POLYGUARD_LANGUAGE_CODES.get(row.metadata.language)
            if code in wanted:
                yield code, row


def read_polyguardprompts(
    paths: Iterable[Path], languages: Iterable[str]
) -> list[tuple[str, PolyGuardPromptsRow]]:
    """
    Read PolyGuardPrompts rows in ``languages``.

    :param paths:       Parquet file paths.
    :param languages:   ISO codes to keep.
    :return:            ``(language code, row)`` pairs.
    """
    wanted = set(languages)
    rows: list[tuple[str, PolyGuardPromptsRow]] = []
    for path in paths:
        for item in parquet_rows(path, None):
            row = PolyGuardPromptsRow.model_validate(item)
            code = POLYGUARD_LANGUAGE_CODES.get(row.language)
            if code in wanted:
                rows.append((code, row))
    return rows


# ---------------------------------------------------------------------------
# Per-source cleaning.
# ---------------------------------------------------------------------------


@dataclass
class _NemotronGroup:
    """
    Repeats of one ``(language, prompt)`` pair within a v3 split.

    :param first:       The first row seen; supplies the id and tag of the record.
    :param labels:      Every label seen.
    :param categories:  Union of the categories in first-seen order.
    :param sources:     Leakage groups of every repeat.
    """

    first: NemotronRow
    labels: set[Label] = field(default_factory=set[Label])
    categories: list[str] = field(default_factory=list[str])
    sources: set[str] = field(default_factory=set[str])

    def add(self, row: NemotronRow, source: str) -> None:
        """
        Fold one repeat into the group.

        :param row:     A row with this group's language and prompt.
        :param source:  The row's leakage group.
        """
        self.labels.add(row.prompt_label)
        self.sources.add(source)
        for category in parse_categories(row.violated_categories):
            if category not in self.categories:
                self.categories.append(category)


@dataclass(frozen=True, slots=True)
class NemotronSplits:
    """
    Cleaned v3 records.

    :param splits:              Records per v3 split (``train``/``validation``/``test``).
    :param english_source:      Normalized English prompt of every v3 id that has one.
    """

    splits: dict[str, list[SafetyRecord]]
    english_source: dict[str, str]


def clean_nemotron(
    raw_rows: Sequence[NemotronRaw],
    languages: Iterable[str],
    count_tokens: TokenCounter,
    max_length: int,
    report: NemotronCleaningReport,
) -> NemotronSplits:
    """
    Build deduplicated, leakage-free v3 splits that fit the token budget.

    Aegis 2.0, and therefore every translation of it, pairs one prompt with several
    responses and repeats some prompts under different ids in different splits. Each
    split is cleaned in five steps:

    1. Drop empty and ``REDACTED`` placeholder prompts, whose text cannot support any label.
    2. Drop prompts longer than ``max_length`` tokens (special tokens included), so
       the collate function never truncates.
    3. Merge repeated prompts of one language into one record whose categories are
       the union over the repeats in first-seen order; ``violated_categories`` is
       already a union over annotators, so the union keeps its meaning.
    4. Drop prompts whose repeats disagree on ``prompt_label``.
    5. Drop prompts that occur in a later split (train, validation, test), either as
       the same text in the same language or through an id whose English source
       prompt occurs in a later split. The second test catches translations of
       Aegis prompts that were duplicated across splits under different ids.

    Translations share the id of their English source row, so English rows define the
    leakage groups even when English is not in ``languages``.

    :param raw_rows:        Rows from :func:`read_nemotron_v3`, English included.
    :param languages:       ISO codes to keep in the output.
    :param count_tokens:    Maps a batch of texts to their token counts.
    :param max_length:      Longest prompt kept, in tokens (inclusive).
    :param report:          Report updated in place.
    :return:                The cleaned splits and the English source of every id.
    """
    keep_languages = set(languages)
    split_rank = {split: rank for rank, split in enumerate(NEMOTRON_SPLIT_FILES)}

    # Leakage groups: every id maps to its English source prompt, or to itself when no
    # English row exists. A group's last split is the latest split it occurs in.
    source_of_id: dict[str, str] = {
        raw.row.id: raw.row.prompt
        for raw in raw_rows
        if raw.language == SOURCE_LANGUAGE and raw.row.prompt != REDACTED_PROMPT
    }
    last_rank_of_source: dict[str, int] = {}
    last_rank_of_text: dict[tuple[str, str], int] = {}
    for raw in raw_rows:
        rank = split_rank[raw.split]
        source = source_of_id.get(raw.row.id, raw.row.id)
        last_rank_of_source[source] = max(last_rank_of_source.get(source, 0), rank)
        text_key = (raw.language, raw.row.prompt)
        last_rank_of_text[text_key] = max(last_rank_of_text.get(text_key, 0), rank)

    # Count tokens once per distinct text of the kept languages.
    kept_rows = [raw for raw in raw_rows if raw.language in keep_languages]
    lengths = count_unique_tokens((raw.row.prompt for raw in kept_rows), count_tokens)

    # Steps 1-3: group each split's usable rows by (language, prompt), collecting every
    # label seen, the union of categories, and the leakage groups of all repeats.
    groups: dict[str, dict[tuple[str, str], _NemotronGroup]] = {
        split: {} for split in NEMOTRON_SPLIT_FILES
    }
    for raw in kept_rows:
        counter_key = (raw.split, raw.language)
        report.raw[counter_key] += 1
        if not raw.row.prompt.strip() or raw.row.prompt.strip().upper() == REDACTED_PROMPT:
            report.redacted[counter_key] += 1
            continue
        if lengths[raw.row.prompt] > max_length:
            report.too_long[counter_key] += 1
            continue
        text_key = (raw.language, raw.row.prompt)
        group = groups[raw.split].get(text_key)
        if group is None:
            group = groups[raw.split][text_key] = _NemotronGroup(raw.row)
        else:
            report.merged[counter_key] += 1
        group.add(raw.row, source_of_id.get(raw.row.id, raw.row.id))

    # Steps 4-5: drop label conflicts and prompts that reappear in a later split.
    splits: dict[str, list[SafetyRecord]] = {split: [] for split in NEMOTRON_SPLIT_FILES}
    for split, split_groups in groups.items():
        rank = split_rank[split]
        for (language, prompt), group in split_groups.items():
            counter_key = (split, language)
            if len(group.labels) > 1:
                report.conflicts[counter_key] += 1
                continue
            leaked = last_rank_of_text[(language, prompt)] > rank or any(
                last_rank_of_source[source] > rank for source in group.sources
            )
            if leaked:
                report.leaked[counter_key] += 1
                continue

            # Every v3 category and the caution label are known targets.
            report.kept[counter_key] += 1
            splits[split].append(
                SafetyRecord(
                    id=group.first.id,
                    source=Source.NEMOTRON_V3,
                    language=language,
                    tag=group.first.tag,
                    prompt=prompt,
                    prompt_label=group.first.prompt_label,
                    violated_categories=tuple(group.categories),
                    caution_known=True,
                    category_coverage=CategoryCoverage.ALL,
                    num_tokens=lengths[prompt],
                )
            )
    english_source = {record_id: normalize_prompt(text) for record_id, text in source_of_id.items()}
    return NemotronSplits(splits, english_source)


def map_mlcommons_codes(raw: str | None) -> tuple[tuple[str, ...], CategoryCoverage]:
    """
    Map PolyGuardMix hazard codes to v3 categories.

    Only the seven codes with a one-to-one v3 counterpart map. When every code maps, the
    mapped categories are complete over :data:`MAPPED_CATEGORIES`; otherwise some of the
    prompt's harm is unknown and no category target is used.

    :param raw:     Space-separated codes such as ``"S1 S10"``; ``None`` or ``"None"``.
    :return:        Mapped v3 categories and their coverage.
    """
    codes = {code for code in (raw or "").split() if code in MLCOMMONS_CODES}
    if not codes or not codes <= MLCOMMONS_TO_NEMOTRON.keys():
        return (), CategoryCoverage.NONE
    return tuple(sorted(MLCOMMONS_TO_NEMOTRON[code] for code in codes)), CategoryCoverage.MAPPED


def stable_id(prefix: str, *parts: str) -> str:
    """
    Build a deterministic id from text parts.

    :param prefix:  Short source prefix.
    :param parts:   Strings identifying the record.
    :return:        ``"{prefix}-{16 hex digits}"``.
    """
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8"), usedforsecurity=False)
    return f"{prefix}-{digest.hexdigest()[:16]}"


@dataclass(slots=True)
class _MixGroup:
    """
    Repeats of one ``(language, prompt)`` pair in PolyGuardMix.

    :param source:  Sub-source of the first repeat (e.g. ``"itw"``).
    :param labels:  Every harm label seen.
    :param codes:   Hazard-code strings of every repeat.
    """

    source: str
    labels: set[str] = field(default_factory=set[str])
    codes: list[str] = field(default_factory=list[str])


@dataclass(frozen=True, slots=True)
class PolyGuardMixRecords:
    """
    Cleaned PolyGuardMix records.

    :param records:         One record per ``(language, prompt)``.
    :param english_keys:    Normalized text of every English prompt read, before any
                            filtering, for the v3 leakage rule.
    """

    records: list[SafetyRecord]
    english_keys: frozenset[str]


def clean_polyguardmix(
    rows: Iterable[tuple[str, PolyGuardMixRow]],
    languages: Iterable[str],
    count_tokens: TokenCounter,
    max_length: int,
    report: PolyGuardCleaningReport,
) -> PolyGuardMixRecords:
    """
    Clean PolyGuardMix into one record per ``(language, prompt)``.

    1. Drop rows without a prompt or a harm label.
    2. Merge repeats of a ``(language, prompt)`` pair (the same WildGuardMix prompt
       translated by two systems often coincides); a pair whose repeats disagree on the
       label is dropped, and hazard codes are unioned.
    3. Drop prompts longer than ``max_length`` tokens.

    The caution label is unknown for every record; categories follow
    :func:`map_mlcommons_codes`.

    :param rows:            ``(language code, row)`` pairs from :func:`read_polyguardmix`.
    :param languages:       ISO codes to keep in the output.
    :param count_tokens:    Maps a batch of texts to their token counts.
    :param max_length:      Longest prompt kept, in tokens.
    :param report:          Report updated in place.
    :return:                The records and the English keys.
    """
    keep_languages = set(languages)
    english_keys: set[str] = set()

    # Steps 1 and 2: group by exact text per language. Only the labels, codes and first
    # sub-source are kept per group, since about two million rows pass through here.
    groups: dict[tuple[str, str], _MixGroup] = {}
    for language, row in rows:
        if language == SOURCE_LANGUAGE and row.prompt:
            english_keys.add(normalize_prompt(row.prompt))
        if language not in keep_languages:
            continue
        report.raw[language] += 1
        if not row.prompt or not row.prompt.strip() or row.prompt_harm_label is None:
            report.missing[language] += 1
            continue
        group = groups.get((language, row.prompt))
        if group is None:
            group = groups[(language, row.prompt)] = _MixGroup(row.metadata.source)
        else:
            report.merged[language] += 1
        group.labels.add(row.prompt_harm_label)
        group.codes.append(row.prompt_safety_categories or "")

    # Drop label conflicts; agreeing repeats keep the union of their hazard codes.
    for (language, prompt), group in list(groups.items()):
        if len(group.labels) > 1:
            report.conflicts[language] += 1
            del groups[(language, prompt)]

    # Step 3: length filter, then conversion to the shared schema.
    lengths = count_unique_tokens((prompt for _, prompt in groups), count_tokens)
    records: list[SafetyRecord] = []
    for (language, prompt), group in groups.items():
        if lengths[prompt] > max_length:
            report.too_long[language] += 1
            continue
        unsafe = "yes" in group.labels
        categories, coverage = (
            map_mlcommons_codes(" ".join(group.codes)) if unsafe else ((), CategoryCoverage.NONE)
        )
        report.kept[language] += 1
        records.append(
            SafetyRecord(
                id=stable_id("pgm", language, prompt),
                source=Source.POLYGUARDMIX,
                language=language,
                tag=group.source,
                prompt=prompt,
                prompt_label=Label.UNSAFE if unsafe else Label.SAFE,
                violated_categories=categories,
                caution_known=False,
                category_coverage=coverage,
                num_tokens=lengths[prompt],
            )
        )
    return PolyGuardMixRecords(records, frozenset(english_keys))


def clean_polyguardprompts(
    rows: Iterable[tuple[str, PolyGuardPromptsRow]],
    count_tokens: TokenCounter,
    max_length: int,
    report: PolyGuardCleaningReport,
) -> list[SafetyRecord]:
    """
    Clean PolyGuardPrompts into one record per ``(language, prompt)``.

    Rows without a human harm label are dropped, repeats of a ``(language, prompt)`` pair
    keep the first id (a pair whose repeats disagree is dropped), and prompts longer than
    ``max_length`` tokens are dropped. The set is used for prompt-level evaluation only,
    so neither caution nor category targets are known.

    :param rows:            ``(language code, row)`` pairs from :func:`read_polyguardprompts`.
    :param count_tokens:    Maps a batch of texts to their token counts.
    :param max_length:      Longest prompt kept, in tokens.
    :param report:          Report updated in place.
    :return:                The records.
    """
    # Group usable rows by exact text per language.
    groups: dict[tuple[str, str], list[PolyGuardPromptsRow]] = defaultdict(list)
    for language, row in rows:
        report.raw[language] += 1
        if not row.prompt or not row.prompt.strip() or row.prompt_harm_label is None:
            report.missing[language] += 1
            continue
        groups[(language, row.prompt)].append(row)

    # Resolve repeats, filter by length, and convert to the shared schema.
    lengths = count_unique_tokens((prompt for _, prompt in groups), count_tokens)
    records: list[SafetyRecord] = []
    for (language, prompt), group in groups.items():
        report.merged[language] += len(group) - 1
        if len({row.prompt_harm_label for row in group}) > 1:
            report.conflicts[language] += 1
            continue
        if lengths[prompt] > max_length:
            report.too_long[language] += 1
            continue
        row = group[0]
        report.kept[language] += 1
        records.append(
            SafetyRecord(
                id=f"pgp-{row.id}",
                source=Source.POLYGUARDPROMPTS,
                language=language,
                tag="adversarial" if row.adversarial else "vanilla",
                prompt=prompt,
                prompt_label=Label.UNSAFE if row.prompt_harm_label == "harmful" else Label.SAFE,
                violated_categories=(),
                caution_known=False,
                category_coverage=CategoryCoverage.NONE,
                num_tokens=lengths[prompt],
            )
        )
    return records


# ---------------------------------------------------------------------------
# Cross-dataset merge.
# ---------------------------------------------------------------------------


def record_key(record: SafetyRecord) -> tuple[str, str]:
    """
    Return the ``(language, normalized prompt)`` key used to match records across datasets.

    :param record:  Any record.
    :return:        The match key.
    """
    return record.language, normalize_prompt(record.prompt)


def merge_sources(
    nemotron: NemotronSplits,
    polyguardmix: PolyGuardMixRecords,
    polyguardprompts: Sequence[SafetyRecord],
    train_tags: Iterable[str],
    jailbreak_tags: Iterable[str],
    report: MergeReport,
) -> dict[SplitName, list[SafetyRecord]]:
    """
    Merge the cleaned sources into the final splits.

    1. Drop PolyGuardMix prompts that match any v3 prompt in the same language; v3's
       human label wins over the LLM label.
    2. Drop v3 validation and test ids whose English source appears in PolyGuardMix,
       since PolyGuardMix keeps translations of that prompt in training.
    3. Route v3 records by tag: ``train_tags`` form the training, validation and test
       splits; ``jailbreak_tags`` in the v3 test split form the jailbreak test.
    4. Drop PolyGuardPrompts ids with any translation that matches a training prompt.
    5. Sort every evaluation split by token count, so length-bucketed evaluation batches
       stay full; the training split is shuffled by the sampler instead.

    :param nemotron:            Output of :func:`clean_nemotron`.
    :param polyguardmix:        Output of :func:`clean_polyguardmix`.
    :param polyguardprompts:    Output of :func:`clean_polyguardprompts`.
    :param train_tags:          v3 tags used for training and in-distribution evaluation.
    :param jailbreak_tags:      v3 tags held out as the jailbreak test.
    :param report:              Report updated in place.
    :return:                    Records per split.
    """
    train_tag_set, jailbreak_tag_set = set(train_tags), set(jailbreak_tags)

    # Step 1: PolyGuardMix records whose text already exists in v3.
    v3_keys = {record_key(record) for records in nemotron.splits.values() for record in records}
    polyguardmix_kept = [r for r in polyguardmix.records if record_key(r) not in v3_keys]
    report.polyguardmix_in_v3 = len(polyguardmix.records) - len(polyguardmix_kept)

    # Steps 2 and 3: filter and route the v3 evaluation splits.
    def is_contaminated(record: SafetyRecord) -> bool:
        source = nemotron.english_source.get(record.id)
        return source is not None and source in polyguardmix.english_keys

    splits: dict[SplitName, list[SafetyRecord]] = {name: [] for name in SplitName}
    splits[SplitName.TRAIN] = [
        record for record in nemotron.splits["train"] if record.tag in train_tag_set
    ] + polyguardmix_kept
    for v3_split in ("validation", "test"):
        for record in nemotron.splits[v3_split]:
            if is_contaminated(record):
                report.v3_eval_in_polyguardmix[v3_split] += 1
            elif record.tag in train_tag_set:
                name = SplitName.VALIDATION if v3_split == "validation" else SplitName.TEST
                splits[name].append(record)
            elif v3_split == "test" and record.tag in jailbreak_tag_set:
                splits[SplitName.JAILBREAK_TEST].append(record)

    # Step 4: PolyGuardPrompts ids with any translation inside the training split.
    train_keys = {record_key(record) for record in splits[SplitName.TRAIN]}
    leaked_ids = {record.id for record in polyguardprompts if record_key(record) in train_keys}
    splits[SplitName.POLYGUARDPROMPTS_TEST] = [
        record for record in polyguardprompts if record.id not in leaked_ids
    ]
    report.polyguardprompts_leaked = len(polyguardprompts) - len(
        splits[SplitName.POLYGUARDPROMPTS_TEST]
    )

    # Step 5: evaluation splits in ascending length; sorted() is stable, so ties keep order.
    for name in SplitName:
        if name is not SplitName.TRAIN:
            splits[name] = sorted(splits[name], key=lambda record: record.num_tokens)
        report.split_sizes[name.value] = len(splits[name])
    return splits


# ---------------------------------------------------------------------------
# Hugging Face datasets, caching, and the top-level builder.
# ---------------------------------------------------------------------------


def records_to_dataset(records: Sequence[SafetyRecord]) -> Dataset:
    """
    Convert records to a Hugging Face dataset.

    ``violated_categories`` is stored comma-separated, as in v3, so :func:`parse_categories`
    reads it back.

    :param records: Records of one split.
    :return:        A dataset with one column per :class:`SafetyRecord` field.
    """
    return Dataset.from_dict(  # pyright: ignore[reportUnknownMemberType]
        {
            "id": [r.id for r in records],
            "source": [r.source.value for r in records],
            "language": [r.language for r in records],
            "tag": [r.tag for r in records],
            "prompt": [r.prompt for r in records],
            "prompt_label": [r.prompt_label.value for r in records],
            "violated_categories": [", ".join(r.violated_categories) for r in records],
            "caution_known": [r.caution_known for r in records],
            "category_coverage": [r.category_coverage.value for r in records],
            "num_tokens": [r.num_tokens for r in records],
        }
    )


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """
    Settings that determine the merged splits; stored in exports so TFB rebuilds them.

    :param tokenizer_name:  Tokenizer used to count tokens.
    :param max_length:      Longest prompt kept, in tokens.
    :param languages:       ISO codes included.
    :param train_tags:      v3 tags used for training and in-distribution evaluation.
    :param jailbreak_tags:  v3 tags held out as the jailbreak test.
    """

    tokenizer_name: str
    max_length: int
    languages: tuple[str, ...] = ALL_LANGUAGES
    train_tags: tuple[str, ...] = HUMAN_TAGS
    jailbreak_tags: tuple[str, ...] = JAILBREAK_TAGS

    def to_json(self) -> dict[str, str | int | list[str]]:
        """
        :return:    A JSON-serializable dictionary.
        """
        return {
            "tokenizer_name": self.tokenizer_name,
            "max_length": self.max_length,
            "languages": list(self.languages),
            "train_tags": list(self.train_tags),
            "jailbreak_tags": list(self.jailbreak_tags),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> DatasetSpec:
        """
        :param data:    Output of :meth:`to_json`.
        :return:        The spec.
        """
        return cls(
            tokenizer_name=str(data["tokenizer_name"]),
            max_length=int(cast(int, data["max_length"])),
            languages=tuple(cast(list[str], data["languages"])),
            train_tags=tuple(cast(list[str], data["train_tags"])),
            jailbreak_tags=tuple(cast(list[str], data["jailbreak_tags"])),
        )


def cache_key(spec: DatasetSpec, revisions: Sequence[str]) -> str:
    """
    Hash everything that changes the merged splits.

    :param spec:        Dataset settings.
    :param revisions:   Snapshot revisions of the source datasets.
    :return:            A 16-character hex digest.
    """
    payload = {"spec": spec.to_json(), "revisions": list(revisions), "version": CLEANING_VERSION}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def format_report(report: CleaningReport) -> str:
    """
    Render a cleaning report as plain-text tables.

    :param report:  Report from :func:`build_safety_splits`.
    :return:        Multi-line text.
    """
    lines: list[str] = []

    # One table for v3, one row per (split, language) pair.
    nemotron = report.nemotron
    lines.append("Nemotron v3 cleaning")
    lines.append(
        f"{'split':<12}{'lang':<6}{'raw':>9}{'redacted':>9}{'merged':>8}{'conflict':>9}"
        f"{'too long':>9}{'leaked':>8}{'kept':>9}"
    )
    for split, language in sorted(
        nemotron.raw, key=lambda key: (list(NEMOTRON_SPLIT_FILES).index(key[0]), key[1])
    ):
        key = (split, language)
        lines.append(
            f"{split:<12}{language:<6}{nemotron.raw[key]:>9}{nemotron.redacted[key]:>9}"
            f"{nemotron.merged[key]:>8}{nemotron.conflicts[key]:>9}{nemotron.too_long[key]:>9}"
            f"{nemotron.leaked[key]:>8}{nemotron.kept[key]:>9}"
        )

    # PolyGuardMix and PolyGuardPrompts share a per-language layout.
    for title, table in (
        ("PolyGuardMix cleaning", report.polyguardmix),
        ("PolyGuardPrompts cleaning", report.polyguardprompts),
    ):
        lines.append("")
        lines.append(title)
        lines.append(
            f"{'lang':<6}{'raw':>9}{'missing':>9}{'merged':>9}{'conflict':>9}{'too long':>9}{'kept':>9}"
        )
        for language in sorted(table.raw):
            lines.append(
                f"{language:<6}{table.raw[language]:>9}{table.missing[language]:>9}"
                f"{table.merged[language]:>9}{table.conflicts[language]:>9}"
                f"{table.too_long[language]:>9}{table.kept[language]:>9}"
            )

    # Cross-dataset rules and the final sizes.
    merge = report.merge
    lines.append("")
    lines.append(
        f"PolyGuardMix prompts dropped as copies of v3 prompts: {merge.polyguardmix_in_v3}"
    )
    lines.append(
        "v3 evaluation records dropped (English source in PolyGuardMix): "
        f"{dict(merge.v3_eval_in_polyguardmix)}"
    )
    lines.append(
        f"PolyGuardPrompts records dropped (translation in training): {merge.polyguardprompts_leaked}"
    )
    lines.append(f"Split sizes: {merge.split_sizes}")
    return "\n".join(lines)


def build_safety_splits(
    spec: DatasetSpec,
    count_tokens: TokenCounter,
    cache_dir: Path,
) -> tuple[dict[SplitName, Dataset], str]:
    """
    Build (or load from cache) the merged splits described by ``spec``.

    The source snapshots are fetched from the Hugging Face cache (downloading them on
    first use), cleaned, merged, and saved under ``cache_dir`` keyed by
    :func:`cache_key`. Later calls with the same spec and snapshots load the saved
    splits, which keeps the training and TFB notebooks on identical data.

    :param spec:            Dataset settings.
    :param count_tokens:    Maps a batch of texts to token counts with ``spec.tokenizer_name``.
    :param cache_dir:       Directory holding cached builds.
    :return:                Datasets per split and the cleaning report text.
    """
    # Resolve the snapshots first; their revision hashes are part of the cache key. Only
    # the data files are requested, so unrelated repository files are never downloaded.
    read_languages = sorted({*spec.languages, SOURCE_LANGUAGE})
    v3_patterns = [f"{code}/*.jsonl" for code in read_languages if code in NEMOTRON_LANGUAGES]
    v3_root = Path(
        snapshot_download(NEMOTRON_V3_ID, repo_type="dataset", allow_patterns=v3_patterns)
    )
    pgm_root = Path(
        snapshot_download(POLYGUARDMIX_ID, repo_type="dataset", allow_patterns=["*.parquet"])
    )
    pgp_root = Path(
        snapshot_download(POLYGUARDPROMPTS_ID, repo_type="dataset", allow_patterns=["*.parquet"])
    )
    target = cache_dir / cache_key(spec, [v3_root.name, pgm_root.name, pgp_root.name])
    report_path = target / "report.txt"
    if report_path.exists():
        datasets = {name: cast(Dataset, load_from_disk(target / name.value)) for name in SplitName}
        return datasets, report_path.read_text(encoding="utf-8")

    # Clean each source; English is always read because the leakage rules key on it.
    report = CleaningReport(
        NemotronCleaningReport(),
        PolyGuardCleaningReport(),
        PolyGuardCleaningReport(),
        MergeReport(),
    )
    nemotron = clean_nemotron(
        read_nemotron_v3(v3_root, [code for code in read_languages if code in NEMOTRON_LANGUAGES]),
        spec.languages,
        count_tokens,
        spec.max_length,
        report.nemotron,
    )
    polyguardmix = clean_polyguardmix(
        read_polyguardmix(sorted(pgm_root.rglob("*.parquet")), read_languages),
        spec.languages,
        count_tokens,
        spec.max_length,
        report.polyguardmix,
    )
    polyguardprompts = clean_polyguardprompts(
        read_polyguardprompts(sorted(pgp_root.rglob("*.parquet")), spec.languages),
        count_tokens,
        spec.max_length,
        report.polyguardprompts,
    )

    # Merge, then persist each split; the report is written last and marks a complete build.
    splits = merge_sources(
        nemotron, polyguardmix, polyguardprompts, spec.train_tags, spec.jailbreak_tags, report.merge
    )
    empty = [name.value for name, records in splits.items() if not records]
    if empty:
        raise ValueError(f"Merged splits {empty} are empty; check the languages and max_length")
    datasets = {name: records_to_dataset(records) for name, records in splits.items()}
    for name, dataset in datasets.items():
        dataset.save_to_disk(str(target / name.value))  # pyright: ignore[reportUnknownMemberType]
    text = format_report(report)
    (target / "spec.json").write_text(json.dumps(spec.to_json(), indent=2), encoding="utf-8")
    report_path.write_text(text, encoding="utf-8")
    return datasets, text


# ---------------------------------------------------------------------------
# Subsetting helpers used by the notebooks.
# ---------------------------------------------------------------------------


def take_subset(dataset: Dataset, size: int | None, seed: int) -> Dataset:
    """
    Keep a seeded random sample of ``size`` records, or every record for ``None``.

    The sample keeps the records' original relative order, so length-sorted evaluation
    splits stay sorted.

    :param dataset: A merged split.
    :param size:    Number of records to keep; ``None`` keeps every record.
    :param seed:    Seed of the sample.
    :return:        The sampled records in their original relative order.
    """
    if size is None:
        return dataset
    if size > len(dataset):
        print(f"Warning: only {len(dataset)} cleaned records available; requested {size}.")
    order = np.random.default_rng(seed).permutation(len(dataset))[: min(size, len(dataset))]
    return dataset.select(sorted(order.tolist()))  # pyright: ignore[reportUnknownMemberType]


def stratified_indices(dataset: Dataset, per_language: int, seed: int) -> list[int]:
    """
    Seeded sample of ``per_language`` records from every language of a split.

    Equal counts per language keep every language equally represented in the
    analyses that run on a subset, whatever the language mix of the full split.

    :param dataset:         A merged split with a ``language`` column.
    :param per_language:    Records drawn from each language (all of them if fewer).
    :param seed:            Seed of the sample.
    :return:                Sorted indices into ``dataset``.
    """
    rng = np.random.default_rng(seed)
    languages = np.asarray(column(dataset, "language"))
    chosen: list[int] = []
    for language in sorted(set(languages.tolist())):
        candidates = np.flatnonzero(languages == language)
        size = min(per_language, len(candidates))
        chosen.extend(int(index) for index in rng.choice(candidates, size=size, replace=False))
    return sorted(chosen)


def column(dataset: Dataset, name: str) -> list[str]:
    """
    Read a string column of a dataset as a list.

    :param dataset: A merged split.
    :param name:    Column name.
    :return:        The column values in row order.
    """
    return [str(value) for value in cast(Iterable[object], dataset[name])]
