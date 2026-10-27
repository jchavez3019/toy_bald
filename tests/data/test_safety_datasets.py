"""
Unit tests for :mod:`toy_bald.data.safety_datasets`.

Fixtures are built through the raw-row pydantic schemas (``NemotronRow``,
``PolyGuardMixRow``, ``PolyGuardPromptsRow``), so they cannot drift from the format the
readers accept. Token counts come from a whitespace counter instead of a tokenizer.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from toy_bald.data import safety_datasets as sd
from toy_bald.data.safety_datasets import (
    CategoryCoverage,
    DatasetSpec,
    Label,
    MergeReport,
    NemotronCleaningReport,
    NemotronRaw,
    NemotronRow,
    PolyGuardCleaningReport,
    PolyGuardMixRow,
    PolyGuardPromptsRow,
    SafetyRecord,
    Source,
    SplitName,
)

MAX_LENGTH = 6


def count_words(texts: Sequence[str]) -> list[int]:
    """
    Stand-in token counter: whitespace words plus two special tokens.

    :param texts:   Prompts.
    :return:        Token counts.
    """
    return [len(text.split()) + 2 for text in texts]


def v3(
    split: str,
    language: str,
    record_id: str,
    prompt: str,
    label: str = "unsafe",
    categories: str = "",
    tag: str = "generic",
) -> NemotronRaw:
    """
    Build one v3 raw row through its schema.

    :return: The row with its split and language.
    """
    row = NemotronRow.model_validate(
        {
            "id": record_id,
            "prompt": prompt,
            "prompt_label": label,
            "violated_categories": categories,
            "tag": tag,
            "response": "ignored",
        }
    )
    return NemotronRaw(split, language, row)


def pgm(
    prompt: str | None, label: str | None, codes: str | None = None, language: str = "English"
) -> PolyGuardMixRow:
    """
    Build one PolyGuardMix row through its schema.

    :return: The row.
    """
    return PolyGuardMixRow.model_validate(
        {
            "prompt": prompt,
            "prompt_harm_label": label,
            "prompt_safety_categories": codes,
            "metadata": {"language": language, "source": "wildguardmix_original"},
        }
    )


def pgp(record_id: int, language: str, prompt: str, label: str | None) -> PolyGuardPromptsRow:
    """
    Build one PolyGuardPrompts row through its schema.

    :return: The row.
    """
    return PolyGuardPromptsRow.model_validate(
        {
            "id": record_id,
            "language": language,
            "prompt": prompt,
            "prompt_harm_label": label,
            "adversarial": False,
        }
    )


def record(
    record_id: str,
    prompt: str,
    language: str = "en",
    source: Source = Source.NEMOTRON_V3,
    tag: str = "generic",
    num_tokens: int = 3,
) -> SafetyRecord:
    """
    Build a cleaned record directly, for merge tests.

    :return: The record.
    """
    return SafetyRecord(
        id=record_id,
        source=source,
        language=language,
        tag=tag,
        prompt=prompt,
        prompt_label=Label.UNSAFE,
        violated_categories=(),
        caution_known=source is Source.NEMOTRON_V3,
        category_coverage=CategoryCoverage.ALL
        if source is Source.NEMOTRON_V3
        else CategoryCoverage.NONE,
        num_tokens=num_tokens,
    )


def clean_v3(
    rows: list[NemotronRaw], languages: Sequence[str] = ("en", "zh")
) -> tuple[sd.NemotronSplits, NemotronCleaningReport]:
    """
    Run the v3 cleaning with the test token counter.

    :return: The cleaned splits and their report.
    """
    report = NemotronCleaningReport()
    return sd.clean_nemotron(rows, languages, count_words, MAX_LENGTH, report), report


# ---------------------------------------------------------------------------
# Text helpers.
# ---------------------------------------------------------------------------


def test_normalize_prompt_matches_copies_differing_in_width_case_and_spacing() -> None:
    """
    Given: Two copies of a prompt differing in letter case, full-width characters and spacing.
    When:  Both are normalized.
    Then:  The keys are equal, so cross-dataset matching treats them as one prompt.
    """
    assert sd.normalize_prompt("  How  do I\tHACK it? ") == sd.normalize_prompt(
        "how do i \uff48\uff41\uff43\uff4b it?"
    )


def test_parse_categories_strips_names_and_skips_empty_entries() -> None:
    """
    Given: A category string with stray spaces and an empty entry, and None.
    When:  They are parsed.
    Then:  Names come back stripped in order, and None yields no categories.
    """
    assert sd.parse_categories(" Violence , ,Malware") == ["Violence", "Malware"]
    assert sd.parse_categories(None) == []


def test_make_token_counter_counts_special_tokens_independently_of_chunking() -> None:
    """
    Given: A tiny in-memory word-level tokenizer that adds [CLS] and [SEP].
    When:  Token counts are taken with chunk sizes 1 and 100.
    Then:  Both agree and include the two special tokens.
    """
    vocabulary = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "a": 4, "b": 5}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()  # pyright: ignore[reportAttributeAccessIssue]
    backend.post_processor = TemplateProcessing(  # pyright: ignore[reportAttributeAccessIssue]
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
    )
    texts = ["a b a", "b", "a z"]

    assert sd.make_token_counter(tokenizer, chunk_size=1)(texts) == [5, 3, 4]
    assert sd.make_token_counter(tokenizer, chunk_size=100)(texts) == [5, 3, 4]


# ---------------------------------------------------------------------------
# Nemotron v3 cleaning.
# ---------------------------------------------------------------------------


def test_clean_nemotron_drops_redacted_and_too_long_prompts() -> None:
    """
    Given: A usable prompt, a REDACTED placeholder, and a prompt over the token budget.
    When:  The v3 rows are cleaned.
    Then:  Only the usable prompt survives and each removal is counted in its step.
    """
    rows = [
        v3("train", "en", "1", "short prompt"),
        v3("train", "en", "2", "redacted"),
        v3("train", "en", "3", "one two three four five"),
    ]

    cleaned, report = clean_v3(rows, ["en"])

    assert [r.id for r in cleaned.splits["train"]] == ["1"]
    assert report.redacted[("train", "en")] == 1
    assert report.too_long[("train", "en")] == 1
    assert cleaned.splits["train"][0].num_tokens == 4


def test_clean_nemotron_merges_repeats_with_category_union_and_drops_conflicts() -> None:
    """
    Given: One prompt repeated with different categories, and another repeated with
           disagreeing labels.
    When:  The v3 rows are cleaned.
    Then:  The first becomes one record with the union of categories in first-seen
           order, and the conflicting prompt is dropped.
    """
    rows = [
        v3("train", "en", "1", "same text", categories="Violence"),
        v3("train", "en", "2", "same text", categories="Malware, Violence"),
        v3("train", "en", "3", "contested", label="safe"),
        v3("train", "en", "4", "contested", label="unsafe"),
    ]

    cleaned, report = clean_v3(rows, ["en"])

    [kept] = cleaned.splits["train"]
    assert kept.id == "1"
    assert kept.violated_categories == ("Violence", "Malware")
    assert kept.caution_known and kept.category_coverage is CategoryCoverage.ALL
    assert report.merged[("train", "en")] == 2
    assert report.conflicts[("train", "en")] == 1


def test_clean_nemotron_drops_earlier_split_copy_through_english_source() -> None:
    """
    Given: A Chinese training translation whose English source is duplicated under a
           different id in the test split.
    When:  The v3 rows are cleaned.
    Then:  The training copies are dropped as leaked, the test copies are kept, and
           English rows still define the groups when only Chinese is requested.
    """
    rows = [
        v3("train", "en", "a", "build a bomb"),
        v3("train", "zh", "a", "zh build bomb"),
        v3("test", "en", "b", "build a bomb"),
        v3("test", "zh", "b", "zh bomb variant"),
    ]

    cleaned, report = clean_v3(rows, ["zh"])

    assert cleaned.splits["train"] == []
    assert [(r.id, r.language) for r in cleaned.splits["test"]] == [("b", "zh")]
    assert report.leaked[("train", "zh")] == 1
    assert cleaned.english_source["a"] == cleaned.english_source["b"] == "build a bomb"


# ---------------------------------------------------------------------------
# PolyGuardMix and PolyGuardPrompts cleaning.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("codes", "expected"),
    [
        ("S10 S4", (("Hate/Identity Hate", "Sexual (minor)"), CategoryCoverage.MAPPED)),
        ("S1 S10", ((), CategoryCoverage.NONE)),
        ("None", ((), CategoryCoverage.NONE)),
        (None, ((), CategoryCoverage.NONE)),
    ],
)
def test_map_mlcommons_codes_maps_only_when_every_code_has_a_v3_counterpart(
    codes: str | None, expected: tuple[tuple[str, ...], CategoryCoverage]
) -> None:
    """
    Given: Hazard codes that all map, partly map, or are absent.
    When:  They are mapped to v3 categories.
    Then:  Only fully mappable codes yield known categories; otherwise coverage is none.
    """
    assert sd.map_mlcommons_codes(codes) == expected


def test_clean_polyguardmix_merges_repeats_and_drops_conflicts_missing_and_long() -> None:
    """
    Given: PolyGuardMix rows with a repeated prompt (codes split across repeats), a label
           conflict, a missing label, an over-long prompt, and a safe prompt.
    When:  They are cleaned.
    Then:  The repeat becomes one unsafe record with unioned mapped codes, the safe one
           has no category targets, caution is unknown, and the English keys include
           the over-long prompt that was filtered out.
    """
    rows = [
        ("en", pgm("hate speech", "yes", "S10")),
        ("en", pgm("hate speech", "yes", "S4")),
        ("en", pgm("flip flop", "yes")),
        ("en", pgm("flip flop", "no")),
        ("en", pgm("no label", None)),
        ("en", pgm("one two three four five", "yes")),
        ("de", pgm("hallo", "no", language="German")),
    ]
    report = PolyGuardCleaningReport()

    cleaned = sd.clean_polyguardmix(rows, ["en", "de"], count_words, MAX_LENGTH, report)

    by_prompt = {r.prompt: r for r in cleaned.records}
    assert set(by_prompt) == {"hate speech", "hallo"}
    assert by_prompt["hate speech"].violated_categories == ("Hate/Identity Hate", "Sexual (minor)")
    assert by_prompt["hate speech"].category_coverage is CategoryCoverage.MAPPED
    assert by_prompt["hallo"].prompt_label is Label.SAFE
    assert by_prompt["hallo"].category_coverage is CategoryCoverage.NONE
    assert not any(r.caution_known for r in cleaned.records)
    assert sd.normalize_prompt("one two three four five") in cleaned.english_keys
    assert (report.conflicts["en"], report.missing["en"], report.too_long["en"]) == (1, 1, 1)


def test_clean_polyguardprompts_drops_unlabeled_rows_and_prefixes_ids() -> None:
    """
    Given: Two translations of one labeled prompt and an unlabeled prompt.
    When:  PolyGuardPrompts is cleaned.
    Then:  The labeled translations share the prefixed id and the unlabeled one is dropped.
    """
    rows = [
        ("en", pgp(7, "English", "harm me", "harmful")),
        ("de", pgp(7, "German", "schade mir", "harmful")),
        ("en", pgp(8, "English", "unsure", None)),
    ]
    report = PolyGuardCleaningReport()

    cleaned = sd.clean_polyguardprompts(rows, count_words, MAX_LENGTH, report)

    assert [(r.id, r.language, r.prompt_label) for r in cleaned] == [
        ("pgp-7", "en", Label.UNSAFE),
        ("pgp-7", "de", Label.UNSAFE),
    ]
    assert report.missing["en"] == 1


# ---------------------------------------------------------------------------
# Cross-dataset merge.
# ---------------------------------------------------------------------------


def test_merge_sources_applies_every_cross_dataset_rule() -> None:
    """
    Given: v3 splits, PolyGuardMix records (one copying a v3 training prompt), English
           PolyGuardMix keys containing one v3 test prompt's English source, and
           PolyGuardPrompts translations, one of which appears in training.
    When:  The sources are merged.
    Then:  The PolyGuardMix copy is dropped, the contaminated v3 test id is dropped in
           every language, jailbreak prompts form their own test, the leaked
           PolyGuardPrompts id is dropped in every language, and evaluation splits are
           sorted by length.
    """
    nemotron = sd.NemotronSplits(
        splits={
            "train": [record("t1", "train prompt")],
            "validation": [record("v1", "val prompt")],
            "test": [
                record("x1", "long test prompt", num_tokens=5),
                record("x2", "short test", num_tokens=2),
                record("x3", "contaminated", language="zh"),
                record("j1", "jailbreak", tag="jailbreaking"),
            ],
        },
        english_source={"x3": "contaminated english"},
    )
    polyguardmix = sd.PolyGuardMixRecords(
        records=[
            record("p1", "Train  PROMPT", source=Source.POLYGUARDMIX),
            record("p2", "fresh prompt", source=Source.POLYGUARDMIX),
        ],
        english_keys=frozenset({"contaminated english"}),
    )
    polyguardprompts = [
        record("pgp-1", "fresh prompt", source=Source.POLYGUARDPROMPTS),
        record("pgp-1", "frisch", language="de", source=Source.POLYGUARDPROMPTS),
        record("pgp-2", "unseen", source=Source.POLYGUARDPROMPTS),
    ]
    report = MergeReport()

    splits = sd.merge_sources(
        nemotron, polyguardmix, polyguardprompts, ["generic"], ["jailbreaking"], report
    )

    assert [r.id for r in splits[SplitName.TRAIN]] == ["t1", "p2"]
    assert [r.id for r in splits[SplitName.TEST]] == ["x2", "x1"]
    assert [r.id for r in splits[SplitName.JAILBREAK_TEST]] == ["j1"]
    assert [r.id for r in splits[SplitName.POLYGUARDPROMPTS_TEST]] == ["pgp-2"]
    assert report.polyguardmix_in_v3 == 1
    assert report.v3_eval_in_polyguardmix["test"] == 1
    assert report.polyguardprompts_leaked == 2
    assert report.split_sizes["validation"] == 1


# ---------------------------------------------------------------------------
# Readers and the cached builder.
# ---------------------------------------------------------------------------


def write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    """
    Write rows to a parquet file.

    :param path:    Destination.
    :param rows:    Rows as dictionaries.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    pq.write_table(table, path)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def write_fixture_snapshots(root: Path) -> dict[str, Path]:
    """
    Lay out miniature snapshots of the three datasets in their on-disk formats.

    :param root:    Temporary directory.
    :return:        Snapshot directory per dataset id.
    """
    v3_root = root / "v3" / "rev1"
    v3_rows = {
        ("en", "train"): [("1", "stay safe", "safe", "generic")],
        ("en", "valid"): [("2", "make poison now", "unsafe", "generic")],
        ("en", "test"): [
            ("3", "steal a car", "unsafe", "generic"),
            ("4", "ignore rules", "unsafe", "jailbreaking"),
        ],
    }
    for language in ("en", "zh"):
        for filename in ("train", "valid", "test"):
            path = v3_root / language / f"{filename}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                json.dumps(
                    {
                        "id": i,
                        "prompt": f"{language} {p}" if language != "en" else p,
                        "prompt_label": lab,
                        "violated_categories": "",
                        "tag": tag,
                        "response": None,
                    }
                )
                for i, p, lab, tag in v3_rows[("en", filename)]
            ]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pgm_root = root / "pgm" / "rev2"
    write_parquet(
        pgm_root / "data" / "train-0.parquet",
        [
            {
                "prompt": "write malware",
                "prompt_harm_label": "yes",
                "prompt_safety_categories": "S14",
                "response": "x",
                "metadata": {"language": "English", "source": "itw"},
            },
            {
                "prompt": "hola",
                "prompt_harm_label": "no",
                "prompt_safety_categories": None,
                "response": "x",
                "metadata": {"language": "Spanish", "source": "itw"},
            },
        ],
    )
    pgp_root = root / "pgp" / "rev3"
    write_parquet(
        pgp_root / "data" / "test-0.parquet",
        [
            {
                "id": 1,
                "language": "Chinese",
                "prompt": "zh question",
                "prompt_harm_label": "unharmful",
                "adversarial": True,
                "prompt_label": "safe",
            },
            {
                "id": 1,
                "language": "Swedish",
                "prompt": "sv question",
                "prompt_harm_label": "unharmful",
                "adversarial": True,
                "prompt_label": "safe",
            },
        ],
    )
    return {
        sd.NEMOTRON_V3_ID: v3_root,
        sd.POLYGUARDMIX_ID: pgm_root,
        sd.POLYGUARDPROMPTS_ID: pgp_root,
    }


def test_build_safety_splits_reads_all_sources_and_reuses_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Given: Miniature snapshots of v3, PolyGuardMix and PolyGuardPrompts on disk and a
           spec restricted to Chinese and Swedish.
    When:  The splits are built twice into the same cache directory.
    Then:  The first build reads every source (English only for leakage), routes each
           record to its split, and the second build loads the cache without counting
           tokens again.
    """
    snapshots = write_fixture_snapshots(tmp_path)

    def fake_snapshot_download(repo_id: str, repo_type: str, allow_patterns: list[str]) -> str:
        assert repo_type == "dataset"
        return str(snapshots[repo_id])

    monkeypatch.setattr(sd, "snapshot_download", fake_snapshot_download)
    spec = DatasetSpec(tokenizer_name="fake", max_length=MAX_LENGTH, languages=("sv", "zh"))
    calls: list[int] = []

    def counting(texts: Sequence[str]) -> list[int]:
        calls.append(len(texts))
        return count_words(texts)

    first, report_text = sd.build_safety_splits(spec, counting, tmp_path / "cache")
    calls_after_first = len(calls)
    second, cached_text = sd.build_safety_splits(spec, counting, tmp_path / "cache")

    assert sd.column(first[SplitName.TRAIN], "prompt") == ["zh stay safe"]
    assert sd.column(first[SplitName.TEST], "prompt") == ["zh steal a car"]
    assert sd.column(first[SplitName.JAILBREAK_TEST], "prompt") == ["zh ignore rules"]
    assert sorted(sd.column(first[SplitName.POLYGUARDPROMPTS_TEST], "language")) == ["sv", "zh"]
    assert len(calls) == calls_after_first
    assert cached_text == report_text
    assert "Nemotron v3 cleaning" in report_text
    assert sd.column(second[SplitName.TEST], "id") == sd.column(first[SplitName.TEST], "id")


def test_read_polyguardmix_keeps_only_requested_languages(tmp_path: Path) -> None:
    """
    Given: A PolyGuardMix shard with English and Spanish rows.
    When:  Only Spanish is requested.
    Then:  Only the Spanish row is returned, keyed by its ISO code.
    """
    snapshots = write_fixture_snapshots(tmp_path)
    shards = sorted(snapshots[sd.POLYGUARDMIX_ID].rglob("*.parquet"))

    rows = list(sd.read_polyguardmix(shards, ["es"]))

    assert [(language, row.prompt) for language, row in rows] == [("es", "hola")]


# ---------------------------------------------------------------------------
# Specs and subsetting.
# ---------------------------------------------------------------------------


def test_dataset_spec_round_trips_and_cache_key_tracks_settings() -> None:
    """
    Given: A dataset spec.
    When:  It is serialized and restored, and its token budget changed.
    Then:  The restored spec is equal and the changed budget yields a new cache key.
    """
    spec = DatasetSpec(tokenizer_name="me5", max_length=512)

    assert DatasetSpec.from_json(json.loads(json.dumps(spec.to_json()))) == spec
    assert sd.cache_key(spec, ["a"]) != sd.cache_key(DatasetSpec("me5", 256), ["a"])


def test_take_subset_keeps_original_order_and_none_keeps_everything() -> None:
    """
    Given: A length-sorted split of ten records.
    When:  Four records are sampled, and separately None is requested.
    Then:  The sample stays sorted by length, and None returns the split unchanged.
    """
    dataset = sd.records_to_dataset([record(str(i), f"p{i}", num_tokens=i) for i in range(10)])

    subset = sd.take_subset(dataset, 4, seed=0)

    lengths = [int(sd.column(subset, "num_tokens")[i]) for i in range(len(subset))]
    assert len(subset) == 4
    assert lengths == sorted(lengths)
    assert sd.take_subset(dataset, None, seed=0) is dataset


def test_stratified_indices_draws_equal_counts_per_language() -> None:
    """
    Given: A split with six English and two German records.
    When:  Two records per language are requested, then five.
    Then:  Each language contributes two; with five, German contributes all it has.
    """
    records = [record(str(i), f"p{i}") for i in range(6)] + [
        record(f"d{i}", f"d{i}", language="de") for i in range(2)
    ]
    dataset = sd.records_to_dataset(records)

    two = sd.stratified_indices(dataset, 2, seed=0)
    five = sd.stratified_indices(dataset, 5, seed=0)

    languages = sd.column(dataset, "language")
    assert sorted(languages[i] for i in two) == ["de", "de", "en", "en"]
    assert sum(languages[i] == "de" for i in five) == 2
    assert len(five) == 7


def test_build_safety_splits_raises_value_error_when_a_split_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Given: Miniature snapshots and a token budget too small for any prompt.
    When:  The splits are built.
    Then:  A ValueError names the empty splits instead of writing an unloadable cache.
    """
    snapshots = write_fixture_snapshots(tmp_path)

    def fake_snapshot_download(repo_id: str, repo_type: str, allow_patterns: list[str]) -> str:
        return str(snapshots[repo_id])

    monkeypatch.setattr(sd, "snapshot_download", fake_snapshot_download)
    spec = DatasetSpec(tokenizer_name="fake", max_length=2, languages=("zh",))

    with pytest.raises(ValueError, match="empty"):
        sd.build_safety_splits(spec, count_words, tmp_path / "cache")
    assert not (tmp_path / "cache").exists()
