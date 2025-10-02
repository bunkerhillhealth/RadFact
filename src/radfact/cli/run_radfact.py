#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import argparse
import json
import logging
import tempfile
from pathlib import Path

import polars as pl

from radfact.cli.pipeline.base import CTMetricGenerationPipeline, XRMetricGenerationPipeline, MetricGenerationPipelineType
from radfact.cloud.client import GCSClient, S3Client
from radfact.cloud.types import GCSPath, S3Path
from radfact.data_utils.grounded_phrase_list import GroundedPhraseList, GroundedPhrase
from radfact.llm_utils.report_to_phrases.processor import StudyIdType
from radfact.metric.bootstrapping import MetricBootstrapper
from radfact.metric.print_utils import print_bootstrap_results, print_results
from radfact.metric.radfact import InputDict, RadFactMetric
from radfact.paths import CONFIGS_DIR

logger = logging.getLogger(__name__)

import os
print(os.getcwd())


def validate_config_file(config_name: str | None) -> None:
    if config_name is not None:
        config_path = CONFIGS_DIR / f"{config_name}"
        print(config_path)
        if not config_path.exists():
            message = (
                f"Config file {config_name} does not exist. "
                "Make sure the config file is saved in the `configs` directory."
            )
            raise FileNotFoundError(message)


def get_candidates_and_references_from_json(
    json_path: Path,
) -> tuple[dict[StudyIdType, GroundedPhraseList], dict[StudyIdType, GroundedPhraseList]]:
    """Reads the json file containing the samples to compute RadFact for and returns the candidates and references in
    the expected format."""
    with open(json_path, "r", encoding="utf-8") as f:
        grounded_reporting_samples = json.load(f)
    logger.info(f"Loaded {len(grounded_reporting_samples)} samples from {json_path}")
    candidates = {
        example["example_id"]: GroundedPhraseList.from_list_of_dicts(example["prediction"])
        for example in grounded_reporting_samples
    }
    references = {
        example["example_id"]: GroundedPhraseList.from_list_of_dicts(example["target"])
        for example in grounded_reporting_samples
    }
    return candidates, references

def get_phrases_from_json_file(input_path: str) -> dict[StudyIdType, GroundedPhraseList]:
    """gets candidates or references from a single json file"""
    with open(input_path, "r", encoding="utf-8") as f:
        phrase_list = json.load(f)

    output = {}
    for item in phrase_list:
        phrases = GroundedPhraseList()
        for sentence in item["sentence_list"]:
            for new_text in sentence["new"]:
                phrases.append(GroundedPhrase(text=new_text, boxes=None))
        output[str(item["id"])] = phrases
    return output


def compute_radfact_scores(
    radfact_config_name: str | None,
    phrases_config_name: str | None,
    candidates: InputDict,
    references: InputDict,
    study_instance_uids: InputDict,
    series_instance_uids: InputDict,
    instance_numbers_current_frontal: InputDict,
    is_narrative_text: bool,
    bootstrap_samples: int,
    ev_text_file_name: str = "system_message_ev_singlephrase_updated_with_reasoning.txt",
    reports_to_phrases_text_file_name: str = "system_message.txt",
    candidate_phrases: dict[StudyIdType, GroundedPhraseList] | None = None,
    reference_phrases: dict[StudyIdType, GroundedPhraseList] | None = None,
    few_shot_examples_reports_to_phrases_filename: str | None = None,
    few_shot_examples_radfact_filename: str | None = None,
) -> dict[str, float]:
    radfact_metric = RadFactMetric(
        nli_config_name=radfact_config_name,
        phrase_config_name=phrases_config_name,
        is_narrative_text=is_narrative_text,
        candidate_phrases=candidate_phrases,
        reference_phrases=reference_phrases,
        few_shot_examples_reports_to_phrases_filename=few_shot_examples_reports_to_phrases_filename,
        few_shot_examples_radfact_filename=few_shot_examples_radfact_filename,
        ev_text_file_name=ev_text_file_name,
        reports_to_phrases_text_file_name=reports_to_phrases_text_file_name,
    )
    assert bootstrap_samples >= 1
    bootstrapper = MetricBootstrapper(metric=radfact_metric, num_samples=bootstrap_samples, seed=42)
    results_per_sample = radfact_metric.compute_results_per_sample(
        candidates, references
    )
    results_per_sample_df = radfact_metric.results_per_sample_to_dataframe(
        results_per_sample, study_instance_uids, series_instance_uids, instance_numbers_current_frontal
    )
    return bootstrapper.compute_bootstrap_metrics(results_per_sample=results_per_sample), results_per_sample_df

def resolve_path(path: str):
    if path.startswith("s3://"):
        return S3Path(path)
    if path.startswith("gs://"):
        return GCSPath(path)
    return Path(path)

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Compute RadFact metric for a set of samples and saves the results to a json file."
    )
    parser.add_argument(
        "--input_path_candidate",
        type=str,
        help="The path to the csv file which is obtained from running fm.maira2.generate.generate . See the generated type at: "
        "https://github.com/bunkerhillhealth/bunkerhill/blob/e1baffb1e194f4c330c4254efe243183004710ff/fm/maira2/generate/generate/types.py#L45.",
        default=None,
    )

    parser.add_argument(
        "--input_path_reference",
        type=str,
        help="Path to the hive-partitioned parquet folder containing the reference reports. The folders are assumed to be "
        "hive-partitioned based on unique ID, for fast lookup. Use glob pattern for hive-partitioned parquet files. "
        "For example: gs://fm-internal-data/evaluation_data/xray/medstar_subset_2025-05-06/**/*.parquet",
        default=None,
    )
    parser.add_argument(
        "--combined_generated_path",
        type=str,
        help="Path to the combined generated reports/ground truth reports.",
        default=None,
    )
    parser.add_argument(
        "--pipeline",
        type=MetricGenerationPipelineType,
        help="The processing pipeline to use for generating the metrics ('ct' or 'xr').",
        default='xr',
    )

    parser.add_argument(
        "--is_narrative_text",
        action="store_true",
        help="Whether the input samples are narrative text or not. If true, the input samples are expected to be "
        "narrative text, otherwise they are expected to be grounded phrases.",
    )
    parser.add_argument(
        "--radfact_config_name",
        type=str,
        help="The name of the config file for RadFact processing. We use the default config file but you can provide a "
        "custom config. Make sure the config follows the same structure as `configs/radfact.yaml` and is saved in the "
        "`configs` directory. This is necessary for hydra initialization from the `configs` directory.",
        default=None,
    )
    parser.add_argument('--few_shot_examples_reports_to_phrases_filename', help='The name of the few shot examples file for splitting reports into phrases. This can be found under the `report_to_phrases/prompts` directory.', default="few_shot_examples_ct_shortened_no_measurements.json")
    parser.add_argument('--few_shot_examples_radfact_filename',  help='The name of the few shot examples file for radfact entailment verification. This can be found under the `nli/prompts` directory.', default=None)
    parser.add_argument(
        "--phrases_config_name",
        type=str,
        help="The name of the config file for reports to phrases conversion. We use the default config file but you "
        "can provide a custom config. Make sure the config follows the same structure as "
        "`configs/report_to_phrases.yaml` and is saved in the `configs` directory. This is necessary for hydra "
        "initialization from the `configs` directory.",
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the directory where the results will be saved as a json file.",
        default="outputs",
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        help="Number of bootstrap samples to use for computing the confidence intervals. Set to 0 to disable "
        "bootstrapping.",
        default=500,
    )
    parser.add_argument(
        "--input_phrases_path_candidate",
        type=str,
        help="Path to the json file containing the phrases from a previous run."
        "if provided, will skip the report to phrases step and use these phrases.",
        default=None,
    )
    parser.add_argument(
        "--input_phrases_path_reference",
        type=str,
        help="Path to the json file containing the phrases from a previous run."
        "if provided, will skip the report to phrases step and use these phrases.",
        default=None,
    )

    parser.add_argument(
        "--ev_text_file_name",
        type=str,
        default="system_message_ev_singlephrase_updated_with_reasoning_negatives_ct.txt",
        help="The name of the system message file for the entailment verification processor. This is used to set up "
        "the entailment verification processor for RadFact. The file should be in the `radfact/llm_utils/nli/prompts` directory.",
    )
    parser.add_argument('--reports_to_phrases_text_file_name', help='The name of the system message file for the report to phrases processor. This is used to set up '
        "the report to phrases processor for RadFact. The file should be in the `report_to_phrases/prompts` directory.", default="system_message_ct_no_measurements.txt")

    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples to process.")

    args = parser.parse_args()

    if (args.input_path_reference is None) != (args.input_path_candidate is None):
        raise ValueError("Both input_path_reference and input_path_candidate must be provided together.")
    if (args.pipeline == MetricGenerationPipelineType.CT) and (args.combined_generated_path is None):
        raise ValueError("combined_generated_path is only supported for CT pipeline.")

    combined_generated_path = resolve_path(args.combined_generated_path) if args.combined_generated_path is not None else None
    input_path_candidate = resolve_path(args.input_path_candidate) if args.input_path_candidate is not None else None
    output_dir = resolve_path(args.output_dir)

    if args.input_path_reference is not None:
        assert args.input_path_reference.startswith(
            "gs://"
        ), "All parquet files are assumed to be in hive-partitioned parquet format on GCS."

    input_path_reference = args.input_path_reference if args.input_path_reference is not None else None
    is_narrative_text = args.is_narrative_text
    radfact_config_name = args.radfact_config_name
    phrases_config_name = args.phrases_config_name
    bootstrap_samples = args.bootstrap_samples
    ev_text_file_name = args.ev_text_file_name
    reports_to_phrases_text_file_name = args.reports_to_phrases_text_file_name
    few_shot_examples_reports_to_phrases_filename = args.few_shot_examples_reports_to_phrases_filename
    few_shot_examples_radfact_filename = args.few_shot_examples_radfact_filename
    pipeline = args.pipeline
    limit = args.limit

    if not is_narrative_text:
        raise NotImplementedError("BH output format for grounded phrases is not yet supported.")

    if input_path_candidate is not None:
        assert input_path_candidate.suffix in [".csv", ".json"], "Input file must be a csv or json file."
        assert input_path_candidate.suffix == ".csv" or not is_narrative_text, (
            "Input file must be a json file for grounded phrases and is_narrative_text must be False. For narrative text, "
            "input file must be a csv file and is_narrative_text must be True."
        )
    validate_config_file(radfact_config_name)
    validate_config_file(phrases_config_name)

    assert ev_text_file_name.endswith(".txt"), "The entailment verification text file must be a .txt file."
    assert reports_to_phrases_text_file_name.endswith(".txt"), "The report to phrases text file must be a .txt file."
    assert few_shot_examples_reports_to_phrases_filename.endswith(".json"), "The few shot examples file must be a .json file."
    if few_shot_examples_radfact_filename is not None:
        assert few_shot_examples_radfact_filename.endswith(".json"), "The few shot examples file must be a .json file."


    candidates: InputDict
    references: InputDict

    if args.input_phrases_path_candidate is not None:
        candidate_phrases = get_phrases_from_json_file(args.input_phrases_path_candidate)
    else:
        candidate_phrases = None
    if args.input_phrases_path_reference is not None:
        reference_phrases = get_phrases_from_json_file(args.input_phrases_path_reference)
    else:
        reference_phrases = None


    if pipeline == MetricGenerationPipelineType.XR:
        candidates, references, study_instance_uids, series_instance_uids, instance_numbers_current_frontal = (
            XRMetricGenerationPipeline.get_candidates_and_references(input_path_candidate, input_path_reference, limit=limit)
        )


    elif pipeline == MetricGenerationPipelineType.CT:
        candidates, references, study_instance_uids, series_instance_uids, instance_numbers_current_frontal = (
            CTMetricGenerationPipeline.get_candidates_and_references(combined_generated_path, limit=limit)
        )
    else:
        raise ValueError(f"Invalid pipeline: {pipeline}")


    results_bootstrap_json, results_df = compute_radfact_scores(
        radfact_config_name=radfact_config_name,
        phrases_config_name=phrases_config_name,
        candidates=candidates,
        references=references,
        is_narrative_text=is_narrative_text,
        bootstrap_samples=bootstrap_samples,
        study_instance_uids=study_instance_uids,
        series_instance_uids=series_instance_uids,
        instance_numbers_current_frontal=instance_numbers_current_frontal,
        ev_text_file_name=args.ev_text_file_name,
        reports_to_phrases_text_file_name=args.reports_to_phrases_text_file_name,
        candidate_phrases=candidate_phrases,
        reference_phrases=reference_phrases,
        few_shot_examples_reports_to_phrases_filename=few_shot_examples_reports_to_phrases_filename,
        few_shot_examples_radfact_filename=few_shot_examples_radfact_filename,
    )
    logger.info(f"Processing {len(candidates)} samples")

    print_fn = print_results if bootstrap_samples == 0 else print_bootstrap_results

    logger.info(f"RadFact scores for narrative text samples - using {few_shot_examples_radfact_filename} for entailment verification")
    print_fn(
        results=results_bootstrap_json,
        metrics=["logical_precision", "logical_recall", "logical_f1", "num_llm_failures"],
    )


    output_path = output_dir / f"radfact_scores.json"

    if isinstance(output_path, (GCSPath, S3Path)):
        with tempfile.TemporaryDirectory() as tempdir:
            with open(Path(tempdir) / "tmp.json", "w", encoding="utf-8") as f:
                json.dump(results_bootstrap_json, f, indent=2)
            if isinstance(output_path, GCSPath):
                GCSClient.upload_file(Path(tempdir) / "tmp.json", output_path)
            elif isinstance(output_path, S3Client):
                S3Client.upload_file(Path(tempdir) / "tmp.json", output_path)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_bootstrap_json, f, indent=2)

    results_df.to_csv(str(output_path)[:-5] + ".csv", index=False)

    logger.info(f"results saved to {output_path}")


if __name__ == "__main__":
    main()
