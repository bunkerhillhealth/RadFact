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

from radfact.cli.pipeline.base import CTMetricGenerationPipeline, XRMetricGenerationPipeline
from radfact.cloud.client import GCSClient, S3Client
from radfact.cloud.types import GCSPath, S3Path
from radfact.data_utils.grounded_phrase_list import GroundedPhraseList
from radfact.llm_utils.report_to_phrases.processor import StudyIdType
from radfact.metric.bootstrapping import MetricBootstrapper
from radfact.metric.print_utils import print_bootstrap_results, print_results
from radfact.metric.radfact import InputDict, RadFactMetric
from radfact.paths import CONFIGS_DIR

logger = logging.getLogger(__name__)


def validate_config_file(config_name: str | None) -> None:
    if config_name is not None:
        config_path = CONFIGS_DIR / f"{config_name}"
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
    allow_omitted_negatives: bool = False,
) -> dict[str, float]:
    radfact_metric = RadFactMetric(
        nli_config_name=radfact_config_name,
        phrase_config_name=phrases_config_name,
        is_narrative_text=is_narrative_text,
    )
    # if bootstrap_samples == 0:
    #     _, results = radfact_metric.compute_metric_score(candidates, references)
    #     return results
    assert bootstrap_samples >= 1
    bootstrapper = MetricBootstrapper(metric=radfact_metric, num_samples=bootstrap_samples, seed=42)
    results_per_sample = radfact_metric.compute_results_per_sample(
        candidates, references, ev_text_file_name, allow_omitted_negatives
    )
    results_per_sample_df = radfact_metric.results_per_sample_to_dataframe(
        results_per_sample, study_instance_uids, series_instance_uids, instance_numbers_current_frontal
    )
    return bootstrapper.compute_bootstrap_metrics(results_per_sample=results_per_sample), results_per_sample_df


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
        "--ev_text_file_name",
        type=str,
        default="system_message_ev_singlephrase_updated_with_reasoning.txt",
        help="The name of the system message file for the entailment verification processor. This is used to set up "
        "the entailment verification processor for RadFact. The file should be in the `radfact/llm_utils/nli/prompts` directory.",
    )

    args = parser.parse_args()

    if args.input_path_candidate.startswith("s3://"):
        input_path_candidate = S3Path(args.input_path_candidate)
    elif args.input_path_candidate.startswith("gs://"):
        input_path_candidate = GCSPath(args.input_path_candidate)
    else:
        input_path_candidate = Path(args.input_path_candidate)

    if args.output_dir.startswith("s3://"):
        output_dir = S3Path(args.output_dir)
    elif args.output_dir.startswith("gs://"):
        output_dir = GCSPath(args.output_dir)
    else:
        output_dir = Path(args.output_dir)

    assert args.input_path_reference.startswith(
        "gs://"
    ), "All parquet files are assumed to be in hive-partitioned parquet format on GCS."

    if (args.input_path_reference is None) != (args.input_path_candidate is None):
        raise ValueError("Both input_path_reference and input_path_candidate must be provided together.")

    input_path_reference = args.input_path_reference

    is_narrative_text = args.is_narrative_text
    radfact_config_name = args.radfact_config_name
    phrases_config_name = args.phrases_config_name
    bootstrap_samples = args.bootstrap_samples
    pipeline = args.pipeline

    assert input_path_candidate.suffix in [".csv", ".json"], "Input file must be a csv or json file."
    assert input_path_candidate.suffix == ".csv" or not is_narrative_text, (
        "Input file must be a json file for grounded phrases and is_narrative_text must be False. For narrative text, "
        "input file must be a csv file and is_narrative_text must be True."
    )
    validate_config_file(radfact_config_name)
    validate_config_file(phrases_config_name)

    assert args.ev_text_file_name.endswith(".txt"), "The entailment verification text file must be a .txt file."

    candidates: InputDict
    references: InputDict

    if is_narrative_text:
        if pipeline == MetricGenerationPipelineType.XR:
            candidates, references, study_instance_uids, series_instance_uids, instance_numbers_current_frontal = (
                XRMetricGenerationPipeline.get_candidates_and_references(input_path_candidate, input_path_reference)
            )
        elif pipeline == MetricGenerationPipelineType.CT:
            candidates, references, study_instance_uids, series_instance_uids, instance_numbers_current_frontal = (
                CTMetricGenerationPipeline.get_candidates_and_references(input_path_candidate, input_path_reference)
            )
        else:
            raise ValueError(f"Invalid pipeline: {pipeline}")

    else:
        # candidates, references = get_candidates_and_references_from_json(input_path)
        raise NotImplementedError("BH output format for grounded phrases is not yet supported.")

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
        allow_omitted_negatives=False,
    )

    results_bootstrap_allow_negs_json, results_allow_negs_df = compute_radfact_scores(
        radfact_config_name=radfact_config_name,
        phrases_config_name=phrases_config_name,
        candidates=candidates,
        references=references,
        is_narrative_text=is_narrative_text,
        bootstrap_samples=bootstrap_samples,
        study_instance_uids=study_instance_uids,
        series_instance_uids=series_instance_uids,
        instance_numbers_current_frontal=instance_numbers_current_frontal,
        ev_text_file_name="system_message_ev_singlephrase_updated_with_reasoning_negatives.txt",  # Hard-code for the negative ommission
        allow_omitted_negatives=True,
    )

    print_fn = print_results if bootstrap_samples == 0 else print_bootstrap_results
    if is_narrative_text:
        print("RadFact scores for narrative text samples - lower bound, penalized ommitted negatives")
        print_fn(
            results=results_bootstrap_json,
            metrics=["logical_precision", "logical_recall", "logical_f1", "num_llm_failures"],
        )
        print("RadFact scores for narrative text samples - upper bound, not penalized ommitted negatives")
        print_fn(
            results=results_bootstrap_allow_negs_json,
            metrics=["logical_precision", "logical_recall", "logical_f1", "num_llm_failures"],
        )
    else:
        print("RadFact scores for grounded phrases samples")
        print_fn(
            results=results_bootstrap_json,
            metrics=[
                "logical_precision",
                "logical_recall",
                "logical_f1",
                "spatial_precision",
                "spatial_recall",
                "spatial_f1",
                "grounding_precision",
                "grounding_recall",
                "grounding_f1",
                "num_llm_failures",
            ],
        )

    output_path_lower_bound = output_dir / f"radfact_scores_lower_bound.json"
    output_path_upper_bound = output_dir / f"radfact_scores_upper_bound.json"

    if isinstance(output_path_lower_bound, (GCSPath, S3Path)):
        with tempfile.TemporaryDirectory() as tempdir:
            with open(Path(tempdir) / "tmp.json", "w", encoding="utf-8") as f:
                json.dump(results_bootstrap_json, f, indent=2)
            if isinstance(output_path_lower_bound, GCSPath):
                GCSClient.upload_file(Path(tempdir) / "tmp.json", output_path_lower_bound)
            elif isinstance(output_path_lower_bound, S3Client):
                S3Client.upload_file(Path(tempdir) / "tmp.json", output_path_lower_bound)
    else:
        with open(output_path_lower_bound, "w", encoding="utf-8") as f:
            json.dump(results_bootstrap_json, f, indent=2)

    results_df.to_csv(str(output_path_lower_bound)[:-5] + ".csv", index=False)

    logger.info(f"Lower bound results saved to {output_path_lower_bound}")

    if isinstance(output_path_upper_bound, (GCSPath, S3Path)):
        with tempfile.TemporaryDirectory() as tempdir:
            with open(Path(tempdir) / "tmp.json", "w", encoding="utf-8") as f:
                json.dump(results_bootstrap_allow_negs_json, f, indent=2)
            if isinstance(output_path_upper_bound, GCSPath):
                GCSClient.upload_file(Path(tempdir) / "tmp.json", output_path_upper_bound)
            elif isinstance(output_path_upper_bound, S3Client):
                S3Client.upload_file(Path(tempdir) / "tmp.json", output_path_upper_bound)
    else:
        with open(output_path_upper_bound, "w", encoding="utf-8") as f:
            json.dump(results_bootstrap_allow_negs_json, f, indent=2)

    results_allow_negs_df.to_csv(str(output_path_upper_bound)[:-5] + ".csv", index=False)
    logger.info(f"Upper bound results saved to {output_path_upper_bound}")


if __name__ == "__main__":
    main()
