from enum import StrEnum
from abc import ABC, abstractmethod
from pathlib import Path

from radfact.llm_utils.report_to_phrases.processor import StudyIdType
import polars as pl
import json

class MetricGenerationPipelineType(StrEnum):
    CT = "ct"
    XR = "xr"

class MetricGenerationPipeline(ABC):
    @abstractmethod
    def get_candidates_and_references(self, input_path_candidate: Path, input_path_reference: str) -> tuple[
    dict[StudyIdType, str],
    dict[StudyIdType, str],
    dict[StudyIdType, str],
    dict[StudyIdType, str],
    dict[StudyIdType, str]
    ]:
        ...

class CTMetricGenerationPipeline(MetricGenerationPipeline):
    def get_candidates_and_references(
    combined_generated_path: Path,
) -> tuple[
    dict[StudyIdType, str],
    dict[StudyIdType, str],
    dict[StudyIdType, str],
    dict[StudyIdType, str],
    dict[StudyIdType, str],
]:
        """Reads the csv file containing the samples to compute RadFact for and returns the candidates and references in
        the expected format."""
        predictions_path_is_glob = "*" in str(combined_generated_path)
        findings_generation_samples = pl.scan_csv(str(combined_generated_path), glob=predictions_path_is_glob).select(
            pl.col(
                "study_datapoint_id",
                "series_datapoint_id",
                "generated_report",
                "ground_truth_report",
            )
        )
        findings_generation_samples = findings_generation_samples.collect()
        n_rows = findings_generation_samples.height
        candidates = dict(enumerate(findings_generation_samples["generated_report"].fill_null("")))

        references = dict(enumerate(findings_generation_samples["ground_truth_report"].fill_null("")))

        study_instance_uid = dict(enumerate(findings_generation_samples["study_datapoint_id"].fill_null("")))

        series_instance_uid = dict(enumerate(findings_generation_samples["series_datapoint_id"].fill_null("")))
        #dummy values for CT
        instance_number = dict(enumerate(range(1, n_rows + 1)))


        return (candidates, references, study_instance_uid, series_instance_uid, instance_number)

class XRMetricGenerationPipeline(MetricGenerationPipeline):

    def get_candidates_and_references(
    csv_path: Path,
    input_path_reference: str,
) -> tuple[
    dict[StudyIdType, str],
    dict[StudyIdType, str],
    dict[StudyIdType, str],
    dict[StudyIdType, str],
    dict[StudyIdType, str],
]:
        """Reads the csv file containing the samples to compute RadFact for and returns the candidates and references in
        the expected format."""
        predictions_path_is_glob = "*" in str(csv_path)
        findings_generation_samples = pl.scan_csv(str(csv_path), glob=predictions_path_is_glob).select(
            pl.col(
                "current_frontal_metadata.study_uid",
                "unique_id",
                "generated_response",
                "current_frontal_metadata.series_uid",
                "current_frontal_metadata.instance_number",
            )
        )
        # Add unique_id_prefix as unique_id.str.slice(0, 2)
        findings_generation_samples = findings_generation_samples.with_columns(
            pl.col("unique_id").str.slice(0, 2).alias("datapoint_id_prefix")
        )

        # Get the list of prefixes for the unique IDs
        prefixes = (
            findings_generation_samples.select(pl.col("datapoint_id_prefix"))
            .unique()
            .collect()
            .get_column("datapoint_id_prefix")
            .to_list()
        )

        # Reference LazyFrame - read using the hive-partitioned parquet format
        combined_lf = (
            pl.scan_parquet(input_path_reference, hive_partitioning=True)
            .filter((pl.col("task") == "report_generation") & pl.col("datapoint_id_prefix").is_in(prefixes))
            .select(
                [
                    "datapoint_id",
                    "task",
                    "annotation.annotated_concepts",
                    "annotation.report_sections.findings",
                    "annotation.report_sections.impression",
                    "datapoint_id_prefix",
                ]
            )
        ).join(
            findings_generation_samples,
            left_on=["datapoint_id", "datapoint_id_prefix"],
            right_on=["unique_id", "datapoint_id_prefix"],
            how="inner",
        )

        # Collect
        combined_df = combined_lf.collect()

        # Get candidates dict in format expected downstream
        candidate_values = (
            combined_df.with_columns(pl.col("generated_response").fill_null("[]"))
            .select("generated_response")
            .to_series()
            .map_elements(lambda gr: " ".join(item["response"] for item in json.loads(gr)), return_dtype=pl.String)
        )
        candidates = dict(enumerate(candidate_values))

        # Get reference dict in format expected downstream. Using findings + impression as the reference text
        reference_values = (
            combined_df.with_columns(
                pl.concat_str(
                    [
                        pl.col("annotation.report_sections.findings").fill_null(""),
                        pl.col("annotation.report_sections.impression").fill_null(""),
                    ],
                    separator=" ",
                )
                .str.replace_all(":", " ")
                .alias("combined_findings")
            )
            .select("combined_findings")
            .to_series()
        )

        # Convert to dict with row index as key
        references = dict(enumerate(reference_values))

        # For study_uid
        study_instance_uid_current_frontal = dict(
            enumerate(combined_df["current_frontal_metadata.study_uid"].fill_null(""))
        )

        # For series_uid
        series_instance_uid_current_frontal = dict(
            enumerate(combined_df["current_frontal_metadata.series_uid"].fill_null(""))
        )

        # For instance_number
        instance_number_current_frontal = dict(
            enumerate(combined_df["current_frontal_metadata.instance_number"].fill_null(0).cast(pl.Int32))
        )

        return (
            candidates,
            references,
            study_instance_uid_current_frontal,
            series_instance_uid_current_frontal,
            instance_number_current_frontal,
        )
