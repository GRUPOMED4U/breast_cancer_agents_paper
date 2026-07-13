from pathlib import Path
import re
from tqdm import tqdm
import jsonlines
import pandas as pd

from logs import get_logger

logger = get_logger(__name__)

data_path = Path("data")
files_to_process = [
    "cases_stage_1.csv",
    "cases_stage_2.csv",
    "cases_stage_3.csv",
    "cases_stage_4.csv",
]


def main():
    # Check if output .jsonl file exists
    output_path = data_path / "case_summaries.jsonl"
    if output_path.exists():
        logger.info(f"Output file {output_path} already exists. Skipping.")
        return
    else:
        output_path.touch()

    case_index = 0

    # Iterate through raw data files
    for file in files_to_process:
        filepath = data_path / file
        current_cases_df = pd.read_csv(filepath, dtype=str).dropna(
            subset=["ID PACIENTE"]
        )

        # Iterate through cases
        for row in tqdm(current_cases_df.to_dict("records"), desc=f"Processing {file}"):
            patient_id = row["ID PACIENTE"]
            complete_summary = row["SUMARIZADO COMPLETO"]

            # Skip empty cases
            if (
                not complete_summary
                or type(complete_summary) is not str
                or complete_summary.strip() == ""
            ):
                continue

            # Clean unwanted substrings
            unwanted_subtrings_patterns = [
                r":contentReference\[oaicite:\d+\]\{index=\d+\}",
            ]

            for pattern in unwanted_subtrings_patterns:
                complete_summary = re.sub(pattern, "", complete_summary)

            # Organize data
            try:
                new_data_entry = {
                    "id": case_index,
                    "filename": file,
                    "breast_cancer_stage": file.split(".")[0][-1],
                    "case_summary": complete_summary.split("---")[0].strip(),
                    "official_recommendations": complete_summary.split("---")[
                        1
                    ].strip(),
                }

                # Save to jsonlines
                with jsonlines.open(output_path, mode="a") as writer:
                    writer.write(new_data_entry)

                case_index += 1

            except Exception as e:
                logger.error(f"Error processing {patient_id}: {e}")
                logger.error(f"Row data: {row}")
                raise

        logger.info(f"Saved {output_path} successfully.")


if __name__ == "__main__":
    main()
