"""Dataset container for loading evaluation entries from disk.

This module defines a lightweight dataset abstraction that loads
`EvaluationEntry` records from a supported file format and exposes
list-like access to the loaded records. It is intended for evaluation
workflows that need to iterate over, index, and inspect structured
entries stored on disk.
"""

from pathlib import Path
from typing import List, Literal
import jsonlines

from data_models import EvaluationEntry


class Dataset(list[EvaluationEntry]):
    """List-like dataset of `EvaluationEntry` objects loaded from a file.

    This class loads evaluation records from a dataset file at initialization
    time and stores them in memory. It supports common container operations
    such as indexing, slicing, iteration, and length retrieval.

    Args:
        file_path: Path to the dataset file to load.

    Attributes:
        file_path: Path to the dataset file.
        records: Loaded evaluation entries.
    """

    def __init__(
        self,
        file_path: str | Path,
    ) -> None:
        """Initialize the dataset and load records from the given file path.

        Args:
            file_path: Path to the dataset file. May be provided as a string
                or `Path`.

        Raises:
            AssertionError: If the provided file path does not exist.
        """
        self.file_path = file_path if isinstance(file_path, Path) else Path(file_path)
        assert self.file_path.exists(), f"File {self.file_path} does not exist"
        self.records: List[EvaluationEntry] = self._load_dataset_from_path()

    def _load_dataset_from_path(self) -> List[EvaluationEntry]:
        """Load dataset records based on the input file extension.

        Currently, only JSONL files are supported.

        Returns:
            A list of loaded `EvaluationEntry` records.

        Raises:
            NotImplementedError: If the file extension is not supported.
        """
        if self.file_path.suffix == ".jsonl":
            return self._load_dataset_from_jsonl()
        else:
            raise NotImplementedError(
                f"Loading dataset from {self.file_path.suffix} is not implemented"
            )

    def _load_dataset_from_jsonl(self) -> List[EvaluationEntry]:
        """Load dataset records from a JSONL file.

        Each line in the file is expected to contain one JSON object compatible
        with the `EvaluationEntry` schema.

        Returns:
            A list of parsed `EvaluationEntry` objects.
        """
        records = []
        with jsonlines.open(self.file_path) as reader:
            for entry in reader:
                records.append(EvaluationEntry(**entry))
        return records

    def __len__(self) -> int:
        """Return the number of records in the dataset.

        Returns:
            The total number of loaded records.
        """
        return len(self.records)

    def __getitem__(
        self, index: int | slice
    ) -> List[EvaluationEntry] | EvaluationEntry:
        """Retrieve one or more records from the dataset by index or slice.

        Args:
            index: Integer index for a single record or slice for multiple
                records.

        Returns:
            A single `EvaluationEntry` when `index` is an integer, or a list of
                `EvaluationEntry` objects when `index` is a slice.
        """
        return self.records[index]

    def __iter__(self) -> List[EvaluationEntry]:
        """Return an iterator over the dataset records.

        Returns:
            An iterator over the loaded `EvaluationEntry` objects.
        """
        return iter(self.records)

    def __repr__(self) -> str:
        """Return the developer-friendly string representation of the dataset.

        Returns:
            A string containing the dataset class name and file path.
        """
        return f"Dataset(file_path={self.file_path})"

    def __str__(self) -> str:
        """Return the informal string representation of the dataset.

        Returns:
            A string representation of the dataset.
        """
        return self.__repr__()

    def save_to_disk(
        self,
        dir_path: str | Path = ".",
        file_format: Literal["jsonl", "json-doccano"] = "jsonl",
        include_labels: bool = True,
    ) -> None:
        """Save the dataset to a JSONL file on disk.

        Args:
            path: Destination file path where the dataset should be written.
        """
        if isinstance(dir_path, str):
            dir_path = Path(dir_path)

        if file_format == "jsonl":
            with jsonlines.open(dir_path / "dataset.jsonl", "w") as writer:
                writer.write_all([record.model_dump() for record in self.records])

        elif file_format == "jsonl-doccano":
            with jsonlines.open(dir_path / "dataset.jsonl", "w") as writer:
                writer.write_all(
                    [
                        {
                            **record.to_doccano_dict(include_labels=include_labels),
                            "prompt_id": record.prompt_id,
                        }
                        for record in self.records
                    ]
                )
        else:
            raise NotImplementedError(
                f"Saving dataset to {file_format} is not implemented"
            )

    def merge(self, other_dataset: "Dataset") -> "Dataset":

        # index by prompt_id
        self_records = {entry.prompt_id: entry for entry in self.records}
        other_records = {entry.prompt_id: entry for entry in other_dataset.records}

        for prompt_id, entry in other_records.items():
            self_records[prompt_id] = entry

        self.records = sorted(self_records.values(), key=lambda x: int(x.prompt_id))

        return self

    def get_rubrics_from_other_dataset(self, other_dataset: "Dataset") -> "Dataset":
        # index by prompt_id
        self_records = {entry.prompt_id: entry for entry in self.records}
        other_records = {entry.prompt_id: entry for entry in other_dataset.records}

        for prompt_id, entry in other_records.items():
            if prompt_id not in self_records:
                continue
            self_records[prompt_id].rubrics = entry.rubrics

        self.records = sorted(self_records.values(), key=lambda x: int(x.prompt_id))

        return self
