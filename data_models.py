from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator


class RubricTag(str, Enum):  # inherits from str to support JSON serialization
    """Enumeration of rubric tag categories used to classify evaluation criteria.

    The enum inherits from `str` to support straightforward JSON serialization.
    """

    SYSTEMIC_THERAPY = "systemic_therapy"
    SURGERY = "surgery"
    RADIOTHERAPY = "radiotherapy"
    COMPLEMENTARY_EXAMS = "complementary_exams"


class Rubric(BaseModel):
    """Represents a rubric item used to evaluate a completion.

    Attributes:
        criterion: Description of the attribute being evaluated.
        points: Score contribution for this rubric item, which may be positive or
            negative.
        tags: Optional tags used to categorize the rubric item.
    """

    criterion: str = Field(
        ...,
        description="Describes the attribute that needs to be present in the text being evaluated in order to apply the points associated with this rubric.",
    )
    points: int = Field(
        ...,
        ge=-10,
        le=10,
        description="Number of points associated with this rubric. Ranges from -10 to 10. Negative points are penalizing, positive points are rewarding.",
    )
    tags: List[RubricTag] = Field(  # noqa: F821
        default_factory=list,
        description="A list of tags that can be used to categorize this rubric.",
        min_items=1,
    )

    def to_string(self) -> str:
        return f"[{'+' + str(self.points) if self.points > 0 else str(self.points)}] [{', '.join(self.tags)}] {self.criterion}"


class RubricList(BaseModel):
    rubrics: List[Rubric]


class Message(BaseModel):
    """Represents a single message in a prompt or conversation.

    Attributes:
        role: Role of the message author, such as user or assistant.
        content: Text content of the message.
    """

    role: str
    content: str


class Prompt(RootModel[List[Message]]):
    pass


class GraderResponse(BaseModel):
    """Represents the output of a grader for a single criterion or evaluation step.

    Attributes:
        explanation: Explanation of the grading decision.
        criteria_met: Whether the evaluated criterion was satisfied.
    """

    explanation: str = ""
    criteria_met: bool = False


class EvaluationEntry(BaseModel):
    """Represents one evaluation example containing a prompt and grading metadata.

    Attributes:
        prompt_id: Unique identifier for the prompt.
        prompt: Ordered list of messages forming the prompt.
        rubrics: List of rubric criteria used for evaluation.
        **extra fields: Additional fields to support extensible evaluation datasets.

    Notes:
        Extra fields are allowed to support extensible evaluation datasets.
    """

    prompt_id: str
    prompt: Prompt
    rubrics: List[Rubric] = Field(
        default_factory=list, description="A list of rubrics for the prompt."
    )
    example_tags: List[str] = Field(
        default_factory=list,
        description="A list of tags that can be used to categorize this entry by theme or evaluation dimensions.",
    )
    grader_responses: List["GraderResponse"] = Field(
        default_factory=list,
        description="A list of grader responses for the prompt.",
    )

    model_config = ConfigDict(extra="allow")

    @field_validator("prompt_id", mode="before")
    def validate_prompt_id(cls, value: Any) -> str:
        if isinstance(value, int):
            return str(value)
        return value


class EvaluationMetrics(BaseModel):
    """Represents aggregate metrics and supporting details for an evaluation run.

    Attributes:
        global_score: Overall evaluation score.
        global_score_std: Standard deviation of the overall score.
        per_example_tag_score: Scores aggregated by example tag.
        per_rubric_tag_score: Scores aggregated by rubric tag.
        token_usage: Token usage statistics for the evaluation run.
        per_example_details: Optional list of per-example evaluation details.
    """

    global_score: float
    global_score_std: float
    per_example_tag_score: Dict[str, float]
    per_rubric_tag_score: Dict[RubricTag, float]
    token_usage_on_inference: Optional[Dict[str, Any]] = None
    token_usage_on_eval: Optional[Dict[str, Any]] = None
    per_example_details: Optional[List[EvaluationEntry]]

    model_config = ConfigDict(extra="allow")

    def __str__(self) -> str:
        """Return a formatted JSON string representation of the evaluation metrics.

        The method serializes the model after removing selected detailed fields from the
        output.

        Returns:
            A pretty-printed JSON string representation of the metrics.
        """
        data = self.model_dump()
        del data["per_example_details"]
        return json.dumps(data, indent=4)

    def save_to_disk(self, path: str | Path) -> None:
        """Save the evaluation metrics to a JSON file on disk.

        Args:
            path: Destination file path where the serialized metrics should be written.
        """
        if isinstance(path, str):
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            file.write(self.model_dump_json(indent=4))
