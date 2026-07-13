from .single_llm_with_web_search import SingleLLMWithWebSearchPipeline
from .single_llm_zero_shot import SingleLLMZeroShotPipeline
from .single_llm_with_pubmed_search import SingleLLMWithPubMedSearchPipeline
from .single_llm_with_pubmed_search_and_full_pmc_articles import (
    SingleLLMWithPubMedSearchAndFullPMCArticlesPipeline,
)
from .divide_and_conquer import DivideAndConquerPipeline
from .divide_and_conquer_with_fact_checker import (
    DivideAndConquerPipelineWithFactChecker,
)
from .divide_and_conquer_with_subagents_auto_spawning import (
    DivideAndConquerPipelineWithSubAgentsAutoSpawning,
)

PIPELINE_REGISTRY = {
    "single_llm_zero_shot": SingleLLMZeroShotPipeline,
    "single_llm_with_web_search": SingleLLMWithWebSearchPipeline,
    "single_llm_with_pubmed_search": SingleLLMWithPubMedSearchPipeline,
    "single_llm_with_pubmed_search_and_full_pmc_articles": SingleLLMWithPubMedSearchAndFullPMCArticlesPipeline,
    "divide_and_conquer": DivideAndConquerPipeline,
    "divide_and_conquer_with_fact_checker": DivideAndConquerPipelineWithFactChecker,
    "divide_and_conquer_with_subagents_auto_spawning": DivideAndConquerPipelineWithSubAgentsAutoSpawning,
}

__all__ = [
    "SingleLLMZeroShotPipeline",
    "SingleLLMWithWebSearchPipeline",
    "SingleLLMWithPubMedSearchPipeline",
    "SingleLLMWithPubMedSearchAndFullPMCArticlesPipeline",
    "DivideAndConquerPipeline",
    "DivideAndConquerPipelineWithFactChecker",
    "DivideAndConquerPipelineWithSubAgentsAutoSpawning",
    "PIPELINE_REGISTRY",
]
