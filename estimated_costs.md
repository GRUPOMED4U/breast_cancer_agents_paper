# OpenAI GPT 5.5

Total estimated cost: 399.62 USD

| pipeline_id                                         |   token_cost_on_inference |   total_token_cost_on_inference (USD) | actual_cost |
|:----------------------------------------------------|--------------------------:|-------------------------------------: |------------:|
| divide_and_conquer                                  |                  0.32     |                        23.16          |             |
| divide_and_conquer_with_fact_checker                |                  1.33     |                        96.04          |             |
| divide_and_conquer_with_subagents_auto_spawning     |                  1.17     |                        84.45          | 25.81       |
| single_llm_with_pubmed_search                       |                  0.77     |                        55.93          | 20.41       |
| single_llm_with_pubmed_search_and_full_pmc_articles |                  1.23     |                        89.07          |             |
| single_llm_with_web_search                          |                  0.60     |                        43.44          |  9.05       |
| single_llm_zero_shot                                |                  0.10     |                         7.50          |  3.67       |

# Anthropic Claude Opus 4.8

Total estimated cost: 365.10 USD

| pipeline_id                                         |   token_cost_on_inference |   total_token_cost_on_inference (USD) |actual_cost  |
|:----------------------------------------------------|--------------------------:|--------------------------------------:|------------:|
| divide_and_conquer                                  |                  0.27     |                        19.83          |             |
| divide_and_conquer_with_fact_checker                |                  1.20     |                        86.54          |             |
| divide_and_conquer_with_subagents_auto_spawning     |                  1.05     |                        76.17          | 80.29       |
| single_llm_with_pubmed_search                       |                  0.72     |                        52.30          |             |
| single_llm_with_pubmed_search_and_full_pmc_articles |                  1.17     |                        84.59          |             |
| single_llm_with_web_search                          |                  0.54     |                        39.37          |             |
| single_llm_zero_shot                                |                  0.08     |                         6.28          |  2.36       |

*cost estimates are based on gemini-3.5-flash token usage on each pipeline