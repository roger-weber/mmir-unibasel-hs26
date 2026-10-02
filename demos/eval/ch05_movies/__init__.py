"""
Evaluation harness for the chapter 5 "Building Semantic Search" demo.

This package factors the five book pipelines (A-E) into reusable functions so the
offline trial runner and the eventual student notebook share one implementation.

Modules:
    pipelines  - index builders and the five retrieval pipelines (A-E)
    run_trial  - offline trial runner: executes every condition over the query set
                 and writes the JSON artifacts (raw_results, pool_to_judge, costs)
"""
