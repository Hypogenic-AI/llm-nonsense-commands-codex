## Project Overview
Research workspace to test whether LLMs can interpret and explain high-perplexity (gibberish) jailbreak prompts. We compared baseline harmful prompts, gibberish suffixes, and fluent bypass framings using a local instruction-tuned model.

## Key Findings
- Gibberish prompts remained interpretable (cosine ≈0.83) and more compliant (ASR 0.67) than baseline (ASR 0.42).
- Fluent bypass prompts drove the highest compliance (ASR 0.92) with slightly lower explanation similarity (0.76).
- Perplexity alone was not predictive of explainability (corr ≈0.14); gibberish raised perplexity 5× without obscuring intent.

Full details: see `REPORT.md`.

## Reproduction
1. Environment: `uv venv && source .venv/bin/activate && uv sync`.
2. Run experiments (defaults: 8 AdvBench + 4 harmful behavior prompts):  
   `python src/run_experiments.py`  
   - Override sample sizes: `SAMPLE_SIZE=12 CONTROL_SIZE=6 python src/run_experiments.py`  
   - Override generation model: `GEN_MODEL="Qwen/Qwen2.5-0.5B-Instruct"` env var.
3. Analyze results and generate plots:  
   `python src/analyze_results.py`

Outputs:
- Raw metrics: `results/metrics.jsonl`
- Aggregates: `results/summary.json`, `results/aggregated_metrics.*`, `results/analysis.json`
- Plots: `results/plots/`

## File Structure
- `planning.md` – detailed research plan.
- `REPORT.md` – full report with results and discussion.
- `src/run_experiments.py` – prompt generation, model calls, metric logging.
- `src/analyze_results.py` – aggregation, statistics, and plots.
- `datasets/`, `code/`, `papers/` – provided resources (unchanged).
