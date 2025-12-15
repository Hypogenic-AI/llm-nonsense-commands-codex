## Resources Catalog

### Summary
Papers, datasets, and code collected for studying whether LLMs understand or can be coerced by nonsense/high-perplexity prompts.

### Papers
Total papers downloaded: 7

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| WordGame: Efficient & Effective LLM Jailbreak via Simultaneous Obfuscation | Zhang et al. | 2024 | papers/2405.14023_wordgame.pdf | Dual obfuscation in query/response to bypass safety |
| FLRT: Fluent Student-Teacher Redteaming | Thompson, Sklar | 2024 | papers/2407.17447_fluent_student_teacher_redteaming.pdf | Low-perplexity optimized prompts; fluent jailbreaks |
| Token-Level Adversarial Prompt Detection Based on Perplexity | Hu et al. | 2023 | papers/2311.11509_token_level_adversarial_prompt_detection.pdf | Per-token perplexity defense |
| LatentBreak: Jailbreaking LLMs through Latent Space Feedback | Mura et al. | 2025 | papers/2510.08604_latentbreak.pdf | Synonym substitution to evade perplexity filters |
| The Crescendo Multi-Turn LLM Jailbreak Attack | Russinovich et al. | 2024 | papers/2404.01833_crescendo_multiturn_jailbreak.pdf | Multi-turn benign-to-malicious escalation |
| DrAttack: Prompt Decomposition and Reconstruction | Li et al. | 2024 | papers/2402.16914_drattack.pdf | Decompose/reconstruct malicious prompts |
| AmpleGCG-Plus: Generative Model of Adversarial Suffixes | Kumar et al. | 2024 | papers/2410.22143_amplegcg_plus.pdf | Fast gibberish suffix generation |

See papers/README.md for details.

### Datasets
Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Harmful Behaviors | mlabonne/harmful_behaviors | 416 train / 104 test | Jailbreak/refusal evaluation | datasets/harmful_behaviors/ | MIT license; harmless vs harmful prompts |
| Augmented AdvBench v3 (filtered) | Baidicoot/augmented_advbench_v3_filtered | 5,230 train | Adversarial prompt set | datasets/augmented_advbench_v3_filtered/ | Diverse augmented jailbreak prompts |

See datasets/README.md for download and loading instructions.

### Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| flrt | https://github.com/Confirm-Solutions/flrt | Fluent perplexity-controlled jailbreak generation | code/flrt/ | Requires PyTorch; scripts for AdvBench |
| DrAttack | https://github.com/turningpoint-ai/DrAttack | Prompt decomposition/reconstruction jailbreak | code/DrAttack/ | Uses HF transformers; includes attack pipeline |

See code/README.md for summaries.

### Resource Gathering Notes
- **Search Strategy**: Queried arXiv titles for jailbreak-related keywords; used Hugging Face API for adversarial datasets; targeted GitHub repos cited in papers.
- **Selection Criteria**: Focused on attacks involving gibberish/perplexity manipulation or detection; preferred 2023–2025 papers with empirical evaluations or code.
- **Challenges Encountered**: Semantic Scholar API rate limits; avoided by using arXiv searches and HF API instead. `wget` missing libcrypto—switched to `curl`.
- **Gaps and Workarounds**: Few open datasets specifically for gibberish prompts; chose AdvBench augmentations and harmful-behavior lists as proxies.

### Recommendations for Experiment Design
1. **Primary datasets**: `augmented_advbench_v3_filtered` for adversarial prompts; `harmful_behaviors` for refusal baselines and negative controls.
2. **Baseline methods**: Implement GCG/BEAST (suffix), FLRT-style perplexity-penalized optimization, DrAttack decomposition, and Crescendo-style multi-turn scripts.
3. **Evaluation metrics**: Attack success rate, prompt/token perplexity, refusal rate, and detection rate of token-level perplexity detector.
4. **Code to adapt/reuse**: Start from `code/flrt` for fluent attacks and `code/DrAttack` for compositional/fragmented prompts; integrate datasets above for benchmarking.
