## Literature Review

### Research Area Overview
Work on LLM jailbreaks increasingly targets prompts that look nonsensical or low-probability to humans (gibberish suffixes, obfuscated tokens) while remaining fluent enough to bypass perplexity-based defenses. Recent papers explore both attack generation (suffixes, synonym substitution, prompt decomposition, multi-turn escalation) and defenses grounded in perplexity or token-level anomaly detection.

### Key Papers

#### WordGame: Efficient & Effective LLM Jailbreak via Simultaneous Obfuscation in Query and Response (2024)
- **Authors**: Zhang et al.
- **Source**: arXiv 2405.14023
- **Key Contribution**: Dual obfuscation—replace malicious tokens with word games in queries and seed benign game-related continuations in responses to hide intent.
- **Methodology**: Construct prompts with word-game substitutions plus auxiliary benign tasks so harmful completion is buried behind benign context.
- **Datasets Used**: Benchmarks across proprietary (Claude3, GPT-4) and open models.
- **Results**: Higher attack success rate vs existing suffix attacks; works across safety-tuned models with lower detection.
- **Code Available**: Not linked.
- **Relevance**: Shows how obfuscated language raises perplexity/rarity to evade safety filters while preserving attack intent.

#### FLRT: Fluent Student-Teacher Redteaming (2024)
- **Authors**: Thompson, Sklar
- **Source**: arXiv 2407.17447; code https://github.com/Confirm-Solutions/flrt
- **Key Contribution**: Distillation-based attack that keeps prompts fluent by penalizing perplexity and repetition.
- **Methodology**: Optimize prompts with GCG/BEAST variants plus multi-model perplexity penalties; token insert/swap/delete operations; universal fluent prompts.
- **Datasets Used**: AdvBench-style harmful instructions; tested on Llama-2/3, Phi-3, Vicuna.
- **Results**: >93% attack success on AdvBench with perplexity <33; universal prompt transfers across tasks/models.
- **Code Available**: Yes.
- **Relevance**: Demonstrates that fluent (low perplexity) adversarial prompts bypass gibberish detectors.

#### Token-Level Adversarial Prompt Detection Based on Perplexity Measures and Contextual Information (2023)
- **Authors**: Hu et al.
- **Source**: arXiv 2311.11509
- **Key Contribution**: Defense—detect adversarial tokens via per-token perplexity and contextual smoothing.
- **Methodology**: Use next-token probabilities to flag high-perplexity spans; optimization and PGM-based detection with heatmap outputs.
- **Datasets Used**: Adversarial prompts from discrete-optimization attacks.
- **Results**: Flags adversarial spans with very low p-values; visualizable token heatmaps.
- **Code Available**: Not specified.
- **Relevance**: Highlights perplexity as a practical signal against gibberish/nonsense prompts.

#### LatentBreak: Jailbreaking Large Language Models through Latent Space Feedback (2025)
- **Authors**: Mura et al.
- **Source**: arXiv 2510.08604
- **Key Contribution**: Low-perplexity jailbreak by synonym substitution guided by latent-space distance to benign prompts.
- **Methodology**: White-box attack that swaps tokens to minimize latent distance while retaining malicious intent; bypasses perplexity filters.
- **Datasets Used**: Benchmarks against safety-aligned models with perplexity filters.
- **Results**: Shorter, fluent prompts outperform suffix-based attacks when perplexity filtering is enabled.
- **Code Available**: Not linked.
- **Relevance**: Shows latent-guided paraphrasing as alternative to gibberish suffixes.

#### Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack (2024)
- **Authors**: Russinovich, Salem, Eldan
- **Source**: arXiv 2404.01833
- **Key Contribution**: Multi-turn benign-to-malicious escalation that leverages model’s own prior responses.
- **Methodology**: Start with abstract benign queries and progressively reference prior outputs to move toward harmful tasks; automated Crescendomation tool.
- **Datasets Used**: AdvBench subset, evaluations on ChatGPT, Gemini, Llama models.
- **Results**: 29–71% higher success than other methods on GPT-4/Gemini; works on multimodal models.
- **Code Available**: Tool integrated in PyRIT.
- **Relevance**: Illustrates conversational path attacks beyond single-shot gibberish prompts.

#### DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLMs Jailbreakers (2024)
- **Authors**: Li et al.
- **Source**: arXiv 2402.16914; code https://github.com/turningpoint-ai/DrAttack
- **Key Contribution**: Decompose malicious prompts into sub-prompts and reconstruct via in-context examples to reduce attention to harmful tokens.
- **Methodology**: Syntactic parsing to split prompts, reconstruct via benign ICL examples, synonym search for sub-prompts.
- **Datasets Used**: Benchmarks on GPT-4, Claude, Gemini, Vicuna.
- **Results**: Up to 80% success on GPT-4, +65% over prior SOTA.
- **Code Available**: Yes.
- **Relevance**: Attack avoids high-perplexity gibberish; exploits compositional prompt structure.

#### AmpleGCG-Plus: A Strong Generative Model of Adversarial Suffixes to Jailbreak LLMs (2024)
- **Authors**: Kumar et al.
- **Source**: arXiv 2410.22143
- **Key Contribution**: Generative model that outputs diverse gibberish suffixes quickly, improving over GCG/AmpleGCG.
- **Methodology**: Train generator on adversarial suffixes, tune training strategies to boost success in few attempts; targets OOD/gibberish space.
- **Datasets Used**: Suffix datasets collected from GCG runs; evaluated on open/closed models.
- **Results**: Up to +17% ASR in white-box Llama-2; triples ASR on GPT-4 in black-box.
- **Code Available**: Hugging Face model link provided.
- **Relevance**: Direct evidence that nonsense/gibberish tokens remain potent jailbreaks.

### Common Methodologies
- Perplexity-aware attacks: penalize or minimize perplexity (FLRT, LatentBreak) to evade detectors.
- Gibberish/adversarial suffix generation: GCG variants and AmpleGCG-Plus generate high-perplexity suffixes.
- Prompt restructuring: decomposition/reconstruction (DrAttack), synonym swaps (LatentBreak), query/response obfuscation (WordGame).
- Multi-turn escalation: Crescendo gradually steers conversation toward harmful content.
- Detection: token-level perplexity and graphical models (Hu et al.) as defense signal.

### Standard Baselines
- GCG/BEAST/AutoDAN suffix-based attacks.
- AdvBench prompts for evaluation.
- Safety-aligned Llama-2/3, GPT-4, Claude, Gemini as target models.
- Perplexity filters or refusal rates as defensive baselines.

### Evaluation Metrics
- Attack Success Rate (ASR) / compliance rate.
- Perplexity (global or token-level) to judge fluency and detectability.
- Query budget (#attempts) and prompt length.
- Transferability across models/tasks; universal prompt success.

### Datasets in the Literature
- AdvBench and augmented variants (used in FLRT, Crescendo).
- Harmful/harmless behavior lists (e.g., Anthropic-style) for evaluation.
- Custom collected adversarial suffix sets (AmpleGCG-Plus).
- Token-level detection examples crafted from discrete optimization attacks.

### Gaps and Opportunities
- Limited open datasets focused on gibberish/high-perplexity prompts paired with safety outcomes.
- Few standardized metrics for detectability beyond perplexity (e.g., entropy spikes, logit lens features).
- Multi-turn jailbreak defenses less explored compared to single-shot detection.
- Need for benchmarks combining fluent and gibberish attacks to test robustness of perplexity-based filters.

### Recommendations for Our Experiment
- **Recommended datasets**: Use `Baidicoot/augmented_advbench_v3_filtered` for diverse adversarial prompts; augment with `mlabonne/harmful_behaviors` for harmless/harmful pairs to test refusal vs jailbreak.
- **Recommended baselines**: Implement GCG/BEAST-style suffix attack and FLRT-style perplexity-penalized prompts; include DrAttack decomposition as compositional baseline.
- **Recommended metrics**: Attack success rate, prompt-level and token-level perplexity, refusal probability, and detection rate by a token-level perplexity detector.
- **Methodological considerations**: Test both gibberish and fluent attacks; include multi-turn Crescendo-style scripts; evaluate defenses using token-level perplexity heatmaps to see where nonsense tokens appear.
