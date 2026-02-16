"""
Shared configuration for disaggregated inference experiments.

Everything here is used by baseline_inference.py, prefill_node.py, and decode_node.py
so all three use the EXACT same prompt, model, and parameters.
"""

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
MAX_NEW_TOKENS = 256
NUM_RUNS = 3
DTYPE = "float16"

# ~512 token prompt — a long passage to make prefill compute-heavy
# and generate a meaningful KV cache (~64 MB at FP16).
LONG_PROMPT = """The development of artificial intelligence has been one of the most transformative technological achievements in human history. From the early days of symbolic AI in the 1950s, when researchers at Dartmouth College first coined the term, to the modern era of deep learning and large language models, the field has undergone several paradigm shifts that have fundamentally changed how we think about computation, intelligence, and the relationship between humans and machines.

The first wave of AI research focused on symbolic reasoning and expert systems. Researchers believed that intelligence could be captured through formal logic and rule-based systems. This approach led to significant advances in areas like theorem proving, chess playing, and medical diagnosis. However, these systems were brittle — they could only operate within narrowly defined domains and failed catastrophically when faced with ambiguity or novel situations.

The second wave brought machine learning to the forefront. Instead of hand-coding rules, researchers developed algorithms that could learn patterns from data. Decision trees, support vector machines, and random forests became the workhorses of practical AI applications. These methods excelled at classification and regression tasks but struggled with unstructured data like images, audio, and natural language.

The third wave — deep learning — changed everything. Inspired by the structure of biological neural networks, researchers developed multi-layer artificial neural networks that could automatically learn hierarchical representations of data. The breakthrough came in 2012 when AlexNet demonstrated that deep convolutional neural networks could dramatically outperform traditional computer vision methods on the ImageNet benchmark. This was followed by rapid advances in natural language processing, with recurrent neural networks and attention mechanisms enabling machines to process sequential data with unprecedented accuracy.

The transformer architecture, introduced in 2017 by Vaswani et al. in the landmark paper "Attention Is All You Need," represented perhaps the most significant architectural innovation in the history of deep learning. By replacing recurrence with self-attention mechanisms, transformers could process entire sequences in parallel, enabling both faster training and better modeling of long-range dependencies. This architecture became the foundation for a new generation of language models that would fundamentally change the AI landscape.

The scaling laws discovered by researchers at OpenAI and DeepMind revealed a remarkable property of transformer-based language models: their performance improves predictably with increases in model size, dataset size, and compute budget. This finding triggered an unprecedented race to build ever-larger models, from GPT-2's 1.5 billion parameters to GPT-3's 175 billion, and eventually to models with hundreds of billions or even trillions of parameters.

The emergence of mixture-of-experts architectures like DeepSeek-MoE and Mixtral demonstrated that not all parameters need to be activated for every input. By routing different tokens to different expert subnetworks, these models achieve the capacity of much larger dense models while requiring only a fraction of the compute per forward pass. This insight has profound implications for the future of model scaling and efficiency.

Today, we stand at the intersection of multiple converging trends: increasingly powerful hardware accelerators, novel model architectures, sophisticated training techniques, and a growing understanding of how to efficiently deploy and serve these models at scale. The challenge is no longer just building larger models — it is building systems that can orchestrate computation across heterogeneous hardware, optimize for different phases of inference, and deliver real-time performance at reasonable cost.

This brings us to the fundamental question that drives our research: can we build a runtime that intelligently splits the prefill and decode phases of language model inference across different processors, each optimized for its respective workload? The prefill phase is compute-bound, benefiting from high TFLOPS. The decode phase is memory-bound, benefiting from high memory bandwidth. By matching each phase to the hardware that excels at it, we hypothesize that disaggregated inference can achieve higher throughput than any single machine alone."""
