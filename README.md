# NavBench: Probing Multimodal Large Language Models for Embodied Navigation

[🌐 Project Website]([https://YanyuanQiao.github.io/NavBench](https://navbench.github.io/)) | [📄 arXiv (coming soon)](#) | [🤖 Hugging Face (coming soon)](#)

NavBench is a benchmark designed to evaluate the **embodied navigation capabilities** of Multimodal Large Language Models (MLLMs) in **zero-shot settings**. It assesses both **comprehension** and **execution** skills within realistic indoor navigation tasks.

---

## 🌟 What's in NavBench?

NavBench consists of two components:

- **Navigation Comprehension**  
  3,200 QA pairs across:
  - Global Instruction Alignment
  - Temporal Progress Estimation
  - Local Observation-Action Reasoning

- **Step-by-Step Execution**  
  432 episodes across 72 Matterport3D scenes, stratified by:
  - Spatial Complexity
  - Cognitive Complexity
  - Execution Complexity

We also introduce an MLLM-to-robot pipeline for real-world deployment.
