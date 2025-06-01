# 🧭 NavBench: Probing Multimodal Large Language Models for Embodied Navigation

[🌐 Project Website](https://navbench.github.io) | [📄 arXiv (coming soon)](#) | [🤖 Hugging Face (coming soon)](#)

**NavBench** is a benchmark designed to evaluate the **embodied navigation abilities** of Multimodal Large Language Models (MLLMs) in **zero-shot settings**. It focuses on both **comprehension** (understanding instructions and visual context) and **execution** (making navigation decisions), providing a fine-grained assessment of MLLM capabilities in realistic indoor environments.

---

## 🌟 Key Features

NavBench consists of two major components:

### 🧠 Navigation Comprehension

This component evaluates whether the model can understand and reason about navigation behaviors across three levels:

- **Global Instruction Alignment**  
  Given a full trajectory and multiple candidate instructions, the model selects the one that best aligns with the path.

- **Temporal Progress Estimation**  
  Given a partial trajectory and segmented instructions, the model identifies which sub-instruction has just been completed.

- **Local Observation–Action Reasoning**  
  The model reasons about action consequences.  
  (1) *Future-Observation Prediction*: predict the resulting next view.  
  (2) *Future-Action Prediction*: predict the action connecting two consecutive views.

### 🚶 Step-by-Step Execution

We evaluate MLLMs’ ability to make step-by-step navigation decisions in a zero-shot setting within the Matterport3D simulator.  
Tasks are stratified into **easy**, **medium**, and **hard** levels across three difficulty dimensions: **spatial**, **cognitive**, and **execution**.

---

## 🤖 Real-World Deployment

We demonstrate the integration of MLLMs into an **MLLM-to-Robot** pipeline, showing their potential for real-world instruction-following tasks.

---

## 🧪 Evaluation & Resources

- **Evaluation Code** – [Coming soon](#)  
- **Hugging Face Dataset** – [Coming soon](#)

---

## 📎 Citation

Coming soon.
