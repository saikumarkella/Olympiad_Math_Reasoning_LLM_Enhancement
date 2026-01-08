# Olympiad Math Reasoning LLM Enhancement

## Abstract

This project focuses on enhancing Large Language Models (LLMs) to excel in Olympiad-level mathematical reasoning. Olympiad mathematics presents unique challenges, requiring not only computational proficiency but also creative problem-solving, logical deduction, and the ability to handle abstract concepts that go beyond standard arithmetic and algebraic manipulations.


### Problem Statement
Current state-of-the-art LLMs demonstrate impressive capabilities in natural language processing and basic mathematical computations. However, they often falter when confronted with the intricate, multi-step reasoning required for International Mathematical Olympiad (IMO) problems or similar competitive mathematics challenges. These problems demand:
- Deep understanding of mathematical concepts
- Ability to construct novel proofs
- Logical reasoning across multiple domains
- Handling of abstract and counterintuitive scenarios

### Methodology
There are 2 methodologies are using to solve this .
1. Test-Time Compute / Inferece-time Reasoning.
2. Post Training ( using SFT + Reinforcement Learning ).



## Test-Time Compute / Inference-time Reasoning.
- Proposed  methods:
    1. Chain of Thoughts (CoT)
    2. Tool Integrated Reasoning (TIR)

- Optimization of Speed Efficincy.
    - Applying optimization techniques like 
        - KV-Cache
        - Quantization (using BitsandBytes)
        - Kernel optimizations like `flash-attention-2`


