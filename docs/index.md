---
layout: home

hero:
  name: "LLM Workings"
  text: "A hands-on exploration of how language models work"
  tagline: Understanding the internals of large language models - not just how to use them, but how they represent and process information
  actions:
    - theme: brand
      text: Get Started
      link: /guide/
    - theme: alt
      text: View Demos
      link: /demos/neural-network
    - theme: alt
      text: Development Docs
      link: /development/

features:
  - title: Architecture Fundamentals
    details: Explore transformer layers, attention mechanisms, residual streams, and MLPs to understand how language models are built
    icon: 🏗️
  - title: Representations & Learning
    details: Investigate how information from training corpora gets encoded into weights and activations through hands-on experiments
    icon: 🧠
  - title: Interpretability Techniques
    details: Apply probing, sparse autoencoders, activation patching, and steering vectors to understand what models learn
    icon: 🔬
  - title: Interactive Demos
    details: Step through neural network training with full visualization of forward passes, gradients, and weight updates
    icon: 🎮
---

## Current Status

This is an active learning project, building understanding through code and experimentation. The focus is on **smaller models** (GPT-2, Pythia, Llama-2-7B) that fit on a 12GB GPU, allowing for local development and hands-on exploration.

### What's Been Built

- **Interactive Neural Network Visualizer**: Step-by-step visualization of XOR training with full mathematical detail
- **Python Neural Network**: From-scratch implementation for understanding fundamentals
- **Custom Development Workflow**: Claude Code commands and agents for TDD-driven learning

[Explore what we've built →](/guide/built)

## Learning Method: Claude as Tutor

This project uses Claude not just as a coding assistant, but as a **Socratic tutor** - prioritizing understanding over answers.

### How It Works

Rather than lectures and explanations, the approach is:

- **Ask questions first** - "What do you already know about attention mechanisms?"
- **Guide discovery** - Lead toward insights rather than stating them directly
- **Encourage prediction** - "What do you think will happen when we change this parameter?"
- **Embrace confusion** - Confusion signals you're at the edge of understanding; we explore it together
- **Experiment before explaining** - Run code, observe results, then build theory from what we see
- **Verify understanding** - "Explain this back to me in your own words"

### Why This Matters

Traditional AI assistance provides answers. This approach builds **intuition** and **mental models**:

> "The residual stream is where information flows between layers..."
> ❌ **Too abstract**

> "You mentioned residuals - what do you already know about skip connections? What problem do you think they solve?"
> ✅ **Builds on existing knowledge**

The goal isn't just working code - it's understanding deep enough to ask the next question yourself.

[Read the full tutoring guidelines →](https://github.com/jflournoy/llm-workings/blob/main/CLAUDE.md)

## Technical Approach

- **Local development** - All experiments run on local hardware (12GB GPU constraint)
- **Practical focus** - Hands-on code over theory alone
- **Test-driven learning** - TDD methodology for building reliable tools
- **Learning in public** - Documenting insights and patterns as we go

## Resources & Inspiration

- [Anthropic's interpretability research](https://www.anthropic.com/research#interpretability)
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) - mechanistic interpretability library
- [ARENA curriculum](https://www.arena.education/) - alignment research training
- NeurIPS proceedings and workshops
