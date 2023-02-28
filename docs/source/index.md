---
hide-toc: true
firstpage:
lastpage:
---

# Decision Transformer Interpretability
**This documentation is in progress, we haven't written most of the doc strings.**

In this project, we intend to apply a mathematical framework for transformer circuits to study transformer models trained on reinforcement tasks to create a simpler context than large language models for understanding high-level concepts, such as goal-directedness, in transformer architecture.

 To begin, we hope to train a small transformer model (maximum of two layers, implemented via the Transformer Lens interpretability library) to solve MiniGrid RL tasks. Once we achieve good performance, we will attempt to find mechanistic explanations for the agent's behaviour, such as how it decides whether to move toward the door or the key.

Interpretability of such circuits may be a valuable step towards understanding high-level concepts such as deception or goal-directedness in transformer circuits.

Future work may include attempting to manually edit decision transformers to modify goals and behavior.

## Write Up

You can find an initial technical report for this project [here](https://www.lesswrong.com/posts/bBuBDJBYHt39Q5zZy/decision-transformer-interpretability).

```{toctree}
:caption: 'Contents:'

```
```{toctree}
:caption: 'Getting Started'
:hidden:
content/installation
```
```{toctree}
:caption: 'Code'
:hidden:

modules.rst
```
```{toctree}
:caption: 'References'
:hidden:
content/references

```
