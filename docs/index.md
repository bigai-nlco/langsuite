<h1 align="center" id="title"> 🏘️ LangSuit⋅E </h1>

<h3 align="center">Controlling, Planning, and Interacting with Large Language Models in Embodied Text Environments</h3>

<p align="center">
    <a href="https://opensource.org/licenses/MIT">
        <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </a>
    <a href="https://www.python.org/downloads/release/python-380/">
        <img alt="Documentation" src="https://img.shields.io/badge/Python-3.8+-blue.svg">
    </a>
</p>

![](https://github.com/bigai-nlco/langsuite/assets/teaser.png)


 **LangSuit⋅E** is a systematic and simulation-free testbed for evaluating embodied capabilities of large language models (LLMs) across different tasks in embodied textual worlds. The highlighted features include:

 - **Simluation-Free Embodied Environments**: The testbed provides a general simulation-free textual world that supports most embodied tasks, including navigation, manipulation, communications. The environment is based on [Gymnasium](https://gymnasium.farama.org/index.html) and inherits the design patterns.
 - **Embodied Observations and Actions**: All agents' observations are designed to be embodied with custimisible `max_view_distance`, `max_manipulate_distance`, `focal_length`, *etc*.
 - **Customizible Embodied Agents**: The agents in LangSuit⋅E are fully-customizable *w.r.t* their action spaces and communicative capabilities, *i.e.*, one can easily adapt the communication and acting strategy from one task to another.
 - **Multi-agent Coopearation**: The testbed supports planning, acting and communication among multiple agents, where each agents can be custimized to have different configurations.
 - **Human-agent Communication**: Besides communication between agents, the testbed supports communication and cooperation between human and agents.
 - **Full support to [LangChain](https://www.langchain.com/) library**: The LangSuitE testbed supports full usage of API language models, Open-source language models, tool usages, Chain-of-Thought (CoT) strategies, *etc.*.




## Table of Contents
- [Overview](#title)
- [Benchmark and Dataset](#benchmark-and-dataset)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Quick Start](#quick-start-commandline-interface-default)
  - [Task Configuration](#task-configuration)
  - [Prompt Template](#prompt-template)
- [Citation](#citation)


## 📦 Benchmark and Dataset

We form a benchmark by adating from existing annotations of simluated embodied engines, a by-product benefit of pursuing a general textual embodied world. Below showcases 6 representative embodied tasks, with variants of the number of rooms, the number of agents, the action spaces of agents (whether they can communicate with each other or ask humans).

<div align="center">
<table>
  <tr>
    <th>Task</th>
    <th>Simulator</th>
    <th># of Scenes</th>
    <th># of Tasks</th>
    <th># of Actions</th>
    <th>Multi-Room</th>
    <th>Multi-Agent</th>
    <th>Communicative</th>
  </tr>
  <tr>
    <td><a href="https://github.com/Farama-Foundation/Minigrid">BabyAI</a></td>
    <td>Mini Grid</td>
    <td align="center">105</td>
    <td align="center">500</td>
    <td align="center">6</td>
    <td align="center">&#10003;</td>
    <td align="center">&#10007;</td>
    <td align="center">&#10007;</td>
  </tr>
  <tr>
    <td><a href="https://github.com/allenai/ai2thor-rearrangement">Rearrange</a></td>
    <td>AI2Thor</td>
    <td align="center">120</td>
    <td align="center">6,000</td>
    <td align="center">8</td>
    <td align="center">&#10007;</td>
    <td align="center">&#10007;</td>
    <td align="center">&#10007;</td>
  </tr>
  <tr>
    <td><a href="https://github.com/danielgordon10/thor-iqa-cvpr-2018">IQA</a></td>
    <td>AI2Thor</td>
    <td align="center">30</td>
    <td align="center">1,920</td>
    <td align="center">5</td>
    <td align="center">&#10007;</td>
    <td align="center">&#10007;</td>
    <td align="center">&#10003;</td>
  </tr>
  <tr>
    <td><a href="https://github.com/askforalfred/alfred">ALFred</a></td>
    <td>AI2Thor</td>
    <td align="center">120</td>
    <td align="center">8,055</td>
    <td align="center">12</td>
    <td align="center">&#10007;</td>
    <td align="center">&#10007;</td>
    <td align="center">&#10007;</td>
  </tr>
  <tr>
    <td><a href="https://github.com/alexa/teach">TEACh</a></td>
    <td>AI2Thor</td>
    <td align="center">120</td>
    <td align="center">3,215</td>
    <td align="center">13</td>
    <td align="center">&#10007;</td>
    <td align="center">&#10003;</td>
    <td align="center">&#10003;</td>
  </tr>
  <tr>
    <td><a href="https://vis-www.cs.umass.edu/Co-LLM-Agents/">CWAH</a></td>
    <td>Virtual Home</td>
    <td align="center">2</td>
    <td align="center">50</td>
    <td align="center">6</td>
    <td align="center">&#10003;</td>
    <td align="center">&#10003;</td>
    <td align="center">&#10003;</td>
  </tr>


</table>

</div>

## Citation

```bibtex
@misc{langsuite2023,
  author    = {Zilong Zheng, Mengmeng Wang, Zixia Jia, Baichen Tong, Jiasheng Gu, Song-Chun Zhu},
  title     = {LangSuit⋅E: Controlling, Planning, and Interacting with Large Language Models in Embodied Text Environments},
  year      = {2023},
  publisher = {GitHub},
  url       = {https://github.com/bigai-nlco/langsuite}
}
```

For any questions and issues, please contact [nlp@bigai.ai](mailto:nlp@bigai.ai).
