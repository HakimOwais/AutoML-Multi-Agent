# AutoML-Agent: Multi-Agent LLM Framework for Full-Pipeline AutoML

Inspired by the research paper [AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML](https://arxiv.org/abs/2410.02958), this project leverages a multi-agent system to fully automate the machine learning pipeline. Using open source models for code generation and guided by well-defined agent prompts, our framework transforms user requirements into a deployable machine learning solution.

## Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
  - [Agent Manager](#agent-manager)
  - [Prompt Agent](#prompt-agent)
  - [Data Agent](#data-agent)
  - [Model Agent](#model-agent)
  - [Operation Agent](#operation-agent)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Overview

This project automates the machine learning lifecycle by utilizing multiple agents, each with its own distinct responsibilities. The system begins by receiving user requirements in JSON format and then proceeds through stages including data preprocessing, model optimization, and finally deployment—all orchestrated by specialized agents.

## Project Architecture

The framework is built around several core agents:

### Agent Manager
- **Role:** Oversees the overall project.
- **Function:** Receives user requirements in JSON format and creates high-level plans for the team.

### Prompt Agent
- **Role:** Acts as the intermediary between user instructions and machine-processable data.
- **Function:** Converts free-form user instructions into structured JSON data, ensuring consistency across the pipeline.

### Data Agent
- **Role:** Handles all data-related tasks.
- **Function:** Retrieves and preprocesses datasets, performs data augmentation, and generates visual insights to facilitate exploratory data analysis.

### Model Agent
- **Role:** Focuses on model development and optimization.
- **Function:** Identifies the best machine learning models for the given dataset, performs hyperparameter tuning, and conducts model profiling to enhance performance.

### Operation Agent
- **Role:** Manages the end-to-end deployment of the machine learning solution.
- **Function:** Orchestrates dataset handling, model optimization, training, and production-level code generation. It also evaluates model performance to ensure reliability in a production environment.

In addition, a set of carefully designed agent prompts guides the large language models to adhere to the project workflow, ensuring a coherent and efficient execution of tasks.

## Features

- **Modular Design:** Each agent specializes in a particular stage of the ML pipeline.
- **Automated Code Generation:** Utilizes open source LLM models to generate production-ready Python code.
- **End-to-End Pipeline:** From data acquisition to deployment, all phases are automated.
- **User-Friendly Interface:** Accepts user requirements in JSON format, simplifying integration and customization.
- **Extensible Architecture:** Designed to be easily expanded or modified to suit various ML project needs.

## Getting Started

### Prerequisites

- **Git:** To clone the repository.
- **Conda:** For managing the Python environment.
- **Python 3.10** (recommended).

### Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   cd <repository-folder>


## Setup Instructions

### 1. Create a Conda Environment
```bash
conda create -n automl-agent python=3.10
```

### 2. Activate the Conda Environment
```bash
conda activate automl-agent
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

---

## Usage

1. **Prepare Your Dataset**:  
   Place your dataset in the `data` folder. Copy the **relative path** to your dataset (e.g., `data/your_dataset.csv`) and update the path in `main.py`.

2. Setup your GROQ API KEY in .env file as:
   GROQ_API_KEY = "your api key"

3. **Run the Pipeline**:
   ```bash
   python main.py
   ```
   The system will prompt you to enter your project requirements in JSON format. Agents will then collaborate to build, optimize, and deploy your ML solution.

---

## Contributing

Contributions are welcome! Follow these steps:  
1. **Fork** the repository.  
2. Create a **new branch** for your feature/bugfix.  
3. **Commit** your changes.  
4. **Push** your branch and open a **pull request**.  

---

## Acknowledgments
- **Research Inspiration**: [AutoML-Agent: A Multi-Agent LLM Framework](https://arxiv.org/abs/2410.02958) .  
- Thanks to all open-source contributors and tools used in this project.  

---

## License  
This project is open source. See [LICENSE](LICENSE) for details.  

---

**Notes**:  
- Update the dataset path in `main.py` (e.g., `dataset_path = "data/your_dataset.csv"`).  
- Ensure the `LICENSE` file exists in your repository.  
