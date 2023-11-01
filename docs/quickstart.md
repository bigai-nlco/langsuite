## 🛠 Getting Started
### Installation
1. Clone this repository
```
git clone https://github.com/langsuite/langsuite.git
cd langsuite
```
2. Create a conda environment with `Python3.8+` and install python requirements
```
conda create -n langsuite python=3.8
conda activate langsuite
pip install -e .
```
3. Export your `OPENAI_API_KEY` by
```bash
export OPENAI_API_KEY="your_api_key_here"
```

4. Download task dataset by
```bash
bash ./data/download.sh <data name>
```
Currently supported datasets include: `alfred`, `babyai`, `cwah`, `iqa`, `rearrange`.


### Quick Start: CommandLine Interface (Default)

```bash
langsuite task <config-file.yml>
```

### Quick Start: Interactive Web UI
1. Start langsuite server

```bash
langsuite serve <config-file.yml>
```

2. Start webui

```bash
langsuite webui
```
The user inferface will run on http://localhost:8501/

## Task Configuration
```yaml
task: ExampleTask:Procthor2DEnv
template: ./langsuite/envs/ai2thor/templates/procthor_rearrange.json

env:
  type: Procthor2DEnv

world:
  type: ProcTHORWorld
  id: test_world
  grid_size: 0.25
  asset_path: ./data/asset-database.json
  metadata_path: ./data/ai2thor-object-metadata.json
  receptacles_path: ./data/receptacles.json

agents:
  - type: ChatGPTAgent
    position: 'random'
    inventory_capacity: 1
    focal_length: 10
    max_manipulate_distance: 1
    max_view_distance: 2
    step_size: 0.25
    llm:
      llm_type: ChatOpenAI
```

## Prompt Template
```json
{
    "intro": {
        "default": [
            "You are an autonomous intelligent agent tasked with navigating a vitual home. You will be given a household task. These tasks will be accomplished through the use of specific actions you can issue. [...]"
        ]
    },
    "example": {
        "default": [
            "Task: go to the red box. \nObs:You can see a blue key in front of you; You can see a red box on your right. \nManipulable object: A blue key.\n>Act: turn_right."
        ]
    },
    "InvalidAction": {
        "failure.invalidObjectName": [
            "Feedback: Action failed. There is no the object \"{object}\" in your view space. Please operate the object in sight.\nObs: {observation}"
        ],
        ...
    },
    ...
}

```
