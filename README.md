# Hevy_AI

AI-powered fitness and health advisor application built with LangChain and LangGraph.

## Overview

Hevy AI is a sophisticated fitness planning system that uses Large Language Models to:
- Interpret user fitness goals from natural language
- Manage user profiles and preferences
- Generate personalized fitness plans
- Orchestrate planning workflows using LangGraph

## Stage 1: Core Orchestration

This implementation includes:

- **Goal Description Tool**: Extracts structured goal information from natural language
- **Goal Interpreter**: LangChain application for conversational goal clarification
- **Planning Graph**: LangGraph state machine for orchestrating planning workflows
- **Profile Management**: JSON-based user profile storage and management

## Project Structure

```
Hevy_AI/
├── src/
│   ├── schemas/          # Pydantic schemas (UserGoal)
│   ├── tools/            # LangChain tools (Goal Description Tool)
│   ├── agents/           # LangChain agents (Goal Interpreter)
│   ├── graph/            # LangGraph state machines
│   └── profiles/         # User profile management
├── config/               # Configuration and LLM setup
├── data/                 # Data storage (profiles)
└── example_usage.py      # Example script
```

## Setup

### 1. Virtual Environment

The project uses a Python virtual environment. To activate it:

```bash
source aiagent-env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000
```

## Usage

### Basic Example

Run the example script to see the system in action:

```bash
python example_usage.py
```

### Programmatic Usage

```python
from src.graph.planning_graph import PlanningGraph

# Initialize the planning graph
graph = PlanningGraph()

# Process user input
user_input = "I want to lose 10kg in 3 months with home workouts"
result = graph.run(user_input, user_id="user123")

# Access results
goal = result["extracted_goal"]
plan = result["plan"]
profile = result["user_profile"]
```

### Goal Extraction

```python
from src.tools.goal_description_tool import GoalDescriptionTool

tool = GoalDescriptionTool()
goal = tool.extract_goal("I want to build muscle in 6 months")
print(goal.goal_type)  # muscle_gain
print(goal.target_metric)  # build muscle
```

### Profile Management

```python
from src.profiles.profile_manager import ProfileManager

manager = ProfileManager()
profile = manager.get_or_create_profile("user123")
profile = manager.update_profile_with_goal("user123", goal)
```

## Features

### Goal Types

Supported goal types:
- `weight_loss`: Weight reduction goals
- `muscle_gain`: Muscle building objectives
- `endurance`: Endurance and cardio goals
- `general_wellness`: General health and wellness

### User Status

Supported status levels:
- `beginner`: Just starting out
- `intermediate`: Some experience
- `advanced`: Experienced practitioner
- `sedentary`: Currently inactive
- `active`: Regularly active

## Requirements

- Python 3.12+
- OpenAI API key (or compatible LLM provider)
- Dependencies listed in `requirements.txt`

## Architecture

### Planning Graph Workflow

1. **Extract Goal**: Uses Goal Description Tool to parse user input
2. **Load Profile**: Retrieves or creates user profile
3. **Generate Plan**: Creates high-level plan based on goal and profile
4. **Validate**: Validates plan completeness
5. **Refine** (if needed): Improves plan quality
6. **Save Profile**: Updates user profile with new goal

### Components

- **UserGoal Schema**: Structured representation of fitness objectives
- **GoalDescriptionTool**: LangChain tool for goal extraction
- **GoalInterpreter**: Conversational agent for goal clarification
- **PlanningGraph**: Orchestrates the complete workflow
- **ProfileManager**: Manages user data persistence

## Development

The codebase is organized for extensibility:
- Schemas can be extended with new fields
- Tools can be added for additional capabilities
- Graph nodes can be modified or extended
- Profile storage can be migrated to databases

## License

See LICENSE file for details.
