# LLM Agent for Fitness Goal Extraction - Complete Guide

## 🎯 What is This Process?

This code creates an **AI-powered fitness goal assistant**. It takes what a user says in plain English (like "I want to lose 10kg in 3 months") and turns it into structured, organized data that a computer can understand and use to create personalized fitness plans.

**Real-world use:** Imagine you're building a fitness app. Users type in their goals casually, but your app needs structured data to create workout plans, track progress, and give personalized advice. This code does that conversion automatically!

---

## 📚 Code Breakdown (Cell by Cell)

### **Cell 1: Importing the Tools** 🛠️

```python
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
```

**What it does:**
- **`Enum`**: Creates a list of fixed options (like a dropdown menu) - prevents typos and ensures valid choices
- **`typing`**: Helps Python understand what type of data we're working with (text, numbers, lists, etc.)
- **`pydantic`**: A library that validates data - ensures information follows a specific structure and catches errors
- **`langchain`**: A framework for building AI applications - provides tools to talk to AI models easily
- **`ChatOpenAI`**: Connects to OpenAI's AI models (like ChatGPT)

**Why it matters:** These are like the building blocks and tools needed to build our AI fitness assistant.

---

### **Cell 2: Defining Goal Types** 🏋️

```python
class GoalType(str, Enum):
    """Types of fitness goals."""

    WEIGHT_LOSS = "weight_loss"
    MUSCLE_GAIN = "muscle_gain"
    ENDURANCE = "endurance"
    GENERAL_WELLNESS = "general_wellness"
```

**What it does:**
- Creates a list of **4 possible fitness goal categories**
- Each goal type has a code name (like `WEIGHT_LOSS`) and a value (like `"weight_loss"`)
- Think of it like a menu with 4 options: you can only pick from these 4

**Why it matters:** Standardizes goal types so the AI always categorizes goals correctly. Instead of users saying "lose weight", "slim down", "get thinner" (all meaning the same thing), everything gets mapped to `weight_loss`.

---

### **Cell 3: Defining User Status Levels** 📊

```python
class UserStatus(str, Enum):
    """User fitness status levels."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    SEDENTARY = "sedentary"
    ACTIVE = "active"
```

**What it does:**
- Creates **5 possible fitness levels** a user can be at
- Similar to the GoalType, this creates standardized categories

**Why it matters:** Helps the AI understand the user's starting point. A workout plan for a beginner will be very different from one for an advanced athlete!

---

### **Cell 4: The UserGoal Model (The Most Important Part!)** 📋

```python
class UserGoal(BaseModel):
    goal_type: GoalType
    target_metric: str
    timeframe: str
    constraints: List[str]
    preferences: List[str]
    current_status: UserStatus
    additional_context: Optional[Dict[str, str]]
```

**What it does:**
- This is like a **form** or **template** that captures all information about a user's fitness goal
- `BaseModel` means it's a validated data structure - if data doesn't fit, it throws an error

**Breaking down each field:**
1. **`goal_type`**: Which of the 4 goal types (weight_loss, muscle_gain, etc.)
2. **`target_metric`**: The specific target (e.g., "lose 10kg", "run 5km", "gain upper body strength")
3. **`timeframe`**: How long they have (e.g., "3 months", "6 weeks")
4. **`constraints`**: Limitations (e.g., ["knee injury", "no gym access", "30 mins/day"])
5. **`preferences`**: What they like/dislike (e.g., ["enjoys yoga", "hates running"])
6. **`current_status`**: Their fitness level (beginner, intermediate, etc.)
7. **`additional_context`**: Any extra info (optional - can be empty)

**Example of what this looks like filled out:**
```python
{
    "goal_type": "weight_loss",
    "target_metric": "lose 10kg",
    "timeframe": "3 months",
    "constraints": ["knee injury", "30 mins/day"],
    "preferences": ["enjoys strength training"],
    "current_status": "beginner",
    "additional_context": {"work_schedule": "9-5 weekdays"}
}
```

**Why it matters:** This structured format is what the rest of the application needs to create personalized plans!

---

### **Cell 5: The AI Tool (The Magic Happens Here!)** ✨

```python
@tool
def goal_description_tool(user_input: str) -> dict:
```

**What `@tool` does:**
- This is a **decorator** - it wraps the function with extra features
- Makes the function callable by AI agents and other LangChain components
- Think of it like adding special powers to a regular function

**What the function does:**
1. Takes natural language input (like "I want to lose 10kg in 3 months")
2. Uses an AI model (GPT-5-nano) to understand it
3. Extracts structured information
4. Returns it as a dictionary matching the UserGoal structure

**The AI Prompt:**
The code creates instructions for the AI:
- "You are an expert fitness advisor"
- "Extract structured information from user statements"
- Lists exactly what to extract (goal type, target, timeframe, etc.)

**Error Handling:**
- If the first method fails, it tries a backup method (using a parser)
- This makes the code more robust - if one way doesn't work, it tries another

**Why it matters:** This is where human language becomes computer-readable data!

---

### **Cell 6: The Wrapper Class (Making It Easier to Use)** 🎁

```python
class GoalDescriptionTool:
```

**What it does:**
- Wraps the tool function in a class for easier, safer use
- Provides two methods:
  1. **`extract_goal()`**: Extracts goal and validates it - throws errors if something's wrong
  2. **`extract_goal_safe()`**: Extracts goal but returns `None` if it fails (instead of crashing)

**Why it matters:** Makes the code easier to use in other parts of your application and handles errors gracefully.

---

## 🔄 How The Process Works (Step by Step)

1. **User Input**: "I want to lose 10kg in 3 months, but I only have 30 minutes a day and I'm a beginner"

2. **AI Processing**: The `goal_description_tool` sends this to an AI model with instructions to extract structured data

3. **Structured Output**: The AI returns:
   ```python
   {
       "goal_type": "weight_loss",
       "target_metric": "lose 10kg",
       "timeframe": "3 months",
       "constraints": ["30 mins/day"],
       "preferences": [],
       "current_status": "beginner",
       "additional_context": None
   }
   ```

4. **Validation**: The code checks that all required fields are present and valid

5. **Ready to Use**: Now your application can use this structured data to create workout plans!

---

## 🚀 Next Steps: What to Build Next

Here's code for the next logical steps in building a complete fitness goal system:

### **Step 1: Create a Workout Plan Generator**

```python
from datetime import datetime, timedelta
from typing import List

class WorkoutPlan(BaseModel):
    """Structured workout plan based on user goals."""
    weekly_schedule: Dict[str, List[str]] = Field(
        description="Workouts for each day of the week"
    )
    duration_per_session: str = Field(
        description="How long each workout should be"
    )
    focus_areas: List[str] = Field(
        description="Body parts or fitness aspects to focus on"
    )
    progression_plan: str = Field(
        description="How to increase difficulty over time"
    )

def generate_workout_plan(user_goal: UserGoal) -> WorkoutPlan:
    """
    Generate a personalized workout plan based on user goals.
    
    Args:
        user_goal: The structured goal information
        
    Returns:
        A personalized workout plan
    """
    llm = ChatOpenAI(model='gpt-4-turbo')
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert fitness coach. Create a personalized workout plan 
        based on the user's goal, constraints, and current fitness level.
        
        Consider:
        - Goal type and target metric
        - Timeframe for achieving the goal
        - User constraints (injuries, time limits, equipment)
        - User preferences
        - Current fitness status
        
        Create a realistic, achievable plan that the user can follow."""),
        ("human", "Create a workout plan for this goal: {goal_data}")
    ])
    
    chain = prompt | llm.with_structured_output(WorkoutPlan)
    
    goal_data_str = f"""
    Goal Type: {user_goal.goal_type}
    Target: {user_goal.target_metric}
    Timeframe: {user_goal.timeframe}
    Constraints: {', '.join(user_goal.constraints) if user_goal.constraints else 'None'}
    Preferences: {', '.join(user_goal.preferences) if user_goal.preferences else 'None'}
    Current Status: {user_goal.current_status}
    """
    
    result = chain.invoke({"goal_data": goal_data_str})
    return result
```

### **Step 2: Create a Meal Plan Generator**

```python
class MealPlan(BaseModel):
    """Structured meal plan based on fitness goals."""
    daily_calories: int = Field(description="Target daily calorie intake")
    macronutrients: Dict[str, int] = Field(
        description="Protein, carbs, and fats in grams"
    )
    meal_suggestions: Dict[str, List[str]] = Field(
        description="Meal ideas for breakfast, lunch, dinner, snacks"
    )
    dietary_restrictions: List[str] = Field(
        default_factory=list,
        description="Any dietary restrictions to follow"
    )

def generate_meal_plan(user_goal: UserGoal) -> MealPlan:
    """
    Generate a personalized meal plan based on user goals.
    """
    llm = ChatOpenAI(model='gpt-4-turbo')
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a nutritionist. Create a meal plan that supports 
        the user's fitness goals. Consider their goal type, constraints, and preferences."""),
        ("human", "Create a meal plan for: {goal_data}")
    ])
    
    chain = prompt | llm.with_structured_output(MealPlan)
    goal_data_str = user_goal.model_dump_json()
    
    result = chain.invoke({"goal_data": goal_data_str})
    return result
```

### **Step 3: Create a Progress Tracker**

```python
class ProgressEntry(BaseModel):
    """Individual progress tracking entry."""
    date: str
    weight: Optional[float] = None
    measurements: Dict[str, float] = Field(default_factory=dict)
    workouts_completed: int
    notes: Optional[str] = None

class ProgressTracker:
    """Tracks user progress toward their goal."""
    
    def __init__(self, user_goal: UserGoal):
        self.user_goal = user_goal
        self.entries: List[ProgressEntry] = []
    
    def add_entry(self, entry: ProgressEntry):
        """Add a new progress entry."""
        self.entries.append(entry)
    
    def get_progress_summary(self) -> str:
        """Get a summary of progress toward the goal."""
        if not self.entries:
            return "No progress entries yet."
        
        llm = ChatOpenAI(model='gpt-4-turbo')
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the user's progress and provide encouraging, 
            actionable feedback. Compare current progress to their goal."""),
            ("human", """Goal: {goal}
            
            Progress entries:
            {progress}
            
            Provide a progress summary and recommendations.""")
        ])
        
        goal_str = self.user_goal.model_dump_json()
        progress_str = "\n".join([e.model_dump_json() for e in self.entries])
        
        chain = prompt | llm
        result = chain.invoke({
            "goal": goal_str,
            "progress": progress_str
        })
        
        return result.content
```

### **Step 4: Create a Complete Fitness Assistant**

```python
class FitnessAssistant:
    """Complete fitness assistant that combines goal extraction, planning, and tracking."""
    
    def __init__(self):
        self.goal_tool = GoalDescriptionTool()
        self.current_goal: Optional[UserGoal] = None
        self.workout_plan: Optional[WorkoutPlan] = None
        self.meal_plan: Optional[MealPlan] = None
        self.progress_tracker: Optional[ProgressTracker] = None
    
    def set_goal_from_text(self, user_input: str) -> UserGoal:
        """Extract and set goal from natural language input."""
        goal = self.goal_tool.extract_goal(user_input)
        self.current_goal = goal
        
        # Generate plans automatically
        self.workout_plan = generate_workout_plan(goal)
        self.meal_plan = generate_meal_plan(goal)
        self.progress_tracker = ProgressTracker(goal)
        
        return goal
    
    def get_complete_plan(self) -> Dict:
        """Get the complete fitness plan (goal + workout + meal)."""
        if not self.current_goal:
            raise ValueError("No goal set yet. Call set_goal_from_text() first.")
        
        return {
            "goal": self.current_goal.model_dump(),
            "workout_plan": self.workout_plan.model_dump() if self.workout_plan else None,
            "meal_plan": self.meal_plan.model_dump() if self.meal_plan else None
        }
    
    def track_progress(self, weight: Optional[float] = None, 
                      measurements: Optional[Dict[str, float]] = None,
                      workouts_completed: int = 0,
                      notes: Optional[str] = None):
        """Record progress."""
        if not self.progress_tracker:
            raise ValueError("No goal set yet.")
        
        entry = ProgressEntry(
            date=datetime.now().isoformat(),
            weight=weight,
            measurements=measurements or {},
            workouts_completed=workouts_completed,
            notes=notes
        )
        
        self.progress_tracker.add_entry(entry)
    
    def get_progress_summary(self) -> str:
        """Get AI-generated progress summary."""
        if not self.progress_tracker:
            return "No progress tracked yet."
        return self.progress_tracker.get_progress_summary()
```

### **Step 5: Example Usage**

```python
# Initialize the assistant
assistant = FitnessAssistant()

# Set a goal from natural language
user_input = "I want to lose 10kg in 3 months. I'm a beginner, only have 30 minutes a day, and I have a knee injury."
goal = assistant.set_goal_from_text(user_input)

# Get the complete plan
plan = assistant.get_complete_plan()
print("Your Fitness Plan:")
print(f"Goal: {plan['goal']['goal_type']} - {plan['goal']['target_metric']}")
print(f"Workout Plan: {plan['workout_plan']}")
print(f"Meal Plan: {plan['meal_plan']}")

# Track progress over time
assistant.track_progress(
    weight=85.5,
    workouts_completed=3,
    notes="Feeling good, knee is holding up well"
)

# Get progress summary
summary = assistant.get_progress_summary()
print(summary)
```

---

## 💡 What Can You Build With This?

1. **Fitness Mobile App**: Users type goals, get personalized plans
2. **Personal Trainer Bot**: Chatbot that creates workout plans
3. **Progress Tracking App**: Track goals and get AI feedback
4. **Corporate Wellness Platform**: Help employees set fitness goals
5. **Healthcare Integration**: Doctors can prescribe fitness goals, system creates plans

---

## 🔧 How to Use This Code

1. **Install Dependencies**:
   ```bash
   pip install langchain langchain-openai pydantic
   ```

2. **Set OpenAI API Key**:
   ```python
   import os
   os.environ["OPENAI_API_KEY"] = "your-api-key-here"
   ```

3. **Run the Code**:
   ```python
   tool = GoalDescriptionTool()
   goal = tool.extract_goal("I want to build muscle in 6 months")
   print(goal)
   ```

---

## 📝 Summary

This code transforms **unstructured human language** into **structured data** that can be used to:
- Generate personalized workout plans
- Create meal plans
- Track progress
- Provide personalized fitness advice

It's the foundation for building any AI-powered fitness application!
