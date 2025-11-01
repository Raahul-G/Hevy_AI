"""User goal schema definitions."""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class GoalType(str, Enum):
    """Types of fitness goals."""

    WEIGHT_LOSS = "weight_loss"
    MUSCLE_GAIN = "muscle_gain"
    ENDURANCE = "endurance"
    GENERAL_WELLNESS = "general_wellness"


class UserStatus(str, Enum):
    """User fitness status levels."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    SEDENTARY = "sedentary"
    ACTIVE = "active"


class UserGoal(BaseModel):
    """
    Structured representation of a user's fitness goal.

    This schema captures all essential information about a user's fitness objectives,
    constraints, preferences, and current status to enable personalized planning.
    """

    goal_type: GoalType = Field(
        description="The primary type of fitness goal the user wants to achieve"
    )
    target_metric: str = Field(
        description="The specific target or metric the user wants to achieve (e.g., '10kg', 'upper body strength', '5k race time', 'improved energy')"
    )
    timeframe: str = Field(
        description="The time period for achieving the goal (e.g., '3 months', '6 weeks', 'ongoing')"
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Limitations or restrictions the user has (e.g., 'limited equipment', 'vegan diet', 'knee injury', '30 mins/day')",
    )
    preferences: List[str] = Field(
        default_factory=list,
        description="User preferences and likes (e.g., 'enjoys strength training', 'prefers home workouts', 'dislikes running')",
    )
    current_status: UserStatus = Field(
        description="The user's current fitness level or activity status"
    )
    additional_context: Optional[Dict[str, str]] = Field(
        default=None,
        description="Any additional context or information about the goal",
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_schema_extra = {
            "example": {
                "goal_type": "weight_loss",
                "target_metric": "lose 10kg",
                "timeframe": "3 months",
                "constraints": ["limited equipment", "30 mins/day"],
                "preferences": ["enjoys strength training", "prefers home workouts"],
                "current_status": "beginner",
                "additional_context": {"work_schedule": "9-5 weekdays"},
            }
        }

