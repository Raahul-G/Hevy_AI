"""User profile management with JSON-based storage."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from config.settings import settings
from src.schemas.user_goal import UserGoal


class UserProfile(BaseModel):
    """
    User profile schema for storing user information and preferences.
    """

    user_id: str = Field(description="Unique identifier for the user")
    name: Optional[str] = Field(default=None, description="User's name")
    age: Optional[int] = Field(default=None, description="User's age")
    preferences: List[str] = Field(
        default_factory=list, description="Workout and lifestyle preferences"
    )
    constraints: List[str] = Field(
        default_factory=list, description="Physical or lifestyle constraints"
    )
    goals_history: List[Dict] = Field(
        default_factory=list, description="History of user goals"
    )
    current_goal: Optional[Dict] = Field(
        default=None, description="Current active goal"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Profile creation timestamp",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Profile last update timestamp",
    )
    additional_info: Dict[str, str] = Field(
        default_factory=dict, description="Additional user information"
    )

    def update_from_goal(self, goal: UserGoal) -> None:
        """
        Update profile preferences and constraints from a UserGoal.

        Args:
            goal: UserGoal object to extract information from
        """
        # Merge goal constraints with existing constraints
        for constraint in goal.constraints:
            if constraint not in self.constraints:
                self.constraints.append(constraint)

        # Merge goal preferences with existing preferences
        for preference in goal.preferences:
            if preference not in self.preferences:
                self.preferences.append(preference)

        # Update current goal
        self.current_goal = goal.model_dump()

        # Add to history
        goal_entry = {
            "goal": goal.model_dump(),
            "created_at": datetime.now().isoformat(),
        }
        self.goals_history.append(goal_entry)

        # Update timestamp
        self.updated_at = datetime.now().isoformat()


class ProfileManager:
    """
    Manages user profiles with JSON file-based storage.

    Provides CRUD operations for user profiles stored as JSON files.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the ProfileManager.

        Args:
            storage_path: Path to the profile storage directory.
                         Defaults to settings.profile_dir
        """
        self.storage_path = Path(storage_path) if storage_path else settings.profile_dir
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_profile_path(self, user_id: str) -> Path:
        """Get the file path for a user profile."""
        return self.storage_path / f"{user_id}.json"

    def create_profile(self, user_id: str, **kwargs) -> UserProfile:
        """
        Create a new user profile.

        Args:
            user_id: Unique identifier for the user
            **kwargs: Additional profile fields

        Returns:
            Created UserProfile object

        Raises:
            ValueError: If profile with user_id already exists
        """
        profile_path = self._get_profile_path(user_id)
        if profile_path.exists():
            raise ValueError(f"Profile for user_id '{user_id}' already exists")

        profile = UserProfile(user_id=user_id, **kwargs)
        self.save_profile(profile)
        return profile

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Retrieve a user profile by user_id.

        Args:
            user_id: Unique identifier for the user

        Returns:
            UserProfile object if found, None otherwise
        """
        profile_path = self._get_profile_path(user_id)
        if not profile_path.exists():
            return None

        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return UserProfile(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(
                f"Failed to load profile for user_id '{user_id}': {str(e)}"
            ) from e

    def get_or_create_profile(self, user_id: str, **kwargs) -> UserProfile:
        """
        Get existing profile or create a new one.

        Args:
            user_id: Unique identifier for the user
            **kwargs: Additional profile fields (used only if creating new profile)

        Returns:
            UserProfile object
        """
        profile = self.get_profile(user_id)
        if profile is None:
            profile = self.create_profile(user_id, **kwargs)
        return profile

    def save_profile(self, profile: UserProfile) -> None:
        """
        Save a user profile to disk.

        Args:
            profile: UserProfile object to save
        """
        profile_path = self._get_profile_path(profile.user_id)
        
        # Update timestamp before saving
        profile.updated_at = datetime.now().isoformat()

        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile.model_dump(), f, indent=2, ensure_ascii=False)

    def update_profile(self, user_id: str, **updates) -> UserProfile:
        """
        Update a user profile with new information.

        Args:
            user_id: Unique identifier for the user
            **updates: Fields to update

        Returns:
            Updated UserProfile object

        Raises:
            ValueError: If profile doesn't exist
        """
        profile = self.get_profile(user_id)
        if profile is None:
            raise ValueError(f"Profile for user_id '{user_id}' not found")

        # Update fields
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        self.save_profile(profile)
        return profile

    def delete_profile(self, user_id: str) -> bool:
        """
        Delete a user profile.

        Args:
            user_id: Unique identifier for the user

        Returns:
            True if profile was deleted, False if it didn't exist
        """
        profile_path = self._get_profile_path(user_id)
        if profile_path.exists():
            profile_path.unlink()
            return True
        return False

    def update_profile_with_goal(self, user_id: str, goal: UserGoal) -> UserProfile:
        """
        Update or create a profile with goal information.

        Args:
            user_id: Unique identifier for the user
            goal: UserGoal object to integrate into profile

        Returns:
            Updated UserProfile object
        """
        profile = self.get_or_create_profile(user_id)
        profile.update_from_goal(goal)
        self.save_profile(profile)
        return profile

    def list_profiles(self) -> List[str]:
        """
        List all user IDs with profiles.

        Returns:
            List of user_id strings
        """
        return [
            path.stem
            for path in self.storage_path.glob("*.json")
            if path.is_file()
        ]

