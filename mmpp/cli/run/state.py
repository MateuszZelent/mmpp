"""
Local state management for tracking submitted simulations.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class SimulationEntry:
    """Information about a submitted simulation."""
    task_id: str
    file_path: str
    original_file: str
    submit_time: str
    status: str = "PENDING"
    server_url: str = ""
    download_path: Optional[str] = None
    completed_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationEntry':
        """Create from dictionary."""
        return cls(**data)


class LocalStateManager:
    """Manager for local simulation state."""
    
    def __init__(self, work_dir: Optional[str] = None):
        """Initialize state manager.
        
        Args:
            work_dir: Working directory. If None, uses current directory.
        """
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        self.state_file = self.work_dir / ".mmpp_simulations.json"
        
    def _load_state(self) -> Dict[str, Dict[str, Any]]:
        """Load state from file."""
        if not self.state_file.exists():
            return {}
            
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load state file: {e}")
            return {}
    
    def _save_state(self, state: Dict[str, Dict[str, Any]]) -> None:
        """Save state to file."""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.error(f"Failed to save state file: {e}")
    
    def add_simulation(self, entry: SimulationEntry) -> None:
        """Add a new simulation entry."""
        state = self._load_state()
        state[entry.task_id] = entry.to_dict()
        self._save_state(state)
        logger.info(f"Added simulation {entry.task_id} to local state")
    
    def update_simulation(self, task_id: str, **updates) -> bool:
        """Update simulation entry.
        
        Args:
            task_id: Task ID to update
            **updates: Fields to update
            
        Returns:
            True if updated, False if not found
        """
        state = self._load_state()
        if task_id not in state:
            return False
            
        state[task_id].update(updates)
        self._save_state(state)
        logger.info(f"Updated simulation {task_id} in local state")
        return True
    
    def get_simulation(self, task_id: str) -> Optional[SimulationEntry]:
        """Get simulation entry by task ID."""
        state = self._load_state()
        if task_id not in state:
            return None
        return SimulationEntry.from_dict(state[task_id])
    
    def get_all_simulations(self) -> List[SimulationEntry]:
        """Get all simulation entries."""
        state = self._load_state()
        return [SimulationEntry.from_dict(data) for data in state.values()]
    
    def get_pending_simulations(self) -> List[SimulationEntry]:
        """Get simulations that are not completed."""
        return [
            sim for sim in self.get_all_simulations()
            if sim.status not in ["COMPLETED", "FAILED", "CANCELLED"]
        ]
    
    def get_completed_simulations(self) -> List[SimulationEntry]:
        """Get simulations that are completed but not downloaded."""
        return [
            sim for sim in self.get_all_simulations()
            if sim.status == "COMPLETED" and sim.download_path is None
        ]
    
    def remove_simulation(self, task_id: str) -> bool:
        """Remove simulation entry.
        
        Args:
            task_id: Task ID to remove
            
        Returns:
            True if removed, False if not found
        """
        state = self._load_state()
        if task_id not in state:
            return False
            
        del state[task_id]
        self._save_state(state)
        logger.info(f"Removed simulation {task_id} from local state")
        return True
    
    def cleanup_old_entries(self, days: int = 30) -> int:
        """Remove entries older than specified days.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of entries removed
        """
        from datetime import datetime, timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        state = self._load_state()
        to_remove = []
        
        for task_id, data in state.items():
            try:
                submit_time = datetime.fromisoformat(data['submit_time'])
                if submit_time < cutoff:
                    to_remove.append(task_id)
            except (KeyError, ValueError):
                # Invalid date format, mark for removal
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del state[task_id]
        
        if to_remove:
            self._save_state(state)
            logger.info(f"Cleaned up {len(to_remove)} old simulation entries")
        
        return len(to_remove)
    
    def get_state_file_path(self) -> Path:
        """Get path to state file."""
        return self.state_file
    
    def has_simulations(self) -> bool:
        """Check if there are any tracked simulations."""
        state = self._load_state()
        return len(state) > 0
