"""
Checkpoint manager for benchmark resumption
"""
import os
import json
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkProgress:
    """Tracks progress of benchmark runs"""
    model_name: str
    representation: str
    strategy: str
    completed_sample_ids: List[str]
    results: List[Dict[str, Any]]
    start_time: str
    last_update: str
    total_samples: int


def get_run_key(model_name: str, representation: str, strategy: str) -> str:
    """Generate a unique key for a benchmark run configuration"""
    key_str = f"{model_name}|{representation}|{strategy}"
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


class CheckpointManager:
    """Manages checkpoints for benchmark resumption"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._progress_cache: Dict[str, BenchmarkProgress] = {}
    
    def _get_checkpoint_path(self, run_key: str) -> str:
        """Get the checkpoint file path for a run"""
        return os.path.join(self.checkpoint_dir, f"checkpoint_{run_key}.json")
    
    def load_progress(
        self, 
        model_name: str, 
        representation: str, 
        strategy: str
    ) -> Optional[BenchmarkProgress]:
        """Load progress for a specific run configuration"""
        run_key = get_run_key(model_name, representation, strategy)
        
        # Check cache first
        if run_key in self._progress_cache:
            return self._progress_cache[run_key]
        
        checkpoint_path = self._get_checkpoint_path(run_key)
        
        if not os.path.exists(checkpoint_path):
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            
            progress = BenchmarkProgress(
                model_name=data['model_name'],
                representation=data['representation'],
                strategy=data['strategy'],
                completed_sample_ids=data['completed_sample_ids'],
                results=data['results'],
                start_time=data['start_time'],
                last_update=data['last_update'],
                total_samples=data['total_samples']
            )
            
            self._progress_cache[run_key] = progress
            logger.info(f"Loaded checkpoint: {len(progress.completed_sample_ids)} samples completed")
            return progress
            
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {e}")
            return None
    
    def save_progress(self, progress: BenchmarkProgress) -> None:
        """Save progress to checkpoint file"""
        run_key = get_run_key(
            progress.model_name, 
            progress.representation, 
            progress.strategy
        )
        
        progress.last_update = datetime.now().isoformat()
        
        checkpoint_path = self._get_checkpoint_path(run_key)
        
        data = {
            'model_name': progress.model_name,
            'representation': progress.representation,
            'strategy': progress.strategy,
            'completed_sample_ids': progress.completed_sample_ids,
            'results': progress.results,
            'start_time': progress.start_time,
            'last_update': progress.last_update,
            'total_samples': progress.total_samples
        }
        
        # Write atomically
        temp_path = checkpoint_path + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, checkpoint_path)
        
        self._progress_cache[run_key] = progress
    
    def get_completed_ids(
        self, 
        model_name: str, 
        representation: str, 
        strategy: str
    ) -> Set[str]:
        """Get set of completed sample IDs for a run"""
        progress = self.load_progress(model_name, representation, strategy)
        if progress is None:
            return set()
        return set(progress.completed_sample_ids)
    
    def create_progress(
        self,
        model_name: str,
        representation: str,
        strategy: str,
        total_samples: int
    ) -> BenchmarkProgress:
        """Create a new progress tracker"""
        return BenchmarkProgress(
            model_name=model_name,
            representation=representation,
            strategy=strategy,
            completed_sample_ids=[],
            results=[],
            start_time=datetime.now().isoformat(),
            last_update=datetime.now().isoformat(),
            total_samples=total_samples
        )
    
    def add_result(
        self,
        progress: BenchmarkProgress,
        sample_id: str,
        result: Dict[str, Any]
    ) -> None:
        """Add a result to progress and save"""
        progress.completed_sample_ids.append(sample_id)
        progress.results.append(result)
        self.save_progress(progress)
    
    def get_all_checkpoints(self) -> List[Tuple[str, str, str, int]]:
        """Get list of all checkpoint configurations with their progress"""
        checkpoints = []
        
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith('checkpoint_') and filename.endswith('.json'):
                checkpoint_path = os.path.join(self.checkpoint_dir, filename)
                try:
                    with open(checkpoint_path, 'r') as f:
                        data = json.load(f)
                    checkpoints.append((
                        data['model_name'],
                        data['representation'],
                        data['strategy'],
                        len(data['completed_sample_ids'])
                    ))
                except Exception:
                    continue
        
        return checkpoints
    
    def clear_checkpoint(
        self,
        model_name: str,
        representation: str,
        strategy: str
    ) -> bool:
        """Clear a specific checkpoint"""
        run_key = get_run_key(model_name, representation, strategy)
        checkpoint_path = self._get_checkpoint_path(run_key)
        
        if run_key in self._progress_cache:
            del self._progress_cache[run_key]
        
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info(f"Cleared checkpoint: {checkpoint_path}")
            return True
        return False
    
    def clear_all_checkpoints(self) -> int:
        """Clear all checkpoints, return count of deleted files"""
        count = 0
        self._progress_cache.clear()
        
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith('checkpoint_') and filename.endswith('.json'):
                os.remove(os.path.join(self.checkpoint_dir, filename))
                count += 1
        
        logger.info(f"Cleared {count} checkpoints")
        return count