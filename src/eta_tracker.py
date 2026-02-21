"""
ETA Tracker: Self-learning duration estimation for long-running operations.

Tracks run metrics, persists history, and uses past data to predict future ETAs.
"""
import os
import json
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

# Use project root from config if available, else derive from file location
try:
    from config import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
HISTORY_FILE = os.path.join(DATA_DIR, 'eta_history.json')


@dataclass
class RunMetrics:
    """Metrics for a single run of an operation."""
    op_name: str
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    params: Optional[Dict[str, Any]] = None
    predicted_eta_seconds: Optional[float] = None
    error_pct: Optional[float] = None


class ETATracker:
    """
    Tracks duration and key metrics for long-running operations.
    Learns from past runs to provide accurate ETAs.
    """
    
    def __init__(self, op_name: str):
        self.op_name = op_name
        self.start_ts: Optional[float] = None
        self.params: Dict[str, Any] = {}
        self.predicted_eta: Optional[float] = None
        self._history: Optional[List[dict]] = None
        
        # Performance tracking (Work Units / Time)
        self.total_work_units: float = 0
        self.work_units_done: float = 0
        self.throughput: Optional[float] = None  # Units per second
        self.ema_alpha = 0.2  # Smoothing for throughput
        
        # Secondary tracker for overall session
        self.session_tracker: Optional['ETATracker'] = None
    
    def _load_history(self) -> List[dict]:
        """Load run history from disk."""
        if self._history is not None:
            return self._history
        
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r') as f:
                    self._history = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._history = []
        else:
            self._history = []
        return self._history
    
    def _save_history(self, history: List[dict]):
        """Save run history to disk."""
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        self._history = history
    
    def _get_past_runs(self) -> List[dict]:
        """Get past runs for this operation."""
        history = self._load_history()
        return [r for r in history if r.get('op_name') == self.op_name and r.get('duration_seconds')]
    
    def predict_eta(self, params: Dict[str, Any]) -> Optional[float]:
        """
        Predict duration based on historical data.
        Uses weighted average scaled by param ratios.
        """
        past_runs = self._get_past_runs()
        if not past_runs:
            return None
        
        # Calculate weighted average (recent runs weighted higher)
        weights = []
        durations = []
        param_ratios = []
        
        for i, run in enumerate(past_runs[-10:]):  # Use last 10 runs
            weight = i + 1  # More recent = higher weight
            weights.append(weight)
            durations.append(run['duration_seconds'])
            
            # Calculate param ratio (current / historical)
            run_params = run.get('params', {})
            ratio = 1.0
            for key, val in params.items():
                if key in run_params and run_params[key] > 0:
                    ratio *= val / run_params[key]
            param_ratios.append(ratio)
        
        if not durations:
            return None
        
        # Weighted average duration
        total_weight = sum(weights)
        avg_duration = sum(d * w for d, w in zip(durations, weights)) / total_weight
        avg_ratio = sum(r * w for r, w in zip(param_ratios, weights)) / total_weight
        
        # Apply correction factor from past prediction errors
        errors = [r.get('error_pct', 0) for r in past_runs[-5:] if r.get('error_pct') is not None]
        correction = 1.0
        if errors:
            avg_error = sum(errors) / len(errors)
            correction = 1.0 - (avg_error / 100)  # If we underestimate, increase prediction
        
        predicted = avg_duration * avg_ratio * correction
        return max(predicted, 1.0)  # At least 1 second
    
    def start_run(self, params: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """
        Start tracking a run. Returns predicted ETA in seconds (or None if no history).
        """
        self.start_ts = time.time()
        self.params = params or {}
        self.predicted_eta = self.predict_eta(self.params)
        
        # Log start
        eta_str = f"{self.predicted_eta / 60:.1f} min" if self.predicted_eta else "unknown"
        print(f"[ETA] Starting {self.op_name} | Predicted: {eta_str}", flush=True)
        
        return self.predicted_eta

    def set_work_units(self, total: float):
        """Sets the total work units for this run (e.g., total samples)."""
        self.total_work_units = total

    def get_live_eta(self, progress: float, work_units_done: float = None) -> Optional[float]:
        """
        Calculates live ETA based on current run performance.
        Preferentially uses work_units_done/total_work_units for higher accuracy.
        """
        if self.start_ts is None:
            return self.predicted_eta

        elapsed = time.time() - self.start_ts
        if elapsed < 5: 
            return self.predicted_eta

        # Determine effective progress
        # If work_units_done is provided, calculate progress based on that
        effective_progress = progress
        if work_units_done is not None and self.total_work_units > 0:
            effective_progress = work_units_done / self.total_work_units
            
        if effective_progress <= 0:
            return self.predicted_eta or 0

        # Calculate throughput (units per second)
        # Using a simple moving average would be better, but for a single run
        # the cumulative average (units/elapsed) is more stable than inst. speed.
        
        current_throughput = effective_progress / elapsed
        
        # Exponential moving average for throughput to react to stage slowdowns
        if self.throughput is None:
            self.throughput = current_throughput
        else:
            self.throughput = (self.ema_alpha * current_throughput) + (1 - self.ema_alpha) * self.throughput

        if self.throughput <= 0:
            return None

        remaining_progress = 1.0 - effective_progress
        if remaining_progress <= 0:
            return 0

        return remaining_progress / self.throughput
    
    def end_run(self) -> float:
        """
        End tracking. Saves metrics to history. Returns actual duration in seconds.
        """
        if self.start_ts is None:
            print("[ETA] Warning: end_run called without start_run")
            return 0.0
        
        end_ts = time.time()
        duration = end_ts - self.start_ts
        
        # Calculate prediction error
        error_pct = None
        if self.predicted_eta and self.predicted_eta > 0:
            error_pct = ((self.predicted_eta - duration) / duration) * 100
        
        # Create metrics record
        metrics = RunMetrics(
            op_name=self.op_name,
            start_time=datetime.fromtimestamp(self.start_ts).isoformat(),
            end_time=datetime.fromtimestamp(end_ts).isoformat(),
            duration_seconds=duration,
            params=self.params,
            predicted_eta_seconds=self.predicted_eta,
            error_pct=error_pct
        )
        
        # Save to history
        history = self._load_history()
        history.append(asdict(metrics))
        self._save_history(history)
        
        # Log completion
        actual_str = f"{duration / 60:.1f} min"
        predicted_str = f"{self.predicted_eta / 60:.1f} min" if self.predicted_eta else "N/A"
        error_str = f"{error_pct:+.1f}%" if error_pct is not None else "N/A"
        
        print(f"[ETA] Completed {self.op_name}", flush=True)
        print(f"      Actual: {actual_str} | Predicted: {predicted_str} | Error: {error_str}", flush=True)
        
        return duration

        return duration

    def write_status(self, progress: float, status_text: str, file_path: str, 
                     horizon_info: dict = None, work_units_done: float = None):
        """
        Writes a Markdown status file with progress bar and ETA.
        
        Args:
            progress: Current task progress (0.0 - 1.0)
            status_text: Current status message
            file_path: Path to status file
            horizon_info: Optional dict with multi-horizon training info
            work_units_done: Optional count of samples/units processed for smarter ETA
        """
        """
        Writes a Markdown status file with progress bar and ETA.
        
        Args:
            progress: Current task progress (0.0 - 1.0)
            status_text: Current status message
            file_path: Path to status file
            horizon_info: Optional dict with multi-horizon training info:
                - current_horizon: int (5, 10, or 30)
                - horizon_idx: int (0-based index)
                - total_horizons: int
        """
        if self.start_ts is None:
            return
            
        live_eta = self.get_live_eta(progress, work_units_done=work_units_done)
        eta_str = format_eta(live_eta) if progress < 1.0 else "Finished"
        
        # Progress bar
        bar_len = 25
        filled = int(progress * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        # Build horizon section
        horizon_section = ""
        overall_section = ""
        
        if horizon_info:
            h = horizon_info.get('current_horizon', 10)
            idx = horizon_info.get('horizon_idx', 0)
            total = horizon_info.get('total_horizons', 1)
            
            # Overall progress across all horizons
            overall_progress = (idx + progress) / total
            overall_filled = int(overall_progress * bar_len)
            overall_bar = "█" * overall_filled + "░" * (bar_len - overall_bar_len if 'overall_bar_len' in locals() else (bar_len - overall_filled))
            
            # Smart Session ETA (Overall)
            if self.session_tracker:
                session_eta = self.session_tracker.get_live_eta(overall_progress)
                overall_eta_str = format_eta(session_eta)
            else:
                # Fallback: simple projection
                session_eta = live_eta * (total - (idx + progress)) if live_eta is not None else None
                overall_eta_str = format_eta(session_eta)

            overall_section = f"""## 🌐 Overall Session Status
**Total Progress**: `[{overall_bar}]` {overall_progress:7.1%} ({idx + 1}/{total} horizons)
**Total ETA Remaining**: **{overall_eta_str}**
"""

            horizon_section = f"""
## 📊 Horizon Status: {h}d
| Horizon | Status | Target |
|---------|--------|--------|
| **5d** (Tactical) | {'🔄 In Progress' if h == 5 else ('✅ Done' if idx > 0 or h > 5 else '⏳ Pending')} | 1.5% |
| **10d** (Swing) | {'🔄 In Progress' if h == 10 else ('✅ Done' if idx > 1 or h > 10 else '⏳ Pending')} | 3% |
| **30d** (Trend) | {'🔄 In Progress' if h == 30 else ('✅ Done' if idx > 2 else '⏳ Pending')} | 8% |
"""
        
        content = f"""# 🤖 ML Training Status

**Operation**: {self.op_name}
**Status**: {status_text}
**Current Horizon ETA**: **{eta_str}**
**Progress**: `[{bar}]` {progress:7.1%}

{overall_section}
{horizon_section}
---
- **Started**: {datetime.fromtimestamp(self.start_ts).strftime('%Y-%m-%d %H:%M:%S')}
- **Update Frequency**: Real-time batch updates
- **Accuracy**: Sample-weighted throughput model
"""
        try:
            # Atomic write using unique temp file to avoid collisions
            temp_path = f"{file_path}.tmp.{os.getpid()}.{uuid.uuid4()}"
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            os.replace(temp_path, file_path)
            
            # Debug log to separate file
            with open("debug_eta.log", "a") as f:
                f.write(f"{datetime.now()}: Status written to {file_path} (Progress: {progress})\\n")
        except Exception as e:
            # Don't crash the main process if status write fails
            with open("debug_eta.log", "a") as f:
                f.write(f"{datetime.now()}: ERROR writing status: {e}\\n")
            print(f"Warning: Could not write status file: {e}", flush=True)


def format_eta(seconds: Optional[float]) -> str:
    """Format seconds as human-readable string."""
    if seconds is None:
        return "unknown"
    if seconds < 1:
        return "< 1s"
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} min"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) / 60
        return f"{hours:.0f}h {mins:.0f}m"
