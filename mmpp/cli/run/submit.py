"""
Simulation submission functionality for MMPP Run module.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests

from ...auth import AuthManager
from .utils import safe_print, safe_print_panel
from .state import LocalStateManager, SimulationEntry
from .downloader import ResultDownloader

import logging
logger = logging.getLogger(__name__)


def _submit_single_file(args: argparse.Namespace) -> Optional[str]:
    """Submit a single file and return task ID without waiting.
    
    Args:
        args: Command arguments with file path
        
    Returns:
        Task ID if successful, None if failed
    """
    file_path = args.file
    
    # Validate file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    # Validate file extension
    if not file_path.endswith('.mx3'):
        print(f"❌ Invalid file type. Expected .mx3 file, got: {file_path}")
        return None
    
    # Get absolute path
    file_path = os.path.abspath(file_path)
    
    try:
        # Initialize components
        auth_manager = AuthManager()
        token = _get_auth_token(auth_manager)
        base_url = _get_base_url(auth_manager)
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # Initialize state manager
        state_manager = LocalStateManager()
        
        # Upload and start simulation
        task_id = _upload_and_start_simulation(
            file_path, base_url, headers, args
        )
        
        if not task_id:
            return None
        
        # Add to local state
        entry = SimulationEntry(
            task_id=task_id,
            file_path=file_path,
            original_file=os.path.basename(file_path),
            submit_time=datetime.now().isoformat(),
            status="PENDING",
            server_url=base_url
        )
        state_manager.add_simulation(entry)
        
        return task_id
        
    except Exception as e:
        logger.error(f"Error submitting file {file_path}: {e}")
        return None


def handle_run_command(args: argparse.Namespace) -> None:
    """Handle 'mmpp run' command with file path."""
    
    file_path = args.file
    
    # Validate file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    # Validate file extension
    if not file_path.endswith('.mx3'):
        print(f"❌ Invalid file type. Expected .mx3 file, got: {file_path}")
        sys.exit(1)
    
    # Get absolute path
    file_path = os.path.abspath(file_path)
    
    try:
        # Initialize components
        auth_manager = AuthManager()
        token = _get_auth_token(auth_manager)
        base_url = _get_base_url(auth_manager)
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # Initialize state manager and downloader
        state_manager = LocalStateManager()
        downloader = ResultDownloader(base_url, headers)
        
        # Upload and start simulation
        task_id = _upload_and_start_simulation(
            file_path, base_url, headers, args
        )
        
        if not task_id:
            sys.exit(1)
        
        # Add to local state
        entry = SimulationEntry(
            task_id=task_id,
            file_path=file_path,
            original_file=os.path.basename(file_path),
            submit_time=datetime.now().isoformat(),
            status="PENDING",
            server_url=base_url
        )
        state_manager.add_simulation(entry)
        
        print(f"\n✅ Simulation submitted with ID: {task_id}")
        
        # Handle different modes
        if getattr(args, 'detach', False):
            # Detached mode - don't wait
            print("🔄 Running in detached mode. Use 'mmpp run check' to check status and download results.")
        else:
            # Wait for completion and download
            print("� Waiting for simulation to complete...")
            _wait_and_download(task_id, downloader, state_manager, file_path)
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user. Simulation continues running on server.")
        print("💡 Use 'mmpp run check' to check status and download results later.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in run command: {e}")
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


def handle_check_command(args: argparse.Namespace) -> None:
    """Handle 'mmpp run check' command."""
    
    try:
        # Initialize components
        auth_manager = AuthManager()
        token = _get_auth_token(auth_manager)
        base_url = _get_base_url(auth_manager)
        
        headers = {"Authorization": f"Bearer {token}"}
        
        state_manager = LocalStateManager()
        downloader = ResultDownloader(base_url, headers)
        
        # Get ALL simulations in this directory
        all_sims = state_manager.get_all_simulations()
        
        if not all_sims:
            print("📭 No simulations found in this directory.")
            return
        
        print(f"🔍 Found {len(all_sims)} simulation(s) in this directory:")
        print("=" * 60)
        
        downloaded_count = 0
        
        for sim in all_sims:
            print(f"\n📋 Task: {sim.task_id}")
            print(f"   File: {os.path.basename(sim.file_path)}")
            print(f"   Submitted: {sim.submit_time}")
            
            # Check if .zarr file already exists locally
            zarr_path = sim.file_path.replace('.mx3', '.zarr')
            zarr_exists = os.path.exists(zarr_path)
            
            if zarr_exists:
                print(f"   📦 Results: Already downloaded ({os.path.basename(zarr_path)})")
                continue
            
            # Check status on server
            try:
                is_completed, status, task_info = downloader.check_task_completion(sim.task_id)
                
                # Update local state
                state_manager.update_simulation(sim.task_id, status=status)
                
                print(f"   📊 Server Status: {status}")
                
                if is_completed and status == "COMPLETED":
                    print(f"   ⬇️  Downloading results...")
                    
                    # Download results
                    success, downloaded_zarr_path, error = downloader.download_results(sim.task_id, sim.file_path)
                    
                    if success and downloaded_zarr_path:
                        state_manager.update_simulation(
                            sim.task_id,
                            download_path=downloaded_zarr_path,
                            completed_time=datetime.now().isoformat()
                        )
                        print(f"   ✅ Results downloaded: {os.path.basename(downloaded_zarr_path)}")
                        downloaded_count += 1
                    else:
                        print(f"   ❌ Failed to download: {error}")
                elif is_completed:
                    print(f"   ⚠️  Simulation failed: {status}")
                else:
                    print(f"   🔄 Still running...")
                    
                    # Show progress if available
                    progress = task_info.get('progress', 0)
                    if progress > 0:
                        print(f"   📈 Progress: {progress}%")
                    
                    node = task_info.get('node')
                    if node:
                        print(f"   🖥️  Node: {node}")
                        
            except Exception as e:
                print(f"   ❌ Error checking status: {e}")
                # Continue with other simulations
        
        print("\n" + "=" * 60)
        if downloaded_count > 0:
            print(f"🎉 Downloaded {downloaded_count} completed simulation(s)!")
        else:
            print("📋 Summary: No new results to download.")
        
    except Exception as e:
        logger.error(f"Error in check command: {e}")
        print(f"❌ Error checking simulations: {e}")
        sys.exit(1)


def _get_auth_token(auth_manager: AuthManager) -> str:
    """Get authentication token."""
    token = auth_manager.get_token()
    if not token:
        print("❌ Authentication required. Please run 'mmpp auth login' first.")
        sys.exit(1)
    return token


def _get_base_url(auth_manager: AuthManager) -> str:
    """Get base URL."""
    base_url = auth_manager.get_base_url()
    if not base_url:
        print("❌ Server URL not configured. Please run 'mmpp auth login' first.")
        sys.exit(1)
    return base_url


def _upload_and_start_simulation(
    file_path: str, 
    base_url: str, 
    headers: dict, 
    args: argparse.Namespace
) -> Optional[str]:
    """Upload file and start simulation."""
    
    print(f"🚀 Uploading and starting simulation: {os.path.basename(file_path)}")
    
    url = f"{base_url}/api/v1/tasks/upload-mx3"
    
    # Prepare task parameters
    task_name = getattr(args, 'name', None) or os.path.basename(file_path).replace('.mx3', '')
    time_limit = getattr(args, 'time', None) or "24:00:00"
    
    # Convert time format if needed (e.g., "10h" -> "10:00:00")
    time_limit = _parse_time_limit(time_limit)
    
    try:
        # Calculate MD5 checksum of the file
        import hashlib
        with open(file_path, 'rb') as file_to_hash:
            md5_hash = hashlib.md5()
            while chunk := file_to_hash.read(8192):
                md5_hash.update(chunk)
            original_md5 = md5_hash.hexdigest()

        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'task_name': task_name,
                'auto_start': 'true',
                'partition': getattr(args, 'partition', 'proxima'),
                'num_cpus': getattr(args, 'cpus', 5),
                'memory_gb': getattr(args, 'memory', 24),
                'num_gpus': getattr(args, 'gpus', 1),
                'time_limit': time_limit,
                'priority': getattr(args, 'priority', 0),
                'original_path': file_path,
                'original_md5': original_md5
            }
            
            response = requests.post(url, headers=headers, files=files, data=data, timeout=60)
        
        if response.status_code in [200, 201]:
            result = response.json()
            task_id = result.get('task_id')
            if task_id:
                print(f"✅ Simulation started with task ID: {task_id}")
                return task_id
            else:
                print("❌ No task ID returned from server")
                return None
        else:
            print(f"❌ Upload failed: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.RequestException as e:
        print(f"❌ Network error during upload: {e}")
        return None
    except OSError as e:
        print(f"❌ File error during upload: {e}")
        return None


def _parse_time_limit(time_str: str) -> str:
    """Parse time limit string and convert to HH:MM:SS format."""
    
    if not time_str:
        return "24:00:00"
    
    # If already in HH:MM:SS format, return as-is
    if ':' in time_str and len(time_str.split(':')) >= 2:
        return time_str
    
    # Handle formats like "10h", "2d", "30m", or just numbers
    import re
    
    # Match patterns like "10h", "2d", "30m"
    match = re.match(r'^(\d+)([hmdHMD]?)$', time_str.lower())
    if match:
        value, unit = match.groups()
        value = int(value)
        
        # If no unit provided, assume hours
        if not unit:
            unit = 'h'
        
        unit = unit.lower()
        
        if unit == 'h':
            return f"{value:02d}:00:00"
        elif unit == 'd':
            hours = value * 24
            return f"{hours:02d}:00:00"
        elif unit == 'm':
            if value < 60:
                return f"00:{value:02d}:00"
            else:
                hours = value // 60
                minutes = value % 60
                return f"{hours:02d}:{minutes:02d}:00"
    
    # Try to parse as pure number (assume hours)
    try:
        hours = int(time_str)
        return f"{hours:02d}:00:00"
    except ValueError:
        pass
    
    # If can't parse, return default
    print(f"⚠️  Warning: Could not parse time format '{time_str}', using 24:00:00")
    return "24:00:00"


def _wait_and_download(task_id: str, downloader: ResultDownloader, state_manager: LocalStateManager, original_file_path: str) -> None:
    """Wait for simulation completion and download results."""
    
    last_status = None
    check_interval = 30  # seconds
    
    while True:
        try:
            # Check task status
            is_completed, status, task_info = downloader.check_task_completion(task_id)
            
            # Update local state
            state_manager.update_simulation(task_id, status=status)
            
            # Show status if changed
            if status != last_status:
                print(f"📊 Status: {status}")
                
                # Show additional info
                progress = task_info.get('progress', 0)
                if progress > 0:
                    print(f"   Progress: {progress}%")
                
                node = task_info.get('node')
                if node:
                    print(f"   Running on: {node}")
                
                last_status = status
            
            if is_completed:
                if status == "COMPLETED":
                    print("\n🎉 Simulation completed! Downloading results...")
                    
                    # Download results
                    success, zarr_path, error = downloader.download_results(task_id, original_file_path)
                    
                    if success and zarr_path:
                        state_manager.update_simulation(
                            task_id,
                            download_path=zarr_path,
                            completed_time=datetime.now().isoformat()
                        )
                        print(f"✅ Results downloaded: {zarr_path}")
                        return
                    else:
                        print(f"❌ Failed to download results: {error}")
                        sys.exit(1)
                else:
                    print(f"\n⚠️  Simulation finished with status: {status}")
                    if status == "FAILED":
                        print("💡 Check logs with: mmpp run logs {task_id}")
                    sys.exit(1)
            
            # Wait before next check
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\n⚠️  Monitoring interrupted. Simulation continues running on server.")
            print(f"� Use 'mmpp run check' to check status and download results later.")
            break
        except Exception as e:
            logger.error(f"Error monitoring task {task_id}: {e}")
            print(f"❌ Error monitoring simulation: {e}")
            break
