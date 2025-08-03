"""
MMPP Run module for simulation execution and cluster management.

This module provides functionality to:
- Run MX3 simulations with automatic upload and download
- Check status of submitted simulations
- Download completed simulation results
"""

import argparse
import sys
import glob
import os

from .status import check_cluster_status
from .submit import handle_run_command as handle_file_submission, handle_check_command, _submit_single_file


def handle_run_command(args: argparse.Namespace) -> None:
    """Handle run-related commands."""
    
    # Get the positional arguments
    files_or_command = getattr(args, 'files_or_command', [])
    
    if not files_or_command:
        # No arguments provided
        _show_usage()
        return
    
    # Check if first argument is a command
    if len(files_or_command) == 1 and files_or_command[0] in ['status', 'check']:
        # It's a command
        command = files_or_command[0]
        if command == "status":
            check_cluster_status()
        elif command == "check":
            handle_check_command(args)
    else:
        # It's file(s) - expand any patterns and validate
        all_files = []
        
        for item in files_or_command:
            if item.endswith('.mx3') and os.path.isfile(item):
                # Direct file path
                all_files.append(item)
            else:
                # Try glob expansion (in case shell didn't expand)
                expanded = _expand_file_pattern(item)
                all_files.extend(expanded)
        
        # Remove duplicates and sort
        unique_files = sorted(list(set(all_files)))
        
        if not unique_files:
            print(f"❌ No .mx3 files found in: {', '.join(files_or_command)}")
            sys.exit(1)
        
        if len(unique_files) == 1:
            # Single file - handle normally
            args.file = unique_files[0]
            handle_file_submission(args)
        else:
            # Multiple files - handle batch submission
            _handle_batch_submission(unique_files, args)


def _expand_file_pattern(pattern: str) -> list[str]:
    """Expand glob pattern to list of .mx3 files."""
    
    # If it's an exact file path, check if it exists
    if os.path.isfile(pattern):
        if pattern.endswith('.mx3'):
            return [pattern]
        else:
            print(f"⚠️  Warning: {pattern} is not an .mx3 file")
            return []
    
    # Try glob expansion
    expanded = glob.glob(pattern)
    
    # Filter for .mx3 files only
    mx3_files = [f for f in expanded if f.endswith('.mx3') and os.path.isfile(f)]
    
    # Sort for consistent ordering
    return sorted(mx3_files)


def _handle_batch_submission(files: list[str], args: argparse.Namespace) -> None:
    """Handle submission of multiple files."""
    
    print(f"🚀 Found {len(files)} files to submit:")
    for i, file in enumerate(files, 1):
        print(f"   {i}. {os.path.basename(file)}")
    
    print()
    
    # Ask for confirmation
    response = input(f"Submit all {len(files)} simulations to queue? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Cancelled by user.")
        sys.exit(0)
    
    print("=" * 60)
    print("📤 Submitting all simulations to server queue...")
    
    submitted_tasks = []
    failed_count = 0
    
    # Submit all files in detached mode (let server handle queueing)
    for i, file in enumerate(files, 1):
        print(f"\n📋 [{i}/{len(files)}] Submitting: {os.path.basename(file)}")
        
        try:
            # Create a copy of args for this file with forced detach mode
            file_args = argparse.Namespace(**vars(args))
            file_args.file = file
            file_args.detach = True  # Force detached mode for batch submissions
            
            # Submit the file and get task ID
            task_id = _submit_single_file(file_args)
            if task_id:
                submitted_tasks.append({
                    'task_id': task_id,
                    'file': file,
                    'name': os.path.basename(file)
                })
                print(f"   ✅ Queued with ID: {task_id}")
            else:
                print("   ❌ Failed to submit")
                failed_count += 1
            
        except KeyboardInterrupt:
            print(f"❌ Interrupted: {os.path.basename(file)}")
            failed_count += 1
            break
        except Exception as e:
            print(f"❌ Failed to submit {os.path.basename(file)}: {e}")
            failed_count += 1
            continue
    
    print("\n" + "=" * 60)
    print("📊 Batch submission summary:")
    print(f"   ✅ Submitted: {len(submitted_tasks)}")
    if failed_count > 0:
        print(f"   ❌ Failed: {failed_count}")
    
    if submitted_tasks:
        print("\n📋 Submitted tasks:")
        for task in submitted_tasks:
            print(f"   • {task['name']} → {task['task_id']}")
        
        print("\n💡 Use 'mmpp run check' to monitor all simulations")
        
        # Ask if user wants to monitor progress
        if not getattr(args, 'detach', False):
            response = input(f"\nMonitor progress of all {len(submitted_tasks)} simulations? [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                _monitor_batch_progress(submitted_tasks, args)


def _show_usage() -> None:
    """Show usage information."""
    print("Usage: mmpp run <file.mx3|pattern> [options]")
    print("       mmpp run <command>")
    print()
    print("Commands:")
    print("  mmpp run <file.mx3>        Upload and run single simulation")
    print("  mmpp run <pattern>         Upload and run multiple simulations")
    print("  mmpp run status            Check cluster status")
    print("  mmpp run check             Check and download completed simulations")
    print()
    print("Options for running simulations:")
    print("  -d, --detach              Don't wait for completion")
    print("  -t, --time TIME           Time limit (e.g., '10h', '2d', '30m')")
    print("  --name NAME               Custom task name")
    print("  --cpus N                  Number of CPUs (default: 5)")
    print("  --memory N                Memory in GB (default: 24)")
    print("  --gpus N                  Number of GPUs (default: 1)")
    print("  --partition NAME          SLURM partition (default: proxima)")
    print()
    print("Examples:")
    print("  mmpp run simulation.mx3")
    print("  mmpp run test*.mx3                    # Submit all files matching pattern")
    print("  mmpp run *.mx3 -t 10h --cpus 8       # Batch with custom settings")
    print("  mmpp run simulation.mx3 -d           # Detached mode")
    print("  mmpp run check")
    print("  mmpp run status")


def _monitor_batch_progress(submitted_tasks: list, _args: argparse.Namespace) -> None:
    """Monitor progress of multiple submitted tasks."""
    
    print("\n🔍 Monitoring batch progress...")
    print("Press Ctrl+C to stop monitoring (simulations will continue running)")
    
    # Initialize components
    from .submit import _get_auth_token, _get_base_url
    from .downloader import ResultDownloader
    from .state import LocalStateManager
    from ...auth import AuthManager
    
    try:
        auth_manager = AuthManager()
        token = _get_auth_token(auth_manager)
        base_url = _get_base_url(auth_manager)
        headers = {"Authorization": f"Bearer {token}"}
        
        downloader = ResultDownloader(base_url, headers)
        state_manager = LocalStateManager()
        
        completed_tasks = set()
        check_interval = 30  # seconds
        
        while len(completed_tasks) < len(submitted_tasks):
            print(f"\n📊 Checking {len(submitted_tasks)} simulations...")
            
            for task in submitted_tasks:
                task_id = task['task_id']
                
                if task_id in completed_tasks:
                    continue
                    
                try:
                    is_completed, status, task_info = downloader.check_task_completion(task_id)
                    
                    # Update local state
                    state_manager.update_simulation(task_id, status=status)
                    
                    print(f"   • {task['name']}: {status}")
                    
                    if is_completed:
                        completed_tasks.add(task_id)
                        
                        if status == "COMPLETED":
                            print("     ⬇️  Downloading results...")
                            
                            # Find original file path
                            original_file = task['file']
                            success, zarr_path, error = downloader.download_results(task_id, original_file)
                            
                            if success and zarr_path:
                                print(f"     ✅ Downloaded: {os.path.basename(zarr_path)}")
                            else:
                                print(f"     ❌ Download failed: {error}")
                        else:
                            print(f"     ⚠️  Failed with status: {status}")
                    else:
                        # Show progress if available
                        progress = task_info.get('progress', 0)
                        if progress > 0:
                            print(f"     📈 {progress}%")
                
                except Exception as e:
                    print(f"   • {task['name']}: ❌ Error checking status: {e}")
            
            if len(completed_tasks) < len(submitted_tasks):
                print(f"\n⏳ Waiting {check_interval}s before next check...")
                import time
                time.sleep(check_interval)
        
        print(f"\n🎉 All {len(submitted_tasks)} simulations completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Monitoring stopped by user.")
        print("💡 Use 'mmpp run check' to check status and download results later.")
    except Exception as e:
        print(f"❌ Error monitoring batch: {e}")


# Export main functions for external use
__all__ = ['handle_run_command', 'check_cluster_status']
