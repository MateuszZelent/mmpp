"""
Result downloading and extraction utilities for simulations.
"""

import os
import zipfile
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import logging

logger = logging.getLogger(__name__)


class ResultDownloader:
    """Handles downloading and extracting simulation results."""
    
    def __init__(self, base_url: str, headers: Dict[str, str]):
        """Initialize downloader.
        
        Args:
            base_url: Server base URL
            headers: Authentication headers
        """
        self.base_url = base_url
        self.headers = headers
    
    def download_results(self, task_id: str, original_file_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Download and extract simulation results.
        
        Args:
            task_id: Task ID to download
            original_file_path: Path to original MX3 file (results will be placed in same directory)
            
        Returns:
            Tuple of (success, zarr_file_path, error_message)
        """
        # Get directory of original file
        original_file = Path(original_file_path)
        output_dir = original_file.parent
        
        try:
            # Download ZIP file
            zip_path = self._download_zip(task_id, output_dir)
            if not zip_path:
                return False, None, "Failed to download results"
            
            # Extract and process
            zarr_file = self._extract_and_process(zip_path, output_dir, original_file.stem)
            
            # Clean up ZIP file
            try:
                os.remove(zip_path)
                logger.info(f"Cleaned up ZIP file: {zip_path}")
            except OSError as e:
                logger.warning(f"Failed to remove ZIP file {zip_path}: {e}")
            
            if zarr_file:
                logger.info(f"Results successfully downloaded and extracted: {zarr_file}")
                return True, str(zarr_file), None
            else:
                return False, None, "No ZARR file found in results"
                
        except Exception as e:
            logger.error(f"Error downloading results for task {task_id}: {e}")
            return False, None, str(e)
    
    def _download_zip(self, task_id: str, output_dir: Path) -> Optional[Path]:
        """Download ZIP file from server.
        
        Args:
            task_id: Task ID
            output_dir: Output directory
            
        Returns:
            Path to downloaded ZIP file or None if failed
        """
        url = f"{self.base_url}/api/v1/tasks/{task_id}/download"
        zip_path = output_dir / f"results_{task_id}.zip"
        
        try:
            logger.info(f"Downloading results for task {task_id}...")
            response = requests.get(url, headers=self.headers, timeout=300)
            
            if response.status_code == 200:
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Downloaded ZIP file: {zip_path}")
                return zip_path
            else:
                logger.error(f"Failed to download results: HTTP {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Network error downloading results: {e}")
            return None
        except OSError as e:
            logger.error(f"File error saving results: {e}")
            return None
    
    def _extract_and_process(self, zip_path: Path, output_dir: Path, base_name: str) -> Optional[Path]:
        """Extract ZIP file and organize results properly.
        
        Args:
            zip_path: Path to ZIP file
            output_dir: Output directory (same as original MX3 file)
            base_name: Base name for output files (e.g., "test" from "test.mx3")
            
        Returns:
            Path to ZARR file or None if not found
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # List contents
                file_list = zip_ref.namelist()
                logger.info(f"ZIP contents: {file_list}")
                
                # Create temporary extraction directory
                temp_extract_dir = output_dir / f"temp_extract_{zip_path.stem}"
                temp_extract_dir.mkdir(exist_ok=True)
                
                # Extract all files to temp directory
                zip_ref.extractall(temp_extract_dir)
                logger.info(f"Extracted files to: {temp_extract_dir}")
                
                # Find ZARR file/directory in extracted content
                zarr_files = []
                metadata_files = []
                
                for root, dirs, files in os.walk(temp_extract_dir):
                    for file in files:
                        if file.endswith('.zarr') or '.zarr' in file:
                            zarr_files.append(Path(root) / file)
                        elif 'metadata' in file.lower() and file.endswith('.json'):
                            metadata_files.append(Path(root) / file)
                    
                    # Also check for directories ending in .zarr
                    for dir_name in dirs:
                        if dir_name.endswith('.zarr'):
                            zarr_files.append(Path(root) / dir_name)
                
                if not zarr_files:
                    logger.warning("No ZARR file found in extraction")
                    # Clean up temp directory
                    import shutil
                    shutil.rmtree(temp_extract_dir, ignore_errors=True)
                    return None
                
                # Use the first ZARR file found
                temp_zarr_path = zarr_files[0]
                
                # Target ZARR path in the same directory as original MX3
                target_zarr_path = output_dir / f"{base_name}.zarr"
                
                # If target already exists, remove it
                if target_zarr_path.exists():
                    import shutil
                    if target_zarr_path.is_dir():
                        shutil.rmtree(target_zarr_path)
                    else:
                        target_zarr_path.unlink()
                
                # Move ZARR to target location
                import shutil
                shutil.move(str(temp_zarr_path), str(target_zarr_path))
                logger.info(f"Moved ZARR to: {target_zarr_path}")
                
                # Move metadata files into ZARR directory (if ZARR is a directory)
                if target_zarr_path.is_dir() and metadata_files:
                    for metadata_file in metadata_files:
                        target_metadata = target_zarr_path / metadata_file.name
                        shutil.move(str(metadata_file), str(target_metadata))
                        logger.info(f"Moved metadata to: {target_metadata}")
                
                # Clean up temp directory
                shutil.rmtree(temp_extract_dir, ignore_errors=True)
                
                # Log other files for reference
                logger.info(f"Results organized: {target_zarr_path}")
                if metadata_files:
                    logger.info(f"Metadata files included in ZARR directory")
                
                return target_zarr_path
                    
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid ZIP file {zip_path}: {e}")
            return None
        except OSError as e:
            logger.error(f"File error extracting {zip_path}: {e}")
            return None
    
    def check_task_completion(self, task_id: str) -> Tuple[bool, str, Dict]:
        """Check if a task is completed and ready for download.
        
        Args:
            task_id: Task ID to check
            
        Returns:
            Tuple of (is_completed, status, task_info)
        """
        url = f"{self.base_url}/api/v1/tasks/{task_id}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                task_info = response.json()
                status = task_info.get('status', 'UNKNOWN')
                is_completed = status in ['COMPLETED', 'FAILED', 'CANCELLED']
                return is_completed, status, task_info
            else:
                logger.error(f"Failed to check task status: HTTP {response.status_code}")
                return False, 'ERROR', {}
                
        except requests.RequestException as e:
            logger.error(f"Network error checking task status: {e}")
            return False, 'ERROR', {}
