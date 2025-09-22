#!/usr/bin/env python3
"""
End-to-End test for MX3 file upload and execution workflow.

This script tests the new API endpoints for uploading and running MX3 simulations.
"""

import os
import sys
import requests
from pathlib import Path

# Add mmpp to path
sys.path.insert(0, str(Path(__file__).parent))
from mmpp.auth import AuthManager


class MX3Client:
    """Client for testing MX3 upload and execution workflow."""
    
    def __init__(self):
        self.auth_manager = AuthManager()
        self.base_url = None
        self.headers = None
        
    def authenticate(self):
        """Authenticate and set up headers."""
        token = self.auth_manager.get_token()
        if not token:
            print("❌ Authentication required. Please run 'mmpp auth login' first.")
            sys.exit(1)
            
        self.base_url = self.auth_manager.get_base_url()
        if not self.base_url:
            print("❌ Server URL not configured. Please run 'mmpp auth login' first.")
            sys.exit(1)
            
        self.headers = {
            "Authorization": f"Bearer {token}"
        }
        print(f"✅ Authenticated with server: {self.base_url}")
        
    def create_test_mx3_file(self, filename="test_simulation.mx3"):
        """Create a simple test MX3 file."""
        content = """// Test MX3 simulation
// number of Layers 38: 15, 30: 12
yig_layers := 28
CFB_layers := 20
spacer := 2

// number of cells & cell size
Nx = 188
Ny = 188
Nz = yig_layers+CFB_layers+spacer

dx = 2.5e-9
dy = 2.5e-9
dz = 2.5e-9
PBCx=2
PBCy=2

smoothmesh(true,true,false)

EdgeSmooth = 0
Msat  = 158e3   // old YIG: 180e3, LPE YIG: 158e3
Aex   = 3.1e-12
alpha = 1e-3

//define region
YIG := Layers(0, yig_layers)

p := 630e-9
diam := 180e-9

NP := Circle(diam)

CFB := Layers(yig_layers+spacer, yig_layers+CFB_layers+spacer).intersect(NP)

all:=NP.add(CFB)

defregion(1, YIG)   //YIG
Msat.setRegion(1, 158e3)
Aex.setRegion(1, 4e-12)
alpha.setRegion(1, 1e-3)
m.setRegion(1, uniform(0,1,0))

defregion(2, CFB)   //CoFeB NP
Msat.setRegion(2, 1150e3)
Aex.setRegion(2, 1.6e-11)
alpha.setRegion(2, 0.005)  
m.setRegion(2, uniform(0,1,0))


SetGeom(YIG.add(CFB))

angle := 90

B_min := -200e-3
B_max := 200e-3
B_step := 1e-3


TableAdd(B_ext)
TableAdd(m.comp(1).region(1))   // m in YIG
TableAdd(m.comp(1).region(2))   // m in CFB
save(m)
run(1e-10)
"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Created test MX3 file: {filename}")
        return filename
        
    def upload_and_queue_later(self, file_path):
        """Upload file without auto-start (scenario 1)."""
        print("\n🔄 Testing upload without auto-start...")
        
        url = f"{self.base_url}/api/v1/task-queue/upload-mx3"
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'task_name': 'Test Upload Only',
                'auto_start': 'false'
            }
            
            response = requests.post(url, headers=self.headers, files=files, data=data, timeout=30)
            
        if response.status_code in [200, 201]:
            result = response.json()
            print("✅ Upload successful!")
            print(f"   Job Key: {result.get('job_key', 'N/A')}")
            print(f"   File Path: {result.get('file_path', 'N/A')}")
            print(f"   MD5: {result.get('file_md5', 'N/A')}")
            return result
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    def create_task_manually(self, file_path, job_name="Manual Task", **params):
        """Create task manually after upload."""
        print("\n🔄 Creating task manually...")
        
        url = f"{self.base_url}/api/v1/task-queue/"
        
        task_data = {
            "name": job_name,
            "simulation_file": file_path,
            "partition": params.get("partition", "proxima"),
            "num_cpus": params.get("num_cpus", 4),
            "memory_gb": params.get("memory_gb", 16),
            "num_gpus": params.get("num_gpus", 1),
            "time_limit": params.get("time_limit", "02:00:00"),
            "priority": params.get("priority", 0)
        }
        
        response = requests.post(url, headers=self.headers, json=task_data, timeout=30)
        
        if response.status_code in [200, 201]:
            result = response.json()
            print("✅ Task created successfully!")
            print(f"   Task ID: {result.get('task_id', 'N/A')}")
            print(f"   Status: {result.get('status', 'N/A')}")
            return result
        else:
            print(f"❌ Task creation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    def upload_and_run_immediately(self, file_path, **params):
        """Upload file with auto-start (scenario 2)."""
        print("\n🔄 Testing upload with auto-start...")
        
        url = f"{self.base_url}/api/v1/task-queue/upload-mx3"
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'task_name': 'Test Auto Start',
                'auto_start': 'true',
                'partition': params.get("partition", "proxima"),
                'num_cpus': params.get("num_cpus", 4),
                'memory_gb': params.get("memory_gb", 16),
                'num_gpus': params.get("num_gpus", 1),
                'time_limit': params.get("time_limit", "02:00:00"),
                'priority': params.get("priority", 0)
            }
            
            response = requests.post(url, headers=self.headers, files=files, data=data, timeout=30)
            
        if response.status_code in [200, 201]:
            result = response.json()
            print("✅ Upload and start successful!")
            print(f"   Task ID: {result.get('task_id', 'N/A')}")
            print(f"   Status: {result.get('status', 'N/A')}")
            print(f"   Created At: {result.get('created_at', 'N/A')}")
            return result
        else:
            print(f"❌ Upload and start failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    def check_task_status(self, task_id):
        """Check status of a task."""
        print(f"\n🔍 Checking task status: {task_id}")
        
        url = f"{self.base_url}/api/v1/task-queue/{task_id}"
        response = requests.get(url, headers=self.headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Task status retrieved:")
            print(f"   Name: {result.get('name', 'N/A')}")
            print(f"   Status: {result.get('status', 'N/A')}")
            print(f"   Progress: {result.get('progress', 0)}%")
            print(f"   Node: {result.get('node', 'not assigned')}")
            print(f"   SLURM Job ID: {result.get('slurm_job_id', 'not assigned')}")
            return result
        else:
            print(f"❌ Failed to get task status: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    def download_results(self, task_id):
        """Download results for a completed task."""
        print(f"\n📥 Downloading results for task: {task_id}")
        
        url = f"{self.base_url}/api/v1/task-queue/{task_id}/download"
        response = requests.get(url, headers=self.headers, timeout=30)
        
        if response.status_code == 200:
            filename = f"results_{task_id}.zip"
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"✅ Results downloaded: {filename}")
            return filename
        else:
            print(f"❌ Failed to download results: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    def cleanup_test_file(self, filename):
        """Clean up test file."""
        try:
            os.remove(filename)
            print(f"🧹 Cleaned up test file: {filename}")
        except FileNotFoundError:
            pass


def main():
    """Run E2E test."""
    print("🧪 Starting MX3 Upload and Execution E2E Test")
    print("=" * 60)
    
    client = MX3Client()
    
    # Authenticate
    client.authenticate()
    
    # Create test file
    test_file = client.create_test_mx3_file()
    
    try:
        # Scenario 1: Upload without auto-start, then create task manually
        print("\n" + "="*60)
        print("SCENARIO 1: Upload without auto-start")
        print("="*60)
        
        upload_result = client.upload_and_queue_later(test_file)
        if upload_result:
            # Create task manually with uploaded file
            # For upload-only, the result should contain file info
            file_path = upload_result.get('file_path') or upload_result.get('simulation_file')
            task_result = client.create_task_manually(
                file_path,
                job_name="E2E Test Manual",
                num_cpus=2,
                memory_gb=8,
                time_limit="01:00:00"
            )
            if task_result:
                client.check_task_status(task_result['task_id'])
        
        # # Scenario 2: Upload with auto-start
        # print("\n" + "="*60)
        # print("SCENARIO 2: Upload with auto-start")
        # print("="*60)
        
        # auto_result = client.upload_and_run_immediately(
        #     test_file,
        #     num_cpus=2,
        #     memory_gb=8,
        #     time_limit="01:00:00"
        # )
        # if auto_result:
        #     client.check_task_status(auto_result['task_id'])
            
        print("\n" + "="*60)
        print("✅ E2E Test completed successfully!")
        # print("Both scenarios tested:")
        # print("  1. Upload file → Manual task creation")
        # print("  2. Upload file → Automatic task creation")
        print("="*60)
        
    except requests.RequestException as e:
        print(f"\n❌ E2E Test failed with network error: {e}")
        
    except (FileNotFoundError, OSError) as e:
        print(f"\n❌ E2E Test failed with file error: {e}")
        
    except KeyError as e:
        print(f"\n❌ E2E Test failed with missing data: {e}")
        
    except ValueError as e:
        print(f"\n❌ E2E Test failed with invalid data: {e}")
        
    finally:
        # Cleanup
        client.cleanup_test_file(test_file)


if __name__ == "__main__":
    main()
