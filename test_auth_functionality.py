#!/usr/bin/env python3
"""
Test script for MMPP Authentication functionality.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_auth_manager():
    """Test the AuthManager functionality."""
    try:
        from mmpp.auth import AuthManager
        
        print("🔧 Testing AuthManager...")
        
        # Create AuthManager
        auth_manager = AuthManager()
        print("✅ AuthManager created successfully")
        
        # Test loading credentials
        credentials = auth_manager.load_credentials()
        if credentials:
            print(f"📋 Found stored credentials for: {credentials.get('server_url', 'unknown')}")
            
            # Test if credentials are still valid
            server_url = credentials.get("server_url")
            token = credentials.get("token")
            
            if server_url and token:
                print(f"🔍 Testing connection to {server_url}...")
                success, info = auth_manager.test_connection(server_url, token)
                
                if success:
                    print("✅ Connection test successful!")
                    user_info = info or {}
                    if "username" in user_info:
                        print(f"👤 Authenticated as: {user_info['username']}")
                    if "max_containers" in user_info:
                        print(f"🐳 Max containers: {user_info['max_containers']}")
                    if "max_gpus" in user_info:
                        print(f"🖥️ Max GPUs: {user_info['max_gpus']}")
                else:
                    error_msg = info.get('error', 'Unknown error') if info else 'Unknown error'
                    print(f"❌ Connection test failed: {error_msg}")
            else:
                print("❌ Invalid stored credentials")
        else:
            print("ℹ️ No stored credentials found")
        
        # Test server connectivity (if credentials exist)
        if credentials and credentials.get("server_url"):
            server_url = credentials["server_url"]
            print(f"\n🌐 Testing server connectivity to {server_url}...")
            
            connectivity_results = auth_manager.test_server_connectivity(server_url)
            
            for test_url, result in connectivity_results.items():
                status_emoji = "✅" if "SUCCESS" in result else "❌"
                print(f"  {status_emoji} {test_url}: {result}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_auth_status():
    """Test authentication status function."""
    try:
        from mmpp.auth import show_auth_status
        
        print("\n" + "="*50)
        show_auth_status()
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing auth status: {e}")
        return False

if __name__ == "__main__":
    print("🚀 MMPP Authentication Test")
    print("=" * 50)
    
    success = test_auth_manager()
    
    if success:
        test_auth_status()
        print("\n✅ All authentication tests completed!")
    else:
        print("\n❌ Authentication tests failed!")
        sys.exit(1)
