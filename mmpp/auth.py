"""
Authentication module for MMPP server connections.
Handles authentication with containers_admin2 server.
"""

import getpass
import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

import requests
import yaml

from .logging_config import get_mmpp_logger

log = get_mmpp_logger("mmpp.auth")


class AuthManager:
    """Manages authentication credentials and server connections."""

    def __init__(self):
        self.config_dir = Path.home() / ".mmpp"
        self.token_file = self.config_dir / "auth.yaml"
        self.config_dir.mkdir(exist_ok=True)

    def load_credentials(self) -> Optional[dict[str, Any]]:
        """Load stored authentication credentials."""
        if not self.token_file.exists():
            return None

        try:
            with open(self.token_file) as f:
                return yaml.safe_load(f)
        except Exception as e:
            log.error(f"Error loading credentials: {e}")
            return None

    def save_credentials(
        self, server_url: str, token: str, user_info: Optional[dict] = None
    ) -> None:
        """Save authentication credentials to file."""
        credentials = {
            "server_url": server_url.rstrip("/"),
            "token": token,
            "user_info": user_info or {},
            "created_at": str(
                Path(self.token_file).stat().st_mtime
                if self.token_file.exists()
                else "new"
            ),
        }

        try:
            with open(self.token_file, "w") as f:
                yaml.safe_dump(credentials, f, default_flow_style=False)

            # Set file permissions to be readable only by owner
            os.chmod(self.token_file, 0o600)
            log.info(f"Credentials saved to {self.token_file}")

        except Exception as e:
            log.error(f"Error saving credentials: {e}")
            raise

    def remove_credentials(self) -> bool:
        """Remove stored authentication credentials."""
        if self.token_file.exists():
            try:
                os.remove(self.token_file)
                log.info("Authentication credentials removed")
                return True
            except Exception as e:
                log.error(f"Error removing credentials: {e}")
                return False
        return True

    def cli_login(self, server_url: str, cli_token: str) -> tuple[bool, Optional[dict]]:
        """Login using CLI token to get JWT access token."""
        try:
            # Normalize server URL
            if not server_url.startswith(("http://", "https://")):
                server_url = f"https://{server_url}"
            
            print(f"🔧 DEBUG: Server URL: {server_url}")
            
            # Remove /login suffix if present (user might copy full login URL)
            if server_url.endswith("/login"):
                server_url = server_url[:-6]
                print(f"🔧 DEBUG: Removed /login suffix: {server_url}")
            
            # Use containers_admin2 cli-login endpoint
            login_url = urljoin(server_url, "/api/v1/auth/cli-login")
            print(f"🔧 DEBUG: Login URL: {login_url}")
            
            headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
            }
            
            # Send cli_token in JSON body, not as Bearer token
            data = {
                "cli_token": cli_token
            }
            
            print(f"🔧 DEBUG: Headers: {headers}")
            print(f"🔧 DEBUG: Data: {data}")
            
            print(f"🔧 DEBUG: Making POST request to {login_url}")
            response = requests.post(login_url, headers=headers, json=data, timeout=10)
            print(f"🔧 DEBUG: Response status: {response.status_code}")
            print(f"🔧 DEBUG: Response headers: {dict(response.headers)}")
            print(f"🔧 DEBUG: Response text: {response.text[:300]}...")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"🔧 DEBUG: Successfully parsed JSON: {result}")
                    
                    if "access_token" in result:
                        return True, {
                            "access_token": result["access_token"],
                            "token_type": result.get("token_type", "bearer"),
                            "login_method": "cli_token"
                        }
                    else:
                        return False, {"error": "No access token in response"}
                        
                except Exception as json_err:
                    print(f"🔧 DEBUG: JSON parse error: {json_err}")
                    return False, {"error": f"Invalid response format: {json_err}"}
            elif response.status_code == 401:
                print("🔧 DEBUG: Authentication failed - invalid CLI token")
                return False, {"error": "Invalid CLI token"}
            elif response.status_code == 404:
                print("🔧 DEBUG: API endpoint not found")
                return False, {"error": "Server API not found - check server URL"}
            else:
                print(f"🔧 DEBUG: Unexpected status code: {response.status_code}")
                return False, {"error": f"Server returned status {response.status_code}"}
                
        except requests.exceptions.ConnectionError as conn_err:
            print(f"🔧 DEBUG: Connection error: {conn_err}")
            return False, {"error": "Cannot connect to server"}
        except requests.exceptions.Timeout as timeout_err:
            print(f"🔧 DEBUG: Timeout error: {timeout_err}")
            return False, {"error": "Connection timeout"}
        except Exception as e:
            print(f"🔧 DEBUG: General exception: {e}")
            return False, {"error": str(e)}

    def test_connection(
        self, server_url: str, token: str
    ) -> tuple[bool, Optional[dict]]:
        """Test connection to containers_admin2 server using proper API endpoints."""
        try:
            # Normalize server URL
            if not server_url.startswith(("http://", "https://")):
                server_url = f"https://{server_url}"

            print(f"🔧 DEBUG: Original server URL: {server_url}")

            # Remove /login suffix if present (user might copy full login URL)
            if server_url.endswith("/login"):
                server_url = server_url[:-6]
                print(f"🔧 DEBUG: Removed /login suffix: {server_url}")

            # Use containers_admin2 /me endpoint for verification
            verify_url = urljoin(server_url, "/api/v1/auth/me")
            print(f"🔧 DEBUG: Verify URL: {verify_url}")

            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            print(f"🔧 DEBUG: Headers: {headers}")

            print(f"🔧 DEBUG: Making GET request to {verify_url}")
            response = requests.get(verify_url, headers=headers, timeout=10)
            print(f"🔧 DEBUG: Response status: {response.status_code}")
            print(f"🔧 DEBUG: Response headers: {dict(response.headers)}")
            print(f"🔧 DEBUG: Response text: {response.text[:300]}...")

            if response.status_code == 200:
                try:
                    user_info = response.json()
                    print(f"🔧 DEBUG: Successfully parsed JSON: {user_info}")
                    return True, user_info
                except Exception as json_err:
                    print(f"🔧 DEBUG: JSON parse error: {json_err}")
                    return True, {"status": "authenticated"}
            elif response.status_code == 401:
                print("🔧 DEBUG: Authentication failed - invalid token")
                return False, {"error": "Invalid token"}
            elif response.status_code == 404:
                print("🔧 DEBUG: API endpoint not found")
                return False, {"error": "Server API not found - check server URL"}
            else:
                print(f"🔧 DEBUG: Unexpected status code: {response.status_code}")
                return False, {
                    "error": f"Server returned status {response.status_code}"
                }

        except requests.exceptions.ConnectionError as conn_err:
            print(f"🔧 DEBUG: Connection error: {conn_err}")
            return False, {"error": "Cannot connect to server"}
        except requests.exceptions.Timeout as timeout_err:
            print(f"🔧 DEBUG: Timeout error: {timeout_err}")
            return False, {"error": "Connection timeout"}
        except Exception as e:
            return False, {"error": str(e)}

    def get_current_status(self) -> dict[str, Any]:
        """Get current authentication status."""
        credentials = self.load_credentials()

        if not credentials:
            return {"authenticated": False, "message": "No stored credentials found"}

        # Test if credentials are still valid
        server_url = credentials.get("server_url")
        token = credentials.get("token")

        if not server_url or not token:
            return {"authenticated": False, "message": "Invalid stored credentials"}

        success, info = self.test_connection(server_url, token)

        if success:
            return {
                "authenticated": True,
                "server_url": server_url,
                "user_info": credentials.get("user_info", {}),
                "message": "Successfully authenticated",
            }
        else:
            return {
                "authenticated": False,
                "server_url": server_url,
                "error": info.get("error", "Unknown error"),
                "message": "Authentication failed",
            }

    def login_via_api(
        self, server_url: str, username: str, password: str
    ) -> tuple[bool, Optional[dict]]:
        """
        Login via API using username/password (optional feature for future).
        Currently just provides framework - most users will use token directly.
        """
        try:
            # Normalize server URL
            if not server_url.startswith(("http://", "https://")):
                server_url = f"https://{server_url}"

            # Remove /login suffix if present
            if server_url.endswith("/login"):
                server_url = server_url[:-6]

            # Use containers_admin2 login endpoint
            login_url = urljoin(server_url, "/api/v1/auth/login")

            data = {"username": username, "password": password}

            response = requests.post(login_url, json=data, timeout=10)

            if response.status_code == 200:
                try:
                    result = response.json()
                    if "token" in result:
                        return True, result
                    else:
                        return False, {"error": "No token in response"}
                except Exception as e:
                    return False, {"error": f"Invalid response format: {e}"}
            elif response.status_code == 401:
                return False, {"error": "Invalid credentials"}
            elif response.status_code == 404:
                return False, {"error": "Login API not found - check server URL"}
            else:
                return False, {
                    "error": f"Login failed with status {response.status_code}"
                }

        except requests.exceptions.ConnectionError:
            return False, {"error": "Cannot connect to server"}
        except requests.exceptions.Timeout:
            return False, {"error": "Connection timeout"}
        except Exception as e:
            return False, {"error": str(e)}

    def test_server_connectivity(self, server_url: str) -> dict:
        """Test basic connectivity to server on different ports/protocols."""
        results = {}
        
        # Test different variations of the URL
        test_urls = []
        
        if server_url.startswith('http'):
            test_urls.append(server_url)
        else:
            test_urls.extend([
                f"https://{server_url}",
                f"http://{server_url}",
                f"https://{server_url}:443",
                f"http://{server_url}:80",
                f"http://{server_url}:8000",
                f"https://{server_url}:8000"
            ])
        
        for test_url in test_urls:
            try:
                response = requests.get(f"{test_url}/", timeout=5)
                results[test_url] = f"SUCCESS - Status: {response.status_code}"
            except requests.exceptions.ConnectionError as e:
                results[test_url] = f"CONNECTION_ERROR - {str(e)[:100]}..."
            except requests.exceptions.Timeout:
                results[test_url] = "TIMEOUT"
            except Exception as e:
                results[test_url] = f"ERROR - {str(e)[:100]}..."
        
        return results


def login_to_server(server_url: str, token: Optional[str] = None) -> bool:
    """Login to containers_admin2 server using CLI token."""
    auth_manager = AuthManager()

    # Get token if not provided
    if not token:
        token = getpass.getpass("Enter your CLI authentication token: ")

    if not token.strip():
        print("❌ Token cannot be empty")
        return False

    print(f"🔍 Logging in to {server_url}...")

    # Use CLI login method
    success, info = auth_manager.cli_login(server_url, token)

    if success:
        print("✅ Login successful!")
        
        # Extract access token from response
        access_token = info.get("access_token")
        if not access_token:
            print("❌ No access token received")
            return False

        # Save credentials with JWT access token
        try:
            auth_manager.save_credentials(server_url, access_token, info)
            print(f"🔐 Credentials saved to {auth_manager.token_file}")
            return True

        except Exception as e:
            print(f"❌ Error saving credentials: {e}")
            return False
    else:
        error_msg = info.get("error", "Unknown error")
        print(f"❌ Authentication failed: {error_msg}")
        return False


def show_auth_status() -> None:
    """Show current authentication status."""
    auth_manager = AuthManager()
    status = auth_manager.get_current_status()

    print("🔐 Authentication Status")
    print("=" * 50)

    if status["authenticated"]:
        print("✅ Status: Authenticated")
        print(f"🌐 Server: {status['server_url']}")

        user_info = status.get("user_info", {})
        if user_info:
            if "username" in user_info:
                print(f"👤 User: {user_info['username']}")
            if "role" in user_info:
                print(f"🏷️  Role: {user_info['role']}")
    else:
        print("❌ Status: Not authenticated")
        print(f"💬 Message: {status['message']}")

        if "server_url" in status:
            print(f"🌐 Server: {status['server_url']}")
            print(f"⚠️  Error: {status.get('error', 'Unknown error')}")


def logout_from_server() -> bool:
    """Logout from server (remove stored credentials)."""
    auth_manager = AuthManager()

    if auth_manager.remove_credentials():
        print("✅ Successfully logged out")
        return True
    else:
        print("❌ Error during logout")
        return False
