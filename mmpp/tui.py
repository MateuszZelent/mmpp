"""
Modern TUI interface for MMPP using Textual framework.

This module provides a clean, modern Text User Interface with:
- Dashboard overview
- Authentication management (primary focus)
- Extensible architecture for future features
"""

import sys
from typing import Optional

try:
    from textual import on
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, Vertical
    from textual.css.query import NoMatches
    from textual.screen import Screen
    from textual.widgets import (
        Button,
        DataTable,
        Footer,
        Header,
        Input,
        Label,
        Log,
        Static,
        TabbedContent,
        TabPane,
    )
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    # Stub classes to prevent NameError
    class Screen:
        pass
    class App:
        pass

if TEXTUAL_AVAILABLE:
    
    class DashboardScreen(Screen):
        """Main dashboard with navigation to other features."""
        
        BINDINGS = [
            Binding("a", "auth", "Auth"),
            Binding("j", "jobs", "Jobs"),
            Binding("q", "quit", "Quit"),
        ]

        def compose(self) -> ComposeResult:
            """Compose the dashboard layout."""
            yield Header()
            
            with Vertical():
                yield Static("🚀 MMPP Dashboard 2", classes="screen-title")
                
                # Main navigation
                with Container(classes="nav-container"):
                    yield Button("🔐 Authentication", id="auth-btn", variant="primary")
                    yield Button("📊 Jobs", id="jobs-btn", variant="primary")
                    yield Button("🔄 Swap (Coming Soon)", id="swap-btn", variant="default", disabled=True)
                    
                # Quick info
                with Container(classes="info-container"):
                    yield Static("Welcome to MMPP TUI!", classes="welcome-text")
                    yield Static("Use 'A' for Authentication or click the button above.", classes="help-text")
                    
            yield Footer()

        @on(Button.Pressed, "#auth-btn")
        def handle_auth_button(self) -> None:
            """Navigate to authentication screen."""
            self.app.push_screen(AuthScreen())

        @on(Button.Pressed, "#jobs-btn")
        def handle_jobs_button(self) -> None:
            """Navigate to jobs screen."""
            self.app.push_screen(JobsScreen())

        def action_auth(self) -> None:
            """Navigate to authentication screen via key binding."""
            self.app.push_screen(AuthScreen())

        def action_jobs(self) -> None:
            """Navigate to jobs screen via key binding."""
            self.app.push_screen(JobsScreen())

        def action_quit(self) -> None:
            """Quit the application."""
            self.app.exit()


    class JobsScreen(Screen):
        """Jobs management screen - displays active containers/jobs."""
        
        BINDINGS = [
            Binding("escape", "back", "Back"),
            Binding("r", "refresh", "Refresh"),
        ]

        def __init__(self):
            super().__init__()
            self.auth_manager = None
            self.jobs_data = []
            self.is_loading = False

        def compose(self) -> ComposeResult:
            """Compose the jobs layout."""
            yield Header()
            
            with Vertical():
                yield Static("📊 Active Jobs & Containers", classes="screen-title")
                
                # Control panel
                with Horizontal(classes="control-panel"):
                    yield Button("🔄 Refresh", id="refresh-jobs", variant="primary")
                    yield Button("⏹️ Stop Selected", id="stop-jobs", variant="error", disabled=True)
                    yield Static("", id="jobs-status", classes="status-text")
                
                # Jobs table
                yield DataTable(id="jobs-table", cursor_type="row")
            
            yield Footer()

        async def on_mount(self) -> None:
            """Initialize jobs screen when mounted."""
            self._initialize_auth_manager()
            self._setup_table()
            await self._load_jobs()

        def _initialize_auth_manager(self) -> None:
            """Initialize authentication manager."""
            try:
                from mmpp.auth import AuthManager
                self.auth_manager = AuthManager()
            except ImportError as e:
                self._update_status(f"❌ Auth manager not available: {e}", "error")

        def _setup_table(self) -> None:
            """Setup the jobs table columns."""
            table = self.query_one("#jobs-table", DataTable)
            
            # Add columns with appropriate styling
            table.add_columns(
                "Job ID",
                "Name", 
                "User",
                "State",
                "Partition",
                "Node",
                "Time Used",
                "Time Left",
                "Memory"
            )

        async def _load_jobs(self) -> None:
            """Load active jobs from the server."""
            if self.is_loading:
                return
                
            self.is_loading = True
            self._update_status("🔄 Loading jobs...", "info")
            
            if not self.auth_manager:
                self._update_status("❌ Auth manager not available", "error")
                self.is_loading = False
                return
            
            # Check credentials
            credentials = self.auth_manager.load_credentials()
            if not credentials:
                self._update_status("❌ Not authenticated. Please login first.", "error")
                self.is_loading = False
                return
            
            server_url = credentials.get("server_url")
            token = credentials.get("token")
            
            if not server_url or not token:
                self._update_status("❌ Invalid credentials. Please login again.", "error")
                self.is_loading = False
                return
            
            try:
                # Import requests here to avoid circular imports
                import requests
                
                # Normalize server URL
                if not server_url.startswith(("http://", "https://")):
                    server_url = f"https://{server_url}"
                
                # Remove /login suffix if present
                if server_url.endswith("/login"):
                    server_url = server_url[:-6]
                
                # Construct API URL
                jobs_url = f"{server_url}/api/v1/jobs/active-jobs"
                
                headers = {
                    "Authorization": f"Bearer {token}",
                    "accept": "application/json",
                }
                
                response = requests.get(jobs_url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    self.jobs_data = response.json()
                    self._populate_table()
                    
                    total_jobs = len(self.jobs_data)
                    running_jobs = len([j for j in self.jobs_data if j.get("state") == "RUNNING"])
                    self._update_status(f"✅ Loaded {total_jobs} jobs ({running_jobs} running)", "success")
                    
                elif response.status_code == 401:
                    self._update_status("❌ Authentication failed. Please login again.", "error")
                elif response.status_code == 404:
                    self._update_status("❌ Jobs API not found on server.", "error")
                else:
                    self._update_status(f"❌ Server error: HTTP {response.status_code}", "error")
                    
            except requests.exceptions.ConnectionError:
                self._update_status(f"❌ Cannot connect to server: {server_url}", "error")
            except requests.exceptions.Timeout:
                self._update_status("❌ Request timeout", "error")
            except Exception as e:
                self._update_status(f"❌ Error loading jobs: {str(e)}", "error")
            
            self.is_loading = False

        def _populate_table(self) -> None:
            """Populate the table with jobs data."""
            table = self.query_one("#jobs-table", DataTable)
            
            # Clear existing rows
            table.clear()
            
            if not self.jobs_data:
                # Show empty state
                table.add_row("", "", "", "No active jobs", "", "", "", "", "")
                return
            
            for job in self.jobs_data:
                # Format state with appropriate styling
                state = job.get("state", "UNKNOWN")
                
                table.add_row(
                    str(job.get("job_id", "N/A")),
                    job.get("name", "N/A"),
                    job.get("user", "N/A"),
                    state,
                    job.get("partition", "N/A"),
                    job.get("node", "N/A"),
                    job.get("time_used", "N/A"),
                    job.get("time_left", "N/A"),
                    job.get("memory_requested", "N/A"),
                )

        def _update_status(self, message: str, status_type: str = "info") -> None:
            """Update status message."""
            try:
                status_widget = self.query_one("#jobs-status", Static)
                status_widget.update(message)
                
                # Update CSS class based on status type
                status_widget.remove_class("error", "success", "info", "warning")
                status_widget.add_class(status_type)
            except NoMatches:
                pass

        @on(Button.Pressed, "#refresh-jobs")
        async def handle_refresh(self) -> None:
            """Handle refresh button press."""
            await self._load_jobs()

        def action_back(self) -> None:
            """Return to dashboard."""
            self.app.pop_screen()

        def action_refresh(self) -> None:
            """Refresh jobs list via key binding."""
            self.run_worker(self._load_jobs())


    class AuthScreen(Screen):
        """Authentication management screen - main focus of the TUI."""
        
        BINDINGS = [
            Binding("escape", "back", "Back"),
            Binding("l", "login", "Login"),
        ]

        def __init__(self):
            super().__init__()
            self.auth_manager = None  # Will be AuthManager instance
            self.is_authenticated = False
            self.current_credentials = None

        def compose(self) -> ComposeResult:
            """Compose the authentication layout."""
            yield Header()
            
            with Vertical():
                yield Static("🔐 Authentication Management", classes="screen-title")
                
                with TabbedContent():
                    with TabPane("Login"):
                        yield self._create_login_panel()
                    
                    with TabPane("Status"):
                        yield self._create_status_panel()
                        
                    with TabPane("Logs"):
                        yield self._create_logs_panel()
                        
            yield Footer()

        def _create_login_panel(self) -> Container:
            """Create the login form."""
            return Container(
                Static("Server Connection", classes="section-title"),
                Label("Server URL:"),
                Input(placeholder="https://server.example.com", id="server-url"),
                Label("Authentication Token:"),
                Input(placeholder="Your authentication token", password=True, id="auth-token"),
                
                Horizontal(
                    Button("🔐 Login", id="login-btn", variant="primary"),
                    Button("🧪 Test Connection", id="test-btn", variant="default"),
                    Button("🔓 Logout", id="logout-btn", variant="error"),
                ),
                
                Static("", id="login-status", classes="status-message"),
                classes="login-panel"
            )

        def _create_status_panel(self) -> Container:
            """Create the status display."""
            return Container(
                Static("Authentication Status", classes="section-title"),
                Static("🔴 Not Authenticated", id="auth-status", classes="status-indicator"),
                Static("Server: Not Connected", id="server-info"),
                Static("Token: None", id="token-info"),
                
                Button("🔄 Refresh Status", id="refresh-btn", variant="default"),
                classes="status-panel"
            )

        def _create_logs_panel(self) -> Container:
            """Create the logs display."""
            return Container(
                Static("Authentication Logs", classes="section-title"),
                Log(highlight=True, id="auth-logs", auto_scroll=True),
                Button("🗑️ Clear Logs", id="clear-logs", variant="default"),
                classes="logs-panel"
            )

        def on_mount(self) -> None:
            """Initialize auth screen when mounted."""
            self._initialize_auth_manager()
            self._check_current_auth_status()
            self._log_activity("🔐 Auth screen initialized")

        def _initialize_auth_manager(self) -> None:
            """Initialize authentication manager."""
            try:
                from mmpp.auth import AuthManager
                self.auth_manager = AuthManager()
                self._log_activity("✅ Auth manager initialized")
            except ImportError as e:
                self._log_activity(f"❌ Failed to initialize auth manager: {e}")

        def _check_current_auth_status(self) -> None:
            """Check current authentication status."""
            if not self.auth_manager:
                self._update_status("❌ Auth manager not available")
                return
                
            try:
                credentials = self.auth_manager.load_credentials()
                if credentials:
                    server_url = credentials.get("server_url")
                    token = credentials.get("token")
                    
                    if server_url and token:
                        # Test connection
                        success, info = self.auth_manager.test_connection(server_url, token)
                        
                        if success:
                            user_info = info or {}
                            username = user_info.get("username", "Unknown")
                            self.is_authenticated = True
                            self.current_credentials = credentials
                            self._update_status(f"✅ Authenticated as {username}")
                            self._log_activity(f"✅ Authenticated as {username} on {server_url}")
                            
                            # Pre-populate server URL
                            try:
                                server_input = self.query_one("#server-url", Input)
                                server_input.value = server_url
                            except:
                                pass
                        else:
                            self.is_authenticated = False
                            self.current_credentials = None
                            error_msg = info.get("error", "Unknown error") if info else "Connection failed"
                            self._update_status(f"❌ Authentication expired: {error_msg}")
                            self._log_activity(f"❌ Auth check failed: {error_msg}")
                    else:
                        self.is_authenticated = False
                        self.current_credentials = None
                        self._update_status("❌ Invalid stored credentials")
                        self._log_activity("❌ Invalid stored credentials")
                else:
                    self.is_authenticated = False
                    self.current_credentials = None
                    self._update_status("⚠️ Not authenticated")
                    self._log_activity("ℹ️ No stored credentials found")
                    
                # Update UI after checking status
                self._update_interface_state()
                    
            except Exception as e:
                self.is_authenticated = False
                self.current_credentials = None
                self._update_status(f"❌ Error checking auth: {str(e)}")
                self._log_activity(f"❌ Auth check error: {str(e)}")
                self._update_interface_state()

        def _update_status(self, message: str) -> None:
            """Update status display."""
            try:
                status_widget = self.query_one("#login-status", Static)
                status_widget.update(message)
            except:
                pass

        def _update_interface_state(self) -> None:
            """Update interface based on authentication state."""
            try:
                # Update login button
                login_btn = self.query_one("#login-btn", Button)
                if self.is_authenticated:
                    login_btn.label = "🔓 Logout"
                    login_btn.variant = "error"
                else:
                    login_btn.label = "🔐 Login"
                    login_btn.variant = "primary"
                
                # Update status in Status tab
                try:
                    auth_status = self.query_one("#auth-status", Static)
                    if self.is_authenticated:
                        auth_status.update("🟢 Authenticated")
                    else:
                        auth_status.update("🔴 Not Authenticated")
                        
                    # Update server info
                    server_info = self.query_one("#server-info", Static)
                    if self.current_credentials:
                        server_url = self.current_credentials.get("server_url", "Not Connected")
                        server_info.update(f"Server: {server_url}")
                        
                        # Update token info (masked)
                        token_info = self.query_one("#token-info", Static)
                        token = self.current_credentials.get("token", "")
                        if token:
                            masked_token = token[:8] + "..." + token[-8:] if len(token) > 16 else "***"
                            token_info.update(f"Token: {masked_token}")
                        else:
                            token_info.update("Token: None")
                    else:
                        server_info.update("Server: Not Connected")
                        token_info = self.query_one("#token-info", Static)
                        token_info.update("Token: None")
                        
                except:
                    pass  # Status tab widgets might not be available yet
                    
            except Exception as e:
                self._log_activity(f"❌ Error updating interface: {str(e)}")

        def _log_activity(self, message: str) -> None:
            """Add message to activity log."""
            try:
                activity_log = self.query_one("#auth-logs", Log)
                from datetime import datetime
                timestamp = datetime.now().strftime("%H:%M:%S")
                activity_log.write_line(f"[dim]{timestamp}[/dim] {message}")
            except Exception:
                # Fallback - just print to console
                print(f"AUTH: {message}")

        @on(Button.Pressed, "#login-btn")
        async def handle_login(self) -> None:
            """Handle login/logout button press."""
            if not self.auth_manager:
                self._log_activity("❌ Auth manager not available")
                return
            
            if self.is_authenticated:
                # Perform logout
                await self._handle_logout()
            else:
                # Perform login
                await self._handle_login()

        async def _handle_login(self) -> None:
            """Perform login process."""
            server_url = self.query_one("#server-url", Input).value.strip()
            token = self.query_one("#auth-token", Input).value.strip()
            
            if not server_url or not token:
                self._update_status("❌ Please fill in both server URL and token")
                self._log_activity("❌ Login failed: Missing server URL or token")
                return
            
            self._update_status("🔄 Logging in...")
            self._log_activity(f"🔄 Attempting login to {server_url}")
            
            try:
                # Use CLI login method
                success, info = self.auth_manager.cli_login(server_url, token)
                
                if success and info:
                    access_token = info.get("access_token")
                    if access_token:
                        # Save credentials
                        self.auth_manager.save_credentials(server_url, access_token, info)
                        self.current_credentials = {
                            "server_url": server_url,
                            "token": access_token,
                            "user_info": info
                        }
                        self.is_authenticated = True
                        
                        username = info.get("username", "Unknown")
                        self._update_status(f"✅ Login successful as {username}!")
                        self._log_activity("✅ Login successful - credentials saved")
                        self._update_interface_state()  # Update UI after login
                    else:
                        self._update_status("❌ No access token received")
                        self._log_activity("❌ Login failed: No access token in response")
                else:
                    error_msg = info.get("error", "Unknown error") if info else "Unknown error"
                    self._update_status(f"❌ Login failed: {error_msg}")
                    self._log_activity(f"❌ Login failed: {error_msg}")
                    
            except Exception as e:
                self._update_status(f"❌ Login error: {str(e)}")
                self._log_activity(f"❌ Login exception: {str(e)}")

        async def _handle_logout(self) -> None:
            """Perform logout process."""
            self._log_activity("🔄 Logging out...")
            
            try:
                success = self.auth_manager.remove_credentials()
                if success:
                    self.current_credentials = None
                    self.is_authenticated = False
                    self._update_status("✅ Logged out successfully")
                    self._log_activity("✅ Logout successful - credentials removed")
                    self._update_interface_state()
                    
                    # Clear form fields
                    try:
                        self.query_one("#server-url", Input).value = ""
                        self.query_one("#auth-token", Input).value = ""
                    except:
                        pass
                else:
                    self._update_status("❌ Logout failed")
                    self._log_activity("❌ Logout failed")
            except Exception as e:
                self._update_status(f"❌ Logout error: {str(e)}")
                self._log_activity(f"❌ Logout exception: {str(e)}")

        @on(Button.Pressed, "#test-btn")
        async def handle_test_connection(self) -> None:
            """Handle test connection button press."""
            if not self.auth_manager:
                self._log_activity("❌ Auth manager not available")
                return
                
            server_url = self.query_one("#server-url", Input).value.strip()
            
            if not server_url:
                self._update_status("❌ Please enter server URL first")
                self._log_activity("❌ Test connection failed: No server URL")
                return
            
            self._update_status("🔄 Testing connection...")
            self._log_activity(f"🔄 Testing connection to {server_url}")
            
            try:
                # Test basic server connectivity
                connectivity_results = self.auth_manager.test_server_connectivity(server_url)
                
                # Find a successful connection
                successful_url = None
                for test_url, result in connectivity_results.items():
                    if "SUCCESS" in result:
                        successful_url = test_url
                        break
                
                if successful_url:
                    self._update_status(f"✅ Server reachable at {successful_url}")
                    self._log_activity(f"✅ Connection test successful: {successful_url}")
                    
                    # Update server URL field with working URL
                    try:
                        self.query_one("#server-url", Input).value = successful_url
                    except:
                        pass
                else:
                    self._update_status("❌ Server not reachable")
                    self._log_activity("❌ Connection test failed: Server not reachable")
                    
            except Exception as e:
                self._update_status(f"❌ Connection test error: {str(e)}")
                self._log_activity(f"❌ Connection test exception: {str(e)}")

        @on(Button.Pressed, "#refresh-btn")
        async def handle_refresh(self) -> None:
            """Handle refresh status button press."""
            self._log_activity("🔄 Refreshing status...")
            self._check_current_auth_status()

        @on(Button.Pressed, "#logout-btn")
        async def handle_logout(self) -> None:
            """Handle logout button press."""
            if not self.auth_manager:
                self._log_activity("❌ Auth manager not available")
                return
                
            self._log_activity("🔄 Logging out...")
            
            try:
                success = self.auth_manager.remove_credentials()
                if success:
                    self.is_authenticated = False
                    self._update_status("✅ Logged out successfully")
                    self._log_activity("✅ Logout successful - credentials removed")
                    
                    # Clear form fields
                    try:
                        self.query_one("#server-url", Input).value = ""
                        self.query_one("#auth-token", Input).value = ""
                    except:
                        pass
                else:
                    self._update_status("❌ Logout failed")
                    self._log_activity("❌ Logout failed")
            except Exception as e:
                self._update_status(f"❌ Logout error: {str(e)}")
                self._log_activity(f"❌ Logout exception: {str(e)}")
            
        @on(Button.Pressed, "#clear-logs")
        async def handle_clear_logs(self) -> None:
            """Handle clear logs button press."""
            try:
                activity_log = self.query_one("#auth-logs", Log)
                activity_log.clear()
                self._log_activity("🗑️ Logs cleared")
            except Exception:
                pass

        def action_back(self) -> None:
            """Return to dashboard."""
            self.app.pop_screen()

        def action_login(self) -> None:
            """Focus on login button."""
            try:
                self.query_one("#login-btn", Button).focus()
            except Exception:
                pass

    class MMPPApp(App):
        """Main MMPP TUI Application."""
        
        CSS = """
        /* Screen title */
        .screen-title {
            margin: 1;
            text-align: center;
            text-style: bold;
            color: #bd93f9;
        }
        
        /* Navigation container */
        .nav-container {
            margin: 2;
            padding: 1;
            background: #44475a;
            border: solid #bd93f9;
            height: 12;
        }
        
        .nav-container Button {
            margin: 1;
            height: 3;
        }
        
        /* Info container */
        .info-container {
            margin: 2;
            padding: 1;
        }
        
        .welcome-text {
            text-align: center;
            text-style: bold;
            color: #50fa7b;
            margin-bottom: 1;
        }
        
        .help-text {
            text-align: center;
            color: #f8f8f2;
        }
        
        /* Section titles */
        .section-title {
            text-style: bold;
            color: #bd93f9;
            margin-bottom: 1;
        }
        
        /* Status messages */
        .status-message {
            margin-top: 1;
            padding: 1;
        }
        
        /* Panels */
        .login-panel, .status-panel, .logs-panel {
            margin: 1;
            padding: 1;
        }
        
        /* Jobs screen styles */
        .control-panel {
            height: 3;
            margin: 1;
            padding: 0 2;
        }
        
        .control-panel Button {
            margin-right: 1;
        }
        
        .status-text {
            margin-left: 2;
            text-align: center;
        }
        
        .status-text.success {
            color: #50fa7b;
        }
        
        .status-text.error {
            color: #ff5555;
        }
        
        .status-text.info {
            color: #8be9fd;
        }
        
        .status-text.warning {
            color: #f1fa8c;
        }
        
        /* Data table styling */
        DataTable {
            margin: 1;
        }
        """
        
        BINDINGS = [
            Binding("ctrl+q", "quit", "Quit"),
            Binding("ctrl+d", "dashboard", "Dashboard"),
        ]

        def on_mount(self) -> None:
            """Initialize app when mounted."""
            self.title = "MMPP TUI - Professional Magnetic Analysis"
            self.sub_title = "Authentication & Management Interface"
            
            # Start with dashboard
            self.push_screen(DashboardScreen())

        def action_quit(self) -> None:
            """Quit the application."""
            self.exit()

        def action_dashboard(self) -> None:
            """Return to dashboard."""
            # Clear screen stack and show dashboard
            while len(self.screen_stack) > 1:
                self.pop_screen()


def main() -> None:
    """Main entry point for the TUI."""
    if not TEXTUAL_AVAILABLE:
        print("❌ Textual not available. This should not happen!")
        return
    
    try:
        app = MMPPApp()
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ TUI Error: {e}")
        print("💡 Please report this issue")
        sys.exit(1)


if __name__ == "__main__":
    main()
