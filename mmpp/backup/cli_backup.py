"""
Modern TUI for MMPP library using Textual framework.

A professional Text User Interface built with Textual, featuring:
- Dracula theme for dark mode aesthetics
- Interactive dashboard with real-time updates
- Modular architecture with separate screens for different functions
- Rich formatting and responsive design
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, Vertical
    from textual.css.query import NoMatches
    from textual.message import Message
    from textual.reactive import reactive
    from textual.screen import Screen
    from textual.widgets import (
        Button,
        DataTable,
        Footer,
        Grid,
        Header,
        Input,
        Label,
        Log,
        Markdown,
        ProgressBar,
        Select,
        Static,
        TabbedContent,
        TabPane,
        Tree,
    )
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    # Provide stub classes to prevent NameError
    class Screen:
        pass
    class Static:
        pass
    class Button:
        pass
    class App:
        pass
    
    print("❌ Textual not available. Install with: pip install textual")
    print("Falling back to traditional CLI...")

# Always import these - needed for both TUI and fallback
from .cli.logging_config import get_mmpp_logger

# Import MMPP version info
try:
    from . import __author__, __version__
except ImportError:
    __version__ = "unknown"
    __author__ = "MMPP Team"

# Only define TUI classes if Textual is available
if TEXTUAL_AVAILABLE:
    # Initialize logger
    log = get_mmpp_logger("mmpp.tui")

    class DashboardScreen(Screen):
    """Main dashboard screen."""

    BINDINGS = [
        Binding("j", "jobs", "Jobs"),
        Binding("a", "auth", "Auth"),
        Binding("s", "swap", "Swap"),
        Binding("i", "info", "Info"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.auth_manager = None
        self.auth_status = "Checking..."

    async def on_mount(self) -> None:
        """Initialize dashboard when mounted."""
        await self._check_authentication_status()

    async def _check_authentication_status(self) -> None:
        """Check authentication status and update display."""
        try:
            from .auth import AuthManager
            self.auth_manager = AuthManager()
            
            credentials = self.auth_manager.load_credentials()
            if credentials:
                server_url = credentials.get("server_url")
                token = credentials.get("token")
                
                if server_url and token:
                    # Test connection in background
                    success, info = self.auth_manager.test_connection(server_url, token)
                    
                    if success:
                        user_info = info or {}
                        username = user_info.get("username", "Unknown")
                        self.auth_status = f"✅ Authenticated as {username}"
                    else:
                        self.auth_status = "❌ Authentication expired"
                else:
                    self.auth_status = "❌ Invalid credentials"
            else:
                self.auth_status = "⚠️ Not authenticated"
                
        except Exception as e:
            self.auth_status = f"❌ Auth error: {str(e)}"
        
        # Update the status display
        try:
            auth_display = self.query_one("#auth-status-display", Static)
            auth_display.update(self.auth_status)
        except NoMatches:
            pass

    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        yield Header()
        
        with Vertical():
            # Auth status bar
            with Horizontal(classes="auth-status-bar"):
                yield Static("🔐 Auth Status:", classes="auth-label")
                yield Static("Checking...", id="auth-status-display", classes="auth-status")
                yield Button("🔄 Refresh", id="refresh-auth", variant="default")
            
            yield Static("🚀 MMPP Dashboard", classes="screen-title")
            
            # Main menu grid
            with Grid(classes="dashboard-grid"):
                yield Button("� Job Management", id="jobs-btn", variant="primary")
                yield Button("� Authentication", id="auth-btn", variant="success")
                yield Button("🔄 Parameter Swap", id="swap-btn", variant="warning")
                yield Button("ℹ️ System Info", id="info-btn", variant="default")
            
            # Quick stats
            with Container(classes="stats-container"):
                yield Static("Quick Stats", classes="section-title")
                yield Static("Loading system information...", id="stats-display")
        
        yield Footer()

    @on(Button.Pressed, "#refresh-auth")
    async def handle_refresh_auth(self) -> None:
        """Refresh authentication status."""
        auth_display = self.query_one("#auth-status-display", Static)
        auth_display.update("🔄 Checking...")
        await self._check_authentication_status()

    @on(Button.Pressed, "#jobs-btn")
    def action_jobs(self) -> None:
        """Open jobs screen."""
        self.app.push_screen(JobsScreen())

    @on(Button.Pressed, "#auth-btn")
    def action_auth(self) -> None:
        """Open authentication screen."""
        self.app.push_screen(AuthScreen())

    @on(Button.Pressed, "#swap-btn")
    def action_swap(self) -> None:
        """Open swap screen."""
        self.app.push_screen(SwapScreen())

    @on(Button.Pressed, "#info-btn")
    def action_info(self) -> None:
        """Open info screen."""
        self.app.push_screen(InfoScreen())

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()

    def _create_welcome_markdown(self) -> Static:
        """Create welcome message as Static widget for better compatibility."""
        welcome_text = f"""🧲 MMPP - Micro Magnetic Post Processing

Version: {__version__} | Author: {__author__}

A professional library for magnetic simulation analysis and post-processing.

Quick Actions:
• A - Authentication management
• J - Job monitoring and control
• S - Simulation parameter swapping
• I - System information and diagnostics"""
        return Static(welcome_text, classes="welcome-text")

    def _create_status_panel(self) -> Container:
        """Create system status panel."""
        return Container(
            Static("🟢 System Ready", classes="status-item"),
            Static("🔗 Connection: Not Connected", id="connection-status", classes="status-item"),
            Static("💾 Cache: Clean", classes="status-item"),
            Static("📂 Working Directory: " + str(Path.cwd()), classes="status-item"),
            classes="status-panel"
        )

    def _create_stats_panel(self) -> Container:
        """Create quick statistics panel."""
        return Container(
            Static("📈 Active Jobs: 0", id="stat-jobs"),
            Static("⚡ FFT Computations: 0", id="stat-fft"),
            Static("🎯 Mode Analysis: 0", id="stat-modes"),
            Static("📊 Plots Generated: 0", id="stat-plots"),
            classes="stats-panel"
        )

    @on(Button.Pressed, "#btn-auth")
    def handle_auth_button(self) -> None:
        """Handle authentication button press."""
        self.app.push_screen("auth")

    @on(Button.Pressed, "#btn-jobs")
    def handle_jobs_button(self) -> None:
        """Handle jobs button press."""
        self.app.push_screen("jobs")

    @on(Button.Pressed, "#btn-swap")
    def handle_swap_button(self) -> None:
        """Handle swap button press."""
        self.app.push_screen("swap")

    @on(Button.Pressed, "#btn-info")
    def handle_info_button(self) -> None:
        """Handle info button press."""
        self.app.push_screen("info")

    def action_auth(self) -> None:
        """Navigate to authentication screen."""
        self.app.push_screen("auth")

    def action_jobs(self) -> None:
        """Navigate to jobs screen."""
        self.app.push_screen("jobs")

    def action_swap(self) -> None:
        """Navigate to swap screen."""
        self.app.push_screen("swap")

    def action_info(self) -> None:
        """Navigate to info screen."""
        self.app.push_screen("info")

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()

    def on_mount(self) -> None:
        """Initialize dashboard when mounted."""
        log.info("Dashboard mounted - initializing system checks")
        self._update_activity_log("🚀 MMPP TUI Started")
        self._check_system_status()

    def _update_activity_log(self, message: str) -> None:
        """Add message to activity log."""
        try:
            activity_log = self.query_one("#activity-log", Log)
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            activity_log.write_line(f"[dim]{timestamp}[/dim] {message}")
        except NoMatches:
            log.warning("Activity log widget not found")

    def _check_system_status(self) -> None:
        """Check and update system status."""
        try:
            # Update connection status
            connection_status = self.query_one("#connection-status", Static)
            # TODO: Implement actual connection check
            connection_status.update("🔗 Connection: Checking...")
            
            self._update_activity_log("🔍 System status check completed")
        except NoMatches:
            log.warning("Status widgets not found")


class AuthScreen(Screen):
    """Authentication management screen."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("l", "login", "Login"),
        Binding("s", "status", "Status"),
        Binding("o", "logout", "Logout"),
    ]

    def __init__(self):
        super().__init__()
        self.auth_manager = None
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
                
                with TabPane("Settings"):
                    yield self._create_settings_panel()
        
        yield Footer()

    def on_mount(self) -> None:
        """Initialize auth screen when mounted."""
        self._initialize_auth_manager()
        self._load_saved_credentials()
        self._update_interface_state()
        self._log_activity("🔐 Auth screen initialized")

    def _initialize_auth_manager(self) -> None:
        """Initialize authentication manager."""
        try:
            from .auth import AuthManager
            self.auth_manager = AuthManager()
            self._log_activity("✅ Auth manager initialized")
        except ImportError as e:
            self._log_activity(f"❌ Failed to initialize auth manager: {e}")
            self._update_status("❌ Authentication module not available", "error")

    def _load_saved_credentials(self) -> None:
        """Load saved credentials if they exist."""
        if not self.auth_manager:
            return
            
        try:
            credentials = self.auth_manager.load_credentials()
            if credentials:
                self.current_credentials = credentials
                self._log_activity("📋 Found saved credentials")
                
                # Pre-populate form fields
                server_url = credentials.get("server_url", "")
                if server_url:
                    try:
                        server_input = self.query_one("#server-url", Input)
                        server_input.value = server_url
                        self._log_activity(f"🌐 Pre-populated server URL: {server_url}")
                    except NoMatches:
                        pass
                
                # Check if credentials are still valid
                self._check_authentication_status()
            else:
                self._log_activity("ℹ️ No saved credentials found")
                self.is_authenticated = False
        except Exception as e:
            self._log_activity(f"❌ Error loading credentials: {e}")

    def _check_authentication_status(self) -> None:
        """Check current authentication status."""
        if not self.auth_manager or not self.current_credentials:
            self.is_authenticated = False
            return
            
        server_url = self.current_credentials.get("server_url")
        token = self.current_credentials.get("token")
        
        if not server_url or not token:
            self.is_authenticated = False
            return
            
        self._log_activity("🔍 Checking authentication status...")
        
        # Test connection
        try:
            success, info = self.auth_manager.test_connection(server_url, token)
            self.is_authenticated = success
            
            if success:
                self._log_activity("✅ Authentication valid")
                user_info = info or {}
                username = user_info.get("username", "Unknown")
                self._update_status(f"✅ Authenticated as {username}", "success")
            else:
                error_msg = info.get("error", "Unknown error") if info else "Unknown error"
                self._log_activity(f"❌ Authentication invalid: {error_msg}")
                self._update_status("❌ Stored credentials are invalid", "error")
                
        except Exception as e:
            self._log_activity(f"❌ Error checking auth status: {e}")
            self.is_authenticated = False

    def _update_interface_state(self) -> None:
        """Update interface based on authentication state."""
        try:
            if self.is_authenticated:
                # Update login button to show logout
                try:
                    login_btn = self.query_one("#login-btn", Button)
                    login_btn.label = "🔓 Logout"
                    login_btn.variant = "error"
                    # Don't change ID - use classes instead
                    login_btn.add_class("logout-mode")
                    login_btn.remove_class("login-mode")
                except NoMatches:
                    pass
                    
                # Update status display
                self._update_auth_status_display()
            else:
                # Update button to show login
                try:
                    login_btn = self.query_one("#login-btn", Button)
                    login_btn.label = "🔐 Login"
                    login_btn.variant = "primary"
                    # Don't change ID - use classes instead
                    login_btn.add_class("login-mode")
                    login_btn.remove_class("logout-mode")
                except NoMatches:
                    pass
        except Exception as e:
            self._log_activity(f"❌ Error updating interface: {e}")

    def _update_auth_status_display(self) -> None:
        """Update authentication status in the Status tab."""
        if not self.current_credentials:
            return
            
        try:
            # Update status indicator
            auth_status = self.query_one("#auth-status", Static)
            if self.is_authenticated:
                auth_status.update("🟢 Authenticated")
                auth_status.remove_class("error")
                auth_status.add_class("success")
            else:
                auth_status.update("🔴 Not Authenticated")
                auth_status.remove_class("success")
                auth_status.add_class("error")
            
            # Update server info
            server_info = self.query_one("#server-info", Static)
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
                
        except NoMatches:
            pass

    def _log_activity(self, message: str) -> None:
        """Add message to activity log."""
        try:
            activity_log = self.query_one("#auth-logs", Log)
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            activity_log.write_line(f"[dim]{timestamp}[/dim] {message}")
        except NoMatches:
            # Fallback to main logger
            log.info(f"AUTH: {message}")

    def _create_logs_panel(self) -> Container:
        """Create logs panel for authentication activities."""
        return Container(
            Static("Authentication Logs", classes="section-title"),
            Log(highlight=True, id="auth-logs", auto_scroll=True),
            Button("🗑️ Clear Logs", id="clear-logs", variant="default"),
            classes="logs-panel"
        )

    def _create_login_panel(self) -> Container:
        """Create login form panel."""
        return Container(
            Static("Server Connection", classes="section-title"),
            Input(placeholder="Server URL (e.g., https://server.example.com)", id="server-url"),
            Input(placeholder="Authentication Token", password=True, id="auth-token"),
            Horizontal(
                Button("🔐 Login", id="login-btn", variant="primary"),
                Button("🧪 Test Connection", id="test-btn", variant="default"),
            ),
            Static("", id="login-status", classes="status-message"),
            classes="login-panel"
        )

    def _create_status_panel(self) -> Container:
        """Create authentication status panel."""
        return Container(
            Static("Authentication Status", classes="section-title"),
            Static("🔴 Not Authenticated", id="auth-status", classes="status-indicator"),
            Static("Server: Not Connected", id="server-info"),
            Static("Token: None", id="token-info"),
            Static("Last Login: Never", id="last-login"),
            
            Button("🔄 Refresh Status", id="refresh-status", variant="default"),
            classes="status-panel"
        )

    def _create_settings_panel(self) -> Container:
        """Create authentication settings panel."""
        return Container(
            Static("Authentication Settings", classes="section-title"),
            
            Label("Auto-save credentials:"),
            Select([("Yes", True), ("No", False)], value=True, id="auto-save"),
            
            Label("Connection timeout (seconds):"),
            Input(value="30", id="timeout"),
            
            Button("💾 Save Settings", id="save-settings", variant="success"),
            classes="settings-panel"
        )

    @on(Button.Pressed, "#login-btn")
    async def handle_login_logout(self) -> None:
        """Handle login/logout button press."""
        if self.is_authenticated:
            # Perform logout
            await self._handle_logout()
        else:
            # Perform login
            await self._handle_login()

    async def _handle_login(self) -> None:
        """Handle login process."""
        server_url = self.query_one("#server-url", Input).value.strip()
        auth_token = self.query_one("#auth-token", Input).value.strip()
        
        if not server_url or not auth_token:
            self._update_status("❌ Please fill in both server URL and token", "error")
            self._log_activity("❌ Login failed: Missing server URL or token")
            return
        
        self._update_status("🔄 Logging in...", "info")
        self._log_activity(f"🔄 Attempting login to {server_url}")
        
        if not self.auth_manager:
            self._update_status("❌ Authentication manager not available", "error")
            self._log_activity("❌ Authentication manager not available")
            return
        
        try:
            # Use CLI login method
            success, info = self.auth_manager.cli_login(server_url, auth_token)
            
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
                    
                    self._update_status("✅ Login successful!", "success")
                    self._log_activity("✅ Login successful - credentials saved")
                    self._update_interface_state()
                    self._update_auth_status_display()
                else:
                    self._update_status("❌ No access token received", "error")
                    self._log_activity("❌ Login failed: No access token in response")
            else:
                error_msg = info.get("error", "Unknown error") if info else "Unknown error"
                self._update_status(f"❌ Login failed: {error_msg}", "error")
                self._log_activity(f"❌ Login failed: {error_msg}")
                
        except Exception as e:
            self._update_status(f"❌ Login error: {str(e)}", "error")
            self._log_activity(f"❌ Login exception: {str(e)}")

    async def _handle_logout(self) -> None:
        """Handle logout process."""
        self._log_activity("🔄 Logging out...")
        
        if self.auth_manager:
            try:
                success = self.auth_manager.remove_credentials()
                if success:
                    self.current_credentials = None
                    self.is_authenticated = False
                    self._update_status("✅ Logged out successfully", "success")
                    self._log_activity("✅ Logout successful - credentials removed")
                    self._update_interface_state()
                    self._update_auth_status_display()
                    
                    # Clear form fields
                    try:
                        self.query_one("#server-url", Input).value = ""
                        self.query_one("#auth-token", Input).value = ""
                    except NoMatches:
                        pass
                else:
                    self._update_status("❌ Logout failed", "error")
                    self._log_activity("❌ Logout failed")
            except Exception as e:
                self._update_status(f"❌ Logout error: {str(e)}", "error")
                self._log_activity(f"❌ Logout exception: {str(e)}")
        else:
            self._update_status("❌ Authentication manager not available", "error")

    @on(Button.Pressed, "#test-btn")
    async def handle_test_connection(self) -> None:
        """Handle test connection button press."""
        server_url = self.query_one("#server-url", Input).value.strip()
        
        if not server_url:
            self._update_status("❌ Please enter server URL first", "error")
            self._log_activity("❌ Test connection failed: No server URL")
            return
        
        self._update_status("🔄 Testing connection...", "info")
        self._log_activity(f"🔄 Testing connection to {server_url}")
        
        if not self.auth_manager:
            self._update_status("❌ Authentication manager not available", "error")
            self._log_activity("❌ Authentication manager not available")
            return
        
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
                self._update_status(f"✅ Server reachable at {successful_url}", "success")
                self._log_activity(f"✅ Connection test successful: {successful_url}")
                
                # Update server URL field with working URL
                try:
                    self.query_one("#server-url", Input).value = successful_url
                except NoMatches:
                    pass
            else:
                self._update_status("❌ Server not reachable", "error")
                self._log_activity("❌ Connection test failed: Server not reachable")
                
                # Log detailed results
                for test_url, result in connectivity_results.items():
                    self._log_activity(f"  {test_url}: {result}")
                    
        except Exception as e:
            self._update_status(f"❌ Connection test error: {str(e)}", "error")
            self._log_activity(f"❌ Connection test exception: {str(e)}")

    @on(Button.Pressed, "#refresh-status")
    async def handle_refresh_status(self) -> None:
        """Handle refresh status button press."""
        self._log_activity("🔄 Refreshing authentication status...")
        self._load_saved_credentials()
        self._update_interface_state()

    @on(Button.Pressed, "#clear-logs")
    async def handle_clear_logs(self) -> None:
        """Handle clear logs button press."""
        try:
            activity_log = self.query_one("#auth-logs", Log)
            activity_log.clear()
            self._log_activity("🗑️ Logs cleared")
        except NoMatches:
            pass

    def _update_status(self, message: str, status_type: str = "info") -> None:
        """Update login status message."""
        try:
            status_widget = self.query_one("#login-status", Static)
            status_widget.update(message)
            
            # Update CSS class based on status type
            status_widget.remove_class("error", "success", "info", "warning")
            status_widget.add_class(status_type)
        except NoMatches:
            log.warning("Status widget not found")

    def action_back(self) -> None:
        """Return to dashboard."""
        self.app.pop_screen()

    def action_login(self) -> None:
        """Focus on login/logout button."""
        try:
            if self.is_authenticated:
                self.query_one("#logout-btn", Button).focus()
            else:
                self.query_one("#login-btn", Button).focus()
        except NoMatches:
            pass

    def action_status(self) -> None:
        """Switch to status tab."""
        try:
            tabbed_content = self.query_one(TabbedContent)
            tabbed_content.active = "tab-2"  # Status tab
        except NoMatches:
            pass

    def action_logout(self) -> None:
        """Perform logout action."""
        if self.is_authenticated:
            # If authenticated, trigger logout
            try:
                logout_btn = self.query_one("#logout-btn", Button)
                logout_btn.press()
            except NoMatches:
                pass
        else:
            self._update_status("🔓 Not currently logged in", "info")


class JobsScreen(Screen):
    """Job monitoring and management screen."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("r", "refresh", "Refresh"),
        Binding("f", "filter", "Filter"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the jobs layout."""
        yield Header()
        
        with Vertical():
            yield Static("📊 Job Management", classes="screen-title")
            
            # Control panel
            with Horizontal(classes="control-panel"):
                yield Button("🔄 Refresh", id="refresh-jobs", variant="primary")
                yield Button("📋 Filter", id="filter-jobs", variant="default")
                yield Button("⏹️ Stop Selected", id="stop-jobs", variant="error")
            
            # Jobs table
            yield DataTable(id="jobs-table")
            
            # Status bar
            yield Static("Ready", id="jobs-status", classes="status-bar")
        
        yield Footer()

    def on_mount(self) -> None:
        """Initialize jobs table when mounted."""
        table = self.query_one("#jobs-table", DataTable)
        
        # Add columns
        table.add_columns(
            "Job ID",
            "Name", 
            "User",
            "State",
            "Partition",
            "Node",
            "Time Used",
            "Memory"
        )
        
        # Add sample data
        sample_jobs = [
            ("12345", "fmr_simulation", "user1", "RUNNING", "gpu", "node01", "02:30:15", "8GB"),
            ("12346", "mode_analysis", "user2", "PENDING", "cpu", "-", "00:00:00", "4GB"),
            ("12347", "fft_compute", "user1", "COMPLETED", "gpu", "node02", "01:45:30", "16GB"),
        ]
        
        for job in sample_jobs:
            table.add_row(*job)

    @on(Button.Pressed, "#refresh-jobs")
    async def handle_refresh(self) -> None:
        """Handle refresh button press."""
        self._update_status("🔄 Refreshing job list...")
        
        # TODO: Implement actual job refresh
        await asyncio.sleep(1)
        self._update_status("✅ Job list updated")

    def _update_status(self, message: str) -> None:
        """Update jobs status message."""
        try:
            status_widget = self.query_one("#jobs-status", Static)
            status_widget.update(message)
        except NoMatches:
            log.warning("Jobs status widget not found")

    def action_back(self) -> None:
        """Return to dashboard."""
        self.app.pop_screen()

    def action_refresh(self) -> None:
        """Refresh jobs list."""
        # Schedule the async function to run
        self.run_worker(self.handle_refresh())

    def action_filter(self) -> None:
        """Focus on filter button."""
        try:
            self.query_one("#filter-jobs", Button).focus()
        except NoMatches:
            pass


class SwapScreen(Screen):
    """Simulation parameter swapping screen."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("i", "init", "Init Template"),
        Binding("r", "run", "Run"),
        Binding("v", "validate", "Validate"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the swap layout."""
        yield Header()
        
        with Vertical():
            yield Static("🔄 Simulation Parameter Swapping", classes="screen-title")
            
            with TabbedContent():
                with TabPane("Initialize"):
                    yield self._create_init_panel()
                
                with TabPane("Run"):
                    yield self._create_run_panel()
                
                with TabPane("Validate"):
                    yield self._create_validate_panel()
        
        yield Footer()

    def _create_init_panel(self) -> Container:
        """Create template initialization panel."""
        return Container(
            Static("Initialize Parameter Template", classes="section-title"),
            Label("Template file:"),
            Input(placeholder="template.mx3", value="template.mx3", id="template-file"),
            Label("Output file:"),
            Input(placeholder="parms.yml", value="parms.yml", id="output-file"),
            Label("Simulation prefix:"),
            Input(placeholder="v1", value="v1", id="sim-prefix"),
            Horizontal(
                Button("📄 Initialize Template", id="init-template", variant="primary"),
                Button("📁 Browse Files", id="browse-files", variant="default"),
            ),
            Static("", id="init-status", classes="status-message"),
            classes="init-panel"
        )

    def _create_run_panel(self) -> Container:
        """Create simulation run panel."""
        return Container(
            Static("Run Simulations", classes="section-title"),
            Label("Configuration file:"),
            Input(placeholder="parms.yml", value="parms.yml", id="config-file"),
            Label("Execution mode:"),
            Select([
                ("Normal", "normal"),
                ("Dry Run", "dry_run"),
                ("Debug", "debug")
            ], value="normal", id="exec-mode"),
            Horizontal(
                Button("▶️ Run Simulations", id="run-sims", variant="success"),
                Button("🧪 Dry Run", id="dry-run", variant="warning"),
            ),
            Static("Progress:", classes="section-title"),
            ProgressBar(id="run-progress"),
            Log(highlight=True, id="run-log"),
            classes="run-panel"
        )

    def _create_validate_panel(self) -> Container:
        """Create validation panel."""
        return Container(
            Static("Validate Configuration", classes="section-title"),
            Label("Configuration file:"),
            Input(placeholder="parms.yml", value="parms.yml", id="validate-file"),
            Button("✅ Validate", id="validate-config", variant="primary"),
            Static("Validation Results:", classes="section-title"),
            Log(highlight=True, id="validation-log"),
            classes="validate-panel"
        )

    def action_back(self) -> None:
        """Return to dashboard."""
        self.app.pop_screen()


class InfoScreen(Screen):
    """System information and diagnostics screen."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the info layout."""
        yield Header()
        
        with Vertical():
            yield Static("ℹ️ System Information", classes="screen-title")
            
            with TabbedContent():
                with TabPane("Library Info"):
                    yield self._create_library_info()
                
                with TabPane("Dependencies"):
                    yield self._create_dependencies_info()
                
                with TabPane("System"):
                    yield self._create_system_info()
        
        yield Footer()

    def _create_library_info(self) -> Markdown:
        """Create library information display."""
        info_text = f"""
# MMPP Library Information

**Version:** {__version__}  
**Author:** {__author__}

## Description
A professional library for Micro Magnetic Post Processing simulation and analysis.

## Features
- 🧲 Magnetic simulation analysis
- 📊 FFT computations and mode analysis  
- 📈 Advanced plotting and visualization
- 🔄 Parameter swapping and batch processing
- 🌐 Remote job management
- 🎨 Modern TUI interface

## Installation
```bash
pip install mmpp
```

## Quick Start
```python
import mmpp
data = mmpp.open("simulation_results.zarr")
data.fft.modes.compute_modes()
```
"""
        return Markdown(info_text)

    def _create_dependencies_info(self) -> Container:
        """Create dependencies information."""
        # TODO: Implement actual dependency checking
        return Container(
            Static("Dependency Status", classes="section-title"),
            Static("✅ numpy: 1.24.0", classes="dep-item"),
            Static("✅ matplotlib: 3.6.0", classes="dep-item"),
            Static("✅ zarr: 2.13.0", classes="dep-item"),
            Static("✅ rich: 13.0.0", classes="dep-item"),
            Static("✅ textual: 0.41.0", classes="dep-item"),
            Static("⚠️ pyfftw: Not installed", classes="dep-item warning"),
            
            Button("🔄 Refresh Dependencies", id="refresh-deps", variant="default"),
            classes="deps-panel"
        )

    def _create_system_info(self) -> Container:
        """Create system information display."""
        import platform
        import sys
        from pathlib import Path
        
        return Container(
            Static("System Information", classes="section-title"),
            Static(f"🐍 Python: {sys.version.split()[0]}", classes="sys-item"),
            Static(f"💻 Platform: {platform.system()} {platform.release()}", classes="sys-item"),
            Static(f"🏗️ Architecture: {platform.machine()}", classes="sys-item"),
            Static(f"📂 Working Directory: {Path.cwd()}", classes="sys-item"),
            Static(f"🏠 Home Directory: {Path.home()}", classes="sys-item"),
            
            Button("🔄 Refresh System Info", id="refresh-sys", variant="default"),
            classes="system-panel"
        )

    def action_back(self) -> None:
        """Return to dashboard."""
        self.app.pop_screen()

    def action_refresh(self) -> None:
        """Refresh system information."""
        # TODO: Implement refresh functionality
        pass


class MMPPApp(App):
    """Main MMPP TUI Application."""
    
    CSS = """
    /* Dracula theme colors */
    $primary: #bd93f9;
    $secondary: #ff79c6;
    $success: #50fa7b;
    $warning: #f1fa8c;
    $error: #ff5555;
    $surface: #44475a;
    $background: #282a36;
    $text: #f8f8f2;
    
    /* Screen title */
    .screen-title {
        margin: 1;
        text-align: center;
        text-style: bold;
        color: $primary;
    }
    
    /* Auth status bar */
    .auth-status-bar {
        height: 3;
        margin: 1;
        padding: 0 2;
        background: $surface;
        border: solid $primary;
    }
    
    .auth-label {
        width: 12;
        text-align: right;
        margin-right: 1;
        color: $text;
    }
    
    .auth-status {
        margin-right: 1;
        color: $success;
    }
    
    /* Dashboard grid */
    .dashboard-grid {
        grid-size: 2 2;
        grid-gutter: 1;
        margin: 2;
    }
    
    .dashboard-grid Button {
        height: 5;
    }
    
    /* Stats container */
    .stats-container {
        margin: 2;
        padding: 1;
        background: $surface;
        border: solid $primary;
    }
    
    /* Section titles */
    .section-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    /* Status messages */
    .status-message {
        margin-top: 1;
    }
    
    .status-message.success {
        color: $success;
    }
    
    .status-message.error {
        color: $error;
    }
    
    .status-message.warning {
        color: $warning;
    }
    
    .status-message.info {
        color: $text;
    }
    """
    
    SCREENS = {
        "dashboard": DashboardScreen,
        "auth": AuthScreen,
        "jobs": JobsScreen,
        "swap": SwapScreen,
        "info": InfoScreen,
    }

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+h", "help", "Help"),
        Binding("ctrl+d", "dashboard", "Dashboard"),
    ]

    def on_mount(self) -> None:
        """Initialize app when mounted."""
        self.title = f"MMPP v{__version__} - Professional Magnetic Analysis"
        self.sub_title = "Modern Text User Interface"
        
        # Start with dashboard
        self.push_screen("dashboard")
        
        log.info(f"MMPP TUI v{__version__} started")

    def action_quit(self) -> None:
        """Quit the application."""
        log.info("MMPP TUI shutting down")
        self.exit()

    def action_help(self) -> None:
        """Show help information."""
        self.push_screen("info")

    def action_dashboard(self) -> None:
        """Return to dashboard."""
        # Clear screen stack and show dashboard
        while len(self.screen_stack) > 1:
            self.pop_screen()


def main() -> None:
    """Main entry point for the new Textual-based CLI."""
    
    # Check if Textual is available
    if not TEXTUAL_AVAILABLE:
        print("📱 Textual TUI not available, using traditional CLI...")
        from .cli.main import main as fallback_main
        return fallback_main()
    
    # Check command line arguments for mode selection
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # If --classic or --cli flag is passed, use traditional CLI
    if "--classic" in args or "--cli" in args:
        print("🖥️ Using traditional CLI interface...")
        # Remove the flag before passing to traditional CLI
        sys.argv = [sys.argv[0]] + [arg for arg in args if arg not in ["--classic", "--cli"]]
        from .cli.main import main as fallback_main
        return fallback_main()
    
    # Check if running with --tui flag (explicit TUI mode)
    if "--tui" in args:
        # Remove the tui flag before starting the app
        sys.argv = [sys.argv[0]] + [arg for arg in args if arg != "--tui"]
    
    # If no arguments or --tui, start the modern TUI
    if not args or "--tui" in args:
        try:
            app = MMPPApp()
            log.info(f"Starting MMPP TUI v{__version__}")
            app.run()
        except KeyboardInterrupt:
            log.info("Application interrupted by user")
            sys.exit(0)
        except Exception as e:
            log.error(f"TUI Application error: {e}")
            print(f"❌ TUI Error: {e}")
            print("💡 Try using traditional CLI with: mmpp --classic")
            sys.exit(1)
    else:
        # For help and other specific commands, use traditional CLI
        from .cli.main import main as fallback_main
        return fallback_main()


if __name__ == "__main__":
    main()
