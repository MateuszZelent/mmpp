#!/usr/bin/env python3
"""
Prosty test TUI do sprawdzenia przycisków
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Button, Header, Footer, Static
from textual.screen import Screen
from textual.binding import Binding

class SimpleScreen(Screen):
    BINDINGS = [
        Binding("a", "auth", "Auth"),
        Binding("j", "jobs", "Jobs"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        
        with Vertical():
            yield Static("🚀 Test Dashboard", classes="title")
            
            with Container(classes="buttons"):
                yield Button("🔐 Authentication", id="auth-btn", variant="primary")
                yield Button("📊 Jobs", id="jobs-btn", variant="primary") 
                yield Button("🔄 Swap", id="swap-btn", variant="default")
                
            yield Static("Wszystkie przyciski powinny być widoczne", classes="info")
            
        yield Footer()

    def action_auth(self) -> None:
        self.app.bell()
        
    def action_jobs(self) -> None:
        self.app.bell()
        
    def action_quit(self) -> None:
        self.app.exit()

class SimpleApp(App):
    CSS = """
    .title {
        text-align: center;
        text-style: bold;
        color: cyan;
        margin: 1;
    }
    
    .buttons {
        margin: 2;
        padding: 1;
        border: solid green;
        height: auto;
    }
    
    .buttons Button {
        margin: 1;
        height: 3;
    }
    
    .info {
        text-align: center;
        margin: 1;
        color: yellow;
    }
    """

    def on_mount(self) -> None:
        self.title = "Simple TUI Test"
        self.push_screen(SimpleScreen())

if __name__ == "__main__":
    app = SimpleApp()
    app.run()
