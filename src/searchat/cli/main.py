#!/usr/bin/env python
"""Simple CLI for Claude Search - no complex TUI, just clean terminal output"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import json

from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.syntax import Syntax
from rich import print as rprint

from searchat.core.unified_search import UnifiedSearchEngine
from searchat.models import AlgorithmType, SearchResult
from searchat.config import Config, PathResolver


class SearchCLI:
    def __init__(self):
        self.console = Console()
        self.config = Config.load()
        search_dir = PathResolver.get_shared_search_dir(self.config)
        self.search_engine = UnifiedSearchEngine(search_dir, self.config)
        self.current_results: List[SearchResult] = []
        self.search_mode = AlgorithmType.HYBRID

    def run(self):
        """Main CLI loop"""
        self.console.clear()
        self.console.print("[bold cyan]Claude Conversation Search[/bold cyan]")
        self.console.print("Enter search query (or 'help' for commands, 'quit' to exit)\n")

        while True:
            try:
                # Get search query
                query = Prompt.ask("[bold green]Search[/bold green]")

                if query.lower() == 'quit':
                    break
                elif query.lower() == 'help':
                    self._show_help()
                    continue
                elif query.lower() == 'mode':
                    self._change_mode()
                    continue
                elif query.lower() == 'all':
                    self._show_all_conversations()
                    if self.current_results:
                        self._display_results()
                        # Ask to view a result
                        while True:
                            choice = Prompt.ask("\n[bold]Enter number to view (or 'b' to go back, 's' for new search)[/bold]")

                            if choice.lower() == 'b' or choice.lower() == 's':
                                break
                            elif choice.isdigit():
                                idx = int(choice) - 1
                                if 0 <= idx < len(self.current_results):
                                    self._view_conversation(self.current_results[idx])
                                    # After viewing, show results again
                                    self._display_results()
                                else:
                                    self.console.print("[red]Invalid selection[/red]")
                            else:
                                self.console.print("[red]Invalid input[/red]")
                    continue
                elif not query.strip():
                    continue

                # Execute search
                self._search(query)

                # Show results
                if self.current_results:
                    self._display_results()

                    # Ask to view a result
                    while True:
                        choice = Prompt.ask("\n[bold]Enter number to view (or 'b' to go back, 's' for new search)[/bold]")

                        if choice.lower() == 'b' or choice.lower() == 's':
                            break
                        elif choice.isdigit():
                            idx = int(choice) - 1
                            if 0 <= idx < len(self.current_results):
                                self._view_conversation(self.current_results[idx])
                                # After viewing, show results again
                                self._display_results()
                            else:
                                self.console.print("[red]Invalid selection[/red]")
                        else:
                            self.console.print("[red]Invalid input[/red]")

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Search interrupted[/yellow]")
                continue
            except EOFError:
                break

        self.console.print("\n[green]Goodbye![/green]")

    def _show_help(self):
        """Show help information"""
        help_text = """
[bold cyan]Commands:[/bold cyan]
  [yellow]search query[/yellow] - Search for conversations
  [yellow]all[/yellow]          - Show all conversations (sorted by length)
  [yellow]mode[/yellow]         - Change search mode (hybrid/semantic/keyword)
  [yellow]help[/yellow]         - Show this help
  [yellow]quit[/yellow]         - Exit

[bold cyan]Search Tips:[/bold cyan]
  • Use specific terms like "batch 15" or "84%"
  • Combine concepts: "test cases openai batch"
  • Exact phrases: "truncation problem"
  • Mode affects results:
    - [green]Hybrid[/green]: Best overall (default)
    - [green]Semantic[/green]: Meaning-based
    - [green]Keyword[/green]: Exact matches
        """
        self.console.print(help_text)

    def _change_mode(self):
        """Change search mode"""
        modes = {
            "1": AlgorithmType.HYBRID,
            "2": AlgorithmType.SEMANTIC,
            "3": AlgorithmType.KEYWORD
        }

        self.console.print("\n[bold]Select search mode:[/bold]")
        self.console.print("1. Hybrid (default)")
        self.console.print("2. Semantic")
        self.console.print("3. Keyword")

        choice = Prompt.ask("Choice", choices=["1", "2", "3"])
        self.search_mode = modes[choice]
        self.console.print(f"[green]Mode changed to: {self.search_mode.value}[/green]")

    def _show_all_conversations(self):
        """Show all conversations sorted by message length"""
        self.console.print("\n[dim]Loading all conversations...[/dim]")

        try:
            # Get all conversations from DuckDB
            cursor = self.search_engine.storage._get_read_cursor()
            rows = cursor.execute("""
                SELECT conversation_id, project_id, title, created_at, updated_at,
                       message_count, file_path, full_text
                FROM conversations
                WHERE message_count > 0
                ORDER BY message_count DESC
            """).fetchall()
            columns = [desc[0] for desc in cursor.description]

            self.current_results = [
                SearchResult(
                    conversation_id=row[columns.index('conversation_id')],
                    project_id=row[columns.index('project_id')],
                    title=row[columns.index('title')],
                    created_at=row[columns.index('created_at')],
                    updated_at=row[columns.index('updated_at')],
                    message_count=row[columns.index('message_count')],
                    file_path=row[columns.index('file_path')],
                    score=0.0,
                    snippet=row[columns.index('full_text')][:200] + ("..." if len(row[columns.index('full_text')]) > 200 else "")
                )
                for row in rows
            ]

            self.console.print(
                f"[green]Showing all {len(self.current_results)} conversations[/green] "
                f"[dim](sorted by message length)[/dim]\n"
            )
        except Exception as e:
            self.console.print(f"[red]Error loading conversations: {e}[/red]")
            self.current_results = []

    def _search(self, query: str):
        """Execute search"""
        self.console.print(f"\n[dim]Searching in {self.search_mode.value} mode...[/dim]")

        try:
            results = self.search_engine.search(query, algorithm=self.search_mode)
            self.current_results = results.results

            self.console.print(
                f"[green]Found {results.total_count} results[/green] "
                f"[dim]({results.search_time_ms:.0f}ms)[/dim]\n"
            )
        except Exception as e:
            self.console.print(f"[red]Search error: {e}[/red]")
            self.current_results = []

    def _display_results(self):
        """Display search results with context"""
        if not self.current_results:
            self.console.print("[yellow]No results found[/yellow]")
            return

        # Show up to 20 results
        for i, result in enumerate(self.current_results[:20], 1):
            # Build result display
            text = Text()

            # Number and source
            text.append(f"{i:2}. ", style="bold cyan")
            source = "WSL" if "/home/" in result.file_path else "WIN"
            text.append(f"[{source}] ", style="magenta" if source == "WSL" else "blue")

            # Date and project
            text.append(result.updated_at.strftime("%b %d %H:%M"), style="dim")
            text.append(f" | {result.project_id[:30]}\n", style="dim")

            # Title
            text.append(f"    {result.title[:70]}", style="bold")
            text.append(f" ({result.message_count} msgs)", style="dim")

            # Message range if available
            if result.message_start_index is not None:
                text.append(f" [msgs {result.message_start_index}-{result.message_end_index}]", style="yellow dim")
            text.append("\n")

            # Snippet with highlighting
            snippet = Text(f"    {result.snippet[:150]}...")
            # Highlight search terms (simple approach)
            query_words = self.current_results[0].snippet[:0]  # Get query from first result
            snippet.stylize("italic dim")

            text.append(snippet)

            # Display with separator
            self.console.print(text)
            self.console.print("    " + "─" * 70, style="dim")

        if len(self.current_results) > 20:
            self.console.print(f"\n[dim]Showing first 20 of {len(self.current_results)} results[/dim]")

    def _view_conversation(self, result: SearchResult):
        """View a conversation with pagination"""
        self.console.clear()

        # Header
        header = Panel(
            f"[bold]{result.title}[/bold]\n"
            f"Project: {result.project_id} | "
            f"Messages: {result.message_count} | "
            f"Updated: {result.updated_at.strftime('%Y-%m-%d %H:%M')}",
            title="Conversation",
            border_style="cyan"
        )
        self.console.print(header)

        # Load conversation
        try:
            with open(result.file_path, 'r', encoding='utf-8') as f:
                lines = [json.loads(line) for line in f]

            messages = []
            for entry in lines:
                if entry.get('type') in ('user', 'assistant'):
                    raw_content = entry.get('message', {}).get('content', '')
                    if isinstance(raw_content, str):
                        content = raw_content
                    elif isinstance(raw_content, list):
                        content = '\n\n'.join(
                            block.get('text', '')
                            for block in raw_content
                            if block.get('type') == 'text'
                        )
                    else:
                        content = ''

                    if content:
                        messages.append({
                            'role': entry.get('type'),
                            'content': content,
                            'timestamp': entry.get('timestamp', '')
                        })

            # Pagination
            page_size = 10
            start_idx = result.message_start_index or 0
            current_page = start_idx // page_size

            while True:
                self.console.clear()
                self.console.print(header)

                # Calculate page range
                start = current_page * page_size
                end = min(start + page_size, len(messages))

                self.console.print(f"\n[dim]Messages {start+1}-{end} of {len(messages)}[/dim]\n")

                # Display messages
                for i in range(start, end):
                    msg = messages[i]

                    # Role header
                    role_color = "cyan" if msg['role'] == 'user' else "green"
                    self.console.print(f"[bold {role_color}][{msg['role'].upper()}] Message {i+1}[/bold {role_color}]")

                    # Content (truncated)
                    content = msg['content'][:1000]
                    if len(msg['content']) > 1000:
                        content += "...[truncated]"
                    self.console.print(content)
                    self.console.print("─" * 80, style="dim")

                # Navigation
                nav_text = "\n[bold]Navigation:[/bold] "
                if current_page > 0:
                    nav_text += "[p]revious page | "
                if end < len(messages):
                    nav_text += "[n]ext page | "
                nav_text += "[j]ump to message | [b]ack to results"

                self.console.print(nav_text)
                choice = Prompt.ask("Choice", default="b")

                if choice.lower() == 'b':
                    break
                elif choice.lower() == 'n' and end < len(messages):
                    current_page += 1
                elif choice.lower() == 'p' and current_page > 0:
                    current_page -= 1
                elif choice.lower() == 'j':
                    msg_num = IntPrompt.ask("Jump to message number", default=1)
                    if 1 <= msg_num <= len(messages):
                        current_page = (msg_num - 1) // page_size
                    else:
                        self.console.print("[red]Invalid message number[/red]")

        except Exception as e:
            self.console.print(f"[red]Error loading conversation: {e}[/red]")
            Prompt.ask("\n[Press Enter to continue]")

        self.console.clear()


def main():
    """Entry point"""
    cli = SearchCLI()
    cli.run()


if __name__ == "__main__":
    main()
