"""
core/learning_mode.py

Handles PDF ingestion for learning-mode.
• Supports up to 15 search results when finding by subject.
• Prompts interactively (TTY or front-end) to pick a PDF.
• Stores the 15-URL list in pdf_options.json in the session folder for front-end APIs.
• Uses input_with_timeout to capture front-end CLI messages or console input.
• Skips gracefully if no input arrives within the timeout.
"""

import os
import sys
import json
import requests
from googlesearch import search
from core.chat import input_with_timeout, INACTIVITY_TIMEOUT

class LearningModeAgent:
    """
    PDF ingestion for learning-mode.

    • If file_path is provided, uses it.
    • Else if subject is provided, searches up to 15 PDFs, stores them, and prompts user to pick one.
    • Utilizes input_with_timeout to accept front-end or console input.
    • Skips ingestion if timeout elapses or invalid selection.
    """

    def __init__(self):
        self.final_pdf_path: str | None = None

    def init_learning_mode(
        self,
        file_path: str | None = None,
        subject:   str | None = None,
    ) -> None:
        # 1) Local file if provided
        if file_path:
            self._use_local_file(file_path)
        # 2) Subject if provided
        elif subject:
            self._subject_flow(subject)
        # 3) Interactive prompt if no args
        else:
            print("Learning Mode activated.")
            try:
                received = input_with_timeout(
                    "Enter 'file (C:\\path\\to.pdf)' or a search subject:\n",
                    timeout=INACTIVITY_TIMEOUT
                )
                if isinstance(received, dict):
                    user_input = received.get("message", "").strip()
                else:
                    user_input = str(received).strip()
            except TimeoutError:
                print("[LearningMode] No input received within timeout; skipping ingestion.")
                return
            except Exception as e:
                print(f"[LearningMode] Error reading input: {e}; skipping ingestion.")
                return

            if user_input.lower().startswith("file"):
                path = self._extract_file_path(user_input)
                self._use_local_file(path)
            else:
                self._subject_flow(user_input)

        # Report the result
        if self.final_pdf_path:
            print(f"[LearningMode] Selected PDF → {self.final_pdf_path}")
        else:
            print("[LearningMode] No PDF could be ingested.")

    def _use_local_file(self, path: str | None) -> None:
        if path and os.path.isfile(path):
            self.final_pdf_path = path
        else:
            print(f"[LearningMode] Local file not found: {path!r}")

    def _subject_flow(self, subject: str) -> None:
        print(f"[LearningMode] Searching for up to 15 PDFs about “{subject}”…")
        urls = self._search_pdfs(subject, max_results=15)
        if not urls:
            print("[LearningMode] No PDF links found.")
            return

        # Store the URL list for API/front-end
        session_folder = os.path.dirname(os.getenv("CHAT_HISTORY_FILE", "chat_history.json"))
        options_path = os.path.join(session_folder, "pdf_options.json")
        try:
            with open(options_path, "w", encoding="utf-8") as f:
                json.dump(urls, f, indent=2)
            print(f"[LearningMode] Stored PDF options to {options_path}")
        except Exception as e:
            print(f"[LearningMode] Failed to write options file: {e}")

        # Prompt user to pick one
        if sys.stdin.isatty():
            for i, u in enumerate(urls, start=1):
                print(f"[{i}] {u}")
            try:
                received = input_with_timeout(
                    f"Pick a number (1-{len(urls)}) to select PDF:\n",
                    timeout=INACTIVITY_TIMEOUT
                )
                if isinstance(received, dict):
                    choice = received.get("message", "").strip()
                else:
                    choice = str(received).strip()
                idx = int(choice) - 1
                if idx < 0 or idx >= len(urls):
                    raise ValueError
                chosen = urls[idx]
            except TimeoutError:
                print("[LearningMode] No selection made within timeout; skipping ingestion.")
                return
            except Exception:
                print("[LearningMode] Invalid selection; skipping ingestion.")
                return
        else:
            chosen = urls[0]
            print(f"[LearningMode] Non-interactive; auto-selecting first result: {chosen}")

        self.final_pdf_path = self._download_pdf(chosen)

    def _search_pdfs(self, query: str, max_results: int = 5) -> list[str]:
        try:
            hits = list(search(f"{query} filetype:pdf", num_results=max_results))
            return [u for u in hits if u.lower().endswith(".pdf")]
        except Exception as e:
            print(f"[LearningMode] Search error: {e}")
            return []

    def _download_pdf(self, url: str) -> str | None:
        print(f"[LearningMode] Downloading: {url}")
        session_folder = os.path.dirname(
            os.getenv("CHAT_HISTORY_FILE", "chat_history.json")
        )
        pdf_folder = os.path.join(session_folder, "pdf")
        os.makedirs(pdf_folder, exist_ok=True)

        try:
            resp = requests.get(url, timeout=15, verify=False)
            resp.raise_for_status()
            name = os.path.basename(url)
            if not name.lower().endswith(".pdf"):
                name = "document.pdf"
            out = os.path.join(pdf_folder, name)
            with open(out, "wb") as f:
                f.write(resp.content)
            return out
        except Exception as e:
            print(f"[LearningMode] Download failed: {e}")
            return None

    def _extract_file_path(self, cmd: str) -> str | None:
        if "(" in cmd and ")" in cmd:
            return cmd.split("(", 1)[1].rsplit(")", 1)[0].strip()
        return None