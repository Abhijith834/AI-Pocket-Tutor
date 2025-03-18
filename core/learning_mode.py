"""
Module: learning_mode.py

When a new learning-mode session starts, the user can either:
1) Provide a local file via 'file (C:\path)', or
2) Specify a subject to perform a PDF search via the Googlesearch library.

Any downloaded PDF is saved in the current chat session folder under a "pdf" subfolder.
For example, if the session is "chat_7", the PDF is stored in:
    chat_7/pdf/filename.pdf

Once the PDF path is determined, it is stored in final_pdf_path, but not ingested here.
"""

import os
import requests
from googlesearch import search

class LearningModeAgent:
    def __init__(self):
        self.final_pdf_path = None

    def init_learning_mode(self):
        """
        Prompts the user for a document (file) or a subject to find a PDF.
        If a PDF is successfully chosen/downloaded, self.final_pdf_path will be set.
        Otherwise, it remains None.
        """
        print("Learning Mode activated.")
        user_input = input("Would you like to 'file (C:\\path)' or type a subject? ").strip()

        if user_input.lower().startswith("file"):
            # Extract local path from user input
            path = self._extract_file_path(user_input)
            if not path or not os.path.exists(path):
                print("Invalid or missing file. Exiting learning mode.")
                return
            self.final_pdf_path = path
        else:
            # Treat user_input as a subject for PDF search
            self.subject_flow(user_input)

        if self.final_pdf_path:
            print(f"[LearningMode] Document found -> final_pdf_path set to {self.final_pdf_path}")
        else:
            print("[LearningMode] No PDF chosen. Exiting learning mode.")

    def _extract_file_path(self, command: str) -> str:
        """
        Extracts the file path from a command of the format:
            file (C:\path\to\doc.pdf)
        """
        start = command.find("(")
        end = command.find(")")
        if start != -1 and end != -1:
            return command[start+1:end].strip()
        return None

    def subject_flow(self, subject: str):
        """
        Uses Google search to find PDF documents related to the subject,
        prompts the user to select one, downloads it, and updates final_pdf_path.
        """
        print(f"[LearningMode] Searching for PDF docs about '{subject}'")
        pdf_urls = self.google_search_pdf(subject, num_results=5)
        if not pdf_urls:
            print("No PDF found for that subject. Exiting.")
            return

        # Display candidate URLs
        for i, url in enumerate(pdf_urls):
            print(f"[{i+1}] {url}")
        choice = input("Which PDF do you want (1-5)? ").strip()
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(pdf_urls):
                print("Invalid selection.")
                return
        except:
            print("Invalid input. Exiting learning mode.")
            return

        chosen_url = pdf_urls[idx]
        downloaded = self.download_pdf(chosen_url)
        if downloaded:
            self.final_pdf_path = downloaded

    def google_search_pdf(self, query: str, num_results: int):
        """
        Uses the googlesearch package to retrieve up to `num_results` PDF links for the given query.
        """
        try:
            full_query = f"{query} filetype:pdf"
            return list(search(full_query, num_results=num_results))
        except Exception as e:
            print(f"Error searching: {e}")
            return []

    def download_pdf(self, url: str):
        """
        Downloads the PDF from the given URL, disabling SSL verification to avoid certificate issues.
        The PDF is saved in the current chat's session folder under "pdf/filename.pdf".
        """
        print(f"[LearningMode] Downloading: {url}")

        # 1) Determine the chat folder from environment
        session_id = os.environ.get("SESSION_ID", "unknown")
        chat_history_file = os.environ.get("CHAT_HISTORY_FILE", "chat_history.json")
        session_folder = os.path.dirname(os.path.abspath(chat_history_file))

        # 2) Create a "pdf" subfolder inside this session folder
        pdf_folder = os.path.join(session_folder, "pdf")
        os.makedirs(pdf_folder, exist_ok=True)

        try:
            # Make the request (disable SSL verify as needed)
            resp = requests.get(url, timeout=15, verify=False)
            resp.raise_for_status()

            # Extract filename from URL
            fname = url.split("/")[-1]
            if not fname.endswith(".pdf"):
                fname = "document.pdf"

            outpath = os.path.join(pdf_folder, fname)
            with open(outpath, "wb") as f:
                f.write(resp.content)

            print(f"[LearningMode] Downloaded PDF to {outpath}")
            return outpath
        except Exception as e:
            print(f"Failed to download: {e}")
            return None
