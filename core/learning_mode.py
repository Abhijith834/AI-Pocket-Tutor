"""
Module: learning_mode.py

When a brand-new session is in learning mode, this module helps the user either
provide a local file or specify a subject, which yields a PDF from the web.
We store the final chosen PDF path in self.final_pdf_path, but do NOT ingest it ourselves.
"""

import os
import requests
from googlesearch import search

class LearningModeAgent:
    def __init__(self):
        self.final_pdf_path = None

    def init_learning_mode(self):
        print("Learning Mode activated.")
        user_input = input("Would you like to 'file (C:\\path)' or type a subject? ").strip()

        if user_input.lower().startswith("file"):
            path = self._extract_file_path(user_input)
            if not path or not os.path.exists(path):
                print("Invalid or missing file. Exiting learning mode.")
                return
            self.final_pdf_path = path
        else:
            self.subject_flow(user_input)

        if self.final_pdf_path:
            print(f"[LearningMode] Document found -> final_pdf_path set to {self.final_pdf_path}")
        else:
            print("[LearningMode] No PDF chosen. Exiting learning mode.")

    def _extract_file_path(self, command: str) -> str:
        start = command.find("(")
        end = command.find(")")
        if start != -1 and end != -1:
            return command[start+1:end].strip()
        return None

    def subject_flow(self, subject: str):
        print(f"[LearningMode] Searching for PDF docs about '{subject}'")
        pdf_urls = self.google_search_pdf(subject, 5)
        if not pdf_urls:
            print("No PDF found for that subject. Exiting.")
            return

        for i, url in enumerate(pdf_urls):
            print(f"[{i+1}] {url}")
        choice = input("Which PDF do you want (1-5)? ").strip()
        try:
            idx = int(choice)-1
            if idx < 0 or idx >= len(pdf_urls):
                print("Invalid selection.")
                return
        except:
            print("Invalid input. Exiting learning mode.")
            return

        pdf_url = pdf_urls[idx]
        downloaded = self.download_pdf(pdf_url)
        if downloaded:
            self.final_pdf_path = downloaded

    def google_search_pdf(self, query: str, num_results: int):
        try:
            full_query = f"{query} filetype:pdf"
            return list(search(full_query, num_results=num_results))
        except Exception as e:
            print(f"Error searching: {e}")
            return []

    def download_pdf(self, url: str):
        try:
            print(f"[LearningMode] Downloading: {url}")
            resp = requests.get(url, timeout=15, verify=False)
            resp.raise_for_status()
            folder = "downloaded_pdfs"
            os.makedirs(folder, exist_ok=True)
            fname = url.split("/")[-1]
            if not fname.endswith(".pdf"):
                fname = "document.pdf"
            outpath = os.path.join(folder, fname)
            with open(outpath, "wb") as f:
                f.write(resp.content)
            print(f"[LearningMode] Downloaded to {outpath}")
            return outpath
        except Exception as e:
            print(f"Failed to download: {e}")
            return None
