#!/usr/bin/env python
"""
Quick Launcher for the Adversarial Studio Web App.
Usage:
    python scripts/launch_studio.py [--port 8000] [--host 127.0.0.1]
"""

import argparse
import sys
import os
import uvicorn

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adv_studio.web.server import app

def main():
    parser = argparse.ArgumentParser(description="Launch Adversarial Studio Web Interface")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()

    print(f"\n========================================================")
    print(f" 🚀 Adversarial Studio Web App starting on http://{args.host}:{args.port}")
    print(f"========================================================\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
