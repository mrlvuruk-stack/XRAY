"""
Swasthya MONAI Medical Imaging Lab - Application Runner
Launches the Streamlit medical imaging interface with auto-configured environment.
"""

import sys
import os
import subprocess
from pathlib import Path


def main():
    project_dir = Path(__file__).resolve().parent
    app_main = project_dir / "app" / "main.py"

    if not app_main.exists():
        print(f"Error: Could not find main application entry point at {app_main}")
        sys.exit(1)

    # Set working directory to project root
    os.chdir(str(project_dir))

    port = os.getenv("PORT", "8501")
    host = os.getenv("HOST", "localhost")

    print("=" * 65)
    print("  SWASTHYA AI - MONAI Medical Imaging Diagnostic Lab")
    print(f"  Platform: Python {sys.version.split()[0]}")
    print(f"  Root Path: {project_dir}")
    print(f"  Server URL: http://{host}:{port}")
    print("=" * 65)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_main),
        f"--server.port={port}",
        f"--server.address={host}",
        "--browser.gatherUsageStats=false",
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Swasthya AI server stopped gracefully.")
    except Exception as e:
        print(f"\n[ERROR] Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
