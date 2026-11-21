import subprocess
import sys


def main():

    print("=" * 50)
    print("PLANT DISEASE DETECTION")
    print("FEDERATED LEARNING PROJECT")
    print("=" * 50)

    print("\nStarting Streamlit Application...")

    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "frontend/app.py"
    ])


if __name__ == "__main__":
    main()