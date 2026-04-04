import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn

def main():
    import app as application
    uvicorn.run(application.app, host="0.0.0.0", port=8000)

def run():
    main()

if __name__ == "__main__":
    main()