"""Development entry point for the Options Trading Copilot API."""

import os

import uvicorn


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8002"))
    uvicorn.run("app.main:app", host="127.0.0.1", port=port, reload=True)
