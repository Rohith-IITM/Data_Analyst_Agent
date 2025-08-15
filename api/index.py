from fastapi import FastAPI
from app.main import app  # Import FastAPI app from your existing logic

# Vercel expects an ASGI app to be exposed. So, we use the FastAPI instance in `app/main.py`.
# This allows Vercel to route traffic to your FastAPI app.

# The `app` here is the FastAPI instance
app = app
