from fastapi import FastAPI

from .agents.router import router as agents_router
from .ping.router import router as ping_router
from .summarize.router import router as summarize_router

app = FastAPI(title="AI RAG Demo", version="1.0.0")

app.include_router(agents_router)
app.include_router(ping_router)
app.include_router(summarize_router)
