"""Separate baseline execution entry point; no frozen module mutation."""
from .worker import serve

if __name__ == '__main__':
    raise SystemExit(serve(route='frozen'))
