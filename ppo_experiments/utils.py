import os

def create_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs/ppo", exist_ok=True)