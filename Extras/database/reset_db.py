import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from database.config import engine
from database.models import Base, Video, TranscriptSegment, Embedding, Scene, SearchQuery

print("Warning: This will DROP ALL DATA in the database.")
confirm = input("Are you sure? (yes/no): ")

if confirm.lower() == "yes":
    print("Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("Recreating tables...")
    Base.metadata.create_all(bind=engine)
    print("Database reset complete. New schema applied.")
else:
    print("Operation cancelled.")
