import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from database.config import get_db
from llm.video_qa import VideoQA

db = next(get_db())
try:
    qa = VideoQA(db)
    print("QA Initialization successful")
    res = qa.ask("What is an oil rig?")
    print("Answer:", res["answer"])
except Exception as e:
    print("QA Initialization failed!")
    import traceback
    traceback.print_exc()
