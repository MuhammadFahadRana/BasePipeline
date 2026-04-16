import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from database.config import get_db
from search.multi_modal_search import MultiModalSearchEngine

db = next(get_db())
engine = MultiModalSearchEngine(db)

# Just test search
import time
t0 = time.time()
res = engine.search_with_fallback("oil rig installation", top_k=5)
print(f"Search took {time.time() - t0:.2f}s")
if res["results"]:
    print("Top result score text:", res["results"][0].text_score)
    print("Top result score combined:", res["results"][0].combined_score)
    print("Filename:", res["results"][0].video_filename)
else:
    print("No results!")
