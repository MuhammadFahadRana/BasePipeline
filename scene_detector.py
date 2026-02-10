import cv2
import numpy as np
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from pathlib import Path
import json
from PIL import Image
import tempfile

class SceneDetector:
    """Detect scenes/shots in videos and extract keyframes."""
    
    def __init__(self, threshold: float = 30.0, min_scene_len: int = 15): 
        """
        Initialize scene detector.
        
        Args:
            threshold: Detection threshold (lower = more sensitive)
                Lower = more cuts
                Higher = fewer, longer scenes
            min_scene_len: Minimum scene length in frames not seconds 
                15 to 20 → aggressive (presentations, slides)
                25 to 35 → normal videos
                40+ → very stable scenes
        """
        self.threshold = threshold
        self.min_scene_len = min_scene_len
    
    def detect_scenes(self, video_path: str, base_output_dir: str = "processed/scenes", force_reprocess: bool = False):
        """
        Detect scenes in a video.
        
        Args:
            video_path: Path to video file
            base_output_dir: Base directory for output (will create subfolder per video)
            force_reprocess: Whether to overwrite existing results
            
        Returns:
            List of scenes with start/end times and keyframes
        """
        video_path = Path(video_path)
        # Create video-specific output directory: processed/scenes/{VideoName}
        output_dir = Path(base_output_dir) / video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        scene_file = output_dir / f"{video_path.stem}_scenes.json"
        
        # Check if already processed
        if scene_file.exists() and not force_reprocess:
            print(f"Scenes already detected for {video_path.name}. Skipping reprocessing.")
            try:
                with open(scene_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Corrupt scene file found for {video_path.name}, reprocessing...")
        
        print(f"Detecting scenes in: {video_path.name}")
        
        # Setup video manager using modern API
        video = open_video(str(video_path))
        scene_manager = SceneManager()
        
        # Add content detector
        scene_manager.add_detector(
            ContentDetector(
                threshold=self.threshold,
                min_scene_len=self.min_scene_len
            )
        )
        
        # Detect scenes
        scene_manager.detect_scenes(frame_source=video)
        scene_list = scene_manager.get_scene_list()
        
        scenes = []
        
        # Process each detected scene
        # Post-process: Split long scenes (e.g., > 60s)
        final_scenes = []
        max_scene_duration = 60.0  # seconds
        
        current_scene_idx = 0
        for i, (start_frame, end_frame) in enumerate(scene_list):
            start_time = start_frame.get_seconds()
            end_time = end_frame.get_seconds()
            duration = end_time - start_time
            
            # If scene is too long, split it
            if duration > max_scene_duration:
                num_splits = int(np.ceil(duration / max_scene_duration))
                split_duration = duration / num_splits
                
                for k in range(num_splits):
                    sub_start = start_time + (k * split_duration)
                    sub_end = min(end_time, sub_start + split_duration)
                    
                    # Create sub-scene
                    keyframe_path = self.extract_keyframe(
                        video_path, 
                        sub_start, 
                        sub_end, 
                        output_dir,
                        scene_idx=current_scene_idx
                    )
                    
                    scene_data = {
                        'scene_id': current_scene_idx,
                        'start_time': sub_start,
                        'end_time': sub_end,
                        'duration': sub_end - sub_start,
                        'keyframe_path': str(keyframe_path) if keyframe_path else None
                    }
                    final_scenes.append(scene_data)
                    print(f"  Scene {current_scene_idx} (split {k+1}/{num_splits}): "
                          f"{sub_start:.1f}s - {sub_end:.1f}s ({scene_data['duration']:.1f}s)")
                    current_scene_idx += 1
            else:
                # Keep original scene
                keyframe_path = self.extract_keyframe(
                    video_path, 
                    start_time, 
                    end_time, 
                    output_dir,
                    scene_idx=current_scene_idx
                )
                
                scene_data = {
                    'scene_id': current_scene_idx,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'start_frame': start_frame.get_frames(),
                    'end_frame': end_frame.get_frames(),
                    'keyframe_path': str(keyframe_path) if keyframe_path else None
                }
                final_scenes.append(scene_data)
                print(f"  Scene {current_scene_idx}: {start_time:.1f}s - {end_time:.1f}s "
                      f"({scene_data['duration']:.1f}s)")
                current_scene_idx += 1
        
        # Save scene information
        with open(scene_file, 'w') as f:
            json.dump(final_scenes, f, indent=2)
        
        print(f"Detected {len(final_scenes)} scenes (after post-processing)")
        print(f"Scene info saved to: {scene_file}")
        
        return final_scenes
    
    def extract_keyframe(self, video_path: Path, start_time: float, 
                        end_time: float, output_dir: Path, scene_idx: int) -> Path:
        """
        Extract a keyframe from the middle of a scene.
        
        Returns:
            Path to saved keyframe image
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame at middle of scene
        mid_time = (start_time + end_time) / 2
        mid_frame = int(mid_time * fps)
        
        # Set video to the middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save as JPEG
            keyframe_path = output_dir / f"{video_path.stem}_scene_{scene_idx}.jpg"
            Image.fromarray(frame_rgb).save(keyframe_path, quality=85)
            return keyframe_path
        
        return None
    
    def visualize_scenes(self, video_path: str, scenes: list, output_file: str = None):
        """
        Create a visualization of scene boundaries.
        
        This generates a strip of thumbnails for each scene.
        """
        import matplotlib.pyplot as plt
        
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        # Create figure
        num_scenes = len(scenes)
        if num_scenes == 0:
            print("No scenes to visualize.")
            return
            
        fig, axes = plt.subplots(1, min(5, num_scenes), figsize=(15, 3))
        if num_scenes == 1:
            axes = [axes]
        
        for i, (scene, ax) in enumerate(zip(scenes[:5], axes)):
            # Get frame at start of scene
            start_frame = int(scene['start_time'] * cap.get(cv2.CAP_PROP_FPS))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ax.imshow(frame_rgb)
                ax.set_title(f"Scene {i}\n{scene['start_time']:.1f}s")
                ax.axis('off')
        
        cap.release()
        
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {output_file}")
        
        plt.show()
    
    def batch_detect(self, video_folder: str = "videos"):
        """Detect scenes for all videos in a folder."""
        video_folder = Path(video_folder)
        videos = list(video_folder.glob("*.*"))
        
        print(f"Found {len(videos)} videos for scene detection")
        
        all_scenes = []
        for i, video_path in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] Processing: {video_path.name}")
            # Skip test audio file and json
            if video_path.suffix in ['.json', '.txt', '.wav'] or 'test_audio' in video_path.name:
                continue

            try:
                # Use new default: processed/scenes/{VideoName}
                scenes = self.detect_scenes(video_path)
                
                all_scenes.append({
                    'video': video_path.name,
                    'scenes_file': f"processed/scenes/{video_path.stem}/{video_path.stem}_scenes.json",
                    'num_scenes': len(scenes),
                    'success': True
                })
            except Exception as e:
                print(f"Failed to detect scenes in {video_path.name}: {str(e)}")
                all_scenes.append({
                    'video': video_path.name,
                    'error': str(e),
                    'success': False
                })
        
        return all_scenes


if __name__ == "__main__":
    detector = SceneDetector(threshold=20.0) 
    
    # # Test with a single video
    # scenes = detector.detect_scenes("videos/Risk management.mp4")
    
    # # Create visualization
    # detector.visualize_scenes("videos/Risk management.mp4", scenes, "processed/scenes/Risk management/visualization.png")
    
    # Run batch detection
    detector.batch_detect()