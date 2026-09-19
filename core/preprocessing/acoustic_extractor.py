import os
import numpy as np

# Lazy load parselmouth so we don't crash if it's missing (e.g. for users who just want TextGrids without audio)
try:
    import parselmouth
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False

class AcousticExtractor:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.snd = None
        self.pitch = None
        self.formant = None
        self._loaded = False
        self.has_parselmouth = HAS_PARSELMOUTH
        
    def load(self):
        if not self.has_parselmouth:
            return False
            
        if self._loaded:
            return True
            
        if not os.path.exists(self.audio_path):
            return False
            
        try:
            # Load audio using parselmouth
            self.snd = parselmouth.Sound(self.audio_path)
            self.pitch = self.snd.to_pitch()
            self.formant = self.snd.to_formant_burg()
            self._loaded = True
            return True
        except Exception as e:
            print(f"Error loading acoustic features for {self.audio_path}: {e}")
            return False
            
    def get_features_for_interval(self, start_time, end_time, time_step=0.01):
        """
        Returns dictionary of acoustic features for the given interval.
        """
        if not self.has_parselmouth or not self.load():
            return {
                'pitch_pattern': None,
                'f0_mean': None,
                'f1_mean': None,
                'f2_mean': None
            }
            
        time_steps = np.arange(start_time, end_time, time_step)
        if len(time_steps) == 0:
            return {
                'pitch_pattern': None,
                'f0_mean': None,
                'f1_mean': None,
                'f2_mean': None
            }
            
        # F0 (Pitch)
        f0_values = []
        for t in time_steps:
            f0 = self.pitch.get_value_at_time(t)
            if not np.isnan(f0):
                f0_values.append(f0)
                
        f0_mean = np.mean(f0_values) if f0_values else None
        
        # Determine contour pattern
        pitch_pattern = None
        if len(f0_values) > 2:
            # Smooth the curve slightly to avoid micro-jitters
            smoothed = np.convolve(f0_values, np.ones(3)/3, mode='valid') if len(f0_values) >= 3 else f0_values
            
            if len(smoothed) > 2:
                global_max_idx = np.argmax(smoothed)
                global_min_idx = np.argmin(smoothed)
                global_max = smoothed[global_max_idx]
                global_min = smoothed[global_min_idx]
                
                start_f0 = smoothed[0]
                end_f0 = smoothed[-1]
                
                n = len(smoothed)
                DIFF_THRESHOLD = 15
                
                if (global_max_idx > n // 4 and global_max_idx < 3 * n // 4) and (global_max - start_f0 > DIFF_THRESHOLD) and (global_max - end_f0 > DIFF_THRESHOLD):
                    pitch_pattern = "peak"
                elif (global_min_idx > n // 4 and global_min_idx < 3 * n // 4) and (start_f0 - global_min > DIFF_THRESHOLD) and (end_f0 - global_min > DIFF_THRESHOLD):
                    pitch_pattern = "valley"
                elif end_f0 - start_f0 > DIFF_THRESHOLD:
                    pitch_pattern = "rising"
                elif start_f0 - end_f0 > DIFF_THRESHOLD:
                    pitch_pattern = "falling"
                else:
                    pitch_pattern = "level"
            else:
                pitch_pattern = "level"
                
        # Formants
        f1_values = []
        f2_values = []
        for t in time_steps:
            f1 = self.formant.get_value_at_time(1, t)
            f2 = self.formant.get_value_at_time(2, t)
            if not np.isnan(f1): f1_values.append(f1)
            if not np.isnan(f2): f2_values.append(f2)
            
        f1_mean = np.mean(f1_values) if f1_values else None
        f2_mean = np.mean(f2_values) if f2_values else None
        
        return {
            'pitch_pattern': pitch_pattern,
            'f0_mean': round(f0_mean, 1) if f0_mean else None,
            'f1_mean': round(f1_mean, 1) if f1_mean else None,
            'f2_mean': round(f2_mean, 1) if f2_mean else None
        }
