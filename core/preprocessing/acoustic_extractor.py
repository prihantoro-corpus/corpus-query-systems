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
        
        self.pitch_values = None
        self.pitch_times = None
        self.f1_values = None
        self.f2_values = None
        self.formant_times = None
        
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
            
            # Precompute numpy arrays for blazing fast slicing!
            self.pitch_values = self.pitch.selected_array['frequency']
            self.pitch_times = self.pitch.xs()
            
            self.formant_times = self.formant.xs()
            # Formant values aren't as easily accessible as a single flat array, so we precompute them in python.
            # This happens exactly ONCE per file, taking ~0.5s instead of doing it per-word!
            self.f1_values = np.array([self.formant.get_value_at_time(1, t) for t in self.formant_times])
            self.f2_values = np.array([self.formant.get_value_at_time(2, t) for t in self.formant_times])
            
            self._loaded = True
            return True
        except Exception as e:
            print(f"Error loading acoustic features for {self.audio_path}: {e}")
            return False
            
    def get_features_for_interval(self, start_time, end_time, time_step=0.01):
        """
        Returns dictionary of acoustic features for the given interval instantly via array slicing.
        """
        if not self.has_parselmouth or not self.load():
            return {
                'pitch_pattern': None,
                'f0_mean': None,
                'f1_mean': None,
                'f2_mean': None
            }
            
        # Slice pitch
        p_mask = (self.pitch_times >= start_time) & (self.pitch_times <= end_time)
        f0_interval = self.pitch_values[p_mask]
        
        # Filter out unvoiced frames (0 Hz) and nan
        f0_interval = f0_interval[(f0_interval > 0) & (~np.isnan(f0_interval))]
        f0_mean = np.mean(f0_interval) if len(f0_interval) > 0 else None
        
        # Determine contour pattern using the sliced array
        pitch_pattern = None
        if len(f0_interval) > 2:
            smoothed = np.convolve(f0_interval, np.ones(3)/3, mode='valid') if len(f0_interval) >= 3 else f0_interval
            
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
                
        # Slice formants
        f_mask = (self.formant_times >= start_time) & (self.formant_times <= end_time)
        f1_interval = self.f1_values[f_mask]
        f2_interval = self.f2_values[f_mask]
        
        f1_interval = f1_interval[~np.isnan(f1_interval)]
        f2_interval = f2_interval[~np.isnan(f2_interval)]
        
        f1_mean = np.mean(f1_interval) if len(f1_interval) > 0 else None
        f2_mean = np.mean(f2_interval) if len(f2_interval) > 0 else None
        
        return {
            'pitch_pattern': pitch_pattern,
            'f0_mean': round(float(f0_mean), 1) if f0_mean is not None else None,
            'f1_mean': round(float(f1_mean), 1) if f1_mean is not None else None,
            'f2_mean': round(float(f2_mean), 1) if f2_mean is not None else None
        }
