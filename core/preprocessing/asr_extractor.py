import os
import numpy as np
import librosa
import whisper
import string
import logging

logger = logging.getLogger(__name__)

def load_audio_librosa(file_path: str, sr: int = 16000) -> np.ndarray:
    """
    Load an audio file into a numpy array using librosa to bypass ffmpeg.
    Whisper expects 16kHz mono float32.
    """
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        return y.astype(np.float32)
    except Exception as e:
        logger.error(f"Error loading audio via librosa: {e}")
        return np.array([], dtype=np.float32)

def transcribe_audio_to_words(audio_path: str, model_size="base") -> list:
    """
    Transcribes audio using Whisper and returns a list of dictionaries containing
    word-level boundaries, simulating the output of textgrid_parser.py
    """
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return []
        
    # SAFETY CHECK: Prevent OOM on Streamlit Cloud
    # Whisper loads the model into RAM and the uncompressed audio array.
    # If the file is >50MB, we refuse to load it to protect the server.
    try:
        from core.utils.profiler import get_environment
        if get_environment() == "STREAMLIT_CLOUD":
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            if file_size_mb > 50:
                logger.error(f"Skipping Whisper ASR for {audio_path}: File too large ({file_size_mb:.1f} MB > 50 MB limit)")
                return []
    except:
        pass
        
    logger.info(f"Loading Whisper model '{model_size}'...")
    # Load model (this downloads the weights on first run)
    model = whisper.load_model(model_size)
    
    logger.info(f"Loading audio file via librosa: {audio_path}")
    audio_data = load_audio_librosa(audio_path)
    
    if len(audio_data) == 0:
        return []
        
    logger.info("Running Whisper transcription with word timestamps...")
    result = model.transcribe(audio_data, word_timestamps=True)
    
    words_data = []
    
    for segment in result.get('segments', []):
        for word_info in segment.get('words', []):
            raw_word = word_info['word'].strip()
            
            # Clean punctuation for the core token column
            clean_token = raw_word.translate(str.maketrans('', '', string.punctuation)).lower()
            
            # If the token is purely punctuation, we might still want to keep it?
            # Usually Concordance handles punctuation separately or strips it. Let's keep it if clean_token is empty just in case.
            if not clean_token and raw_word:
                clean_token = raw_word.lower()
                
            if not clean_token:
                continue
                
            words_data.append({
                'token': clean_token,
                'ort_d': raw_word, # Save the original orthography
                'start_time': word_info['start'],
                'end_time': word_info['end']
            })
            
    logger.info(f"ASR extracted {len(words_data)} words.")
    return words_data
