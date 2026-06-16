import os
import sys
import logging
import threading
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class AlertSystem(ABC):
    """
    Abstract interface for triggerable system alerts.
    Follows Single Responsibility Principle (SRP) and Open/Closed Principle (OCP).
    """
    
    @abstractmethod
    def start_alarm(self) -> None:
        """Trigger the alarm sound."""
        pass
        
    @abstractmethod
    def stop_alarm(self) -> None:
        """Silence the alarm sound."""
        pass


class AudioAlertSystem(AlertSystem):
    """
    Asynchronous audio alert system that plays alarm sounds on a background thread
    or via OS-native non-blocking APIs.
    """
    
    def __init__(self, sound_path: str):
        self.sound_path = os.path.abspath(sound_path)
        self.is_active = False
        self.is_windows = sys.platform == "win32"
        self._beep_thread = None
        self._stop_event = threading.Event()
        
    def start_alarm(self) -> None:
        if self.is_active:
            return
            
        logger.info("Activating alertness alarm...")
        self.is_active = True
        
        sound_exists = os.path.exists(self.sound_path)
        
        if self.is_windows:
            if sound_exists:
                try:
                    import winsound
                    # SND_ASYNC plays sound asynchronously. SND_LOOP loops the sound. SND_FILENAME specifies it's a file.
                    winsound.PlaySound(
                        self.sound_path, 
                        winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP
                    )
                    logger.info("Playing Windows native WAV alarm in async loop.")
                    return
                except Exception as e:
                    logger.error(f"Failed to play sound via winsound: {e}. Falling back to system beep.")
            
            # If sound doesn't exist or winsound fails, run system beep thread
            self._start_beep_thread()
            
        else:
            # Non-Windows systems: Use a thread to run playsound (or system bell)
            self._start_playsound_thread(sound_exists)
            
    def stop_alarm(self) -> None:
        if not self.is_active:
            return
            
        logger.info("Silencing alertness alarm.")
        self.is_active = False
        self._stop_event.set()
        
        if self.is_windows:
            try:
                import winsound
                # SND_PURGE stops all playing sounds associated with the calling thread
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception as e:
                logger.error(f"Error purging winsound sound: {e}")
                
        # Clean up any threads
        if self._beep_thread is not None:
            self._beep_thread.join(timeout=0.5)
            self._beep_thread = None
            
    def _start_beep_thread(self) -> None:
        self._stop_event.clear()
        self._beep_thread = threading.Thread(target=self._beep_loop, daemon=True)
        self._beep_thread.start()
        logger.info("Started system beep alarm thread.")
        
    def _beep_loop(self) -> None:
        import winsound
        while not self._stop_event.is_set() and self.is_active:
            try:
                # 1500 Hz tone for 400 milliseconds
                winsound.Beep(1500, 400)
                # Pause brief moment between beeps
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Beep alarm error: {e}")
                time.sleep(0.5)
                
    def _start_playsound_thread(self, sound_exists: bool) -> None:
        self._stop_event.clear()
        self._beep_thread = threading.Thread(
            target=self._playsound_loop, 
            args=(sound_exists,), 
            daemon=True
        )
        self._beep_thread.start()
        logger.info("Started cross-platform sound alarm thread.")
        
    def _playsound_loop(self, sound_exists: bool) -> None:
        while not self._stop_event.is_set() and self.is_active:
            try:
                if sound_exists:
                    from playsound import playsound
                    playsound(self.sound_path)
                else:
                    # Output terminal bell character on Unix/Mac
                    sys.stdout.write('\a')
                    sys.stdout.flush()
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"Cross-platform sound playback error: {e}")
                time.sleep(1.0)
