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
        self._playsound_thread = None
        self._stop_event = threading.Event()
        
    def start_alarm(self) -> None:
        # 1. Prevent starting if already active
        if self.is_active:
            return
            
        self.is_active = True
        self._stop_event.clear()
        
        sound_exists = os.path.exists(self.sound_path)
        
        if self.is_windows:
            if sound_exists:
                # 3. If on Windows and WAV file exists, use winsound.PlaySound in async loop
                try:
                    import winsound
                    winsound.PlaySound(
                        self.sound_path,
                        winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP
                    )
                    logger.info(f"Started playing WAV alarm asynchronously: {self.sound_path}")
                except Exception as e:
                    logger.error(f"Failed to play WAV with winsound: {e}. Falling back to beep loop.")
                    self._start_beep_thread()
            else:
                logger.warning(f"WAV alarm file not found at {self.sound_path}. Falling back to beep loop.")
                self._start_beep_thread()
        else:
            logger.info("Non-Windows platform. Starting playsound background thread.")
            self._start_playsound_thread(sound_exists)
            
    def stop_alarm(self) -> None:
        # 1. Return if not currently active
        if not self.is_active:
            return
            
        self.is_active = False
        self._stop_event.set()
        
        if self.is_windows:
            sound_exists = os.path.exists(self.sound_path)
            if sound_exists:
                try:
                    import winsound
                    winsound.PlaySound(None, winsound.SND_PURGE)
                    logger.info("Stopped playing WAV alarm.")
                except Exception as e:
                    logger.error(f"Failed to stop WAV sound: {e}")
        
        # Join threads if they were started
        if self._beep_thread and self._beep_thread.is_alive():
            self._beep_thread.join(timeout=0.5)
            self._beep_thread = None
            
        if self._playsound_thread and self._playsound_thread.is_alive():
            self._playsound_thread.join(timeout=0.5)
            self._playsound_thread = None
            
    def _start_beep_thread(self) -> None:
        if self._beep_thread is None or not self._beep_thread.is_alive():
            self._beep_thread = threading.Thread(target=self._beep_loop, daemon=True)
            self._beep_thread.start()
            logger.info("Started background beep thread.")
        
    def _beep_loop(self) -> None:
        while self.is_active and not self._stop_event.is_set():
            try:
                import winsound
                # Play high pitched beep (frequency 2000Hz, duration 400ms)
                winsound.Beep(2000, 400)
            except Exception as e:
                logger.error(f"Error in winsound.Beep: {e}")
            # Wait 100ms before beep again or until stopped
            self._stop_event.wait(0.1)
                
    def _start_playsound_thread(self, sound_exists: bool) -> None:
        if self._playsound_thread is None or not self._playsound_thread.is_alive():
            self._playsound_thread = threading.Thread(
                target=self._playsound_loop, 
                args=(sound_exists,), 
                daemon=True
            )
            self._playsound_thread.start()
            logger.info("Started background playsound thread.")
        
    def _playsound_loop(self, sound_exists: bool) -> None:
        while self.is_active and not self._stop_event.is_set():
            if sound_exists:
                try:
                    import playsound
                    playsound.playsound(self.sound_path)
                except Exception as e:
                    logger.error(f"Error playing sound with playsound: {e}. Falling back to terminal bell.")
                    sys.stdout.write('\a')
                    sys.stdout.flush()
                    self._stop_event.wait(1.0)
            else:
                # Terminal bell fallback
                sys.stdout.write('\a')
                sys.stdout.flush()
                self._stop_event.wait(1.0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    # Define a dummy wav path for testing fallbacks
    dummy_wav = "assets/alarm.wav"
    print("=========================================================")
    print("         TESTING ASYNC AUDIO ALERT SYSTEM SYSTEM         ")
    print("=========================================================")
    print(f"Khoi tao AudioAlertSystem voi duong dan: {dummy_wav}")
    alert = AudioAlertSystem(dummy_wav)
    
    # Run test loop: alarm ON for 3 seconds, OFF for 3 seconds
    try:
        for i in range(2):
            print(f"\n--- Chu ky {i+1} ---")
            print("Kich hoat coi bao dong trong 3 giay...")
            alert.start_alarm()
            
            # Print messages from main thread to prove it is non-blocking (async)
            for j in range(3):
                print(f"  [Main Thread] Dang chay song song - giay thu {j+1}")
                time.sleep(1.0)
                
            print("Tat coi bao dong va giu im lang trong 3 giay...")
            alert.stop_alarm()
            
            for j in range(3):
                print(f"  [Main Thread] Dang chay song song - giay thu {j+1}")
                time.sleep(1.0)
                
        print("\nTest hoan thanh thanh cong.")
    except KeyboardInterrupt:
        print("\nDa ngat boi nguoi dung.")
    finally:
        alert.stop_alarm()

