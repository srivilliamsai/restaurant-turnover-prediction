"""Bound stale remote stages without exposing authentication or touching models."""
import json,time
from pathlib import Path

class HeartbeatGuard:
    def __init__(self,path,stage,maximum_silence_seconds=300,now=None):
        self.path=Path(path);self.stage=stage;self.maximum_silence_seconds=maximum_silence_seconds
        self.last_advance=time.monotonic() if now is None else now;self.marker=None
    def observe(self,now=None):
        now=time.monotonic() if now is None else now
        try:record=json.loads(self.path.read_text())
        except (FileNotFoundError,json.JSONDecodeError):record={}
        if record.get('stage')==self.stage:
            marker=(record.get('pid'),record.get('elapsed_seconds'))
            if marker!=self.marker:self.marker=marker;self.last_advance=now
        if now-self.last_advance>self.maximum_silence_seconds:
            raise RuntimeError('Colab heartbeat stopped advancing for '+self.stage+'; preserved completed checkpoints and bounded the stale connection.')
