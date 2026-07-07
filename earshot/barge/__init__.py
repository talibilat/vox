"""Barge-in: interrupt the agent's speech by speaking over it."""

from earshot.barge.interrupt import InterruptibleVoiceLoop
from earshot.barge.vad import SpeechOnsetDetector

__all__ = ["InterruptibleVoiceLoop", "SpeechOnsetDetector"]
