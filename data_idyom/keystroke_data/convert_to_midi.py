import csv
import mido
from mido import Message, MidiFile, MidiTrack, bpm2tempo

def csv_to_midi(csv_filename, midi_filename='output.mid', note_duration=0.3, tempo_bpm=120):
    """
    Convert a CSV with columns 'time' and 'midi_values' into a MIDI file.
    
    Parameters:
    - csv_filename: str, path to the input CSV file.
    - midi_filename: str, path to save the output MIDI file.
    - note_duration: float, duration (in seconds) each note should be held.
    - tempo_bpm: int, tempo in beats per minute (affects timing).
    """
    # Load notes from CSV
    notes = []
    with open(csv_filename, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            notes.append((float(row['time']), int(row['midi_values'])))
    
    if not notes:
        raise ValueError("No notes found in CSV")
    
    # Create MIDI file and track
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    tempo = bpm2tempo(tempo_bpm)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    
    ticks_per_beat = mid.ticks_per_beat
    time0 = notes[0][0]
    
    for i, (time, note) in enumerate(notes):
        # Calculate delta time from previous note
        delta_time_sec = time - (notes[i - 1][0] if i > 0 else time0)
        delta_ticks = int(mido.second2tick(delta_time_sec, ticks_per_beat, tempo))
        
        # Note ON
        track.append(Message('note_on', note=note, velocity=64, time=delta_ticks))
        
        # Note OFF after note_duration seconds
        off_ticks = int(mido.second2tick(note_duration, ticks_per_beat, tempo))
        track.append(Message('note_off', note=note, velocity=64, time=off_ticks))
    
    # Save MIDI file
    mid.save(midi_filename)
    print(f"MIDI file saved as {midi_filename}")

subjects_to_process = ['13','14','15','16','17','18','19','20']

for sub in subjects_to_process:
    csv_to_midi(f'../midi_analysis/keystroke_data/keystrokes_{sub}_pre.csv', f'./sub_recordings/audio_{sub}_pre.mid', note_duration=0.4, tempo_bpm=100)
