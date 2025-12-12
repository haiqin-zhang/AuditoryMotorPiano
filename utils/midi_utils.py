import mido 


def midi_target(midi_file, chunksize=4):
    target_midi = mido.MidiFile(midi_file)
    target_notes = []
    for message in target_midi.tracks[0]:
        if message.type == "note_on":
            target_notes.append(message.note)

    split_list = [target_notes[i:i + chunksize] for i in range(0, len(target_notes), chunksize)]
    return split_list
