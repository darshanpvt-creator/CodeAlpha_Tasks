from pathlib import Path
import random
import numpy as np
import pretty_midi
import torch
from torch import nn

BASE = Path(__file__).parent
DATA = BASE / "data" / "training"
MODELS = BASE / "models"
DATA.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

LOW, HIGH = 48, 84
DEVICE = torch.device("cpu")
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def create_training_data(count=24):
    """Create small structured MIDI examples for the prototype."""
    scale_patterns = [
        [0, 2, 4, 5, 7, 9, 11, 12],
        [0, 2, 3, 5, 7, 9, 10, 12],
        [0, 2, 4, 7, 9, 12, 14, 16],
    ]

    for index in range(count):
        root = random.choice([48, 50, 52, 53, 55, 57, 60])
        scale = random.choice(scale_patterns)
        notes = []
        pitch = root

        for step in range(40):
            if step % 8 == 0:
                pitch = root + random.choice(scale)
            else:
                pitch += random.choice([-2, -1, 0, 1, 2, 3, -3])
                pitch = min(max(pitch, root), root + 16)

            pitch = min(max(pitch, LOW), HIGH)
            start = step * 0.35
            duration = random.choice([0.2, 0.25, 0.3])
            notes.append(
                pretty_midi.Note(
                    velocity=random.randint(75, 105),
                    pitch=int(pitch),
                    start=start,
                    end=start + duration
                )
            )

        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        instrument.notes = notes
        midi.instruments.append(instrument)
        midi.write(str(DATA / f"training_{index + 1:02d}.mid"))

def read_midi_sequences():
    sequences = []
    for file in DATA.glob("*.mid"):
        midi = pretty_midi.PrettyMIDI(str(file))
        notes = sorted(
            [n.pitch for inst in midi.instruments for n in inst.notes],
        )
        notes = [p - LOW for p in notes if LOW <= p <= HIGH]
        if len(notes) >= 10:
            sequences.append(notes)
    return sequences

class MelodyLSTM(nn.Module):
    def __init__(self, vocab_size=HIGH - LOW + 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        return self.fc(output)

def train_model():
    create_training_data()
    sequences = read_midi_sequences()

    window = 8
    inputs, targets = [], []
    for sequence in sequences:
        for i in range(len(sequence) - window):
            inputs.append(sequence[i:i + window])
            targets.append(sequence[i + window])

    x = torch.tensor(inputs, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)

    model = MelodyLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(6):
        permutation = torch.randperm(len(x))
        total_loss = 0.0

        for start in range(0, len(x), 64):
            indices = permutation[start:start + 64]
            batch_x, batch_y = x[indices], y[indices]
            optimizer.zero_grad()
            predictions = model(batch_x)[:, -1, :]
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/6 - loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODELS / "melody_lstm.pth")
    print(f"Training complete. MIDI files: {len(sequences)}")
    print(f"Training examples: {len(x)}")
    print(f"Model saved: {MODELS / 'melody_lstm.pth'}")

def generate_melody(length=48, temperature=0.9):
    model_path = MODELS / "melody_lstm.pth"
    if not model_path.exists():
        train_model()

    model = MelodyLSTM()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    seed = torch.tensor([[0, 2, 4, 7, 9, 7, 4, 2]], dtype=torch.long)
    generated = seed[0].tolist()

    with torch.no_grad():
        for _ in range(length):
            context = torch.tensor([generated[-8:]], dtype=torch.long)
            logits = model(context)[:, -1, :] / max(temperature, 0.1)
            probabilities = torch.softmax(logits, dim=-1)
            next_note = torch.multinomial(probabilities, 1).item()
            generated.append(next_note)

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    for index, note in enumerate(generated):
        start = index * 0.3
        instrument.notes.append(
            pretty_midi.Note(
                velocity=95,
                pitch=int(note + LOW),
                start=start,
                end=start + 0.25
            )
        )

    midi.instruments.append(instrument)
    output = BASE / "generated"
    output.mkdir(exist_ok=True)
    filename = output / "melodyforge_generated.mid"
    midi.write(str(filename))
    return filename

if __name__ == "__main__":
    train_model()
