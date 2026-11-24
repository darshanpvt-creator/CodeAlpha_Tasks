import streamlit as st
from train import train_model, generate_melody, MODELS

st.set_page_config(page_title="MelodyForge AI", page_icon="🎵", layout="centered")

st.title("MelodyForge AI")
st.subheader("LSTM-Powered Music Generation")
st.write("Train a recurrent neural network and generate original MIDI melodies.")

st.info(
    "Prototype note: the initial training examples are procedurally generated "
    "MIDI patterns. Replace them with a licensed MIDI collection for broader training."
)

if MODELS.joinpath("melody_lstm.pth").exists():
    st.success("Trained LSTM model found.")
else:
    st.warning("The model has not been trained yet.")

if st.button("Train AI Model", use_container_width=True):
    with st.spinner("Creating MIDI examples and training the LSTM..."):
        train_model()
    st.success("Training completed. Your model is ready!")

st.divider()
st.subheader("Generate a melody")

length = st.slider("Melody length (notes)", 16, 96, 48, step=8)
temperature = st.slider("Creativity", 0.3, 1.5, 0.9, step=0.1)

if st.button("Generate Music", type="primary", use_container_width=True):
    with st.spinner("Generating your melody..."):
        file_path = generate_melody(length, temperature)
    st.success("Your MIDI melody is ready!")
    with open(file_path, "rb") as midi_file:
        st.download_button(
            "Download generated MIDI",
            data=midi_file.read(),
            file_name="melodyforge_generated.mid",
            mime="audio/midi",
            use_container_width=True
        )

st.caption("Built with Python, PyTorch LSTM, Streamlit and pretty_midi.")
