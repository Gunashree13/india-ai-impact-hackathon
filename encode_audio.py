import sys
import os
from pydub import AudioSegment
import base64

def convert_and_encode(file_path, output_b64="b64.txt"):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    # Determine format
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.wav':
        # Convert WAV to MP3
        audio = AudioSegment.from_wav(file_path)
        audio.export('temp.mp3', format='mp3')
        mp3_path = 'temp.mp3'
    elif ext == '.mp3':
        # Already MP3
        mp3_path = file_path
    else:
        print("Error: Unsupported file format. Only .wav and .mp3 are supported.")
        return

    # Encode to Base64
    with open(mp3_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()

    # Save to file
    with open(output_b64, 'w') as f:
        f.write(b64)

    print(f"Base64 encoded and saved to '{output_b64}' (length: {len(b64)} characters).")

    # Clean up temp file if created
    if mp3_path == 'temp.mp3' and os.path.exists('temp.mp3'):
        os.remove('temp.mp3')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python encode_audio.py <audio_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    convert_and_encode(file_path)