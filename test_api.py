import requests
import os

# API endpoint
url = "https://sonex-voice-api-n7v3.onrender.com/api/clone-voice"

# File path (update this if needed)
file_path = r"C:\Users\DT\Downloads\speechma_audio_Andrew Multilingual_at_4_55_39 PM_on_March_31st_2025.mp3"

# Check if file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit(1)

# Prepare the form data
files = {
    'audio': open(file_path, 'rb')
}
data = {
    'text': 'hello world'
}

print("Sending request to voice cloning API...")
print(f"URL: {url}")
print(f"Text: {data['text']}")
print(f"Audio file: {file_path}")

# Send the request
try:
    response = requests.post(url, files=files, data=data)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Save the response content to a file
        output_file = "cloned_voice.wav"
        with open(output_file, 'wb') as f:
            f.write(response.content)
        print(f"Success! Cloned voice saved to {output_file}")
    else:
        print(f"Error: Received status code {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {str(e)}")
finally:
    # Close the file
    files['audio'].close()
