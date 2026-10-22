import os
import time
from gemini_client import get_gemini_client
from google.genai.errors import ServerError

client = get_gemini_client()

video_path = "data/Vehicle_clip2.mp4"
srt_path = "data/Vehicle_clip2.srt"

print("🚀 Uploading video and SRT telemetry to Google AI Studio...")
video_file = client.files.upload(file=video_path)
srt_file = client.files.upload(file=srt_path)

print("🔍 Generating comprehensive post-flight intelligence report...")

# Patient retry loop with longer backoff for heavy traffic spikes
max_retries = 4
response = None

for attempt in range(max_retries):
  try:
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[
            video_file,
            srt_file,
            (
                "Analyze this drone surveillance video alongside its "
                "corresponding SRT telemetry log. "
                "1. Cross-reference the moving vehicles and trucks with the "
                "timestamps and telemetry data in the SRT file. "
                "2. Pinpoint the exact timestamps where vehicles and trucks "
                "are active. "
                "3. Compile a comprehensive, structured intelligence report "
                "mapping out timestamps, vehicle classifications, and location "
                "data."
            ),
        ],
    )
    break
  except ServerError as e:
    if attempt < max_retries - 1:
      print(
          f"⚠️ Server high demand (Attempt {attempt + 1}/{max_retries})."
          " Pausing for 25 seconds to let traffic clear..."
      )
      time.sleep(25)
    else:
      raise e

print("\n" + "=" * 40)
print("📋 AI STUDIO SURVEILLANCE REPORT:")
print("=" * 40)
print(response.text)