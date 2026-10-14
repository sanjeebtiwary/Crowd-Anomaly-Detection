import os
import tempfile

import cv2
import pandas as pd
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration & API Settings
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Crowd Anomaly Detector",
    page_icon="🚨",
    layout="wide"
)

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = int(os.getenv("API_TIMEOUT", "300"))

st.sidebar.title("⚙️ API Configuration")
api_url = st.sidebar.text_input("Base API URL", value=DEFAULT_API_URL)

if st.sidebar.button("Check API Health"):
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        if response.status_code == 200:
            st.sidebar.success("API is healthy and reachable! 🟢")
        else:
            st.sidebar.error(f"API responded with status code: {response.status_code} 🔴")
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Failed to connect to API: {e}")


# -----------------------------------------------------------------------------
# Helper Function: Extract Anomaly Frames
# -----------------------------------------------------------------------------
def extract_anomaly_frames(video_bytes, flags, max_display=12, suffix=".mp4"):
    """
    Saves video to a temporary file, reads frames using OpenCV,
    and returns a list of RGB image arrays for frames flagged as anomalous.
    """
    # Write uploaded video bytes to a temporary file for OpenCV
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(video_bytes)
        tmp_file_path = tmp_file.name

    cap = cv2.VideoCapture(tmp_file_path)
    anomaly_frame_images = []

    # Identify indices where flags are True
    flagged_indices = [idx for idx, flagged in enumerate(flags) if flagged]

    # Sample indices if there are too many to avoid UI slowdowns
    if len(flagged_indices) > max_display:
        step = len(flagged_indices) // max_display
        sampled_indices = flagged_indices[::step][:max_display]
    else:
        sampled_indices = flagged_indices

    current_frame = 0
    sampled_set = set(sampled_indices)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in sampled_set:
            # OpenCV loads frames as BGR; convert to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            anomaly_frame_images.append((current_frame + 1, frame_rgb))

            if len(anomaly_frame_images) == len(sampled_indices):
                break

        current_frame += 1

    cap.release()
    os.remove(tmp_file_path)  # Clean up temp file

    return anomaly_frame_images


# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------
st.title("🚨 Crowd Anomaly Detection Dashboard")
st.markdown("Upload a video clip to run anomaly scoring and inspect anomalous video frames.")

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov", "mkv"],
    help="Upload a video file for processing."
)

if uploaded_file is not None:
    col_vid, col_down = st.columns([3, 1])

    with col_vid:
        st.video(uploaded_file)

    with col_down:
        st.markdown("### 💾 Video File Options")
        st.download_button(
            label="⬇️ Download Source Video",
            data=uploaded_file.getvalue(),
            file_name=uploaded_file.name,
            mime=uploaded_file.type or "video/mp4",
            width="stretch"
        )

    if st.button("🔍 Analyze Video", type="primary"):
        with st.spinner("Processing video and extracting frame-level scores..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                predict_endpoint = f"{api_url.rstrip('/')}/predict"

                response = requests.post(predict_endpoint, files=files, timeout=REQUEST_TIMEOUT)

                if response.status_code == 200:
                    data = response.json()
                    st.success("Analysis Complete!")
                    st.divider()

                    # 1. Summary Metrics
                    st.subheader("📊 Processing Summary")
                    col1, col2, col3, col4 = st.columns(4)

                    anomaly_detected = data.get("anomaly_detected", False)
                    col1.metric("Filename", data.get("filename", uploaded_file.name))
                    col2.metric("Total Frames", data.get("frames_processed", 0))
                    col3.metric("Anomaly Frames", data.get("anomaly_frames", 0))
                    col4.metric(
                        "Anomaly Status",
                        "DETECTED 🚨" if anomaly_detected else "NORMAL ✅",
                        delta="Action Required" if anomaly_detected else "Safe",
                        delta_color="inverse" if anomaly_detected else "normal"
                    )

                    # 2. Score Chart
                    st.subheader("📈 Frame-Level Anomaly Scores")
                    scores = data.get("scores", [])
                    flags = data.get("flags", [])
                    threshold = data.get("threshold", 0.5)

                    if scores:
                        df_chart = pd.DataFrame({
                            "Frame": list(range(1, len(scores) + 1)),
                            "Anomaly Score": scores,
                            "Threshold": [threshold] * len(scores)
                        }).set_index("Frame")

                        st.line_chart(df_chart, color=["#FF4B4B", "#008000"])
                        st.caption(f"Horizontal threshold line set at **{threshold}**.")

                    # 3. Anomaly Frame Gallery
                    st.subheader("🖼️ Detected Anomaly Frame Snapshots")

                    if flags and any(flags):
                        with st.spinner("Extracting flagged frames from video..."):
                            video_bytes = uploaded_file.getvalue()
                            frame_snapshots = extract_anomaly_frames(video_bytes, flags, max_display=12,
                                                              suffix=os.path.splitext(uploaded_file.name)[1] or ".mp4")

                            if frame_snapshots:
                                # Render snapshots in a 4-column grid layout
                                cols = st.columns(4)
                                for idx, (frame_num, img_array) in enumerate(frame_snapshots):
                                    with cols[idx % 4]:
                                        st.image(
                                            img_array,
                                            caption=f"Frame #{frame_num} (Score: {scores[frame_num - 1]:.2f})",
                                            width="stretch"
                                        )
                            else:
                                st.warning("Could not decode flagged frames from the video file.")
                    else:
                        st.info("No frames exceeded the anomaly threshold in this video.")

                    # 4. Anomaly Segments Detail & Export
                    st.subheader("🚨 Detected Anomaly Segments")
                    segments = data.get("segments", [])

                    if segments:
                        df_segments = pd.DataFrame(segments)
                        df_segments["Duration (Frames)"] = df_segments["end_frame"] - df_segments["start_frame"] + 1
                        df_segments.rename(columns={
                            "start_frame": "Start Frame",
                            "end_frame": "End Frame"
                        }, inplace=True)

                        st.dataframe(df_segments, width="stretch")

                        csv_data = df_segments.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Export Segments as CSV",
                            data=csv_data,
                            file_name=f"{data.get('filename', 'video')}_anomaly_segments.csv",
                            mime="text/csv",
                            type="secondary"
                        )

                elif response.status_code == 422:
                    st.error("Validation Error (422): Please check the uploaded file format.")
                    st.json(response.json())
                else:
                    st.error(f"Error {response.status_code}: {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"API Request Failed: {e}")
