# FYP-2 Enhancement Proposals

This document outlines 10 potential features to enhance the Facial Emotion Recognition (FER) project for FYP-2. These features focus on technical depth, user utility, and real-world application.

## 1. Session Timeline Analysis (Time-Series Visualization)
**Concept:** Instead of just a single "Dominant Emotion" for a session, record the emotion probabilities every second.
**Implementation:**
-   Store an array of `{ timestamp, emotion, confidence }` in the database.
-   Use a line chart (Recharts) to visualize how emotions changed over the duration of the session.
**Value:** Allows users to see triggers and emotional shifts (e.g., "I got stressed 2 minutes into the meeting").

## 2. Smart Recommendations Engine
**Concept:** Provide actionable feedback based on detected emotions.
**Implementation:**
-   Create a rule engine: If `Stress > 70%` for `> 10 seconds` -> Trigger Recommendation.
-   **Actions:**
    -   "Take a deep breath" (Visual breathing guide).
    -   Suggest a calming Spotify playlist.
    -   Prompt to take a break.
**Value:** Transforms the app from a passive monitor to an active wellness tool.

## 3. Mood Calendar & Long-term Insights
**Concept:** Track emotional trends over days, weeks, and months.
**Implementation:**
-   **Calendar View:** Color-code days based on the average dominant emotion.
-   **Weekly Report:** "You were 20% happier this week compared to last week."
-   **Pattern Recognition:** "You tend to be most anxious on Monday mornings."
**Value:** Provides long-term mental health insights.

## 4. Real-time Multi-Face Detection (Group Analysis)
**Concept:** Analyze the emotions of multiple people in the frame simultaneously.
**Implementation:**
-   Upgrade the detection loop to handle multiple bounding boxes.
-   Display separate emotion labels for each face.
-   **Use Case:** Classroom engagement monitoring or meeting sentiment analysis.
**Value:** Expands the use case from personal to social/professional settings.

## 5. Multimodal Analysis (Voice + Face)
**Concept:** Combine facial expression analysis with voice tone analysis for higher accuracy.
**Implementation:**
-   Capture audio input alongside video.
-   Use a lightweight audio model (like a simple speech-emotion-recognition model) to detect tone.
-   Combine the weighted average of Face + Voice confidence.
**Value:** Significant technical depth and improved accuracy (e.g., detecting sarcasm or hidden distress).

## 6. Privacy-First "Incognito" Mode
**Concept:** A mode where no images are ever sent to the server or stored.
**Implementation:**
-   Toggle switch for "Incognito Mode".
-   Ensure all processing happens client-side (TensorFlow.js).
-   Disable the "Save Report" feature or only save numerical data (no snapshots).
**Value:** Addresses privacy concerns, a critical topic in AI ethics.

## 7. Exportable Professional Reports (PDF)
**Concept:** Generate high-quality PDF reports for therapists or doctors.
**Implementation:**
-   Use a library like `jspdf` or `react-pdf`.
-   Include charts, summary statistics, and snapshots in a clean layout.
-   Add a "Notes" section for the user to add context before exporting.
**Value:** Bridges the gap between self-help and professional care.

## 8. Gamification & Wellness Streaks
**Concept:** Encourage positive habits through gamification.
**Implementation:**
-   **Streaks:** "Logged mood for 7 days in a row!"
-   **Badges:** "Zen Master" (Maintained calm for 10 mins), "Positivity Pro" (High happiness score).
-   **Daily Challenges:** "Try to smile for 1 minute today."
**Value:** Increases user retention and engagement.

## 9. Custom Emotion Calibration
**Concept:** Allow the user to "teach" the model their specific facial expressions.
**Implementation:**
-   **Calibration Mode:** Ask user to "Make a Happy Face" -> Capture image -> Fine-tune or offset the model's baseline for that user.
-   Store user-specific calibration weights.
**Value:** Personalization that improves accuracy for unique facial structures.

## 10. Emotion-Based Content Filtering
**Concept:** Suggest content (movies, music, articles) based on the current mood.
**Implementation:**
-   Integrate with a 3rd party API (e.g., TMDB for movies, Spotify for music).
-   **Logic:**
    -   If `Sad` -> Suggest "Uplifting Movies".
    -   If `Happy` -> Suggest "Party Playlist".
**Value:** Fun, consumer-facing feature that demonstrates API integration skills.
