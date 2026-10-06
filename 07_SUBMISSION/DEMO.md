# End-to-end demonstration

The included recording is `04_DEMO/recording/bootcamp_test_meeting.wav`. A genuine successful run's outputs are included under `04_DEMO/outputs/`; the application can also be run again to reproduce them.

1. From the repository root, create and activate a Python 3.10+ virtual environment:
   `python3 -m venv .venv` then `source .venv/bin/activate`.
2. Install dependencies: `python -m pip install -r 03_DEPENDENCIES/requirements.txt`.
3. Create `.env` from `03_DEPENDENCIES/.env.example` only if it is absent; set `GROQ_API_KEY` without overwriting an existing credentials file. OpenAI credentials are not required. Do not share or commit API keys.
4. Start the interactive application with `python run.py`.
5. Open the local Gradio URL printed by the command.
6. Upload `04_DEMO/recording/bootcamp_test_meeting.wav` in the recording control.
7. Optionally enter comma-separated names, acronyms, or technical terms in the glossary field.
8. Select **Start processing** and wait for every displayed pipeline stage to complete.
9. Open **Transcript → Raw transcript** to inspect the complete timestamped local Whisper result.
10. Open **Transcript → Refined transcript** to inspect the complete refined transcript; it remains separate from the raw result.
11. Open **Overview** and **Minutes** for the summary and minutes.
12. Open **Decisions** to inspect confirmed decisions and proposals that remain undecided.
13. Open **Action items** and review any `Needs Review` statuses. Owners and deadlines are only retained when supported by the quoted evidence.
14. Open **Meeting record** to inspect the complete structured record as JSON and Markdown.
15. Select **Download complete results** or use the individual controls in **Export** to download the outputs.

The static files under `04_DEMO/outputs/` are copied directly from a successful run. For this recording, the raw and refined text are identical because no safe terminology correction was made; they are still separately generated and displayed stages.

To test local transcription of the included recording without running the Groq stages or launching Gradio, run this from the repository root:

```bash
python -c "import sys; sys.path.insert(0, '01_APPLICATION'); from app.pipeline.transcription import transcribe_audio; result = transcribe_audio('04_DEMO/recording/bootcamp_test_meeting.wav'); print(result['text']); print(result['segments'])"
```
