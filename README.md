# 🎙️ AI Meeting Assistant

<p align="center">
  <strong>From raw meeting audio to a grounded, structured, and actionable meeting record.</strong>
</p>

<p align="center">
  <em>Inter IIT Bootcamp — Phase 2 ML</em>
</p>

---

## 🚀 Overview

Meetings contain some of the most important information in a project:

- technical discussions
- decisions
- commitments
- deadlines
- responsibilities
- project updates
- unresolved issues
- follow-up tasks

However, this information is usually buried inside a long audio recording.

A conventional workflow looks like:

```text
Meeting Recording
       ↓
Listen to Recording
       ↓
Write Transcript
       ↓
Correct Technical Terms
       ↓
Identify Decisions
       ↓
Identify Action Items
       ↓
Identify Owners / Deadlines
       ↓
Write Meeting Minutes
The AI Meeting Assistant turns this process into an end-to-end multimodel AI pipeline:

                    🎙️ MEETING AUDIO
                           │
                           ▼
                 ┌───────────────────┐
                 │  Input Validation  │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   faster-whisper  │
                 │   Speech → Text   │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   RAW TRANSCRIPT  │
                 │                   │
                 │ What was actually │
                 │      spoken       │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   GPT-OSS-120B    │
                 │                   │
                 │ Domain-Aware      │
                 │ Transcript        │
                 │ Refinement        │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ REFINED TRANSCRIPT│
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   GPT-OSS-20B     │
                 │                   │
                 │ Documentation &   │
                 │ Structured        │
                 │ Extraction        │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │    EVIDENCE       │
                 │    VALIDATION     │
                 └─────────┬─────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │       STRUCTURED RECORD        │
          │                                │
          │  Summary                       │
          │  Decisions                     │
          │  Action Items                  │
          │  Owners                        │
          │  Deadlines                     │
          │  Evidence                      │
          └────────────────────────────────┘
The application exposes the three core representations directly:

Raw Transcript → Refined Transcript → Final Output

🎯 Problem
The system is designed for recorded English-language meetings where the final output should contain:

an accurate transcript
domain-aware terminology refinement
a structured meeting summary
confirmed decisions
actionable tasks
owners where explicitly stated
deadlines where explicitly stated
evidence supporting extracted decisions and actions
The challenge is not simply generating a fluent summary.

The real challenge is:

How do we improve and structure a meeting conversation without inventing information that was never actually said?

That question drives the architecture of the entire system.

🧠 Core Design Philosophy
Preserve First. Improve Second. Extract Third.
The pipeline is intentionally divided into three major semantic representations:

                 AUDIO
                   │
                   ▼
          ┌─────────────────┐
          │ Raw Transcript  │
          └────────┬────────┘
                   │
              refinement
                   │
                   ▼
          ┌─────────────────┐
          │ Refined         │
          │ Transcript      │
          └────────┬────────┘
                   │
              extraction
                   │
                   ▼
          ┌─────────────────┐
          │ Meeting Record  │
          └─────────────────┘
Each representation answers a different question.

Raw Transcript
What did the speech-to-text system hear?

Refined Transcript
How can the transcript be made more accurate and readable while preserving what was said?

Meeting Record
What decisions, tasks, and important information can safely be extracted from the conversation?

This separation prevents the final documentation model from simultaneously rewriting the underlying conversation and summarizing it.

🏗️ Complete Architecture
┌───────────────────────────────────────────────────────────┐
│                         USER                              │
│                                                           │
│              Upload Meeting Recording                    │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                    INPUT VALIDATION                       │
│                                                           │
│  • File validation                                        │
│  • Audio readability                                      │
│  • Empty/corrupt input handling                           │
│  • Unsupported input handling                             │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                    STAGE 1 — STT                          │
│                                                           │
│                    faster-whisper                         │
│                                                           │
│               Audio → Timestamped Text                    │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                    RAW TRANSCRIPT                         │
│                                                           │
│  Preserved representation of the recorded conversation.  │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                 STAGE 2 — REFINEMENT                      │
│                                                           │
│                    GPT-OSS-120B                            │
│                                                           │
│  • Domain terminology                                     │
│  • Technical vocabulary                                   │
│  • Names                                                  │
│  • Numbers                                                │
│  • Monetary values                                        │
│  • Negations                                              │
│  • Commitments                                            │
│  • Grammar / readability                                  │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                  REFINED TRANSCRIPT                       │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│               STAGE 3 — DOCUMENTATION                     │
│                                                           │
│                    GPT-OSS-20B                             │
│                                                           │
│  • Summary                                                │
│  • Decisions                                              │
│  • Action Items                                           │
│  • Owners                                                 │
│  • Deadlines                                              │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                 EVIDENCE VALIDATION                        │
│                                                           │
│  Extracted claims are checked against transcript evidence.│
│                                                           │
│  Proposal ≠ Decision                                      │
│  Suggestion ≠ Assignment                                  │
│  Mentioned date ≠ Deadline                                │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────┐
│                 FINAL MEETING RECORD                       │
│                                                           │
│  Human-readable + machine-readable outputs                 │
└───────────────────────────────────────────────────────────┘
🎙️ Stage 1 — Speech Recognition
faster-whisper
The first stage converts the recorded English meeting into timestamped transcript segments.

Meeting Audio
      ↓
faster-whisper
      ↓
Timestamped Segments
      ↓
Raw Transcript
The raw transcript is preserved as its own artifact.

It is not silently replaced by later processing.

Why preserve the raw transcript?
Keeping the raw transcript makes the pipeline inspectable.

If an error appears in the final meeting record, the intermediate representations make it possible to determine whether the problem originated from:

Audio
  ↓
Speech Recognition
  ↓
Refinement
  ↓
Documentation
  ↓
Evidence Validation
rather than treating the complete application as one black box.

✨ Stage 2 — Domain-Aware Transcript Refinement
openai/gpt-oss-120b
The second stage performs controlled transcript refinement.

Inference is performed through Groq.

The model's responsibility is:

Improve transcript quality without changing the underlying meaning.

The refinement stage focuses on:

technical terminology
domain-specific vocabulary
names
acronyms
numbers
monetary values
negations
commitments
grammatical cleanup
terminology consistency
It is deliberately not responsible for producing the final meeting summary.

🔒 Refinement Safety
Certain pieces of meeting information are especially sensitive.

Names
Names should not be silently replaced with invented or more common names.

Numbers
Numerical information should be preserved accurately.

For example:

2.5 lakh
must not become:

25 lakh
Negation
Consider:

"We will not deploy this version."
The refined transcript must retain the negative commitment.

It must not become:

"We will deploy this version."
Commitments
Statements such as:

"We will complete this tomorrow."
should preserve their commitment semantics.

🧩 Glossary Extraction
The refinement pipeline can extract domain-specific terminology before transcript refinement.

The glossary can contain:

technical terms
product names
project names
domain vocabulary
important names
acronyms
The glossary is treated as an optional enrichment stage.

Therefore:

Glossary Extraction
        │
        ├── Success
        │      ↓
        │   Use terms
        │
        └── Failure
               ↓
        Continue refinement
A failure in optional terminology extraction should not terminate the complete meeting-processing pipeline.

⏱️ Long Meeting Processing
One of the major engineering challenges appeared while processing a full approximately 15-minute meeting recording.

Large transcripts can exceed the practical token budget of a single structured LLM request.

Instead of assuming a fixed meeting length, the final system uses token-aware transcript chunking.

                 LONG TRANSCRIPT
                        │
                        ▼
               Token-aware Chunking
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
       Chunk 1       Chunk 2       Chunk N
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                  GPT-OSS-120B
                        │
                        ▼
              Ordered Reconstruction
                        │
                        ▼
                Refined Transcript
Chunks are processed sequentially and reconstructed in their original order.

This avoids relying on a hardcoded meeting-duration assumption.

📝 Stage 3 — Structured Meeting Documentation
openai/gpt-oss-20b
The third stage receives the refined transcript and generates the structured meeting record.

Its responsibility is different from the refinement model.

The documentation model asks:

What useful meeting information can safely be extracted from this conversation?

It produces:

📋 Summary
A concise representation of what happened during the meeting.

✅ Decisions
Only decisions supported by the conversation.

📌 Action Items
Tasks supported by the conversation.

👤 Owners
Only when an owner is actually stated.

📅 Deadlines
Only when a deadline is actually stated.

🔎 Evidence
Important decisions and actions remain grounded in transcript evidence.

🛡️ Evidence-Based Extraction
A meeting assistant must distinguish between discussion and commitment.

For example:

"We could move this to next week."
                │
                ▼
             Proposal
                ≠
             Decision
Likewise:

"Maybe Jatin can handle it."
                │
                ▼
            Suggestion
                ≠
            Assignment
The system therefore treats:

Proposal        ≠ Decision
Suggestion      ≠ Assignment
Mentioned date  ≠ Deadline
Possibility     ≠ Commitment
The evidence layer validates extracted decisions and tasks against the transcript before they become part of the final meeting record.

🔐 Structured Output Reliability
During development, the glossary extraction stage encountered Groq structured-output validation failures such as:

400
json_validate_failed
Failed to validate JSON
The final implementation introduced multiple protections:

strict JSON schema
synchronized prompts
explicit empty-array handling
schema → JSON-object fallback
retries with backoff
JSON boundary recovery
graceful glossary failure
API-key sanitization
An important architectural decision was made:

Optional enrichment must never bring down the mandatory pipeline.

If glossary extraction fails, transcript refinement can continue using fallback terminology.

🧠 Reasoning Token Control
The selected gpt-oss models are reasoning models.

During long-recording testing, reasoning-token consumption could consume the available generation budget before the model produced the required structured JSON.

This could result in:

Generation reaches limit
        ↓
No valid JSON produced
        ↓
Groq JSON validation failure
The final implementation therefore controls reasoning behavior for supported reasoning models.

The goal is not to expose or depend on hidden reasoning.

The goal is to make structured generation reliable.

🔁 Retry and Fallback Architecture
Structured LLM requests follow a resilient request strategy.

                  LLM Request
                       │
                       ▼
                    Success
                       │
                       ▼
                  Parse JSON
                       │
                       │
                Validation Error
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Fallback       Retry       Backoff
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
                  Try Again
Different pipeline stages have different criticality.

Optional stage
Glossary extraction:

Failure → Warning → Continue
Required stages
Refinement and documentation:

Failure → Explicit Error → Safe Pipeline Stop
The system does not silently produce an incomplete final record.

🔐 API Key Protection
API errors can sometimes contain sensitive information.

The application sanitizes exceptions before logging them.

For example:

gsk_xxxxxxxxxxxxxxxxxxxxx
        ↓
[REDACTED_API_KEY]
This prevents credentials from accidentally appearing in terminal output or application error messages.

📡 Stage-Level Observability
The pipeline reports explicit processing stages.

A successful run can look like:

[STAGE] received
[STAGE] validated
[STAGE] transcribing
[STAGE] raw_ready
[STAGE] refining
[STAGE] glossary
[STAGE] refinement chunk 1/6
[STAGE] refinement chunk 2/6
[STAGE] refinement chunk 3/6
[STAGE] refinement chunk 4/6
[STAGE] refinement chunk 5/6
[STAGE] refinement chunk 6/6
[STAGE] refined
[STAGE] documentation
[STAGE] validating
[STAGE] evidence
[STAGE] exporting
[STAGE] complete
This is particularly important for long recordings because a generic:

Processing failed
does not tell the developer or evaluator where the failure occurred.

🎨 User Interface
The final interface intentionally hides unnecessary implementation complexity from the user.

The primary workflow is:

Upload
   ↓
Process
   ↓
Inspect
   ↓
Export
The output interface is organized around the three core representations.

Tab 1 — Raw Transcript
Shows the direct transcription output.

This allows the user to inspect what the speech-recognition model produced before any LLM refinement.

Tab 2 — Refined Transcript
Shows the domain-aware refined transcript.

The user can compare:

Raw Transcript
      ↕
Refined Transcript
instead of blindly trusting a final summary.

Tab 3 — Final Output
Shows the structured meeting record containing:

summary
decisions
action items
owners
deadlines
evidence
📱 Responsive Design
The final interface was designed for different screen sizes.

It includes:

responsive header layout
mobile stacking
readable text
accessible controls
explicit light-theme styling
high-contrast text
no unnecessary horizontal scrolling
clear tab hierarchy
The landing section communicates the workflow:

Audio → Refine → Meeting Record
🎨 Light Theme Reliability
During development, the Gradio interface experienced a visibility issue where system/browser dark-mode preferences caused light text to appear against light backgrounds.

The result was effectively:

Light Background
       +
Light Text
       =
Unreadable Interface
The final implementation explicitly controls the relevant theme variables and typography styles.

This ensures:

body text remains visible
Markdown remains readable
tab labels remain readable
transcript text remains readable
final output remains readable
🧪 Testing Strategy
The project contains a dedicated test suite:

06_TESTS/
Run the complete suite with:

pytest 06_TESTS

Current verification:

============================== 43 passed ==============================
🔬 Test Coverage
test_pipeline.py
Covers pipeline-level behavior including:

audio validation
transcription flow
pipeline execution
output generation
package creation
test_refinement.py
Covers refinement behavior including:

transcript chunking
chunk ordering
edit safety
glossary fallback
structured JSON handling
reasoning configuration
retry behavior
sanitization
stage logging
test_safety.py
Covers preservation of important semantic information including:

negation
monetary values
safety-sensitive transcript transformations
test_evidence.py
Covers evidence-related behavior including:

decision classification
evidence matching
quote extraction
test_ui.py
Covers interface behavior including:

status streaming
error presentation
three-stage output layout
📊 End-to-End Verification
The final project was tested using a real approximately 15-minute meeting recording.

The complete pipeline successfully reached:

received
    ↓
validated
    ↓
transcribing
    ↓
raw_ready
    ↓
refining
    ↓
refined
    ↓
documenting
    ↓
validating
    ↓
exporting
    ↓
complete
The successful run produced:

Raw transcript      → 14,073 characters
Refined transcript  → 14,104 characters
Confirmed decisions → 2
Action items        → 4
The complete output bundle was successfully generated.

🎬 Demo
All demo material is kept under:

04_DEMO/
The structure is:

04_DEMO/
│
├── recording/
│   ├── bootcamp_test_meeting.wav
│   └── README.md
│
├── outputs/
│   ├── raw_transcript.txt
│   ├── refined_transcript.txt
│   ├── minutes.md
│   ├── decisions.json
│   ├── action_items.json
│   ├── meeting_record.json
│   ├── meeting_record.md
│   └── refinement_audit.json
│
└── DEMO.md
The demo demonstrates the complete workflow:

Upload
  ↓
Validation
  ↓
Transcription
  ↓
Raw Transcript
  ↓
Refinement
  ↓
Refined Transcript
  ↓
Documentation
  ↓
Evidence Validation
  ↓
Final Meeting Record
  ↓
Export
📦 Output Contract
Raw Transcript
raw_transcript.txt
Contains the speech-recognition output.

Refined Transcript
refined_transcript.txt
Contains the domain-aware refined transcript.

Minutes
minutes.md
Human-readable meeting minutes.

Decisions
decisions.json
Machine-readable confirmed decisions.

Action Items
action_items.json
Machine-readable action items.

Meeting Record
meeting_record.json
Complete structured machine-readable meeting record.

Human-Readable Meeting Record
meeting_record.md
Human-readable representation of the structured meeting record.

Refinement Audit
refinement_audit.json
Audit information associated with transcript refinement.

🧩 Technology Stack
Layer	Technology
Programming Language	Python
Speech-to-Text	faster-whisper
Refinement Model	openai/gpt-oss-120b
Documentation Model	openai/gpt-oss-20b
LLM Inference	Groq
User Interface	Gradio
Structured Output	JSON / JSON Schema
Testing	Pytest
💡 Why faster-whisper?
The speech-to-text stage could be implemented using a hosted transcription API.

The project instead uses local faster-whisper because it provides:

local speech recognition
timestamped segments
no separate hosted STT API dependency
no external STT quota during inference
direct control over transcription output
a predictable interface for downstream processing
This keeps the speech-recognition stage independent from the Groq language-model stages.

💡 Why Groq?
Groq is used for the language-model stages because it provides fast inference for the selected models.

This is useful for an interactive meeting assistant where the user expects the pipeline to move through multiple processing stages without unnecessary latency.

The LLM client is abstracted from the pipeline logic so that the inference provider can be changed without redesigning the complete application.

💡 Why Two Language Models?
The two language models intentionally perform different jobs.

GPT-OSS-120B — Refinement
The question answered by this stage is:

What was said, and how can we represent it more accurately without changing its meaning?

Responsible for:

transcript refinement
terminology
names
numbers
negation preservation
commitment preservation
GPT-OSS-20B — Documentation
The question answered by this stage is:

Given the refined conversation, what decisions and actionable information can safely be extracted?

Responsible for:

summary
decisions
action items
owners
deadlines
structured meeting record
🔄 Why Not One LLM?
A single LLM could theoretically perform:

Transcript
    ↓
Summary + Decisions + Tasks
However, this makes it difficult to identify where an error was introduced.

The model would simultaneously be responsible for:

interpreting imperfect speech recognition
correcting terminology
determining what was said
summarizing the meeting
determining decisions
determining action items
The multi-stage architecture instead provides:

STT
 ↓
Raw Transcript
 ↓
Refinement Model
 ↓
Refined Transcript
 ↓
Documentation Model
 ↓
Evidence Validation
 ↓
Final Record
Each stage has a clear responsibility.

🧱 Code Architecture
The core implementation is organized into separate layers:

01_APPLICATION/
│
├── app/
│   │
│   ├── pipeline/
│   │   ├── transcription.py
│   │   ├── refinement.py
│   │   ├── documentation.py
│   │   └── evidence.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── clients.py
│   │   └── utils.py
│   │
│   └── ui/
│       └── gradio_app.py
│
└── run.py
transcription.py
Responsible for:

audio processing
faster-whisper inference
segment handling
timestamp preservation
raw transcript generation
refinement.py
Responsible for:

glossary extraction
token-aware chunking
refinement prompts
Groq structured calls
JSON handling
retry behavior
fallback behavior
refined transcript reconstruction
documentation.py
Responsible for:

documentation prompt
structured meeting-record generation
summary generation
decision extraction
action-item extraction
evidence.py
Responsible for:

validating extracted information
evidence matching
decision classification
action-item validation
config.py
Centralizes:

model names
environment variables
token limits
pipeline configuration
clients.py
Provides the model/client abstraction used by the application.

utils.py
Contains shared utilities such as:

error sanitization
reusable helper functions
common pipeline utilities
gradio_app.py
Responsible for:

file upload
pipeline status
processing controls
raw transcript display
refined transcript display
final record display
error presentation
responsive UI styling
📝 Prompt Architecture
Prompts are maintained separately under:

02_PROMPTS/
02_PROMPTS/
├── glossary_prompt.txt
├── refinement_prompt.txt
└── documentation_prompt.txt
Keeping prompts separate provides:

easier inspection
independent prompt versioning
separation of prompt engineering from application logic
clearer model-role documentation
easier evaluator review
📂 Repository Structure
AI_Meeting_Assistant/
│
├── README.md
│
├── 01_APPLICATION/
│   ├── app/
│   │   ├── main.py
│   │   │
│   │   ├── pipeline/
│   │   │   ├── transcription.py
│   │   │   ├── refinement.py
│   │   │   ├── documentation.py
│   │   │   └── evidence.py
│   │   │
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── clients.py
│   │   │   └── utils.py
│   │   │
│   │   └── ui/
│   │       └── gradio_app.py
│   │
│   └── run.py
│
├── 02_PROMPTS/
│   ├── glossary_prompt.txt
│   ├── refinement_prompt.txt
│   └── documentation_prompt.txt
│
├── 03_DEPENDENCIES/
│   ├── requirements.txt
│   ├── environment_setup.md
│   └── .env.example
│
├── 04_DEMO/
│   ├── recording/
│   ├── outputs/
│   └── DEMO.md
│
├── 05_TECHNICAL_DESCRIPTION/
│   ├── TECHNICAL_DESCRIPTION.md
│   └── ARCHITECTURE.md
│
├── 06_TESTS/
│   ├── test_safety.py
│   ├── test_evidence.py
│   ├── test_pipeline.py
│   ├── test_refinement.py
│   └── test_ui.py
│
├── .gitignore
├── run.py
└── submission_notes.md
🚀 Running the Project
1. Create a virtual environment
python -m venv .venv

macOS / Linux
source .venv/bin/activate

Windows
.venv\Scripts\activate

2. Install dependencies
pip install -r 03_DEPENDENCIES/requirements.txt

3. Configure the API key
Create .env using:

03_DEPENDENCIES/.env.example
Then add:

GROQ_API_KEY=your_groq_api_key
Never commit .env or a real API key.

4. Start the application
python run.py

The Gradio interface will start locally.

5. Run the test suite
pytest 06_TESTS

🔐 Environment & Security
The repository intentionally contains:

.env.example
instead of the real .env.

The real API key should remain local.

The following should never be committed:

.env
or any other file containing a live API credential.

The repository .gitignore is configured to exclude sensitive and generated development artifacts.

🧪 Reliability Philosophy
The project follows a simple engineering principle:

A meeting assistant should fail explicitly rather than silently invent information.

Bad failure
LLM fails
   ↓
Generate plausible meeting record anyway
Desired failure
LLM fails
   ↓
Identify failed stage
   ↓
Show clear error
   ↓
Preserve previous valid outputs
Similarly:

Bad extraction
"We might deploy next Friday."

        ↓

Decision:
Deploy next Friday.
Desired extraction
Discussion:
Possible deployment next Friday.

Decision:
Not confirmed.
The system prioritizes evidence over fluency.

📈 Engineering Journey
The final architecture was shaped by actual testing rather than being designed only as a theoretical pipeline.

Several important failure modes appeared during development:

UI presentation problems
        ↓
Structured JSON failures
        ↓
Long-recording token exhaustion
        ↓
Insufficient stage visibility
        ↓
Landing-page responsiveness
        ↓
Repository path inconsistencies
        ↓
Production hardening
Each failure resulted in a corresponding engineering change.

This process transformed the project from a basic prototype into a more robust end-to-end application.

🛠️ Major Engineering Lessons
1. A pipeline needs observable stages
A generic:

"Processing failed"
is not enough for a multi-stage AI application.

The final implementation reports stage-level progress and errors.

2. Optional components should fail gracefully
Glossary extraction is useful, but it is not the core purpose of the system.

Therefore:

Glossary failure
       ↓
Warning
       ↓
Continue refinement
rather than:

Glossary failure
       ↓
Entire pipeline crashes
3. Token budgets are engineering constraints
A model can technically accept a long transcript while still failing to produce the required structured output.

The final implementation therefore uses:

token-aware chunking
explicit output budgets
reasoning control
retries
structured-output fallback
4. LLM fluency is not evidence
A sentence that sounds convincing is not necessarily supported by the meeting.

Therefore the documentation stage is followed by evidence validation.

5. Intermediate outputs matter
Keeping:

Raw Transcript
Refined Transcript
Final Record
makes the system easier to debug and gives the user transparency into what the AI changed.

📊 Final Verification
The current project verification includes:

Test Suite
──────────
43 / 43 tests passing

End-to-End
──────────
15-minute real meeting successfully processed

Pipeline
────────
received
validated
transcribing
raw_ready
refining
refined
documenting
validating
exporting
complete
The successful long-recording run generated:

Raw transcript      → 14,073 characters
Refined transcript  → 14,104 characters
Confirmed decisions → 2
Action items        → 4
The final pipeline completed successfully and produced the expected output artifacts.

📌 Current Capabilities
The system currently supports:

✅ Recorded English meeting audio
✅ Local speech recognition
✅ Timestamped transcription
✅ Raw transcript preservation
✅ Domain-aware transcript refinement
✅ Technical terminology handling
✅ Name and number preservation
✅ Negation preservation
✅ Commitment preservation
✅ Token-aware transcript chunking
✅ Structured LLM outputs
✅ Retry and fallback handling
✅ Evidence-based decision extraction
✅ Evidence-based action-item extraction
✅ Unspecified owner/deadline handling
✅ Human-readable outputs
✅ Machine-readable outputs
✅ Interactive Gradio interface
✅ Responsive UI
✅ Stage-level pipeline visibility
✅ Regression test suite
⚠️ Important Notes
API Requirement
The refinement and documentation stages currently use Groq inference.

A valid:

GROQ_API_KEY
is therefore required for the complete LLM pipeline.

Model Roles
The models are intentionally separated:

GPT-OSS-120B
    ↓
Transcript Refinement

GPT-OSS-20B
    ↓
Meeting Documentation
They should not be treated as interchangeable stages.

Local STT
Speech recognition is performed locally through faster-whisper.

The application therefore does not require a separate hosted STT API key for transcription.

🗺️ Future Extensions
The current implementation focuses on the requirements of the meeting-assistant problem.

Potential future extensions include:

multilingual meeting support
speaker diarization
persistent meeting history
semantic meeting search
cross-meeting action-item tracking
calendar integration
task-management integration
organization-specific terminology memory
domain-specific model adaptation
streaming transcription
real-time meeting assistance
stronger evidence visualization
human approval workflows for extracted decisions
These are intentionally outside the core implementation so that the current pipeline remains focused, inspectable, and evaluator-friendly.

🏆 Final Status
Component	Status
Audio validation	✅
Local speech recognition	✅
Raw transcript	✅
Domain refinement	✅
Long-recording chunking	✅
Structured documentation	✅
Evidence validation	✅
Decision extraction	✅
Action-item extraction	✅
Owner/deadline handling	✅
Human-readable outputs	✅
Machine-readable outputs	✅
Interactive UI	✅
Responsive UI	✅
Error handling	✅
Regression tests	✅ 43/43
Real 15-minute run	✅
Demo artifacts	✅
🎯 The Goal
This project is not designed to make an LLM "sound smart."

It is designed to make meeting information usable without losing trust in what was actually said.

Listen.
   ↓
Transcribe.
   ↓
Refine.
   ↓
Validate.
   ↓
Structure.
   ↓
Act.
👨‍💻 Project
AI Meeting Assistant
Inter IIT Bootcamp

Listen once. Refine accurately. Record what actually happened.
