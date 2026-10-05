# Architecture

```mermaid
flowchart TD
    A[Meeting audio upload] --> B{Validate type, existence, readability, non-empty}
    B -->|valid| C[Local faster-whisper small, CPU int8]
    B -->|invalid| X[Explicit input error]
    C --> D[Raw timestamped transcript retained unchanged]
    D --> E[Token-aware glossary chunks in source order]
    E --> F[Groq gpt-oss-120b: conservative refinement chunks]
    E --> F
    F --> G[Deterministic safety checks on proposed edits]
    G --> H[Reassembled refined transcript + audit]
    H --> I[Groq gpt-oss-20b: meeting documentation]
    I --> J[Evidence and owner/deadline validation]
    J --> K[Meeting record JSON and Markdown]
    D --> L[Download package]
    H --> L
    J --> L
    K --> L
    L --> M[Gradio: raw, refined, minutes, decisions, actions, final record, ZIP]
```

## Information movement

The raw transcript is stored independently and is passed to the refinement stage alongside extracted glossary terms. The documentation model sees the refined transcript, not the audio. Python validates edit safety and evidence after model responses. The final ZIP includes the unchanged raw transcript, refined text, structured decisions/action items, minutes, complete record, audit, and source audio.
