 Atto Sense Multimodal Bot Trainer MK1

A production-ready, multimodal Natural Language Understanding (NLU) microservice. It leverages the Python SDK for ultra-fast, zero-shot intent classification across text, audio, and visual inputs.


To prevent local UI resource crashes (the "Silent Kill"), the application is physically decoupled into two distinct layers:


A headless API server that handles strict data routing (via Pydantic) and communicates with LPUs.


A modern, dashboard featuring a chat UI, voice capture, image upload, and a real-time analytics sidebar.


Zero-Shot Intent Routing: Classifies user intent without needing thousands of training examples.


Multimodal Pipelines: Native support for text, microphone audio arrays, and camera/image uploads.


Discovery Inbox: An active learning feature that quarantines low-confidence queries (<70%) for developer review.


Headless Analytics: Generates `matplotlib` confusion matrices and FPDF audit reports without invoking OS-level GUI threads.



How to Run

Because this is a microservices architecture, you must run the backend and frontend simultaneously.

Terminal 1 (Start the Backend Engine)

```bash

\# Activate your environment

venv\\Scripts\\activate

uvicorn backend.api:app --reload --port 8000



