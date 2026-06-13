# BIOS Check — High-Level Architecture Diagram

Paste the Mermaid code below into https://mermaid.live to generate the diagram image.

```mermaid
flowchart TD
    User(["👤 User / Browser"])

    subgraph Vercel ["☁️ Vercel — Frontend"]
        direction TB
        React["React + Vite SPA"]
        subgraph Pages ["Pages"]
            P1["🏠 Home"]
            P2["🔍 JD Bias Reducer"]
            P3["🤖 Hiring AI"]
            P4["📊 Fair Hiring Index"]
            P5["📬 Contact"]
        end
        React --> Pages
    end

    subgraph Railway ["🚂 Railway — Backend (Flask)"]
        direction TB
        API["Flask REST API"]
        subgraph Routes ["API Routes"]
            R1["/api/bias-reducer/analyze"]
            R2["/api/hiring-ai/compare"]
            R3["/api/contact/submit"]
        end
        Detector["detector.py\nRule-based + Semantic\nBias Detection"]
        subgraph Agents ["agents.py — LLM Agents"]
            A1["SuggestionAgent"]
            A2["RewriteAgent"]
            A3["PIIStripper"]
            A4["FitEvaluator"]
        end
        API --> Routes
        R1 --> Detector
        R1 --> A1
        R1 --> A2
        R2 --> A3
        R2 --> A4
    end

    subgraph AILayer ["🧠 AI Layer"]
        GPT2["Fine-tuned GPT-2\n(Gender Bias Detection)"]
        OpenAI["OpenAI GPT-4.1-nano\n(Text Generation & Evaluation)"]
        Gmail["📧 Gmail SMTP\n(Contact Form)"]
    end

    User -->|"visits"| React
    Pages -->|"VITE_API_BASE_URL"| API
    Detector -->|"model inference"| GPT2
    A1 & A2 & A3 & A4 -->|"API calls"| OpenAI
    R3 -->|"sends email"| Gmail
    Gmail -->|"delivers to"| User
```

---

## Plain-English Summary (for slides)

| Layer | Technology | Hosted On |
|-------|-----------|-----------|
| Frontend | React + Vite (SPA) | Vercel |
| Backend API | Python / Flask | Railway |
| Bias Detection | Rule-based detector + Fine-tuned GPT-2 | Railway |
| AI Agents | OpenAI GPT-4.1-nano | OpenAI API |
| Contact | Gmail SMTP | — |

**Data flow:**
1. User pastes a job description in the browser
2. Frontend sends it to the Flask API on Railway
3. `detector.py` scores it using rules + fine-tuned model
4. LLM agents generate suggestions and an inclusive rewrite
5. Results stream back to the browser
