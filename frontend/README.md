# SU-GPT Frontend (React + Vite + shadcn/ui)

A React frontend for SU-GPT that talks to the existing FastAPI backend.
Mirrors what the Streamlit `client/` does, but with proper login, signup, and
chat pages built on shadcn/ui + Tailwind.

## Stack

- Vite + React 18 + TypeScript
- Tailwind CSS (dark theme tuned to Sabancı navy)
- shadcn/ui primitives (Button, Input, Label, Card, Checkbox, Separator)
- React Router v6 (`/login`, `/signup`, `/`)
- `sonner` for toasts, `lucide-react` for icons
- `AuthContext` backed by `localStorage` (visual-only auth — any non-empty
  credentials sign you in)

## Setup

```bash
cd frontend
npm install
cp .env.example .env       # only if you want to point at a non-default backend
npm run dev
```

Open <http://localhost:5173>.

## Backend

Run the FastAPI server as before, from `server/`:

```bash
python -m uvicorn main:app --reload
```

The frontend hits `http://127.0.0.1:8000` by default. Override via
`VITE_API_URL` in `.env`.

CORS is already wide open on the backend (`allow_origins=["*"]`) so no
changes are needed there.

## Endpoints used

- `POST /upload_documents/` — multi-format upload (PDF/PPTX/DOCX/MD/TXT)
- `POST /ask/` — chat question, returns `{ response, sources }`
- `GET /test` — health check

## Routes

| Path      | Component    | Notes                                          |
| --------- | ------------ | ---------------------------------------------- |
| `/login`  | `LoginPage`  | Campus background, username + password         |
| `/signup` | `SignupPage` | Campus background, name + email + password + ToS |
| `/`       | `ChatPage`   | Sidebar + chat shell, requires auth            |

Logging out clears the `su-gpt-auth` localStorage key and bounces you back to
`/login`.

## Files

```
frontend/
├── public/assets/        # campus.jpg, sabanci_logo.png, sugptlogo.png
├── src/
│   ├── components/
│   │   ├── ui/           # shadcn primitives
│   │   └── chat/         # Sidebar, ChatHeader, ChatMessages, ChatInput
│   ├── contexts/AuthContext.tsx
│   ├── lib/{api,utils}.ts
│   ├── pages/{LoginPage,SignupPage,ChatPage}.tsx
│   ├── App.tsx           # Router + guards
│   ├── main.tsx
│   └── index.css         # Tailwind base + theme tokens
├── index.html
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── vite.config.ts
```
