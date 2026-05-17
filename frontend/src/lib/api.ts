const API_URL: string =
  (import.meta.env.VITE_API_URL as string) || "http://127.0.0.1:8000";

export interface AskResponse {
  response: string;
  sources: string[];
}

export interface UploadResponse {
  message: string;
  chunks: number;
  accepted_files: string[];
  skipped_files: string[];
  supported_extensions: string[];
}

async function parseJsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}${text ? ` — ${text}` : ""}`);
  }
  return res.json() as Promise<T>;
}

export async function askQuestion(question: string): Promise<AskResponse> {
  const form = new FormData();
  form.append("question", question);
  const res = await fetch(`${API_URL}/ask/`, { method: "POST", body: form });
  return parseJsonOrThrow<AskResponse>(res);
}

export async function uploadDocuments(files: File[]): Promise<UploadResponse> {
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);
  const res = await fetch(`${API_URL}/upload_documents/`, {
    method: "POST",
    body: form,
  });
  return parseJsonOrThrow<UploadResponse>(res);
}

export async function healthCheck(): Promise<{ message: string }> {
  const res = await fetch(`${API_URL}/test`);
  return parseJsonOrThrow(res);
}

export const SUPPORTED_EXTENSIONS = [".pdf", ".pptx", ".docx", ".md", ".txt"];
