const API_URL: string =
  (import.meta.env.VITE_API_URL as string) || "http://127.0.0.1:8000";

export interface AskResponse {
  response: string;
  sources: string[];
}

export interface AuthResponse {
  username: string;
  role: string;
}

export interface Course {
  id: string;
  code: string;
  subject: string;
  number: string;
  title: string;
  su_credits?: number | null;
  ects?: number | null;
  engineering_ects?: number | null;
  basic_science_ects?: number | null;
  description?: string;
  prerequisites?: string;
  corequisites?: string;
  source_url?: string;
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
  const rawAuth = localStorage.getItem("su-gpt-auth");
  if (rawAuth) {
    try {
      const user = JSON.parse(rawAuth) as { username?: string };
      if (user.username) form.append("username", user.username);
    } catch {
      // ignore corrupt local auth state
    }
  }
  const res = await fetch(`${API_URL}/ask/`, { method: "POST", body: form });
  return parseJsonOrThrow<AskResponse>(res);
}

export async function login(username: string, password: string): Promise<AuthResponse> {
  const res = await fetch(`${API_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  return parseJsonOrThrow<AuthResponse>(res);
}

export async function fetchCourses(search = ""): Promise<Course[]> {
  const params = new URLSearchParams();
  if (search) params.set("search", search);
  const res = await fetch(`${API_URL}/courses/?${params.toString()}`);
  const data = await parseJsonOrThrow<{ courses: Course[] }>(res);
  return data.courses;
}

export async function fetchUserCourses(username: string): Promise<Course[]> {
  const res = await fetch(`${API_URL}/users/${encodeURIComponent(username)}/courses`);
  const data = await parseJsonOrThrow<{ courses: Course[] }>(res);
  return data.courses;
}

export async function saveUserCourses(username: string, courseIds: string[]): Promise<Course[]> {
  const res = await fetch(`${API_URL}/users/${encodeURIComponent(username)}/courses`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ course_ids: courseIds }),
  });
  const data = await parseJsonOrThrow<{ courses: Course[] }>(res);
  return data.courses;
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
