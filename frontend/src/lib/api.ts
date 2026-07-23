const API_URL: string =
  (import.meta.env.VITE_API_URL as string) || "http://127.0.0.1:8000";

/**
 * Retrieval modes (CLAUDE.md Section 3).
 *
 * NOT surfaced in the product UI: students should not have to choose a retrieval strategy, and
 * the server already runs the configuration our benchmark measured as best. These stay exported
 * because `/ask/` still accepts a `mode`, which the evaluation harness relies on.
 */
export const RETRIEVAL_MODES = [
  { value: "hybrid", label: "Hybrid RAG" },
  { value: "hybrid_rerank", label: "Hybrid + CrossEncoder Reranking" },
  { value: "dense", label: "Dense RAG" },
  { value: "bm25", label: "BM25 RAG" },
  { value: "llm_only", label: "LLM-only baseline" },
] as const;

export type RetrievalMode = (typeof RETRIEVAL_MODES)[number]["value"];

export interface AskResponse {
  response: string;
  sources: string[];
  intent?: string;
  profile_required?: boolean;
  curriculum_unavailable?: boolean;
  mode?: RetrievalMode;
  top_k?: number;
  prompt_strategy?: string;
  expert_mode?: string;
  warning?: string;
  reranked?: boolean;
  num_retrieved_chunks?: number;
  num_final_context_chunks?: number;
}

export interface AcademicProfile {
  major: string | null;
  degree_code: string | null;
  admission_term: string | null;
  curriculum_term: string | null;
  minor_codes: string[];
  profile_status: string;
}

export interface CurriculumRow {
  program: string;
  degree_code: string;
  program_name: string;
  program_type: string;
  curriculum_term: string;
  admit_term_label: string;
  total_min_su_credits: number | null;
  is_minor: boolean;
}

export interface DegreeAuditCategory {
  category: string;
  required_su_credits: number | null;
  completed_su_credits: number;
  remaining_su_credits: number | null;
}

export interface DegreeAuditEcts {
  category: string;
  required_ects: number;
  completed_ects: number;
  remaining_ects: number;
}

export interface DegreeAudit {
  status: string;
  reliability: string;
  program: string;
  curriculum_term: string;
  program_name?: string;
  total_min_su_credits?: number | null;
  completed_su_credits?: number;
  remaining_su_credits?: number | null;
  categories?: DegreeAuditCategory[];
  ects_requirements?: DegreeAuditEcts[];
  missing_required_courses?: string[];
  warnings?: string[];
  message?: string;
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

function getSessionId(): string {
  let sid = localStorage.getItem("advisu-session");
  if (!sid) {
    sid = `s-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    localStorage.setItem("advisu-session", sid);
  }
  return sid;
}

export function resetSession(): void {
  localStorage.removeItem("advisu-session");
}

/** Start a fresh chat thread and return its id. */
export function startNewSession(): string {
  localStorage.removeItem("advisu-session");
  return getSessionId();
}

/** Switch the active thread to an existing conversation. */
export function useSession(sessionId: string): void {
  localStorage.setItem("advisu-session", sessionId);
}

export function currentSessionId(): string {
  return getSessionId();
}

export interface ConversationSummary {
  session_id: string;
  title: string;
  updated_at: string | null;
  message_count: number;
}

export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
  sources: string[];
  at: string | null;
}

export interface Conversation extends ConversationSummary {
  messages: ConversationMessage[];
}

export async function listConversations(username: string): Promise<ConversationSummary[]> {
  const res = await fetch(`${API_URL}/users/${encodeURIComponent(username)}/conversations`);
  const data = await parseJsonOrThrow<{ conversations: ConversationSummary[] }>(res);
  return data.conversations ?? [];
}

export async function getConversation(sessionId: string): Promise<Conversation> {
  const res = await fetch(`${API_URL}/conversations/${encodeURIComponent(sessionId)}`);
  return parseJsonOrThrow<Conversation>(res);
}

export async function deleteConversation(sessionId: string): Promise<void> {
  const res = await fetch(`${API_URL}/conversations/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
  await parseJsonOrThrow(res);
}

export async function askQuestion(
  question: string,
  options: { mode?: RetrievalMode; topK?: number } = {}
): Promise<AskResponse> {
  const form = new FormData();
  form.append("question", question);
  form.append("session_id", getSessionId());
  if (options.mode) form.append("mode", options.mode);
  if (options.topK) form.append("top_k", String(options.topK));
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

export async function getCurricula(): Promise<{
  programs: string[];
  majors: CurriculumRow[];
  minors: CurriculumRow[];
}> {
  const res = await fetch(`${API_URL}/curricula/`);
  return parseJsonOrThrow(res);
}

export async function getProfile(username: string): Promise<AcademicProfile> {
  const res = await fetch(`${API_URL}/users/${encodeURIComponent(username)}/profile`);
  const data = await parseJsonOrThrow<{ profile: AcademicProfile }>(res);
  return data.profile;
}

export async function saveProfile(
  username: string,
  profile: Partial<AcademicProfile>,
): Promise<AcademicProfile> {
  const res = await fetch(`${API_URL}/users/${encodeURIComponent(username)}/profile`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(profile),
  });
  const data = await parseJsonOrThrow<{ profile: AcademicProfile }>(res);
  return data.profile;
}

export async function getDegreeAudit(username: string): Promise<DegreeAudit> {
  const res = await fetch(`${API_URL}/users/${encodeURIComponent(username)}/degree-audit`);
  return parseJsonOrThrow<DegreeAudit>(res);
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
