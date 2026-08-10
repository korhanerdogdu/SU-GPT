import {
  normaliseScheduleDocument,
  scheduleToBackendPayload,
  type ScheduleCatalogSection,
  type ScheduleDocument,
} from "./schedule";
import { getStoredLocale, translate } from "@/localization/resources";

export const API_URL: string =
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

/** Daily question quota. `exempt` is true for the admin account and for unauthenticated callers. */
export interface UsageStatus {
  limit: number;
  used: number;
  remaining: number;
  resets_at: string;
  exempt: boolean;
  allowed: boolean;
}

export interface AskResponse {
  response: string;
  /** Present whenever the request was metered, so the header can update without a refetch. */
  usage?: UsageStatus;
  /** Set when the content-safety gate answered instead of the advisor. */
  safety_category?: string;
  rate_limited?: boolean;
  summary?: string;
  sources: string[];
  structured_content?: StructuredContent | null;
  schedule?: Record<string, unknown> | null;
  export_links?: Record<string, string>;
  profile_updated?: boolean;
  course_history?: {
    courses: Course[];
    course_count: number;
    credit_eligible_course_count: number;
    total_su_credits: number;
  };
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

export interface StructuredTable {
  id: string;
  title: string;
  columns: Array<{ key: string; label: string }>;
  rows: Array<Record<string, string | number | null>>;
  exportable?: boolean;
}

export interface StructuredContent {
  kind: string;
  status?: string;
  headline?: Record<string, string | number | null>;
  tables?: StructuredTable[];
  missing_required_courses?: string[];
  warnings?: string[];
  sections?: StructuredContent[];
  // Weekly-schedule (kind === "course_schedule"): the chosen CRNs, for the "copy CRNs" control.
  crns?: string[];
  term?: string;
}

export interface AcademicProfile {
  major: string | null;
  degree_code: string | null;
  admission_term: string | null;
  curriculum_term: string | null;
  academic_year: number | null;
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
  access_token: string;
  token_type: "bearer";
  expires_at: number;
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
  faculty?: string | null;
  status?: "completed" | "enrolled" | "failed" | "withdrawn" | "transfer" | "exempted";
  grade?: string | null;
  description?: string;
  prerequisites?: string;
  corequisites?: string;
  source_url?: string;
}

/** The transcript's own grading vocabulary (letter grades + non-GPA administrative codes),
 *  kept in sync with server/modules/transcript_parser.py's ALL_GRADES. */
export const ALL_GRADES = [
  "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "F",
  "S", "T", "W", "I", "IP", "P", "NP", "NA", "SL", "UL", "EL", "U",
] as const;
export const GPA_ELIGIBLE_GRADES = new Set([
  "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "F",
]);

export interface TranscriptSummary {
  id: string;
  filename: string;
  uploaded_at: string;
  course_count: number;
  gpa: number | null;
  total_su_credits: number | null;
  total_ects: number | null;
  warnings: string[];
}

export interface TranscriptUploadResult {
  summary: TranscriptSummary;
  matched_course_codes: string[];
  unmatched_course_codes: string[];
  courses: Course[];
  profile_major_update: { from: string | null; to: string } | null;
}

export interface GpaResult {
  gpa: number | null;
  gpa_eligible_course_count: number;
  gpa_eligible_su_credits: number;
  courses: { code: string; title: string; grade: string; su_credits?: number | null }[];
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
    await res.text().catch(() => "");
    throw new Error(translate(getStoredLocale(), "errors.http", { status: res.status }));
  }
  return res.json() as Promise<T>;
}

function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  try {
    const raw = localStorage.getItem("su-gpt-auth");
    const auth = raw ? (JSON.parse(raw) as { accessToken?: string }) : null;
    return auth?.accessToken
      ? { ...extra, Authorization: `Bearer ${auth.accessToken}` }
      : extra;
  } catch {
    return extra;
  }
}

/** The active thread id is scoped per logged-in account. A single shared "advisu-session" key
 *  meant that using the login page's own student/admin demo-account shortcuts in one browser
 *  reused the other account's thread id -- the backend correctly rejects that as a conversation
 *  ownership mismatch (403), which read as "queries don't work" until the user manually started
 *  a new chat. Namespacing by username means switching accounts never collides, and each account
 *  keeps its own last-active thread. */
function sessionStorageKey(): string {
  try {
    const raw = localStorage.getItem("su-gpt-auth");
    const username = raw ? (JSON.parse(raw) as { username?: string }).username : null;
    return username ? `advisu-session:${username}` : "advisu-session";
  } catch {
    return "advisu-session";
  }
}

function getSessionId(): string {
  const key = sessionStorageKey();
  let sid = localStorage.getItem(key);
  if (!sid) {
    sid = `s-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    localStorage.setItem(key, sid);
  }
  return sid;
}

export function resetSession(): void {
  localStorage.removeItem(sessionStorageKey());
}

/** Start a fresh chat thread and return its id. */
export function startNewSession(): string {
  localStorage.removeItem(sessionStorageKey());
  return getSessionId();
}

/** Switch the active thread to an existing conversation. */
export function useSession(sessionId: string): void {
  localStorage.setItem(sessionStorageKey(), sessionId);
}

export function currentSessionId(): string {
  return getSessionId();
}

export interface ConversationSummary {
  session_id: string;
  title: string;
  updated_at: string | null;
  message_count: number;
  pinned: boolean;
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
  const res = await fetch(`${API_URL}/users/${encodeURIComponent(username)}/conversations`, {
    headers: authHeaders(),
  });
  const data = await parseJsonOrThrow<{ conversations: ConversationSummary[] }>(res);
  return data.conversations ?? [];
}

export async function getConversation(sessionId: string): Promise<Conversation> {
  const res = await fetch(`${API_URL}/conversations/${encodeURIComponent(sessionId)}`, {
    headers: authHeaders(),
  });
  return parseJsonOrThrow<Conversation>(res);
}

export async function deleteConversation(sessionId: string): Promise<void> {
  const res = await fetch(`${API_URL}/conversations/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  await parseJsonOrThrow(res);
}

export async function updateConversation(
  sessionId: string,
  changes: { title?: string; pinned?: boolean },
): Promise<ConversationSummary> {
  const res = await fetch(`${API_URL}/conversations/${encodeURIComponent(sessionId)}`, {
    method: "PATCH",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(changes),
  });
  return parseJsonOrThrow<ConversationSummary>(res);
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
  const res = await fetch(`${API_URL}/ask/`, { method: "POST", headers: authHeaders(), body: form });
  return parseJsonOrThrow<AskResponse>(res);
}

export async function askQuestionStream(
  question: string,
  onToken: (token: string) => void,
  options: { mode?: RetrievalMode; topK?: number; promptStrategy?: string } = {},
): Promise<AskResponse> {
  const form = new FormData();
  form.append("question", question);
  form.append("session_id", getSessionId());
  if (options.mode) form.append("mode", options.mode);
  if (options.topK) form.append("top_k", String(options.topK));
  if (options.promptStrategy) form.append("prompt_strategy", options.promptStrategy);
  const rawAuth = localStorage.getItem("su-gpt-auth");
  if (rawAuth) {
    try {
      const user = JSON.parse(rawAuth) as { username?: string };
      if (user.username) form.append("username", user.username);
    } catch {
      // ignore corrupt local auth state
    }
  }

  const res = await fetch(`${API_URL}/ask/stream`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  if (!res.ok || !res.body) {
    await res.text().catch(() => "");
    throw new Error(translate(getStoredLocale(), "errors.http", { status: res.status }));
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let metadata: Omit<AskResponse, "response"> = { sources: [] };
  let response = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line) as {
        type: "metadata" | "token" | "done";
        data?: Omit<AskResponse, "response">;
        text?: string;
      };
      if (event.type === "metadata" && event.data) metadata = event.data;
      if (event.type === "token" && event.text) {
        response += event.text;
        onToken(event.text);
      }
    }
    if (done) break;
  }
  return { ...metadata, response };
}

export function absoluteApiUrl(path: string): string {
  return path.startsWith("http") ? path : `${API_URL}${path}`;
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
  const res = await fetch(`${API_URL}/users/${encodeURIComponent(username)}/profile`, {
    headers: authHeaders(),
  });
  const data = await parseJsonOrThrow<{ profile: AcademicProfile }>(res);
  return data.profile;
}

export async function saveProfile(
  username: string,
  profile: Partial<AcademicProfile>,
): Promise<AcademicProfile> {
  const res = await fetch(`${API_URL}/users/${encodeURIComponent(username)}/profile`, {
    method: "PUT",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(profile),
  });
  const data = await parseJsonOrThrow<{ profile: AcademicProfile }>(res);
  return data.profile;
}

export async function getUsage(username: string): Promise<UsageStatus> {
  const res = await fetch(`${API_URL}/users/${encodeURIComponent(username)}/usage`, {
    headers: authHeaders(),
  });
  return parseJsonOrThrow<UsageStatus>(res);
}

export async function getDegreeAudit(username: string): Promise<DegreeAudit> {
  const res = await fetch(`${API_URL}/users/${encodeURIComponent(username)}/degree-audit`, {
    headers: authHeaders(),
  });
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
  const res = await fetch(`${API_URL}/users/${encodeURIComponent(username)}/courses`, {
    headers: authHeaders(),
  });
  const data = await parseJsonOrThrow<{ courses: Course[] }>(res);
  return data.courses;
}

export async function saveUserCourses(
  username: string,
  courseIds: string[],
  statuses: Record<string, Course["status"]> = {},
  grades: Record<string, string> = {},
): Promise<Course[]> {
  const res = await fetch(`${API_URL}/users/${encodeURIComponent(username)}/courses`, {
    method: "PUT",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ course_ids: courseIds, statuses, grades }),
  });
  const data = await parseJsonOrThrow<{ courses: Course[] }>(res);
  return data.courses;
}

/** Uploads an official transcript PDF; the backend parses it and immediately applies every
 *  recognised course (status + grade) to the student's course history — no separate save step. */
export async function uploadTranscript(username: string, file: File): Promise<TranscriptUploadResult> {
  const form = new FormData();
  form.append("file", file, file.name);
  const res = await fetch(`${API_URL}/users/${encodeURIComponent(username)}/transcript`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || translate(getStoredLocale(), "errors.http", { status: res.status }));
  }
  return res.json() as Promise<TranscriptUploadResult>;
}

export async function fetchGpa(username: string): Promise<GpaResult> {
  const res = await fetch(`${API_URL}/users/${encodeURIComponent(username)}/gpa`, {
    headers: authHeaders(),
  });
  return parseJsonOrThrow<GpaResult>(res);
}

export async function uploadDocuments(files: File[]): Promise<UploadResponse> {
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);
  const res = await fetch(`${API_URL}/upload_documents/`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  return parseJsonOrThrow<UploadResponse>(res);
}

export async function healthCheck(): Promise<{ message: string }> {
  const res = await fetch(`${API_URL}/test`);
  return parseJsonOrThrow(res);
}

export const SUPPORTED_EXTENSIONS = [".pdf", ".pptx", ".docx", ".md", ".txt"];

export interface ScheduleSectionsResponse {
  term: string;
  term_label: string;
  total: number;
  limit: number;
  has_more: boolean;
  sections: ScheduleCatalogSection[];
}

export interface ScheduleCourseSectionsResponse {
  term: string;
  term_label: string;
  course_id: string;
  title: string;
  sections: ScheduleCatalogSection[];
}

export interface UserScheduleResponse {
  schedule: ScheduleDocument | null;
  revision: number;
  updatedAt: string | null;
}

export async function fetchScheduleSections(
  search = "",
  term = "202601",
  limit = 80,
): Promise<ScheduleSectionsResponse> {
  const params = new URLSearchParams({ term, limit: String(limit) });
  if (search.trim()) params.set("search", search.trim());
  const res = await fetch(`${API_URL}/schedule/sections?${params.toString()}`);
  return parseJsonOrThrow<ScheduleSectionsResponse>(res);
}

export async function fetchScheduleCourseSections(
  courseCode: string,
  term = "202601",
): Promise<ScheduleCourseSectionsResponse> {
  const params = new URLSearchParams({ term, language: "tr" });
  const res = await fetch(
    `${API_URL}/schedule/courses/${encodeURIComponent(courseCode)}/sections?${params.toString()}`,
  );
  return parseJsonOrThrow<ScheduleCourseSectionsResponse>(res);
}

export async function getUserSchedule(username: string): Promise<UserScheduleResponse> {
  const res = await fetch(`${API_URL}/users/${encodeURIComponent(username)}/schedule`, {
    headers: authHeaders(),
  });
  const data = await parseJsonOrThrow<{
    schedule?: unknown;
    revision?: number;
    updated_at?: string | null;
  }>(res);
  return {
    schedule: normaliseScheduleDocument(data.schedule),
    revision: data.revision ?? 0,
    updatedAt: data.updated_at ?? null,
  };
}

export async function saveUserSchedule(
  username: string,
  schedule: ScheduleDocument,
  expectedRevision?: number,
): Promise<UserScheduleResponse> {
  const body: Record<string, unknown> = { schedule: scheduleToBackendPayload(schedule) };
  if (expectedRevision != null) body.expected_revision = expectedRevision;
  const res = await fetch(`${API_URL}/users/${encodeURIComponent(username)}/schedule`, {
    method: "PUT",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  const data = await parseJsonOrThrow<{
    schedule?: unknown;
    revision?: number;
    updated_at?: string | null;
  }>(res);
  return {
    schedule: normaliseScheduleDocument(data.schedule) ?? schedule,
    revision: data.revision ?? (expectedRevision ?? 0) + 1,
    updatedAt: data.updated_at ?? schedule.updatedAt,
  };
}

export const COURSE_REVIEW_DIMENSIONS = [
  "difficulty",
  "workload",
  "learning_value",
  "organization",
  "overall_satisfaction",
] as const;

export type CourseReviewDimension = (typeof COURSE_REVIEW_DIMENSIONS)[number];

export interface CourseReviewPolicy {
  enabled: boolean;
  course_only: true;
  instructor_ratings_allowed: false;
  private_chat_ingestion_allowed: false;
  explicit_consent_required: true;
  minimum_aggregate_reviews: number;
  rating_dimensions: CourseReviewDimension[];
  message: string;
}

export interface CourseReviewSubmission {
  course_code: string;
  difficulty: number;
  workload: number;
  learning_value: number;
  organization: number;
  overall_satisfaction: number;
  comment?: string;
  consent: true;
  consent_version: "course-review-v1";
}

export interface CourseReviewPublic {
  reviewId: string;
  courseCode: string;
  ratings: Record<CourseReviewDimension, number>;
  moderationState: "pending" | "approved" | "rejected";
}

export interface CourseReviewAggregate {
  available: boolean;
  message?: string;
  courseCode?: string;
  reviewCount?: number;
  averages?: Record<CourseReviewDimension, number>;
  distributions?: Record<CourseReviewDimension, Record<string, number>>;
}

export async function fetchCourseReviewPolicy(locale = getStoredLocale()): Promise<CourseReviewPolicy> {
  const res = await fetch(`${API_URL}/course-reviews/policy?language=${locale}`);
  return parseJsonOrThrow<CourseReviewPolicy>(res);
}

export async function submitCourseReview(
  payload: CourseReviewSubmission,
  locale = getStoredLocale(),
): Promise<{ review: CourseReviewPublic; message: string }> {
  const res = await fetch(`${API_URL}/course-reviews?language=${locale}`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
  });
  return parseJsonOrThrow(res);
}

export async function fetchCourseReviewAggregate(
  courseCode: string,
  locale = getStoredLocale(),
): Promise<CourseReviewAggregate> {
  const res = await fetch(
    `${API_URL}/course-reviews/${encodeURIComponent(courseCode)}/aggregate?language=${locale}`,
    { headers: authHeaders() },
  );
  return parseJsonOrThrow(res);
}

export async function deleteMyCourseReview(
  courseCode: string,
  locale = getStoredLocale(),
): Promise<{ deleted: true; message: string }> {
  const res = await fetch(
    `${API_URL}/course-reviews/${encodeURIComponent(courseCode)}?language=${locale}`,
    { method: "DELETE", headers: authHeaders() },
  );
  return parseJsonOrThrow(res);
}
