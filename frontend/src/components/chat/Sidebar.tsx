import { useRef, useState } from "react";
import { File as FileIcon, Loader2, LogOut, Upload, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { SUPPORTED_EXTENSIONS, uploadDocuments } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";
import CourseSelector from "./CourseSelector";

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function Sidebar() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const { user, signOut } = useAuth();

  function onPick(e: React.ChangeEvent<HTMLInputElement>) {
    const picked = e.target.files ? Array.from(e.target.files) : [];
    if (picked.length === 0) return;
    setFiles((prev) => {
      const seen = new Set(prev.map((f) => f.name + f.size));
      const merged = [...prev];
      for (const p of picked) {
        const key = p.name + p.size;
        if (!seen.has(key)) {
          merged.push(p);
          seen.add(key);
        }
      }
      return merged;
    });
    if (inputRef.current) inputRef.current.value = "";
  }

  function removeFile(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }

  async function handleUpload() {
    if (files.length === 0) return;
    setUploading(true);
    try {
      const result = await uploadDocuments(files);
      toast.success(
        `Uploaded ${result.accepted_files.length} file(s) — ${result.chunks} chunks indexed`
      );
      if (result.skipped_files.length > 0) {
        toast.warning(`Skipped: ${result.skipped_files.join(", ")}`);
      }
      setFiles([]);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  return (
    <aside className="flex h-screen w-72 flex-col border-r border-border bg-background/70 backdrop-blur-md">
      {/* Brand */}
      <div className="flex items-center gap-3 border-b border-border px-4 py-4">
        <img
          src="/assets/sabanci_logo.png"
          alt="Sabancı"
          className="h-11 w-auto rounded-md"
        />
        <div className="leading-tight">
          <div className="text-base font-bold">SU-GPT</div>
          <div className="text-xs text-muted-foreground">
            Course-Aware RAG Assistant
          </div>
        </div>
      </div>

      {/* Uploader */}
      <div className="flex-1 overflow-y-auto scrollbar-thin px-4 py-4">
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Upload Course Documents
        </h3>

        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="flex w-full flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border bg-secondary/40 px-3 py-5 text-center text-sm text-muted-foreground transition hover:bg-secondary/60 hover:text-foreground"
        >
          <Upload className="h-5 w-5" />
          <span>
            <span className="font-medium text-foreground">Click to browse</span>
            <br />
            PDF, PPTX, DOCX, MD, TXT
          </span>
        </button>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={SUPPORTED_EXTENSIONS.join(",")}
          className="hidden"
          onChange={onPick}
        />

        {files.length > 0 && (
          <ul className="mt-3 space-y-2">
            {files.map((f, i) => (
              <li
                key={f.name + i}
                className="flex items-center gap-2 rounded-lg border border-border bg-secondary/60 px-2.5 py-2"
              >
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-background/60">
                  <FileIcon className="h-4 w-4 text-muted-foreground" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium text-foreground">
                    {f.name}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {formatSize(f.size)}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => removeFile(i)}
                  className="rounded-full p-1 text-muted-foreground hover:bg-background hover:text-foreground"
                  aria-label="Remove file"
                >
                  <X className="h-4 w-4" />
                </button>
              </li>
            ))}
          </ul>
        )}

        <Button
          type="button"
          className="mt-4 w-full"
          onClick={handleUpload}
          disabled={files.length === 0 || uploading}
        >
          {uploading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Indexing…
            </>
          ) : (
            <>Upload to SU-GPT</>
          )}
        </Button>

        <CourseSelector />
      </div>

      {/* User / logout */}
      <div className="border-t border-border px-4 py-3">
        <Separator className="mb-2" />
        <div className="mb-2 text-xs text-muted-foreground">
          Signed in as <span className="text-foreground font-medium">{user?.username}</span>
        </div>
        <Button
          variant="outline"
          size="sm"
          className="w-full"
          onClick={signOut}
        >
          <LogOut className="h-4 w-4" />
          Log out
        </Button>
      </div>
    </aside>
  );
}
