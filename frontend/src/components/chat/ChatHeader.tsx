import { AlertTriangle } from "lucide-react";
import { Link } from "react-router-dom";

interface ChatHeaderProps {
  /** True once the student has saved a major + curriculum term. */
  profileReady: boolean;
}

/**
 * Carries no logo (the brand sits in the sidebar) and no retrieval picker — choosing a retrieval
 * strategy is not a student's job, and the server already runs the configuration our benchmark
 * measured as best. The one thing worth surfacing here is whether answers can be personalised.
 */
export default function ChatHeader({ profileReady }: ChatHeaderProps) {
  return (
    <header className="border-b border-white/10 bg-[#0a1830]/70 backdrop-blur-xl">
      <div className="flex items-center gap-6 px-6 py-4">
        <div className="min-w-0 flex-1">
          <h1 className="truncate text-sm font-semibold tracking-tight text-white">
            Academic advising
          </h1>
          <p className="mt-0.5 truncate text-xs text-sabanci-light/45">
            Answers grounded in official Sabancı curriculum data
          </p>
        </div>
      </div>

      {!profileReady && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-t border-sabanci-gold/25 bg-sabanci-gold/10 px-6 py-2.5">
          <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-sabanci-gold" />
          <p className="text-xs leading-relaxed text-sabanci-gold/90">
            Set your major and curriculum term so answers can use your own requirements and course
            history.
          </p>
          <Link
            to="/profile"
            className="rounded-md border border-sabanci-gold/40 px-2 py-0.5 text-xs font-medium text-sabanci-gold transition-colors hover:bg-sabanci-gold/15 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sabanci-gold"
          >
            Open profile &amp; degree audit
          </Link>
        </div>
      )}
    </header>
  );
}
