export default function ChatHeader() {
  return (
    <header className="flex items-center gap-3 border-b border-border bg-background/40 px-6 py-3 backdrop-blur-sm">
      <img
        src="/assets/sugptlogo.png"
        alt="SU-GPT"
        className="h-9 w-9 rounded-md drop-shadow-[0_3px_10px_rgba(0,75,147,0.45)]"
      />
      <div className="leading-tight">
        <h1 className="text-base font-bold tracking-tight">
          SU-GPT — Course-Aware RAG Assistant
        </h1>
        <p className="text-xs text-muted-foreground">
          Sabancı University academic document assistant
        </p>
      </div>
    </header>
  );
}
