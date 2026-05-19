import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { Eye, EyeOff, Lock, User } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/contexts/AuthContext";

export default function LoginPage() {
  const navigate = useNavigate();
  const { signIn } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!username.trim() || !password.trim()) {
      toast.error("Please enter both a username and a password.");
      return;
    }
    setSubmitting(true);
    try {
      await signIn(username.trim(), password);
      navigate("/", { replace: true });
    } catch {
      toast.error("Invalid credentials. Use admin / admin for this demo.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div
      className="h-screen w-screen overflow-hidden bg-cover bg-center bg-fixed"
      style={{
        backgroundImage:
          "linear-gradient(rgba(8,17,30,0.62), rgba(15,31,61,0.7)), url(/assets/campus.jpg)",
      }}
    >
      <div className="flex h-full w-full items-center justify-center px-4">
        <div className="w-full max-w-sm rounded-2xl border border-white/10 bg-card/70 p-6 shadow-2xl backdrop-blur-md">
          <div className="mb-6 flex flex-col items-center text-center">
            <img
              src="/assets/sugptlogo.png"
              alt="SU-GPT"
              className="mb-3 h-12 w-12 rounded-lg object-cover drop-shadow-[0_4px_14px_rgba(0,75,147,0.45)]"
            />
            <h1 className="text-2xl font-bold tracking-tight">Welcome back</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Sign in to continue to SU-GPT
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="username">Username</Label>
              <div className="relative">
                <User className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  id="username"
                  autoComplete="username"
                  placeholder="admin"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Lock className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  autoComplete="current-password"
                  placeholder="admin"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="pl-9 pr-9"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((s) => !s)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>

            <Button type="submit" size="xl" className="w-full" disabled={submitting}>
              {submitting ? "Signing in..." : "Sign in"}
            </Button>
          </form>

          <p className="mt-5 text-center text-sm text-muted-foreground">
            Demo credentials: <span className="font-semibold text-foreground">admin / admin</span>
          </p>

          <p className="mt-6 text-center text-xs text-muted-foreground/70">
            Sabancı University · CS 455 Project · Demo build
          </p>
        </div>
      </div>
    </div>
  );
}
