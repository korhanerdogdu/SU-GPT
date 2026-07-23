import { createContext, useContext, useState, type ReactNode } from "react";
import { login } from "@/lib/api";

interface AuthUser {
  username: string;
  email?: string;
}

interface AuthContextValue {
  user: AuthUser | null;
  isAuthenticated: boolean;
  signIn: (username: string, password: string) => Promise<void>;
  signUp: (username: string, email: string) => void;
  signOut: () => void;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

const STORAGE_KEY = "su-gpt-auth";

export function AuthProvider({ children }: { children: ReactNode }) {
  // Read storage during the FIRST render, not in an effect. Loading it in useEffect made the
  // initial render unauthenticated, so RequireAuth bounced any deep link (or refresh) on a
  // guarded route to /login, and /login then redirected to "/" — silently dropping /profile.
  const [user, setUser] = useState<AuthUser | null>(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? (JSON.parse(raw) as AuthUser) : null;
    } catch {
      return null; // ignore corrupt storage
    }
  });

  function persist(next: AuthUser | null) {
    setUser(next);
    if (next) localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    else localStorage.removeItem(STORAGE_KEY);
  }

  const value: AuthContextValue = {
    user,
    isAuthenticated: !!user,
    signIn: async (username, password) => {
      const result = await login(username, password);
      persist({ username: result.username });
    },
    signUp: (username, email) => persist({ username, email }),
    signOut: () => persist(null),
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
