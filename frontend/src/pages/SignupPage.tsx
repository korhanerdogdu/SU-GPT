import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Eye, EyeOff, Lock, Mail, User } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { useAuth } from "@/contexts/AuthContext";
import { useLocale } from "@/contexts/LocaleContext";
import LanguageToggle from "@/components/LanguageToggle";

export default function SignupPage() {
  const navigate = useNavigate();
  const { signUp } = useAuth();
  const { t } = useLocale();
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [agree, setAgree] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!fullName.trim() || !email.trim() || !password) {
      toast.error(t("signup.missing"));
      return;
    }
    if (!agree) {
      toast.error(t("signup.mustAgree"));
      return;
    }
    try {
      signUp(fullName.trim(), email.trim());
      navigate("/", { replace: true });
    } catch {
      toast.error(t("signup.disabled"));
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
      <div className="mx-auto flex h-full w-full max-w-6xl flex-col items-center justify-center gap-8 px-4 lg:flex-row lg:justify-between lg:gap-16">
        {/* Brand panel — matches the login screen */}
        <div className="flex w-full max-w-xl shrink flex-col items-center text-center lg:items-start lg:text-left">
          {/* Light panel: the lockup's navy type is illegible on the dark backdrop. */}
          <div className="rounded-2xl bg-white/95 px-6 py-5 shadow-2xl backdrop-blur-sm">
            <img
              src="/assets/adviSU-logo.png"
              alt={t("login.logoAlt")}
              className="w-[min(100%,30rem)]"
            />
          </div>
          <p className="mt-4 hidden max-w-md text-sm leading-relaxed text-white/80 lg:block">
            {t("signup.description")}
          </p>
        </div>

        <div className="w-full max-w-sm shrink-0 rounded-2xl border border-white/10 bg-card/70 p-6 shadow-2xl backdrop-blur-md">
          <LanguageToggle className="mb-4 ml-auto flex" />
          <div className="mb-5">
            <h1 className="text-2xl font-bold tracking-tight">
              {t("signup.title")}
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">
              {t("signup.subtitle")}
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-3.5">
            <div className="space-y-1.5">
              <Label htmlFor="fullname">{t("signup.fullName")}</Label>
              <div className="relative">
                <User className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  id="fullname"
                  autoComplete="name"
                  placeholder="John Doe"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="email">{t("signup.email")}</Label>
              <div className="relative">
                <Mail className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  id="email"
                  type="email"
                  autoComplete="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="password">{t("signup.password")}</Label>
              <div className="relative">
                <Lock className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  autoComplete="new-password"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="pl-9 pr-9"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((s) => !s)}
                  aria-label={showPassword ? t("login.hidePassword") : t("login.showPassword")}
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

            <label className="flex items-center gap-2 text-sm text-foreground/90 cursor-pointer select-none pt-1">
              <Checkbox
                checked={agree}
                onCheckedChange={(v) => setAgree(v === true)}
              />
              <span>{t("signup.agree")}</span>
            </label>

            <Button type="submit" size="xl" className="w-full mt-1">
              {t("signup.submit")}
            </Button>
          </form>

          <p className="mt-5 text-center text-sm text-muted-foreground">
            {t("signup.haveAccount")}{" "}
            <Link
              to="/login"
              className="font-semibold text-foreground hover:underline"
            >
              {t("signup.login")}
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
