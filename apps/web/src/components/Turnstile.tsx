import { useEffect, useRef } from "react";

declare global {
  interface Window {
    turnstile?: { render(element: HTMLElement, options: { sitekey: string; callback: (token: string) => void; theme: string }): string; remove(id: string): void };
  }
}

export function Turnstile({ onToken, theme }: { onToken: (token: string) => void; theme: "light" | "dark" }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const sitekey = import.meta.env.VITE_TURNSTILE_SITE_KEY;
    if (!sitekey) { onToken("dev-token"); return; }
    const scriptId = "turnstile-script";
    const render = () => {
      if (!ref.current || !window.turnstile) return;
      const widget = window.turnstile.render(ref.current, { sitekey, callback: onToken, theme });
      return () => window.turnstile?.remove(widget);
    };
    if (window.turnstile) return render();
    let script = document.getElementById(scriptId) as HTMLScriptElement | null;
    if (!script) {
      script = document.createElement("script");
      script.id = scriptId;
      script.src = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
      script.async = true;
      document.head.append(script);
    }
    script.addEventListener("load", render);
    return () => script?.removeEventListener("load", render);
  }, [onToken, theme]);
  return <div className="turnstile" ref={ref} aria-label="Human verification" />;
}
