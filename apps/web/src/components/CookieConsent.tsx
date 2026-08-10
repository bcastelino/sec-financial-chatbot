import { useEffect, useState } from "react";
import { CONSENT_KEY, updateConsent } from "../lib/analytics";

type ConsentState = "granted" | "denied";

export function CookieConsent() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem(CONSENT_KEY);
    if (stored === "granted" || stored === "denied") {
      updateConsent(stored as ConsentState);
    } else {
      setVisible(true);
    }
  }, []);

  function choose(state: ConsentState) {
    localStorage.setItem(CONSENT_KEY, state);
    updateConsent(state);
    setVisible(false);
  }

  if (!visible) return null;

  return (
    <div className="cookie-consent" role="dialog" aria-live="polite" aria-label="Cookie consent">
      <p>
        This site uses cookies to measure traffic with Google Analytics. Analytics
        cookies are only set if you accept.
      </p>
      <div className="cookie-consent-actions">
        <button type="button" onClick={() => choose("denied")}>Reject</button>
        <button type="button" className="consent-accept" onClick={() => choose("granted")}>Accept</button>
      </div>
    </div>
  );
}
