import { useEffect, useState } from "react";
import { Route, Routes, useLocation } from "react-router-dom";
import { Header } from "./components/Header";
import { AboutPage } from "./pages/AboutPage";
import { CompanyPage } from "./pages/CompanyPage";
import { LandingPage } from "./pages/LandingPage";
import { ResearchPage } from "./pages/ResearchPage";

type Theme = "light" | "dark";

export function App() {
  const location = useLocation();
  const [theme, setTheme] = useState<Theme>(() => (localStorage.getItem("filing-room-theme") as Theme | null) ?? (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem("filing-room-theme", theme); }, [theme]);
  return (
    <div className="app">
      <a href="#main-content" className="skip-link">Skip to content</a>
      <Header theme={theme} onTheme={() => setTheme((value) => value === "light" ? "dark" : "light")} />
      <div id="main-content">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/company/:ticker" element={<CompanyPage />} />
          <Route path="/research" element={<ResearchPage theme={theme} />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="*" element={<LandingPage />} />
        </Routes>
      </div>
      {location.pathname !== "/research" && <footer className="site-footer"><span>Filing Room · Built by Brian Castelino</span><span>Not affiliated with the SEC · Not investment advice</span></footer>}
    </div>
  );
}
