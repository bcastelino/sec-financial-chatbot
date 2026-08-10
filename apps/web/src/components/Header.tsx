import { Moon, Sun } from "lucide-react";
import { Link, NavLink } from "react-router-dom";
import { Logo } from "./Logo";

export function Header({ theme, onTheme }: { theme: "light" | "dark"; onTheme: () => void }) {
  return (
    <header className="site-header">
      <Link to="/" className="logo-link"><Logo /></Link>
      <nav aria-label="Primary navigation">
        <NavLink to="/research">Research</NavLink>
        <NavLink to="/about">Methodology</NavLink>
        <a href="https://github.com/bcastelino/sec-filing-room" target="_blank" rel="noreferrer">GitHub</a>
      </nav>
      <button className="icon-button" onClick={onTheme} aria-label={`Use ${theme === "light" ? "dark" : "light"} theme`}>
        {theme === "light" ? <Moon size={18} /> : <Sun size={18} />}
      </button>
    </header>
  );
}
