import type { SourceRef } from "@filing-room/contracts";
import ReactMarkdown from "react-markdown";
import rehypeSanitize from "rehype-sanitize";
import remarkGfm from "remark-gfm";

export function CitationText({ text, sources, onSource }: { text: string; sources: SourceRef[]; onSource: (source: SourceRef) => void }) {
  const parts = text.split(/(\[S\d+\])/g);
  return (
    <div className="markdown">
      {parts.map((part, index) => {
        const match = part.match(/^\[(S\d+)\]$/);
        const source = match ? sources.find((item) => item.id === match[1]) : undefined;
        if (source) return <button key={`${part}-${index}`} className="citation" onClick={() => onSource(source)}>{source.id}</button>;
        return <ReactMarkdown key={index} remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSanitize]}>{part}</ReactMarkdown>;
      })}
    </div>
  );
}
