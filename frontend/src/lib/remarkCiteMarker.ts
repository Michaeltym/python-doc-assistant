import type { Root } from "mdast";
import type { Plugin } from "unified";
import { SKIP, visit } from "unist-util-visit";

// Match `[1]`, `[1, 3]`, `[1,2,3]` — bracket containing one or more
// integers separated by commas (with optional whitespace). Matches our
// backend's _extract_citations parser exactly.
const CITE_PATTERN = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

/**
 * remarkCiteMarker — turns `[N]` / `[N, M]` text spans into `<span
 * class="cite-marker">[N]</span>` so they can be styled distinctly
 * from body prose without dropping any information.
 *
 * Operates on the mdast text nodes BEFORE conversion to hast. The HTML
 * emission needs `rehype-raw` downstream so the raw HTML is parsed
 * back into hast elements (otherwise ReactMarkdown escapes it to
 * literal text — that was the original bug).
 *
 * `code` and `inlineCode` mdast nodes hold their content in a `value`
 * string (no text children), so this visitor naturally does not touch
 * them — `[N]`-shaped tokens inside code blocks render verbatim.
 */
export const remarkCiteMarker: Plugin<[], Root> = () => {
  return (tree) => {
    visit(tree, "text", (node, index, parent) => {
      if (!parent || index === undefined) return;

      const text = node.value;
      const out: Array<{ type: "text"; value: string } | { type: "html"; value: string }> = [];
      let last = 0;
      CITE_PATTERN.lastIndex = 0;
      let m: RegExpExecArray | null;
      while ((m = CITE_PATTERN.exec(text)) !== null) {
        if (m.index > last) {
          out.push({ type: "text", value: text.slice(last, m.index) });
        }
        out.push({
          type: "html",
          value: `<span class="cite-marker">${m[0]}</span>`,
        });
        last = CITE_PATTERN.lastIndex;
      }
      if (out.length === 0) return;
      if (last < text.length) {
        out.push({ type: "text", value: text.slice(last) });
      }
      // mdast typing: text inside a Paragraph permits text + html nodes.
      // `unknown` cast keeps TS happy across the various Parent unions
      // visit() may yield.
      (parent as { children: unknown[] }).children.splice(index, 1, ...out);
      return [SKIP, index + out.length];
    });
  };
};
