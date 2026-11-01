// The Markdown an answer can use, built as React elements and never as raw HTML.

const FENCE = /^\s*```/;
const HEADING = /^(#{1,6})\s+(.*?)\s*#*\s*$/;
const RULE = /^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/;
const QUOTE = /^\s*>\s?/;
const ITEM = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
const TABLE_DIVIDER = /^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?\s*$/;
const INLINE = /(`[^`\n]+`|\*\*(?:[^*\n]|\*(?!\*))+\*\*|\*[^*\s](?:[^*\n]*[^*\s])?\*|(?<![\w)])\[\d+\]|\[[^\]\n]+\]\(https?:\/\/[^\s)]+\))/g;
export const CITATION = /(?<![\w)])\[(\d+)\]/g;

function isTableStart(lines, i) {
  return lines[i].includes("|") && i + 1 < lines.length && lines[i + 1].includes("-") && TABLE_DIVIDER.test(lines[i + 1]);
}

function startsBlock(lines, i) {
  const line = lines[i];
  return FENCE.test(line) || HEADING.test(line) || RULE.test(line) || QUOTE.test(line) || ITEM.test(line) || isTableStart(lines, i);
}

function tableCells(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
}

function parseList(lines, start) {
  const first = lines[start].match(ITEM);
  const indent = first[1].length;
  const list = { type: "list", ordered: /\d/.test(first[2]), items: [] };
  let i = start;
  while (i < lines.length) {
    if (!lines[i].trim()) { // a blank line only continues the list if another item follows
      const next = lines.findIndex((line, n) => n > i && line.trim());
      if (next === -1 || !ITEM.test(lines[next]) || lines[next].match(ITEM)[1].length < indent) break;
      i = next;
      continue;
    }
    const match = lines[i].match(ITEM);
    if (match && match[1].length < indent) break;
    if (match && match[1].length > indent && list.items.length) {
      const [nested, after] = parseList(lines, i);
      list.items[list.items.length - 1].children.push(nested);
      i = after;
    } else if (match) {
      if (/\d/.test(match[2]) !== list.ordered) break; // bullets after numbers start a new list
      list.items.push({ text: match[3], children: [] });
      i += 1;
    } else if (/^\s/.test(lines[i]) && list.items.length) { // an indented line carries on the item
      list.items[list.items.length - 1].text += " " + lines[i].trim();
      i += 1;
    } else {
      break;
    }
  }
  return [list, i];
}

export function parseMarkdown(text) {
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i += 1;
    } else if (FENCE.test(line)) {
      const code = [];
      for (i += 1; i < lines.length && !FENCE.test(lines[i]); i += 1) code.push(lines[i]);
      i += 1; // a fence still being typed simply runs to the end
      blocks.push({ type: "code", text: code.join("\n") });
    } else if (HEADING.test(line)) {
      const [, marks, heading] = line.match(HEADING);
      blocks.push({ type: "heading", level: Math.min(marks.length + 1, 4), text: heading });
      i += 1;
    } else if (RULE.test(line)) {
      blocks.push({ type: "rule" });
      i += 1;
    } else if (QUOTE.test(line)) {
      const quote = [];
      for (; i < lines.length && QUOTE.test(lines[i]); i += 1) quote.push(lines[i].replace(QUOTE, ""));
      blocks.push({ type: "quote", blocks: parseMarkdown(quote.join("\n")) });
    } else if (ITEM.test(line)) {
      const [list, after] = parseList(lines, i);
      blocks.push(list);
      i = after;
    } else if (isTableStart(lines, i)) {
      const head = tableCells(line);
      const rows = [];
      for (i += 2; i < lines.length && lines[i].includes("|"); i += 1) rows.push(tableCells(lines[i]));
      blocks.push({ type: "table", head, rows });
    } else {
      const paragraph = [line.trim()];
      for (i += 1; i < lines.length && lines[i].trim() && !startsBlock(lines, i); i += 1) paragraph.push(lines[i].trim());
      blocks.push({ type: "paragraph", text: paragraph.join(" ") });
    }
  }
  return blocks;
}

function renderInline(text, ctx) {
  return text.split(INLINE).map((part, i) => {
    if (!part || i % 2 === 0) return part;
    if (part.startsWith("`")) return React.createElement("code", { key: i }, part.slice(1, -1));
    if (part.startsWith("**")) return React.createElement("strong", { key: i }, renderInline(part.slice(2, -2), ctx));
    if (part.startsWith("*")) return React.createElement("em", { key: i }, renderInline(part.slice(1, -1), ctx));
    const link = part.match(/^\[([^\]]+)\]\((.+)\)$/);
    if (link) return React.createElement("a", { key: i, href: link[2], target: "_blank", rel: "noopener" }, renderInline(link[1], ctx));
    const source = ctx.sources[Number(part.slice(1, -1)) - 1];
    if (!source) return part;
    return React.createElement("button", {
      key: i,
      type: "button",
      className: "cite",
      title: `${source.title} at ${source.timestamp}`,
      onClick: () => ctx.onPick(source),
    }, part.slice(1, -1));
  });
}

function renderList(list, key, ctx) {
  return React.createElement(list.ordered ? "ol" : "ul", { key }, list.items.map((item, i) =>
    React.createElement("li", { key: i }, renderInline(item.text, ctx), item.children.map((child, n) => renderList(child, n, ctx)))));
}

function renderBlocks(blocks, ctx) {
  return blocks.map((block, i) => {
    switch (block.type) {
      case "heading": return React.createElement(`h${block.level}`, { key: i }, renderInline(block.text, ctx));
      case "code": return React.createElement("pre", { key: i }, React.createElement("code", null, block.text));
      case "rule": return React.createElement("hr", { key: i });
      case "quote": return React.createElement("blockquote", { key: i }, renderBlocks(block.blocks, ctx));
      case "list": return renderList(block, i, ctx);
      case "table": return React.createElement("div", { key: i, className: "table-wrap" }, React.createElement("table", null,
        React.createElement("thead", null, React.createElement("tr", null, block.head.map((cell, n) => React.createElement("th", { key: n }, renderInline(cell, ctx))))),
        React.createElement("tbody", null, block.rows.map((row, r) => React.createElement("tr", { key: r }, row.map((cell, n) => React.createElement("td", { key: n }, renderInline(cell, ctx))))))));
      default: return React.createElement("p", { key: i }, renderInline(block.text, ctx));
    }
  });
}

export function renderMarkdown(text, sources, onPick) {
  return renderBlocks(parseMarkdown(text), { sources, onPick });
}

// A half-revealed citation or marker stays hidden until the rest of it is on screen.
export function hideUnfinished(text) {
  if ((text.split("```").length - 1) % 2) return text; // inside a code block, shown as typed
  let out = text.replace(/\[\d*$/, "");
  const outsideFences = () => out.replace(/```[\s\S]*?```/g, "");
  if ((outsideFences().split("`").length - 1) % 2) out = out.slice(0, out.lastIndexOf("`"));
  if ((outsideFences().replace(/`[^`\n]*`/g, "").split("**").length - 1) % 2) out = out.slice(0, out.lastIndexOf("**"));
  return out;
}
