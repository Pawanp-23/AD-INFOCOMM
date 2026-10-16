// Builds the submission Word document from the Markdown final report.
//
//   cd scripts/report && npm install && npm run build
//   node build.js [input.md] [output.docx]
//
// Supports the Markdown subset used in docs/reports/Final-Report.md: YAML-like front matter
// (title page), #/##/### headings, paragraphs, - and 1. lists, pipe tables, ``` code blocks,
// > quotes, *caption* lines, and inline **bold**, *italic*, `code`.

const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, Footer, Header, HeadingLevel, LevelFormat, Packer,
  PageNumber, Paragraph, ShadingType, Table, TableCell, TableOfContents, TableRow,
  TextRun, WidthType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const INPUT = process.argv[2] || path.join(ROOT, "docs", "reports", "Final-Report.md");
const OUTPUT = process.argv[3] || path.join(ROOT, "docs", "reports", "RAMEI-Final-Report.docx");

const FONT = "Calibri";
const MONO = "Consolas";
const ACCENT = "1F3A5F";
const PAGE_W = 11906; // A4
const MARGIN = 1440;
const CONTENT_W = PAGE_W - 2 * MARGIN;

// ------------------------------------------------------------------ inline formatting
function runs(text, base = {}) {
  // Unfinished placeholders are highlighted so they cannot slip into the submission.
  const parts = text.split(/(\[PENDING[^\]]*\])/);
  if (parts.length > 1) {
    return parts.flatMap((p) => (p.startsWith("[PENDING") ? [new TextRun({ ...base, text: p, highlight: "yellow" })] : inline(p, base)));
  }
  return inline(text, base);
}

function inline(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*)/g;
  let last = 0;
  for (const m of text.matchAll(re)) {
    if (m.index > last) out.push(new TextRun({ ...base, text: text.slice(last, m.index) }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ ...base, text: t.slice(2, -2), bold: true }));
    else if (t.startsWith("`")) out.push(new TextRun({ ...base, text: t.slice(1, -1), font: MONO, size: 19 }));
    else out.push(new TextRun({ ...base, text: t.slice(1, -1), italics: true }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ ...base, text: text.slice(last) }));
  return out;
}

// ------------------------------------------------------------------ block builders
const cellBorder = { style: BorderStyle.SINGLE, size: 4, color: "B7C3D0" };
const borders = { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder };

function splitRow(line) {
  return line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
}

function table(lines) {
  const rows = lines.filter((l, i) => !(i === 1 && /^\|?\s*:?-{3,}/.test(l))).map(splitRow);
  const n = rows[0].length;
  // Column widths proportional to the longest cell text, with a floor so nothing collapses.
  const lens = Array.from({ length: n }, (_, c) =>
    Math.max(6, ...rows.map((r) => Math.min(60, (r[c] || "").replace(/[*`]/g, "").length))),
  );
  const total = lens.reduce((a, b) => a + b, 0);
  const widths = lens.map((l) => Math.floor((l / total) * CONTENT_W));
  widths[n - 1] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map(
      (r, ri) =>
        new TableRow({
          tableHeader: ri === 0,
          children: widths.map(
            (w, c) =>
              new TableCell({
                width: { size: w, type: WidthType.DXA },
                borders,
                margins: { top: 60, bottom: 60, left: 100, right: 100 },
                shading: ri === 0 ? { type: ShadingType.CLEAR, color: "auto", fill: "E3EAF2" } : undefined,
                children: [
                  new Paragraph({
                    spacing: { after: 0 },
                    children: runs(r[c] || "", { size: 19, bold: ri === 0 || undefined }),
                  }),
                ],
              }),
          ),
        }),
    ),
  });
}

function codeBlock(lines) {
  return lines.map(
    (l, i) =>
      new Paragraph({
        shading: { type: ShadingType.CLEAR, color: "auto", fill: "F3F5F8" },
        spacing: { before: i === 0 ? 120 : 0, after: i === lines.length - 1 ? 160 : 0, line: 240 },
        indent: { left: 120, right: 120 },
        children: [new TextRun({ text: l || " ", font: MONO, size: 17 })],
      }),
  );
}

// ------------------------------------------------------------------ markdown -> blocks
function parse(md) {
  let meta = {};
  const fm = md.match(/^---\n([\s\S]*?)\n---\n/);
  if (fm) {
    for (const line of fm[1].split("\n")) {
      const m = line.match(/^(\w+):\s*"?(.*?)"?\s*$/);
      if (m) meta[m[1]] = m[2];
    }
    md = md.slice(fm[0].length);
  }
  const lines = md.split(/\r?\n/);
  const blocks = [];
  let listInstance = 0;
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (line.startsWith("```")) {
      const body = [];
      for (i++; i < lines.length && !lines[i].startsWith("```"); i++) body.push(lines[i]);
      i++;
      blocks.push(...codeBlock(body));
      continue;
    }
    const h = line.match(/^(#{1,3})\s+(.*)$/);
    if (h) {
      const level = [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3][h[1].length - 1];
      // Every top-level chapter starts on a new page.
      blocks.push(new Paragraph({ heading: level, pageBreakBefore: h[1].length === 1, children: runs(h[2]) }));
      i++;
      continue;
    }
    if (line.trim().startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(lines[i++]);
      blocks.push(table(rows));
      blocks.push(new Paragraph({ spacing: { after: 60 }, children: [] }));
      continue;
    }
    if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
      const numbered = /^\s*\d+\./.test(line);
      listInstance++;
      while (i < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[i])) {
        let text = lines[i].replace(/^\s*([-*]|\d+\.)\s+/, "");
        for (i++; i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*]|\d+\.)\s/.test(lines[i]); i++) {
          text += " " + lines[i].trim(); // wrapped continuation line
        }
        blocks.push(
          new Paragraph({
            numbering: numbered
              ? { reference: "numbers", level: 0, instance: listInstance }
              : { reference: "bullets", level: 0 },
            spacing: { after: 60 },
            children: runs(text),
          }),
        );
      }
      continue;
    }
    if (line.startsWith(">")) {
      let text = "";
      while (i < lines.length && lines[i].startsWith(">")) text += " " + lines[i++].replace(/^>\s?/, "");
      blocks.push(
        new Paragraph({
          indent: { left: 567 },
          border: { left: { style: BorderStyle.SINGLE, size: 12, color: "8FA6BF", space: 8 } },
          children: runs(text.trim(), { italics: true }),
        }),
      );
      continue;
    }
    // Paragraph: join wrapped lines until a blank line or another block starts.
    let text = line.trim();
    for (i++; i < lines.length && lines[i].trim() && !/^(#|\||```|>|\s*([-*]|\d+\.)\s)/.test(lines[i]); i++) {
      text += " " + lines[i].trim();
    }
    const caption = /^\*[^*].*\*$/.test(text);
    blocks.push(
      new Paragraph({
        alignment: caption ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
        children: runs(caption ? text.slice(1, -1) : text, {
          ...(caption ? { italics: true, size: 19, color: "555555" } : {}),
        }),
      }),
    );
  }
  return { meta, blocks };
}

// ------------------------------------------------------------------ document
function titlePage(meta) {
  const line = (text, opts = {}) =>
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new TextRun({ text, ...opts })] });
  const [name, ...rest] = (meta.title || "Report").split(" — ");
  return [
    new Paragraph({ spacing: { before: 2400 }, children: [] }),
    line(name, { bold: true, size: 64, color: ACCENT }),
    line(rest.join(" — "), { size: 30, color: "333333" }),
    new Paragraph({
      border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: ACCENT, space: 1 } },
      spacing: { after: 600 },
      children: [],
    }),
    line(meta.subtitle || "", { size: 28 }),
    new Paragraph({ spacing: { before: 1200 }, children: [] }),
    line("Submitted by", { size: 22, color: "666666" }),
    line(meta.author || "", { size: 26, bold: true }),
    new Paragraph({ spacing: { before: 400 }, children: [] }),
    line("Under the guidance of", { size: 22, color: "666666" }),
    line(meta.guide || "", { size: 26, bold: true }),
    new Paragraph({ spacing: { before: 1000 }, children: [] }),
    line(meta.institution || "", { size: 24 }),
    line(meta.date || "", { size: 24 }),
  ];
}

async function main() {
  const { meta, blocks } = parse(fs.readFileSync(INPUT, "utf8").replace(/\r\n/g, "\n"));
  const shortTitle = (meta.title || "").split(" — ")[0];
  const doc = new Document({
    creator: meta.author || "RAMEI project",
    title: meta.title,
    features: { updateFields: true }, // Word refreshes the table of contents on open
    styles: {
      default: { document: { run: { font: FONT, size: 22 }, paragraph: { spacing: { after: 140, line: 288 } } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 34, bold: true, color: ACCENT, font: FONT },
          paragraph: { spacing: { before: 0, after: 240 }, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 27, bold: true, color: ACCENT, font: FONT },
          paragraph: { spacing: { before: 300, after: 120 }, outlineLevel: 1, keepNext: true } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 23, bold: true, color: "333333", font: FONT },
          paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2, keepNext: true } },
      ],
    },
    numbering: {
      config: [
        { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 567, hanging: 283 } } } }] },
        { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 567, hanging: 340 } } } }] },
      ],
    },
    sections: [
      {
        properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
        children: titlePage(meta),
      },
      {
        properties: {
          page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN },
            pageNumbers: { start: 1 } },
        },
        headers: {
          default: new Header({
            children: [new Paragraph({ alignment: AlignmentType.RIGHT,
              children: [new TextRun({ text: `${shortTitle} — Internship Project Report`, size: 17, color: "888888" })] })],
          }),
        },
        footers: {
          default: new Footer({
            children: [new Paragraph({ alignment: AlignmentType.CENTER,
              children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "888888" })] })],
          }),
        },
        children: [
          new Paragraph({ spacing: { after: 240 }, children: [new TextRun({ text: "Contents", bold: true, size: 34, color: ACCENT })] }),
          new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
          ...blocks,
        ],
      },
    ],
  });
  fs.writeFileSync(OUTPUT, await Packer.toBuffer(doc));
  console.log(`wrote ${OUTPUT}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
