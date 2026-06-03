// Slide 2 — The Problem (background + cited claims + cliffhanger question)
// Theme: Claude Code UI — cream background, coral accent, serif title.
// Layout: LAYOUT_WIDE (13.333" × 7.5").

const pptxgen = require("pptxgenjs");

const C = {
  bg:        "FAF9F5",
  text:      "1A1A1A",
  muted:     "5B5B5B",
  faint:     "9C9489",
  accent:    "CC785C",
  accentDim: "E8B6A4",
};

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.title  = "The Problem";
  pres.author = "Aabid Karim";
  const slide = pres.addSlide();
  slide.background = { color: C.bg };

  // ------------------------------------------------------------------
  // Geometry
  // ------------------------------------------------------------------
  const LEFT_X       = 0.7;
  const RIGHT_PAD    = 0.7;
  const SLIDE_W      = 13.333;
  const CONTENT_W    = SLIDE_W - LEFT_X - RIGHT_PAD;     // ~11.93
  const BULLET_NUM_W = 0.55;
  const BULLET_TEXT_X = LEFT_X + 0.7 + BULLET_NUM_W;     // indent body past number
  const BULLET_TEXT_W = SLIDE_W - BULLET_TEXT_X - RIGHT_PAD;

  // ------------------------------------------------------------------
  // Header: small label + title
  // ------------------------------------------------------------------
  slide.addText("[  BACKGROUND  ·  02  ]", {
    x: LEFT_X, y: 0.55, w: CONTENT_W, h: 0.3,
    fontSize: 10, fontFace: "Menlo", bold: true,
    color: C.accent, charSpacing: 4, margin: 0, valign: "middle",
  });

  slide.addText("The Problem", {
    x: LEFT_X, y: 0.95, w: CONTENT_W, h: 0.9,
    fontSize: 44, fontFace: "Georgia", bold: true,
    color: C.text, align: "left", valign: "top", margin: 0,
  });

  // Coral rule under title
  slide.addShape(pres.shapes.RECTANGLE, {
    x: LEFT_X, y: 1.92, w: 0.9, h: 0.03,
    fill: { color: C.accent }, line: { type: "none" },
  });

  // ------------------------------------------------------------------
  // Bullets — six numbered claims, each with optional inline citation
  // ------------------------------------------------------------------
  const BULLET_Y0 = 2.10;
  const BULLET_DY = 0.50;

  const bullets = [
    {
      text: "Language models like GPT, Gemma, and Claude now generate impressive natural-language output across nearly every domain.",
      cites: ["Brown et al., 2020", "Team Gemma, 2024"],
    },
    {
      text: "Yet despite their fluency, we still don’t understand what is happening inside their hidden layers.",
      cites: ["Olah et al., 2020"],
    },
    {
      text: "Their internal computations remain opaque — the original meaning of the phrase “black box”.",
      cites: ["Räuker et al., 2023"],
    },
    {
      text: "Individual neurons are polysemantic: a single neuron can fire for many unrelated concepts at once.",
      cites: ["Elhage et al., 2022"],
    },
    {
      text: "This makes it impossible to interpret, trust, or control these models reliably — a core obstacle for safety and deployment.",
      cites: ["Bereska & Gavves, 2024"],
    },
    {
      text: "Two leading techniques have emerged to peek inside the residual stream: linear probes and sparse autoencoders.",
      cites: ["Alain & Bengio, 2017", "Bricken et al., 2023"],
    },
  ];

  bullets.forEach((b, i) => {
    const y = BULLET_Y0 + i * BULLET_DY;
    // index number in coral monospace
    const n = String(i + 1).padStart(2, "0");
    slide.addText(n, {
      x: LEFT_X, y, w: BULLET_NUM_W, h: 0.45,
      fontSize: 14, fontFace: "Menlo", bold: true,
      color: C.accent, align: "left", valign: "top", margin: 0,
      charSpacing: 1,
    });
    // body + inline cite
    const runs = [{ text: b.text, options: { color: C.text } }];
    if (b.cites && b.cites.length) {
      runs.push({
        text: "  (" + b.cites.join("; ") + ")",
        options: { italic: true, color: C.accent },
      });
    }
    slide.addText(runs, {
      x: BULLET_TEXT_X, y, w: BULLET_TEXT_W, h: 0.5,
      fontSize: 13, fontFace: "Georgia",
      align: "left", valign: "top", margin: 0,
      lineSpacingMultiple: 1.25,
    });
  });

  // ------------------------------------------------------------------
  // Cliffhanger question — the FYP question, visually distinct
  // ------------------------------------------------------------------
  const Q_Y = BULLET_Y0 + bullets.length * BULLET_DY + 0.12;

  // pale coral block on the left
  slide.addShape(pres.shapes.RECTANGLE, {
    x: LEFT_X, y: Q_Y, w: 0.10, h: 0.50,
    fill: { color: C.accent }, line: { type: "none" },
  });
  slide.addText(
    "But which one actually works better?",
    {
      x: LEFT_X + 0.25, y: Q_Y - 0.05, w: CONTENT_W - 0.25, h: 0.6,
      fontSize: 20, fontFace: "Georgia", italic: true, bold: true,
      color: C.accent, align: "left", valign: "middle", margin: 0,
    }
  );

  // ------------------------------------------------------------------
  // References block (APA style) at bottom
  // ------------------------------------------------------------------
  const REF_Y = Q_Y + 0.85;
  slide.addText("REFERENCES", {
    x: LEFT_X, y: REF_Y, w: 2.0, h: 0.22,
    fontSize: 8, fontFace: "Menlo", bold: true,
    color: C.faint, charSpacing: 3, margin: 0, valign: "middle",
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: LEFT_X, y: REF_Y + 0.22, w: CONTENT_W, h: 0.012,
    fill: { color: C.faint }, line: { type: "none" },
  });

  const refs = [
    "Alain, G., & Bengio, Y. (2017). Understanding intermediate layers using linear classifier probes. ICLR Workshop.",
    "Bereska, L., & Gavves, E. (2024). Mechanistic interpretability for AI safety: A review. TMLR.",
    "Bricken, T., Templeton, A., Batson, J., et al. (2023). Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, Anthropic.",
    "Brown, T. B., Mann, B., Ryder, N., et al. (2020). Language models are few-shot learners. NeurIPS, 33, 1877–1901.",
    "Elhage, N., Hume, T., Olsson, C., et al. (2022). Toy models of superposition. Transformer Circuits Thread, Anthropic.",
    "Olah, C., Cammarata, N., Schubert, L., et al. (2020). Zoom in: An introduction to circuits. Distill, 5(3).",
    "Räuker, T., Ho, A., Casper, S., & Hadfield-Menell, D. (2023). Toward transparent AI: A survey on interpreting the inner structures of deep neural networks. IEEE SaTML.",
    "Team Gemma. (2024). Gemma 2: Improving open language models at a practical size. Google DeepMind technical report.",
  ];

  // two columns
  const COL_W = (CONTENT_W - 0.4) / 2;
  const ROW_Y0 = REF_Y + 0.35;
  const ROW_H = 0.22;
  refs.forEach((r, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    slide.addText(r, {
      x: LEFT_X + col * (COL_W + 0.4),
      y: ROW_Y0 + row * ROW_H,
      w: COL_W, h: ROW_H,
      fontSize: 8, fontFace: "Georgia",
      color: C.muted, align: "left", valign: "top", margin: 0,
      lineSpacingMultiple: 1.1,
    });
  });

  // ------------------------------------------------------------------
  // Footer
  // ------------------------------------------------------------------
  slide.addText("Aabid Karim   ·   ID 30468176", {
    x: LEFT_X, y: 7.10, w: 6, h: 0.30,
    fontSize: 9, fontFace: "Menlo",
    color: C.muted, align: "left", valign: "middle", margin: 0,
    charSpacing: 2,
  });
  slide.addText("02", {
    x: SLIDE_W - RIGHT_PAD - 0.5, y: 7.10, w: 0.4, h: 0.30,
    fontSize: 10, fontFace: "Menlo", bold: true,
    color: C.accent, align: "right", valign: "middle", margin: 0,
  });

  await pres.writeFile({ fileName: "slide_02.pptx" });
  console.log("Saved slide_02.pptx");
})();
