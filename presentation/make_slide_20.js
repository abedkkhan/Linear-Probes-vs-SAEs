// Slide 20 — Limitations & Conclusion (combined)
// Theme: Claude Code UI — matches slides 1, 2, 15–19.
// Layout: LAYOUT_WIDE (13.333" × 7.5").

const pptxgen = require("pptxgenjs");

const C = {
  bg:        "FAF9F5",
  text:      "1A1A1A",
  muted:     "5B5B5B",
  faint:     "9C9489",
  accent:    "CC785C",
  panel:     "EFECE4",
  panelEdge: "D6D2C7",
};

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.title  = "Limitations & Conclusion";
  pres.author = "Aabid Karim";
  const slide = pres.addSlide();
  slide.background = { color: C.bg };

  // ------------------------------------------------------------------
  // Geometry
  // ------------------------------------------------------------------
  const SLIDE_W   = 13.333;
  const LEFT_X    = 0.7;
  const RIGHT_PAD = 0.7;
  const CONTENT_W = SLIDE_W - LEFT_X - RIGHT_PAD;

  // ------------------------------------------------------------------
  // Header
  // ------------------------------------------------------------------
  slide.addText("[  LIMITATIONS  ·  CONCLUSION  ·  20  ]", {
    x: LEFT_X, y: 0.55, w: CONTENT_W, h: 0.3,
    fontSize: 10, fontFace: "Menlo", bold: true,
    color: C.accent, charSpacing: 4, margin: 0, valign: "middle",
  });

  slide.addText("What we found — and where it stops.", {
    x: LEFT_X, y: 0.95, w: CONTENT_W, h: 0.85,
    fontSize: 34, fontFace: "Georgia", bold: true,
    color: C.text, align: "left", valign: "top", margin: 0,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: LEFT_X, y: 1.92, w: 0.9, h: 0.03,
    fill: { color: C.accent }, line: { type: "none" },
  });

  // ------------------------------------------------------------------
  // Two columns: Conclusion (left) + Limitations (right)
  // ------------------------------------------------------------------
  const COL_GAP = 0.6;
  const COL_W   = (CONTENT_W - COL_GAP) / 2;
  const COL_Y   = 2.20;
  const COL_H   = 4.30;

  const COL_LEFT_X  = LEFT_X;
  const COL_RIGHT_X = LEFT_X + COL_W + COL_GAP;

  // ---- Left column header (Conclusion) ----
  slide.addText("CONCLUSION", {
    x: COL_LEFT_X, y: COL_Y, w: COL_W, h: 0.35,
    fontSize: 11, fontFace: "Menlo", bold: true,
    color: C.accent, charSpacing: 5, margin: 0, valign: "middle",
  });
  slide.addText("what the experiments tell us", {
    x: COL_LEFT_X, y: COL_Y + 0.30, w: COL_W, h: 0.28,
    fontSize: 11, fontFace: "Georgia", italic: true,
    color: C.muted, margin: 0, valign: "middle",
  });

  // ---- Right column header (Limitations) ----
  slide.addText("LIMITATIONS", {
    x: COL_RIGHT_X, y: COL_Y, w: COL_W, h: 0.35,
    fontSize: 11, fontFace: "Menlo", bold: true,
    color: C.accent, charSpacing: 5, margin: 0, valign: "middle",
  });
  slide.addText("scope this work does not cover", {
    x: COL_RIGHT_X, y: COL_Y + 0.30, w: COL_W, h: 0.28,
    fontSize: 11, fontFace: "Georgia", italic: true,
    color: C.muted, margin: 0, valign: "middle",
  });

  // Subtle thin divider between columns (vertical)
  slide.addShape(pres.shapes.RECTANGLE, {
    x: LEFT_X + COL_W + COL_GAP / 2 - 0.005,
    y: COL_Y,
    w: 0.01, h: COL_H + 0.10,
    fill: { color: C.panelEdge }, line: { type: "none" },
  });

  // ------------------------------------------------------------------
  // Conclusion bullets (left)
  // ------------------------------------------------------------------
  const BULLET_Y0 = COL_Y + 0.85;
  const BULLET_DY = 0.62;
  const NUM_W = 0.45;

  const conclusionItems = [
    {
      head: "Sentiment is linear",
      body: "at Gemma layer 12 — both methods recover the same concept as a single direction.",
    },
    {
      head: "Probe wins on accuracy",
      body: "92.0% vs 89.8% — supervised training beats unsupervised features by ~2 pp.",
    },
    {
      head: "Sentiment is distributed",
      body: "in the SAE basis — ~5–100 features carry the signal, not one alone.",
    },
    {
      head: "Directions agree",
      body: "|cos(w, top SAE)| ≈ 0.22, 13× above the random baseline — same concept, two views.",
    },
    {
      head: "The real trade-off",
      body: "probes for accuracy, SAEs for individual-feature interpretability.",
    },
  ];

  conclusionItems.forEach((it, i) => {
    const y = BULLET_Y0 + i * BULLET_DY;
    slide.addText(String(i + 1).padStart(2, "0"), {
      x: COL_LEFT_X, y, w: NUM_W, h: 0.30,
      fontSize: 12, fontFace: "Menlo", bold: true,
      color: C.accent, align: "left", valign: "top", margin: 0,
    });
    slide.addText(
      [
        { text: it.head + "  ", options: { bold: true, color: C.text } },
        { text: "— " + it.body, options: { color: C.muted } },
      ],
      {
        x: COL_LEFT_X + NUM_W, y: y - 0.02, w: COL_W - NUM_W, h: 0.6,
        fontSize: 12, fontFace: "Georgia",
        align: "left", valign: "top", margin: 0,
        lineSpacingMultiple: 1.25,
      }
    );
  });

  // ------------------------------------------------------------------
  // Limitations bullets (right)
  // ------------------------------------------------------------------
  const limitations = [
    {
      head: "One dataset",
      body: "SST-2 sentiment only — findings may not transfer to other concepts.",
    },
    {
      head: "One layer",
      body: "Layer 12 of 26 — other layers may favour each method differently.",
    },
    {
      head: "One model",
      body: "Gemma 2 2B — larger models or other families could shift the gap.",
    },
    {
      head: "One SAE variant",
      body: "Width 16K, canonical sparsity — wider SAEs (65K, 1M) may close the gap.",
    },
    {
      head: "Empirical choices",
      body: "BOS dropping & mean-pooling were fixed early — not ablated systematically.",
    },
  ];

  limitations.forEach((it, i) => {
    const y = BULLET_Y0 + i * BULLET_DY;
    slide.addText(String(i + 1).padStart(2, "0"), {
      x: COL_RIGHT_X, y, w: NUM_W, h: 0.30,
      fontSize: 12, fontFace: "Menlo", bold: true,
      color: C.accent, align: "left", valign: "top", margin: 0,
    });
    slide.addText(
      [
        { text: it.head + "  ", options: { bold: true, color: C.text } },
        { text: "— " + it.body, options: { color: C.muted } },
      ],
      {
        x: COL_RIGHT_X + NUM_W, y: y - 0.02, w: COL_W - NUM_W, h: 0.6,
        fontSize: 12, fontFace: "Georgia",
        align: "left", valign: "top", margin: 0,
        lineSpacingMultiple: 1.25,
      }
    );
  });

  // ------------------------------------------------------------------
  // Bottom takeaway / closing line
  // ------------------------------------------------------------------
  const TAKE_Y = 6.55;
  slide.addShape(pres.shapes.RECTANGLE, {
    x: LEFT_X, y: TAKE_Y, w: 0.10, h: 0.4,
    fill: { color: C.accent }, line: { type: "none" },
  });
  slide.addText(
    "Both methods see the same model — they just describe it differently.",
    {
      x: LEFT_X + 0.25, y: TAKE_Y - 0.04, w: CONTENT_W - 0.5, h: 0.5,
      fontSize: 16, fontFace: "Georgia", italic: true, bold: true,
      color: C.accent, align: "left", valign: "middle", margin: 0,
    }
  );

  // ------------------------------------------------------------------
  // Footer
  // ------------------------------------------------------------------
  slide.addText("Aabid Karim   ·   ID 30468176", {
    x: LEFT_X, y: 7.10, w: 6, h: 0.30,
    fontSize: 9, fontFace: "Menlo",
    color: C.muted, align: "left", valign: "middle", margin: 0,
    charSpacing: 2,
  });
  slide.addText("20", {
    x: SLIDE_W - RIGHT_PAD - 0.5, y: 7.10, w: 0.4, h: 0.30,
    fontSize: 10, fontFace: "Menlo", bold: true,
    color: C.accent, align: "right", valign: "middle", margin: 0,
  });

  await pres.writeFile({ fileName: "slide_20.pptx" });
  console.log("Saved slide_20.pptx");
})();
