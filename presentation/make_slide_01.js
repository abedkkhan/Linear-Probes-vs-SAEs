// Slide 1 — Opening the Black Box (FYP title + hook + transformer diagram)
// Theme: Claude Code UI — cream background, coral accent, serif title.
// Layout: LAYOUT_WIDE (13.333" × 7.5").

const pptxgen = require("pptxgenjs");

const C = {
  bg:         "FAF9F5",
  text:       "1A1A1A",
  muted:      "6E6E6E",
  faint:      "AFAAA0",
  accent:     "CC785C",
  accentSoft: "EFD4C8",
  layer:      "EFECE4",
  layerEdge:  "D6D2C7",
};

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.title  = "Opening the Black Box";
  pres.author = "Aabid Karim";
  const slide = pres.addSlide();
  slide.background = { color: C.bg };

  // ------------------------------------------------------------------
  // Geometry constants (one source of truth)
  // ------------------------------------------------------------------
  const SLIDE_W = 13.333;
  const SLIDE_H = 7.5;

  const LEFT_X  = 0.7;
  const LEFT_W  = 7.6;        // hard ceiling on left-column width

  const STACK_X     = 8.9;    // moved left from before
  const STACK_W     = 2.4;
  const STACK_TOP_Y = 1.30;
  const LAYER_H     = 0.34;
  const LAYER_GAP   = 0.08;
  const ROW_STRIDE  = LAYER_H + LAYER_GAP;
  const N_ROWS      = 7;
  const STACK_BOT_Y = STACK_TOP_Y + N_ROWS * ROW_STRIDE - LAYER_GAP;

  const CALLOUT_X = STACK_X + STACK_W + 0.20;
  const CALLOUT_W = SLIDE_W - CALLOUT_X - 0.4;   // fits comfortably

  const TOK_TOP_Y = STACK_BOT_Y + 0.22;
  const TOK_H     = 0.32;

  // ------------------------------------------------------------------
  // LEFT BLOCK
  // ------------------------------------------------------------------
  slide.addText("[  FINAL YEAR PROJECT  ·  2026  ]", {
    x: LEFT_X, y: 0.55, w: LEFT_W, h: 0.3,
    fontSize: 10, fontFace: "Menlo", bold: true,
    color: C.accent, charSpacing: 4, margin: 0, valign: "middle",
  });

  // Title broken to two lines so it never bleeds into the right column
  slide.addText("Opening the\nBlack Box", {
    x: LEFT_X, y: 1.0, w: LEFT_W, h: 2.4,
    fontSize: 60, fontFace: "Georgia", bold: true,
    color: C.text, align: "left", valign: "top", margin: 0,
    lineSpacingMultiple: 0.95,
  });

  // Subtitle
  slide.addText(
    "Comparing Linear Probes and Sparse\nAutoencoders for Understanding\nLanguage Models",
    {
      x: LEFT_X + 0.02, y: 3.55, w: LEFT_W, h: 1.6,
      fontSize: 19, fontFace: "Georgia",
      color: C.muted, align: "left", valign: "top", margin: 0,
      lineSpacingMultiple: 1.3,
    }
  );

  // Coral rule + hook
  slide.addShape(pres.shapes.RECTANGLE, {
    x: LEFT_X + 0.02, y: 5.65, w: 0.9, h: 0.03,
    fill: { color: C.accent }, line: { type: "none" },
  });
  slide.addText(
    "“We use AI everywhere — but we still don’t know how it thinks.”",
    {
      x: LEFT_X + 0.02, y: 5.8, w: LEFT_W, h: 0.6,
      fontSize: 16, fontFace: "Georgia", italic: true,
      color: C.text, align: "left", valign: "middle", margin: 0,
    }
  );

  // Footer
  slide.addText("Aabid Karim   ·   ID 30468176", {
    x: LEFT_X, y: 6.95, w: 6, h: 0.35,
    fontSize: 10, fontFace: "Menlo",
    color: C.muted, align: "left", valign: "middle", margin: 0,
    charSpacing: 2,
  });

  // ------------------------------------------------------------------
  // RIGHT BLOCK — Transformer diagram
  // ------------------------------------------------------------------
  // Header (no overlap with title: it's above and to the right)
  slide.addText("[  gemma-2-2b  ]", {
    x: STACK_X - 0.3, y: 0.55, w: STACK_W + 0.6, h: 0.3,
    fontSize: 10, fontFace: "Menlo", bold: true,
    color: C.accent, align: "center", valign: "middle", margin: 0,
    charSpacing: 3,
  });
  slide.addText("26 decoder layers", {
    x: STACK_X - 0.3, y: 0.85, w: STACK_W + 0.6, h: 0.3,
    fontSize: 11, fontFace: "Georgia", italic: true,
    color: C.muted, align: "center", valign: "middle", margin: 0,
  });

  // Faint residual-stream band running through stack + tokens
  const BAND_W = 0.36;
  const BAND_X = STACK_X + (STACK_W - BAND_W) / 2;
  slide.addShape(pres.shapes.RECTANGLE, {
    x: BAND_X,
    y: STACK_TOP_Y - 0.10,
    w: BAND_W,
    h: (TOK_TOP_Y + TOK_H) - (STACK_TOP_Y - 0.10) + 0.05,
    fill: { color: C.accentSoft, transparency: 25 },
    line: { type: "none" },
  });

  // Forward-pass arrow on the left
  const ARROW_X = STACK_X - 0.42;
  slide.addShape(pres.shapes.LINE, {
    x: ARROW_X,
    y: STACK_TOP_Y + 0.10,
    w: 0,
    h: STACK_BOT_Y - STACK_TOP_Y - 0.20,
    line: { color: C.faint, width: 1.25, beginArrowType: "triangle", endArrowType: "none" },
  });
  slide.addText("forward pass", {
    x: ARROW_X - 1.00,
    y: (STACK_TOP_Y + STACK_BOT_Y) / 2 - 0.15,
    w: 0.95, h: 0.3,
    fontSize: 9, fontFace: "Menlo",
    color: C.muted, align: "right", valign: "middle", margin: 0,
    charSpacing: 1,
  });

  // Rows
  const rows = [
    { kind: "layer", label: "26" },
    { kind: "layer", label: "25" },
    { kind: "dots"  },
    { kind: "layer", label: "12", highlight: true },
    { kind: "dots"  },
    { kind: "layer", label: "2"  },
    { kind: "layer", label: "1"  },
  ];

  rows.forEach((r, i) => {
    const y = STACK_TOP_Y + i * ROW_STRIDE;
    if (r.kind === "dots") {
      slide.addText("· · ·", {
        x: STACK_X, y, w: STACK_W, h: LAYER_H,
        fontSize: 14, fontFace: "Menlo", bold: true,
        color: C.faint, align: "center", valign: "middle", margin: 0,
      });
      return;
    }
    const isHi = !!r.highlight;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: STACK_X, y, w: STACK_W, h: LAYER_H,
      fill: { color: isHi ? C.accent : C.layer },
      line: { color: isHi ? C.accent : C.layerEdge, width: 0.75 },
    });
    slide.addText(
      [
        { text: "layer ", options: { fontFace: "Georgia", italic: true } },
        { text: r.label, options: { fontFace: "Menlo",   bold: true   } },
      ],
      {
        x: STACK_X, y, w: STACK_W, h: LAYER_H,
        fontSize: 12,
        color: isHi ? C.bg : C.text,
        align: "center", valign: "middle", margin: 0,
        charSpacing: 1,
      }
    );
  });

  // ---------- Callout to highlighted Layer 12 ----------
  const HI_INDEX = 3;
  const HI_Y_CENTER = STACK_TOP_Y + HI_INDEX * ROW_STRIDE + LAYER_H / 2;

  // Connector line
  slide.addShape(pres.shapes.LINE, {
    x: STACK_X + STACK_W,
    y: HI_Y_CENTER,
    w: 0.22, h: 0,
    line: { color: C.accent, width: 1 },
  });

  // "we look here" text — sized to fit inside CALLOUT_W
  slide.addText("we look here", {
    x: CALLOUT_X,
    y: HI_Y_CENTER - 0.18,
    w: CALLOUT_W, h: 0.30,
    fontSize: 11, fontFace: "Georgia", italic: true,
    color: C.accent, align: "left", valign: "middle", margin: 0,
  });
  slide.addText("residual stream", {
    x: CALLOUT_X,
    y: HI_Y_CENTER + 0.05,
    w: CALLOUT_W, h: 0.20,
    fontSize: 9, fontFace: "Menlo",
    color: C.muted, align: "left", valign: "middle", margin: 0,
    charSpacing: 1,
  });
  slide.addText("2304-dim vector", {
    x: CALLOUT_X,
    y: HI_Y_CENTER + 0.22,
    w: CALLOUT_W, h: 0.20,
    fontSize: 9, fontFace: "Menlo",
    color: C.muted, align: "left", valign: "middle", margin: 0,
    charSpacing: 1,
  });

  // ---------- Token strip below stack ----------
  const tokens = ["the", "movie", "was", "great"];
  const TOK_GAP = 0.06;
  const TOK_W = (STACK_W - (tokens.length - 1) * TOK_GAP) / tokens.length;
  tokens.forEach((t, i) => {
    const tx = STACK_X + i * (TOK_W + TOK_GAP);
    slide.addShape(pres.shapes.RECTANGLE, {
      x: tx, y: TOK_TOP_Y, w: TOK_W, h: TOK_H,
      fill: { color: C.layer }, line: { color: C.layerEdge, width: 0.75 },
    });
    slide.addText(t, {
      x: tx, y: TOK_TOP_Y, w: TOK_W, h: TOK_H,
      fontSize: 10, fontFace: "Menlo",
      color: C.text, align: "center", valign: "middle", margin: 0,
    });
  });
  slide.addText("input tokens", {
    x: STACK_X, y: TOK_TOP_Y + TOK_H + 0.06, w: STACK_W, h: 0.22,
    fontSize: 9, fontFace: "Georgia", italic: true,
    color: C.muted, align: "center", valign: "middle", margin: 0,
  });

  await pres.writeFile({ fileName: "slide_01.pptx" });
  console.log("Saved slide_01.pptx");
})();
