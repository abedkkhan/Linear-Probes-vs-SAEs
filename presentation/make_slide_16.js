// Slide 16 — SAE · Finding the Sentiment Feature (standalone, no comparison)
// Theme: Claude Code UI — matches slides 1, 2, 15 exactly.
// Layout: LAYOUT_WIDE (13.333" × 7.5").

const pptxgen = require("pptxgenjs");
const path = require("path");

const C = {
  bg:        "FAF9F5",
  text:      "1A1A1A",
  muted:     "5B5B5B",
  faint:     "9C9489",
  accent:    "CC785C",
  panel:     "EFECE4",
  panelEdge: "D6D2C7",
};

const PLOTS = "../results/plots";

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.title  = "SAE — Finding the Sentiment Feature";
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
  slide.addText("[  RESULTS  ·  SAE DISCOVERY  ·  16  ]", {
    x: LEFT_X, y: 0.55, w: CONTENT_W, h: 0.3,
    fontSize: 10, fontFace: "Menlo", bold: true,
    color: C.accent, charSpacing: 4, margin: 0, valign: "middle",
  });

  slide.addText("SAE — Finding the Sentiment Feature", {
    x: LEFT_X, y: 0.95, w: CONTENT_W, h: 0.85,
    fontSize: 36, fontFace: "Georgia", bold: true,
    color: C.text, align: "left", valign: "top", margin: 0,
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: LEFT_X, y: 1.85, w: 0.9, h: 0.03,
    fill: { color: C.accent }, line: { type: "none" },
  });

  // ------------------------------------------------------------------
  // Stat row — three big callouts (same template as slide 15)
  // ------------------------------------------------------------------
  const STAT_Y    = 2.05;
  const STAT_H    = 1.20;
  const STAT_GAP  = 0.35;
  const STAT_W    = (CONTENT_W - 2 * STAT_GAP) / 3;

  const stats = [
    {
      big: "16,384",
      label: "FEATURES SEARCHED",
      sub: "every SAE feature scored vs labels",
    },
    {
      big: "#14733",
      label: "TOP FEATURE",
      sub: "fires on negative sentiment",
    },
    {
      big: "−0.50",
      label: "CORRELATION (TRAIN)",
      sub: "highest |r| of any feature",
    },
  ];

  stats.forEach((s, i) => {
    const sx = LEFT_X + i * (STAT_W + STAT_GAP);
    slide.addShape(pres.shapes.RECTANGLE, {
      x: sx, y: STAT_Y, w: STAT_W, h: STAT_H,
      fill: { color: C.panel }, line: { color: C.panelEdge, width: 0.75 },
    });
    slide.addShape(pres.shapes.RECTANGLE, {
      x: sx, y: STAT_Y, w: 0.07, h: STAT_H,
      fill: { color: C.accent }, line: { type: "none" },
    });
    slide.addText(s.big, {
      x: sx + 0.25, y: STAT_Y + 0.08, w: STAT_W - 0.4, h: 0.7,
      fontSize: 40, fontFace: "Georgia", bold: true,
      color: C.text, align: "left", valign: "top", margin: 0,
    });
    slide.addText(s.label, {
      x: sx + 0.25, y: STAT_Y + 0.72, w: STAT_W - 0.4, h: 0.25,
      fontSize: 10, fontFace: "Menlo", bold: true,
      color: C.accent, charSpacing: 4, align: "left", valign: "middle", margin: 0,
    });
    slide.addText(s.sub, {
      x: sx + 0.25, y: STAT_Y + 0.95, w: STAT_W - 0.4, h: 0.22,
      fontSize: 10, fontFace: "Georgia", italic: true,
      color: C.muted, align: "left", valign: "middle", margin: 0,
    });
  });

  // ------------------------------------------------------------------
  // Plot row — three figures side by side
  // ------------------------------------------------------------------
  const PLOT_Y   = STAT_Y + STAT_H + 0.30;
  const PLOT_H   = 2.35;
  const PLOT_W   = STAT_W;
  const PLOT_GAP = STAT_GAP;
  const CAP_H    = 0.30;

  const plots = [
    {
      file: path.join(PLOTS, "all_feature_correlations_hist.png"),
      cap: "Distribution of correlations across all 16,384 features",
    },
    {
      file: path.join(PLOTS, "sae_feature_correlations.png"),
      cap: "Top-20 features by |correlation| — #14733 leads",
    },
    {
      file: path.join(PLOTS, "feature_14733_distribution.png"),
      cap: "Feature 14733 fires on negative test sentences",
    },
  ];

  plots.forEach((p, i) => {
    const px = LEFT_X + i * (PLOT_W + PLOT_GAP);
    slide.addShape(pres.shapes.RECTANGLE, {
      x: px, y: PLOT_Y, w: PLOT_W, h: PLOT_H,
      fill: { color: "FFFFFF" }, line: { color: C.panelEdge, width: 0.75 },
    });
    slide.addImage({
      path: p.file,
      x: px + 0.05, y: PLOT_Y + 0.05,
      w: PLOT_W - 0.10, h: PLOT_H - 0.10,
      sizing: { type: "contain", w: PLOT_W - 0.10, h: PLOT_H - 0.10 },
    });
    slide.addText(p.cap, {
      x: px, y: PLOT_Y + PLOT_H + 0.05, w: PLOT_W, h: CAP_H,
      fontSize: 9, fontFace: "Menlo",
      color: C.muted, align: "center", valign: "middle", margin: 0,
      charSpacing: 1,
    });
  });

  // ------------------------------------------------------------------
  // Takeaway
  // ------------------------------------------------------------------
  const TAKE_Y = PLOT_Y + PLOT_H + CAP_H + 0.15;
  slide.addShape(pres.shapes.RECTANGLE, {
    x: LEFT_X, y: TAKE_Y, w: 0.07, h: 0.35,
    fill: { color: C.accent }, line: { type: "none" },
  });
  slide.addText(
    "Out of 16,384 unsupervised features, one — #14733 — emerges as a sentiment detector, discovered with no labels.",
    {
      x: LEFT_X + 0.22, y: TAKE_Y - 0.03, w: CONTENT_W - 0.22, h: 0.4,
      fontSize: 14, fontFace: "Georgia", italic: true,
      color: C.text, align: "left", valign: "middle", margin: 0,
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
  slide.addText("16", {
    x: SLIDE_W - RIGHT_PAD - 0.5, y: 7.10, w: 0.4, h: 0.30,
    fontSize: 10, fontFace: "Menlo", bold: true,
    color: C.accent, align: "right", valign: "middle", margin: 0,
  });

  await pres.writeFile({ fileName: "slide_16.pptx" });
  console.log("Saved slide_16.pptx");
})();
