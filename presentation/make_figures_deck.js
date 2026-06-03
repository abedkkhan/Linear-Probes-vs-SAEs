// Figures Discussion Deck — one figure per slide, plain-English explanation +
// an "if asked" answer for the supervisor Q&A. Claude Code UI theme.
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
  qbg:       "F3E9E3",   // pale coral panel for the Q&A box
};

const PLOTS = "../results/plots";
const SLIDE_W = 13.333;
const LEFT_X = 0.6;

let pres;

function addFigureSlide(cfg, pageNum) {
  const slide = pres.addSlide();
  slide.background = { color: C.bg };

  // Header label
  slide.addText(`[  ${cfg.tag}  ·  ${String(pageNum).padStart(2, "0")}  ]`, {
    x: LEFT_X, y: 0.45, w: 11, h: 0.3,
    fontSize: 10, fontFace: "Menlo", bold: true,
    color: C.accent, charSpacing: 4, margin: 0, valign: "middle",
  });

  // Title
  slide.addText(cfg.title, {
    x: LEFT_X, y: 0.78, w: SLIDE_W - 1.2, h: 0.7,
    fontSize: 26, fontFace: "Georgia", bold: true,
    color: C.text, align: "left", valign: "top", margin: 0,
  });

  // ---- Figure (left) ----
  const FIG_X = LEFT_X;
  const FIG_Y = 1.65;
  const FIG_W = 7.3;
  const FIG_H = 5.05;
  slide.addShape(pres.shapes.RECTANGLE, {
    x: FIG_X, y: FIG_Y, w: FIG_W, h: FIG_H,
    fill: { color: "FFFFFF" }, line: { color: C.panelEdge, width: 0.75 },
  });
  slide.addImage({
    path: path.join(PLOTS, cfg.figure),
    x: FIG_X + 0.1, y: FIG_Y + 0.1,
    w: FIG_W - 0.2, h: FIG_H - 0.2,
    sizing: { type: "contain", w: FIG_W - 0.2, h: FIG_H - 0.2 },
  });

  // ---- Right column ----
  const RX = FIG_X + FIG_W + 0.45;       // 8.35
  const RW = SLIDE_W - RX - 0.5;          // ~4.48

  // "WHAT THIS SHOWS"
  slide.addText("WHAT THIS SHOWS", {
    x: RX, y: FIG_Y, w: RW, h: 0.3,
    fontSize: 11, fontFace: "Menlo", bold: true,
    color: C.accent, charSpacing: 3, margin: 0, valign: "middle",
  });
  // bullets
  const bulletRuns = cfg.shows.map((t, i) => ({
    text: t,
    options: { bullet: { code: "2022", indent: 14 }, breakLine: true,
               color: C.text, paraSpaceAfter: 6 },
  }));
  slide.addText(bulletRuns, {
    x: RX, y: FIG_Y + 0.35, w: RW, h: 2.6,
    fontSize: 13, fontFace: "Georgia",
    align: "left", valign: "top", margin: 0,
    lineSpacingMultiple: 1.12,
  });

  // "IF ASKED" panel
  const QY = FIG_Y + 3.15;
  const QH = FIG_H - 3.15;
  slide.addShape(pres.shapes.RECTANGLE, {
    x: RX, y: QY, w: RW, h: QH,
    fill: { color: C.qbg }, line: { type: "none" },
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: RX, y: QY, w: 0.07, h: QH,
    fill: { color: C.accent }, line: { type: "none" },
  });
  slide.addText("IF ASKED", {
    x: RX + 0.22, y: QY + 0.14, w: RW - 0.4, h: 0.28,
    fontSize: 10, fontFace: "Menlo", bold: true,
    color: C.accent, charSpacing: 3, margin: 0, valign: "middle",
  });
  slide.addText(
    [
      { text: "Q  ", options: { bold: true, color: C.accent } },
      { text: cfg.q, options: { italic: true, color: C.text, breakLine: true } },
      { text: "\n", options: { breakLine: true } },
      { text: "A  ", options: { bold: true, color: C.muted } },
      { text: cfg.a, options: { color: C.muted } },
    ],
    {
      x: RX + 0.22, y: QY + 0.45, w: RW - 0.45, h: QH - 0.55,
      fontSize: 12, fontFace: "Georgia",
      align: "left", valign: "top", margin: 0,
      lineSpacingMultiple: 1.18,
    }
  );

  // footer
  slide.addText("Aabid Karim · ID 30468176 · Figures Discussion Deck", {
    x: LEFT_X, y: 7.12, w: 9, h: 0.3,
    fontSize: 8, fontFace: "Menlo", color: C.faint,
    align: "left", valign: "middle", margin: 0, charSpacing: 1,
  });
}

(async () => {
  pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.title = "Figures Discussion Deck";
  pres.author = "Aabid Karim";

  // ---- Cover slide ----
  const cover = pres.addSlide();
  cover.background = { color: C.bg };
  cover.addText("[  FIGURES · DISCUSSION DECK  ]", {
    x: LEFT_X, y: 2.3, w: 11, h: 0.4,
    fontSize: 12, fontFace: "Menlo", bold: true,
    color: C.accent, charSpacing: 5, margin: 0,
  });
  cover.addText("Results, Figure by Figure", {
    x: LEFT_X, y: 2.8, w: 12, h: 1.1,
    fontSize: 48, fontFace: "Georgia", bold: true,
    color: C.text, margin: 0,
  });
  cover.addText(
    "Linear Probes vs Sparse Autoencoders  ·  Gemma 2 2B  ·  SST-2 sentiment\nEach slide: what the figure shows, in plain terms — and the question to expect.",
    {
      x: LEFT_X + 0.02, y: 4.0, w: 11.5, h: 1.0,
      fontSize: 16, fontFace: "Georgia", italic: true,
      color: C.muted, margin: 0, lineSpacingMultiple: 1.3,
    }
  );
  cover.addShape(pres.shapes.RECTANGLE, {
    x: LEFT_X + 0.02, y: 3.85, w: 0.9, h: 0.03,
    fill: { color: C.accent }, line: { type: "none" },
  });
  cover.addText("Aabid Karim · ID 30468176", {
    x: LEFT_X, y: 6.9, w: 8, h: 0.3,
    fontSize: 10, fontFace: "Menlo", color: C.muted, margin: 0, charSpacing: 2,
  });

  // ---- Figure slides ----
  const slides = [
    {
      tag: "LINEAR PROBE",
      title: "The probe separates positive from negative",
      figure: "score_distribution_test.png",
      shows: [
        "Each sentence gets one score from the probe.",
        "Negative sentences (red) score low; positive (blue) score high.",
        "The two groups barely overlap — clean separation.",
      ],
      q: "What is a “probe score”?",
      a: "One number the probe gives each sentence. Below the dashed line = predicted negative; above = positive. The gap between the two colours is what makes it accurate.",
    },
    {
      tag: "LINEAR PROBE",
      title: "92% correct on unseen sentences",
      figure: "confusion_test.png",
      shows: [
        "500 test sentences the probe never saw in training.",
        "Only 40 mistakes total (27 + 13).",
        "197 negatives and 263 positives correctly identified.",
      ],
      q: "Is 92% good?",
      a: "Yes. Our success target was 75%, and 92% is competitive with published results on this dataset. Random guessing would be 50%.",
    },
    {
      tag: "LINEAR PROBE",
      title: "Near-perfect ranking (ROC curve)",
      figure: "roc_curve_test.png",
      shows: [
        "The curve hugging the top-left corner means very few errors.",
        "AUC = 0.97 (1.0 would be perfect, 0.5 is random).",
      ],
      q: "What is AUC?",
      a: "If you pick one positive and one negative sentence at random, AUC is the chance the probe scores the positive one higher. 0.97 means it gets the order right 97% of the time.",
    },
    {
      tag: "LAYER CHOICE",
      title: "Why layer 12? We tested all 26",
      figure: "layer_sweep_accuracy.png",
      shows: [
        "We trained a probe at every one of Gemma’s 26 layers.",
        "Accuracy peaks in the middle layers (10–13).",
        "Layer 12 gives the best test accuracy of all.",
      ],
      q: "Why did you pick layer 12 specifically?",
      a: "Not arbitrarily — the data shows the middle is best. Early layers handle raw words; late layers focus on predicting the next word. Sentiment lives cleanest in the middle, and layer 12 is the peak.",
    },
    {
      tag: "SAE DISCOVERY",
      title: "Searching 16,384 features for sentiment",
      figure: "all_feature_correlations_hist.png",
      shows: [
        "The SAE breaks each activation into 16,384 features.",
        "Most features have nothing to do with sentiment (pile near zero).",
        "A few stand out — feature 14733 is the strongest (red line).",
      ],
      q: "How did you find the sentiment feature?",
      a: "We measured how strongly each of the 16,384 features lines up with the positive/negative labels, then picked the single strongest one. No labels were used to build the SAE itself.",
    },
    {
      tag: "SAE DISCOVERY",
      title: "The top sentiment features",
      figure: "sae_feature_correlations.png",
      shows: [
        "Top 20 features ranked by how well they track sentiment.",
        "Red bars fire on negative sentences, green on positive.",
        "Feature 14733 leads the ranking.",
      ],
      q: "Why are most of them ‘negative’ features?",
      a: "Negative sentiment in reviews tends to use sharper, more distinctive language, so the SAE gives it cleaner dedicated features. Positive sentiment is spread more thinly.",
    },
    {
      tag: "SAE DISCOVERY",
      title: "Feature 14733 fires on negative sentences",
      figure: "feature_14733_distribution.png",
      shows: [
        "How strongly the feature fires across test sentences.",
        "Negative sentences (red) push it high; positive (blue) stay near zero.",
        "The dashed line is the cutoff we use to classify.",
      ],
      q: "Where did the cutoff come from?",
      a: "From the validation set, not the test set — we chose the line that best separated the classes on data held aside, then applied it untouched to the test set.",
    },
    {
      tag: "SAE INTERPRETABILITY",
      title: "What the feature actually responds to",
      figure: "feature_14733_top_sentences.png",
      shows: [
        "The training sentences that make feature 14733 fire hardest.",
        "Almost all are clearly negative movie reviews.",
        "Evidence the feature is genuinely about negativity — not noise.",
      ],
      q: "Two of them are labelled positive — isn’t that a problem?",
      a: "Those two are hedged (“its occasional charms”) — faint praise that reads half-negative. The feature tracks negativity strongly but not perfectly, which is expected and honest.",
    },
    {
      tag: "SAE CLASSIFIER",
      title: "More features → higher accuracy",
      figure: "sae_topk_only.png",
      shows: [
        "Using just feature 14733: 77% accuracy.",
        "Using the top 5: 85%. Top 100: ~90%.",
        "Sentiment is spread across a handful of features, not one.",
      ],
      q: "Why not just use the single best feature?",
      a: "Because sentiment isn’t one single thing — intensity, negation, specific words all contribute. Each is a separate SAE feature, so combining a few gives a big jump.",
    },
    {
      tag: "COMPARISON",
      title: "Probe vs SAE — head to head",
      figure: "accuracy_comparison_bar.png",
      shows: [
        "Linear probe: 92%. Best SAE (top-100): ~90%.",
        "Single SAE feature: 77%. Random guessing: 50%.",
        "The probe wins on accuracy by a small margin.",
      ],
      q: "So the SAE lost?",
      a: "On raw accuracy, yes — by about 2 points. But the SAE was never given labels, and it gives interpretable features you can read and inspect. It’s a trade-off, not a defeat.",
    },
    {
      tag: "COMPARISON",
      title: "Do both methods point the same way?",
      figure: "sae_all_cosines_hist.png",
      shows: [
        "How aligned each SAE feature is with the probe’s direction.",
        "Almost all features are unrelated (pile near zero).",
        "Feature 14733 (red line) sits far out in the tail.",
      ],
      q: "Is −0.22 really ‘aligned’? It sounds small.",
      a: "In 2,304 dimensions, random alignment is essentially zero. −0.22 is over 10 standard deviations above chance — statistically, an extremely strong signal, even if the number looks modest.",
    },
    {
      tag: "COMPARISON",
      title: "How much of the probe lives in SAE features?",
      figure: "direction_subspace_projection.png",
      shows: [
        "How much of the probe’s direction the top SAE features recreate.",
        "Top 100 features capture ~half of it (blue), vs ~20% for random (grey).",
        "Real overlap — but no small set fully reproduces the probe.",
      ],
      q: "Do the two methods find the same thing or not?",
      a: "Both — they clearly point the same way (far above chance), but the SAE spreads sentiment across many features, so no handful exactly equals the probe’s single direction. Same concept, described differently.",
    },
  ];

  slides.forEach((cfg, i) => addFigureSlide(cfg, i + 1));

  await pres.writeFile({ fileName: "figures_deck.pptx" });
  console.log(`Saved figures_deck.pptx (${slides.length + 1} slides)`);
})();
