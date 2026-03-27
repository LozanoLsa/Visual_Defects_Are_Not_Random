# Visual Defects Are Not Random — Naive Bayes

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/02-Naive-Bayes-Surface-Defect/blob/main/02_Naive_Bayes_Surface_Defect_Inspection.ipynb)

> *"Surface defects are deceptive — they appear at final inspection, but they originate at injection, accumulate during transport, and get revealed by the paint. By the time a defect is detected visually, the part has already traveled through three process stages."*

---

## 🎯 Business Problem

In painted plastic part manufacturing, a visual defect at final inspection is never just a quality event — it's a failure that already happened upstream. The part was scratched in storage, contaminated during handling, or damaged in transit, long before it entered the paint cabin. Traditional inspection catches the symptom. This project targets the cause.

With a **21.9% defect rate** across 1,500 painted parts, the question isn't whether defects are costly — it's whether the process signals that precede them are detectable. They are.

---

## 📊 Dataset

- **1,500 painted plastic part records** from a simulated injection-to-inspection production line
- **Target:** `surface_defect` (binary) — 1 = visual defect detected at final inspection
- **Class balance:** 21.9% defective (moderate imbalance)
- **Source:** Simulated operational data spanning four process stages

| Stage | Features |
|-------|----------|
| Injection | `regrind_pct`, `resin_temp_c`, `cooling_time_s` |
| Paint Cabin | `paint_viscosity`, `film_thickness_um`, `booth_humidity_pct` |
| Internal Logistics | `pre_paint_storage_hrs`, `num_handlings` |
| Packaging | `container_type`, `part_protection` |

**The key EDA finding:** Injection parameters (resin temp, cooling time, regrind) show near-zero Cohen's d. The defect doesn't originate at injection — it accumulates in **internal logistics**. That's the story the data tells, and the model confirms it.

---

## 🤖 Model

**Algorithm:** Gaussian Naive Bayes — `sklearn.naive_bayes.GaussianNB`

GaussianNB models each feature as an independent Gaussian distribution per class. No StandardScaler required — it estimates mean and variance directly from the data. This is one of its practical advantages in production: fewer preprocessing steps, fewer failure points.

The probabilistic output (0–1 defect probability) enables **pre-paint risk scoring** — flagging high-risk parts before they enter the paint cabin, not after they fail inspection.

**Preprocessing:** OneHotEncoder on categorical features (container_type, part_protection), passthrough on numerics — all inside a Pipeline with ColumnTransformer.

---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| Accuracy | 76.2% |
| ROC-AUC | 0.629 |
| Precision (Defect) | 28.6% |
| Recall (Defect) | 6.1% |
| F1 (Defect) | 0.101 |

**Honest interpretation:** Low recall at 0.5 threshold. The model's real value is the **probability score**, not the binary verdict. At lower thresholds, recall improves significantly — and in a manufacturing context, flagging more parts for visual pre-check is a worthwhile trade against the cost of a defective part reaching the customer.

---

## 🔍 Top Defect Drivers (Effect Size Analysis)

| Feature | Cohen's d | Defect Rate |
|---------|-----------|-------------|
| `pre_paint_storage_hrs` | 0.189 | Top numeric driver |
| `num_handlings` | 0.178 | Strong numeric driver |
| `part_protection` | Dominant (MI) | Unprotected: 28.8% vs Protected: 16.8% |
| `container_type` | Secondary (MI) | Cardboard: 28.9% vs Plastic box: 15.6% |
| `resin_temp_c` | 0.063 | Near-zero — injection is not the problem |

The practical implication is direct: cap pre-paint storage at 4 hours, make part protection mandatory, migrate from cardboard pallets to plastic boxes. The injection engineer is not the one to call — the logistics supervisor is.

---

## 🗂️ Repository Structure

```
02-Naive-Bayes-Surface-Defect/
├── 02_Naive_Bayes_Surface_Defect_Inspection.ipynb  # Notebook (no outputs)
├── surface_defect_inspection_data.csv              # Sample dataset (250 rows)
├── README.md
└── requirements.txt
```

> 📦 **Full Project Pack** — complete dataset (1,500 rows), notebook with full outputs, presentation deck (PPTX + PDF), and `app.py` part inspection simulator available on [Gumroad](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Google Colab:** Click the badge above.

**Option 2 — Local:**
```bash
pip install -r requirements.txt
jupyter notebook 02_Naive_Bayes_Surface_Defect_Inspection.ipynb
```

---

## 💡 Key Learnings

1. **Where the defect appears ≠ where it originates** — inspection finds logistics problems. That shift in framing changes which manager you call.
2. **Cohen's d beats correlation for class separation** — pooled effect size tells you whether the distribution actually shifts between classes, not just whether there's a linear relationship.
3. **No scaling is a feature, not a limitation** — GaussianNB's direct distribution modeling makes it one of the most deployment-friendly classifiers for production environments.
4. **Mutual Information captures what correlation misses** — container type and protection status are categorical. MI quantifies their contribution without assuming linearity.
5. **A 0.5 threshold is rarely optimal** — in defect detection, lowering the threshold trades precision for recall. That trade is almost always worth it when defects reach customers.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning  
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
