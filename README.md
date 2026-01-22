# 🎨 Surface Defect Prediction (Pass / Fail) using Naive Bayes

## Project Overview
This is a **practical machine learning exercise inspired by a real manufacturing quality problem**.

Let’s be clear from the start 🙂  
The goal here is **not** to chase the highest possible metric or pretend this model will magically eliminate cosmetic defects.  
The goal is to **learn**, to **analyze**, and to **think probabilistically** about why surface defects appear after injection + transport + painting.

Naive Bayes was used here as a **tool for understanding**, not as an end in itself.

---

## Problem Statement
Cosmetic defects in painted plastic parts are a recurring pain point in manufacturing.

They rarely come from a single cause.  
Instead, they emerge from a mix of process variables across the full chain: injection conditions, storage time, handling intensity, container type, protection practices, and paint booth environment.

Rather than treating rejects as isolated events, this project approaches them as a **probabilistic outcome influenced by process variables**.

If this sounds familiar… you’ve probably lived it.

---

## Objective 🎯
The main objectives of this project are to:

- Build a **binary classification model** to estimate the probability of a surface defect (Pass / Fail).
- Use Naive Bayes as a **probability-based, fast and interpretable model** for operational risk screening.
- Create a **learning-oriented foundation** to explore how upstream and handling variables influence defect risk.

This project is intentionally designed as a **foundation for analysis**, not as a production-ready solution.

No magic. Just probabilities.

---

## Dataset Description 📊
The dataset represents a **realistic manufacturing environment**, reflecting common conditions found in day-to-day operations for painted plastic parts.

Each row corresponds to a single part / lot outcome, described by process conditions collected across multiple stages.

### Features (X)
The input variables include:

#### Injection (3)
- **Regrind_pct**: Percentage of regrind material used in the mix (%).
- **Resin_temp_C**: Resin temperature at injection (°C).
- **Cooling_time_s**: Cooling time in seconds (s).

#### Painting (3)
- **Paint_viscosity**: Paint viscosity (e.g., Ford-cup style units / seconds).
- **Film_thickness_um**: Coating film thickness (µm).
- **Booth_humidity_pct**: Paint booth relative humidity (%).

#### Storage / Transport / Handling (4)
- **Prepaint_storage_time_h**: Storage time before painting (hours).
- **Handling_moves**: Number of handling / transport moves before inspection.
- **Container_type**: Container category (`metal_rack`, `plastic_box`, `cardboard_pallet`).
- **Part_protection**: Protection condition (`with_protection`, `no_protection`).

### Target Variable (Y)
- **Surface_defect**: Inspection outcome  
  - **0 = Pass (OK)**  
  - **1 = Fail (Surface defect)**

### Data Origin (Real-World Perspective)
Each variable reflects information that typically comes from **different systems in a real manufacturing environment**:

- **Regrind_pct, Resin_temp_C, Cooling_time_s**  
  → Injection molding machine parameters (PLC / MES), material records, process sheets.

- **Paint_viscosity, Film_thickness_um, Booth_humidity_pct**  
  → Paint shop logs, booth sensors, QC lab checks, inline measurement systems.

- **Prepaint_storage_time_h, Handling_moves, Container_type, Part_protection**  
  → WIP tracking, internal logistics scans (barcode/RFID), warehouse systems, standard work / packaging records.

- **Surface_defect (target variable)**  
  → Final inspection database (visual inspection station), quality disposition records, rework/scrap logs.

> In real-world operations, this type of dataset rarely exists in a single system.  
> It is typically built by **integrating data across multiple sources**, which is why understanding process context is just as important as modeling. 🙂

---

## Modeling Approach 🧠
A **Naive Bayes classifier** was selected because:

- It models **probability**, not certainty  
- It is extremely fast and can be imagined as a near real-time risk screener  
- Many inputs are treated as quasi-independent in practice (injection settings, humidity, handling intensity, storage time, etc.)

This model doesn’t try to be clever.  
It tries to be **honest**.

---

## Why this case is a great fit for Naive Bayes ✅
- **Clear binary classification:** Pass / Fail.
- **Many variables can be treated as quasi-independent in practice:** injection parameters, paint booth humidity, storage time, handling intensity, packaging conditions.
- **Very fast model → easy to imagine in near real-time operations:**
  - “With current injection + painting parameters, surface defect risk is 27%.  
     If we increase viscosity and reduce humidity, risk drops to 8%.”
- **Executive-friendly explanation:**
  - “We’re not guessing. Based on historical parts, for this exact combination of conditions,  
     the model estimates a probability of visual scrap of X%.”

---

## Key Results 📈
Standard classification metrics are reported, but they are **not the headline**.

What really matters here is that the model allows us to:
- Identify **high-impact operational drivers**
- Visualize **risk patterns** across injection, paint, and handling conditions
- Understand how defect probability changes when conditions drift

Accuracy matters.  
Understanding matters more.

---

## Simulation & Scenarios
A simple **scenario simulator** is included in the notebook.

It allows you to:
- Define hypothetical process and handling conditions
- Estimate the probability of a surface defect
- Ask “what if?” without pretending to predict the future

This is where the model becomes a **conversation tool**, not just code.

---

## Business Insights
Some patterns become very clear:

- Handling-related variables (moves, storage time, protection, container type) are major risk drivers
- Paint environment (humidity) amplifies cosmetic risk
- Injection conditions (regrind %, temperature, cooling time) contribute, but often upstream + handling reveals the defect at the surface

Nothing revolutionary.  
Just quantified.

---

## Project Outputs 📂
This repository contains:
- A dataset ('.csv') with process and handling variables
- A Jupyter Notebook with the full analysis, simulation, and visualizations
- A PDF summary with results and conclusions for non-technical readers

---

## Next Steps 🚀
If you wanted to push this further:
- Tune decision thresholds for scrap vs rework strategies
- Add time-based drift monitoring (humidity seasons, shift effects)
- Compare with non-linear models (tree-based) and explainability tooling
- Integrate into a decision-support workflow for paint shop and internal logistics

But that’s a different conversation.

---

—
Not magic. Just probabilities.  
**Where f(x) meets Kaizen**  
LozanoLsa  
Regards from MX