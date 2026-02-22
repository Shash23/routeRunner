# RunSafe — Load-Aware Route Planning

RunSafe is a personalized running route recommendation system that generates routes based on how much physical stress a runner’s body can safely handle today.

Instead of recommending a distance or pace, the system models how hard a run will feel and chooses terrain that produces the correct training stimulus.

---

## Core Idea

Two runs of the same distance are not equally stressful.
Elevation, fatigue, and recent training history determine injury risk and perceived effort.

RunSafe learns how a specific runner responds to training and then solves the inverse problem:

> Given a desired effort level today, what route should the runner take?

---

## What the System Does

1. Learns a personal effort model
   Using historical running data (distance, elevation gain, pace, heart rate, and perceived exertion), the system trains a regression model that predicts how difficult a run will feel.

2. Estimates current fatigue
   Recent training load (last 7 days of effort) is computed to determine readiness.

3. Evaluates possible routes
   Each candidate route has measurable physical cost (distance + elevation profile).

4. Recommends the correct route
   The system selects the route whose predicted effort matches the target training zone (easy, moderate, hard).

---

## Key Principle

This is not a fitness tracker or mileage planner.

It is a decision system:

> Predict effort → match terrain → prevent overload

The goal is to keep the runner in an adaptation zone rather than an injury zone.

---

## Inputs

* Historical running activities (Strava export)
* Perceived exertion (RPE)
* Distance and elevation gain
* Pace / speed
* Recent training load

## Outputs

* Recommended running route
* Expected effort level
* Warning when terrain exceeds safe load

---

## Why This Matters

Current running apps optimize navigation and distance.
RunSafe optimizes biomechanical stress exposure.

It acts as an adaptive training partner that adjusts routes based on recovery state, not just goals.
