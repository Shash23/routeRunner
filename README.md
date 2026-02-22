RolledBadger — Load-Aware Route Recommendation System

Overview
RolledBadger is a personalized running route recommendation engine that selects routes based on the physical stress a runner’s body can safely handle on a given day. Instead of prescribing distance or pace, the system predicts how hard a run will feel and chooses terrain that produces the correct training stimulus.

Core Concept
Two runs of the same distance are not equivalent. Elevation, intensity, and recent training history determine perceived effort and injury risk. The system learns how an individual runner responds to training load and solves the inverse problem:

Given a target effort today → choose the route that produces that effort.

System Pipeline

1. Data Ingestion
   User logs in with Strava.
   We collect historical activities:

* distance
* elapsed time
* pace (derived)
* date
* optional heart rate / relative effort

2. Training Load Calculation
   Each run is converted into a training load value representing physical stress.

TrainingLoad = distance × intensity
Intensity ≈ athlete average pace / run pace

3. Fatigue / Readiness Model
   Recent runs matter more than older runs.
   We compute a decayed cumulative load:

Fatigue_today = Σ TrainingLoad_i · exp(-λ · days_since_run)

This estimates the runner’s current recovery state.

4. Personal Effort Model
   We train a personalized regression model:

Effort = f(distance, pace, fatigue)

The output is predicted perceived exertion (1–10 scale).
This represents how hard a run will feel for THIS athlete, not an average runner.

5. Route Evaluation
   Each candidate route has measurable cost:

* distance
* elevation profile

We predict effort for every route using the trained model.

6. Decision (Inverse Optimization)
   We select the route whose predicted effort best matches the desired training zone:

argmin_route | PredictedEffort(route) − TargetEffort |

Output

* Recommended route
* Expected effort level
* Warning if route exceeds safe load

Key Idea
Most fitness apps track what you did.
RolledBadger decides what you should do today based on recovery state.

Technologies
* **Language:** Python 3
* **Core model:** pandas, numpy, scikit-learn (GradientBoostingRegressor), joblib
* **Route builder:** requests, networkx, numpy, polyline, geopy
* **Maps & routing:** OpenStreetMap (Overpass API) for road/way data; Leaflet + OSM tiles for map display; Google-style polyline encoding
* **APIs:** Overpass (OSM), ipapi.co / ip-api.com for IP-based geolocation
* **Data:** Strava-style CSV; pickle/joblib for saved models (e.g. global_model.pkl)
* **Backend (optional):** FastAPI, Strava API (stravalib), SQLAlchemy, python-dotenv, uvicorn

Goal
Keep runners in the adaptation zone instead of the injury zone by prescribing terrain rather than mileage.
