def build_explanation(context, recommendation):
    fatigue = context["fatigue_today"]
    alpha = context["alpha"]
    baseline_pace = context["baseline_pace"]

    if fatigue > 0.7:
        decision = "Reduced intensity to avoid overload"
    elif fatigue < 0.3:
        decision = "Increased intensity due to recovery"
    else:
        decision = "Maintained steady aerobic development"

    return {
        "fatigue_score": round(fatigue, 3),
        "baseline_pace": baseline_pace,
        "personalization_strength": round(alpha, 3),
        "decision": decision,
        "model_type": "personalized + global prior",
    }
