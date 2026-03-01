def interpret_score(score):
    if score < 0.3:
        return "Low Developmental Risk"
    elif score < 0.6:
        return "Monitor & Collect More Data"
    else:
        return "Recommend Clinician Review"