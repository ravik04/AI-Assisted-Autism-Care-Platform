def generate_explanation(output):
    return (
        f"Recommendation: {output['recommendation']}.\n"
        f"Confidence: {output['confidence']:.2f}.\n"
        "Decision derived from temporal behavioral patterns across the video sequence."
    )