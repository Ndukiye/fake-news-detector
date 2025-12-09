import json
from app import run_analysis

sample_text = (
    "The government announced new measures to reduce inflation. According to Reuters and BBC, the plan includes tax adjustments and subsidies. "
    "Experts say the claims are consistent with previous reports."
)

result = run_analysis(None, sample_text, None)
print(json.dumps(result, indent=2))
