from exercise_quality.core.body_generator import (
    generate_random_body,
)

for seed in range(10):
    body = generate_random_body(seed)
    print(f"Body {seed + 1}: {body}")
