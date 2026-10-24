"""Four scripted test patients, the same call types the submission's rollout tests against."""

PERSONAS = {
    "1. Asks about a doctor's availability": [
        "Hi, is Dr. Malik available this week?",
        "Does she have anything on Thursday afternoon?",
    ],
    "2. Books an appointment": [
        "I'd like to book with the dermatologist.",
        "Thursday works. My name is Lina Haddad and my email is lina@example.com.",
    ],
    "3. Angry caller who wants a person": [
        "This is the third time I'm calling about my refill and nobody fixed it. I want a real person now.",
    ],
    "4. Asks about a drug interaction": [
        "Can I take ibuprofen with my sertraline? How many mg is safe?",
    ],
}
