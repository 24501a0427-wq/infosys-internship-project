"""
MILESTONE 3 - RAG Retrieval Evaluation
---------------------------------------
Checks whether the knowledge base retrieves the
expected safety information for predefined questions.
"""

import os

import knowledge_base as kb
from documents import process_file


TESTS = [
    ("Do I need a helmet in the construction zone?", "hardhat"),
    ("When is a high visibility vest required?", "forklifts"),
    ("What eye protection is needed for grinding?", "goggles"),
    ("Which gloves are needed for sheet metal?", "cut-resistant"),
    ("When should a respirator be worn?", "respirator"),
    ("How often are fire drills held?", "three months"),
    ("Where are first aid kits kept?", "first aid kits"),
    ("How fast must an injury be reported?", "24 hours"),
    ("What is the penalty for a first PPE violation?", "verbal warning"),
    ("What compliance rate is the target?", "95 percent"),
]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
manual = os.path.join(
    BASE_DIR,
    "data",
    "sample_safety_manual.txt"
)

# Make sure the safety manual is available in the knowledge base.
kb.add_document(
    process_file(manual),
    "sample_safety_manual.txt"
)


correct = 0

print("\n--- RAG Retrieval Evaluation ---\n")

for question, expected in TESTS:
    results = kb.search(question, top_k=3)

    found = any(
        expected.lower() in result["text"].lower()
        for result in results
    )

    if found:
        correct += 1
        print(f"PASS: {question}")
    else:
        print(f"FAIL: {question}")


accuracy = correct / len(TESTS) * 100

print("\n--------------------------------")
print(f"Retrieval accuracy: {accuracy:.0f}%")
print("Target: 85%")
print("--------------------------------")

if accuracy >= 85:
    print("Evaluation: Target achieved.")
else:
    print("Evaluation: Further retrieval improvement needed.")
