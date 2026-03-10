import subprocess
import re
import numpy as np

runs = 5
accuracies = []

print("\nRunning multiple trainings...\n")

for i in range(runs):
    print(f"\n===== RUN {i+1} =====\n")

    # run training script
    result = subprocess.run(
        ["python", "train_model.py"],
        capture_output=True,
        text=True
    )

    output = result.stdout

    # find validation accuracy
    matches = re.findall(r"val_accuracy:\s([0-9.]+)", output)

    if matches:
        acc = float(matches[-1])
        accuracies.append(acc)
        print(f"Validation Accuracy: {acc}")

# final statistics
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

print("\n==============================")
print(f"Average Accuracy: {mean_acc:.4f}")
print(f"Std Deviation: ±{std_acc:.4f}")
print("==============================")