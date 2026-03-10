import os

LABEL_DIR = "dataset_detection/labels"

for root, dirs, files in os.walk(LABEL_DIR):
    for file in files:
        if file.endswith(".txt"):
            path = os.path.join(root, file)

            with open(path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    parts[0] = "0"   # force class id = 0
                    new_lines.append(" ".join(parts) + "\n")

            with open(path, "w") as f:
                f.writelines(new_lines)

print("✅ All labels fixed!")