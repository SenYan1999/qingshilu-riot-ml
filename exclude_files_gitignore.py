import os

# Define the size threshold (in bytes)
SIZE_LIMIT = 100 * 1024 * 1024  # 100 MB
gitignore_path = ".gitignore"

# Collect files larger than 100MB
large_files = []
for root, _, files in os.walk("."):
    for file in files:
        filepath = os.path.join(root, file)
        try:
            if os.path.getsize(filepath) > SIZE_LIMIT:
                large_files.append(filepath)
        except FileNotFoundError:
            pass  # skip files that might disappear during scanning

# Normalize paths and remove leading './'
large_files = [f[2:] if f.startswith("./") else f for f in large_files]

if large_files:
    with open(gitignore_path, "a") as f:
        f.write("\n# Automatically ignored files >100MB\n")
        for lf in large_files:
            f.write(f"{lf}\n")

    print(f"Added {len(large_files)} file(s) >100MB to {gitignore_path}")
else:
    print("No files larger than 100MB found.")

