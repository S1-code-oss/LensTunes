import os

folders = [
    'music/happy',
    'music/sad',
    'music/angry',
    'music/neutral',
    'music/fear',
    'music/surprise',
    'music/disgust'
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created: {folder}")

