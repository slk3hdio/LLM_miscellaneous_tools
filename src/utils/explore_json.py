import json
from pathlib import Path


def find_unknown_system_reminder(data):
    skip_key_words = ['Generate','Input','User','AI','Expected output','API-Request','API descriptions', "{\"name\""]
    known_system_reminders = [
        'The current time is',
        'The current year is',
        # 'The current month is',
        # 'The current day is',
    ]
    for item in data:
        for line in item['instruction'].splitlines():
            if not line:
                continue
            need_output = True
            for skip_word in skip_key_words:
                if line.strip().startswith(skip_word):
                    need_output = False
                    break
            for known_reminder in known_system_reminders:
                if known_reminder in line:
                    need_output = False
                    break
            if need_output:
                print(line)

if __name__ == "__main__":
    for file in Path("./data/API-Bank/test-data").rglob("*.json"):
        level = int(file.name.split('-')[1].split('.')[0])
        if level != 2:
            continue
        
        print(file.name)
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        find_unknown_system_reminder(data)

            
            
            