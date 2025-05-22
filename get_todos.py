from api import get

PROJECT_ID = "your_project_id_here"
res = get(f"buckets/{PROJECT_ID}/todosets.json")

if res.status_code == 200:
    for ts in res.json():
        print(f"To-do Set: {ts['name']} (ID: {ts['id']})")
else:
    print(f"Error: {res.status_code} - {res.text}")
