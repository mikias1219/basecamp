from api import get

def get_todolist_ids(project_id):
    res = get(f"buckets/{project_id}/todolists.json")
    if res.status_code == 200:
        return [(lst['name'], lst['id']) for lst in res.json()]
    else:
        return []
