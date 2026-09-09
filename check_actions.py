import urllib.request
import json

url = "https://api.github.com/repos/deser1/NppAI/actions/runs"
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req)
    data = json.loads(response.read().decode('utf-8'))
    
    print("\n--- STATUS GITHUB ACTIONS ---")
    runs = data.get('workflow_runs', [])
    for r in runs[:3]:
        print(f"Workflow: {r['name']}")
        print(f"  Status:     {r['status']}")
        print(f"  Conclusion: {r['conclusion']}")
        print(f"  Commit:     {r['head_commit']['message'][:50]}...")
        print("-" * 30)
except Exception as e:
    print(f"Błąd podczas pobierania: {e}")
