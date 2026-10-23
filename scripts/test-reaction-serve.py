import urllib.request
print(urllib.request.urlopen("http://127.0.0.1:8100/health", timeout=10).read().decode())
