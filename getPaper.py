import requests
import json
import os

paperId = "8a162e36592c158770ac0d6089f05b950cc7309f"
url = f"https://api.semanticscholar.org/graph/v1/paper/{paperId}"

query_params = {
        'fields': 'title,authors.name,fieldsOfStudy,s2FieldsOfStudy',
}
headers = {
    'x-api-key': "n9k3NKbRXh9Nkl95rkMtW1ZFNakKUa211QPe0E5Z"
}
response = requests.get(url, params=query_params, headers=headers)

print(response)