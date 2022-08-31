import requests
import urllib.parse
def get_iconclass_labels():
	labels = []
	response_m = requests.get("https://iconclass.org/11H(...).jsonld")
	response_f = requests.get("https://iconclass.org/11HH(...).jsonld")
	for urls in response_m.json()['graph'][0]['narrower']:
		label = urllib.parse.unquote(urls['uri']).split("/")[-1]
		labels.append(label)

	for urls in response_f.json()['graph'][0]['narrower']:
		label = urllib.parse.unquote(urls['uri']).split("/")[-1]
		labels.append(label)

	labels = labels[10:192] + labels[208:-6] + ["11F(VIRGIN MARY)"] + ["11F4(MADONNA)"] + ["11D(CHRIST)"]
	return labels