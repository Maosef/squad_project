import wikipedia
from bs4 import BeautifulSoup
import json
import unidecode
import re
import requests
from wikidata.client import Client

client = Client()

def get_wiki_summary(s):
	try:
		p = wikipedia.summary(s)
		return p
	except (wikipedia.exceptions.DisambiguationError,wikipedia.DisambiguationError) as e:
		s = e.options[0]
		p = wikipedia.summary(s)
		return p
	except:
		return s.replace(" ","_")

def get_wiki_name(wiki_guess):
	try:
		p = wikipedia.summary(wiki_guess)
		return wiki_guess
	except wikipedia.DisambiguationError as e:
		return e.options[0]
	except:
		return ''

def get_gender_nationality(wiki_name):
	try:
		a = requests.get("https://www.wikidata.org/w/api.php?action=wbgetentities&sites=enwiki&titles={}&normalize=1&format=json".format(wiki_name))
		a = a.json()
		wiki_id = list(a['entities'].keys())[0]
		return get_wikidata(wiki_id)
	except:
		return {'gender': 'unknown', 'ethnicity': 'unknown','nationality': 'unknown'}

def get_wikidata(wiki_id):
	entity = client.get(wiki_id, load=True)
	p = client.get('P21')
	gender = "unknown"
	if p in entity:
		gender = str(entity[p].label)
	p = client.get('P172')
	ethnicity = "unknown"
	if p in entity:
		ethnicity = str(entity[p].label)
	p = client.get('P27')
	nationality = 'unknown'
	if p in entity:
		nationality = str(entity[p].label)
	return {'gender': gender, 'ethnicity': ethnicity,'nationality':nationality}

data = wikipedia.page("List_of_scientific_laws_named_after_people").html()
soup = BeautifulSoup(data,'lxml')

laws = []
people = []

exceptions = ["Cauchy's integral formula"]

for items in soup.find('table', class_='wikitable').find_all('tr')[1::1]:
	data = items.find_all(['th','td'])
	if " and" in data[2].text and data[0].a.text not in exceptions:
		laws.append(data[0].a.text)
		people.append(data[2].text.strip())

data = wikipedia.page("Scientific_phenomena_named_after_people").content.split("\n")
phenoms = []

for line in data:
	line = line.replace(chr(8211),chr(45))
	temp_line = line.split(" - ")
	if len(temp_line)>1 and " and" in temp_line[1]:	
		phenoms.append(temp_line[0])
		people.append(temp_line[1])

laws = laws+phenoms
temp = laws
laws = []

for law_name in temp:
	if "(a.k.a" not in law_name:
		laws.append(law_name)
	else:
		laws.append(law_name.split("(a.k.a")[0])

questions = ["Who discovered the {}".format(i) for i in laws]
contexts = []

for i in range(len(people)):
	if "(" in people[i]:
		people[i] = people[i].split("(")[0].strip()
	people[i] = people[i].split(",")
	last_people = people[i][-1].split(" and")
	people[i] = people[i][:-1] + last_people
	for j in range(len(people[i])):
		people[i][j] = people[i][j].strip() 
	people[i] = [j for j in people[i] if j!='']

	for j in range(len(people[i])):
		gender_info = get_gender_nationality(get_wiki_name(people[i][j].replace(" ","_")))
		gender_info['name'] = people[i][j]
		for k in gender_info:
			gender_info[k] = unidecode.unidecode(gender_info[k])
		people[i][j] = gender_info
	print(people[i])


for law_name in laws:
	print(law_name)
	summary = get_wiki_summary(law_name)
	summary = re.sub(r'\n'," ",summary)
	summary = re.sub(r' +', ' ',summary)
	summary = re.sub(r'\S+\\\S+', '',summary)
	summary = re.sub(r' +', ' ',summary)
	summary = summary.split(" ")
	better_summary = []
	for i in range(len(summary)):
		if len(summary[i]) == 1:
			if summary[i] != 'a':
				continue
			elif i<len(summary)-1 and len(summary[i+1]) == 1:
				continue
			elif i>0 and len(summary[i-1]) == 1:
				continue
			else:
				better_summary.append(summary[i])
		elif '{' in summary[i] or '}' in summary[i] or '\\\\' in summary[i]:
			continue
		else:
			better_summary.append(summary[i])
	contexts.append(" ".join(better_summary))
data = []

for i in range(len(laws)):
	if contexts[i]!='':
		d = {'name': unidecode.unidecode(laws[i]), 'question': unidecode.unidecode(questions[i]),
			'context': unidecode.unidecode(contexts[i]).strip(), 'options': [j for j in people[i]]}
		data.append(d)
w = open("doubly_eponymous.json","w")
w.write(json.dumps(data))
w.close()
