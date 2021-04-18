import json
import csv
from collections import Counter

dev_set = "dev-v2.0.json"
topic_file = "topics.txt"
answers = "dev_submission.csv"

dev_data = json.load(open(dev_set))['data']

topic_number = 0
topics = {}
f = open(topic_file)
line = f.readline()
while line!='':
	if line.strip() == "Topic {}".format(topic_number+1):
		topic_number+=1
	else:
		topics[line.strip()] = topic_number
	line = f.readline()
		 
answer_list = csv.reader(open(answers), delimiter=',')
answers = [i for i in answer_list][1:]
answers = [{'id':int(i[0]),'predicted':i[1],'confidence':i[-3],'F1':i[-2],'AvNA':i[-1],'actual':','.join(i[2:-3])} for i in answers]


question_topics = {}
num = 1
topic_list = set()
for i in dev_data:
	page = i['title']
	topic_list.add(page)
	for j in i['paragraphs']:
		for k in j['qas']:
			if page in topics:
				question_topics[num] = topics[page]
			elif page+"s" in topics:
				question_topics[num] = topics[page+"s"]
			else:
				question_topics[num] = topics["Sky_UK"]
			num+=1
	
print(Counter(list(question_topics.values())))
print(topic_list)
