choices = [
	{
		"name": "I've never smoked regularly",
		"value": 0
	},
	{
		"name": "I used to smoke",
		"value": 1
	},
	{
		"name": "I currently smoke",
		"value": 2
	}
]


[print(item.get('name')) for item in choices]
